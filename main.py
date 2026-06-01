import argparse
import os
os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/matplotlib")
import supervision as sv
import cv2
import numpy as np
from collections import Counter


from player_tracking_id import (
    StableTracker,
    PlayerState,
    build_model,
    build_bytetrack,
    get_class_ids,
    filter_players,
    MIN_TRACK_AGE_FOR_VOTES,
    PRUNE_INTERVAL,
    PLAYER_CLASS_NAME,
    BALL_CLASS_NAME,
    REFEREE_CLASS_NAME,
    CONF_THRESHOLD,
    NMS_THRESHOLD,
    INFERENCE_IMGSZ,
)

from classifier import (
    extract_jersey_color,
    collect_color_samples,
    train_team_model,
    predict_team,
    team_box_color,
    boxes_overlap,
    OVERLAP_IOU_THRESHOLD,
    MIN_COLOR_SAMPLES,
)


DEFAULT_MODEL  = "trained_YOLO.pt"
DEFAULT_VIDEO  = "sample.mp4"
DEFAULT_OUTPUT = "football_out.mp4"

COLOR_REFRESH_FRAMES = 5    



def run_pipeline(model_path: str, video_path: str, output_path: str):
    
    print("Loading model...")
    model, _ = build_model(model_path)
    byte_tracker = build_bytetrack()
    class_names = model.names
    print("Classes:", class_names)

    player_class_ids = get_class_ids(class_names, {PLAYER_CLASS_NAME})
    if not player_class_ids:
        raise RuntimeError(f"Class '{PLAYER_CLASS_NAME}' not found in {class_names}")

    
    print("Collecting jersey colour samples for team model...")
    samples = collect_color_samples(model, video_path, player_class_ids,
                                    max_frames=800)
    if len(samples) < 2:
        raise RuntimeError("Not enough colour samples to build team model.")
    print(f"  Collected {len(samples)} samples.")

    
    team_model = train_team_model(samples)
    for i, c in enumerate(team_model.cluster_centers_):
        r, g, b = c.astype(int)
        print(f"  Team {i + 1} cluster centre → R={r:3d}  G={g:3d}  B={b:3d}")

    
    print("Processing video...")
    stable_tracker = StableTracker()
    color_cache: dict[int, tuple[int, np.ndarray | None]] = {}

    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
    W   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(output_path,
                          cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))
    frame_count = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_count += 1
            if frame_count % 150 == 0:
                print(f"  Frame {frame_count}")

            
            res  = model(frame, conf=CONF_THRESHOLD, imgsz=INFERENCE_IMGSZ,
                         verbose=False)[0]
            dets = sv.Detections.from_ultralytics(res)
            dets = dets.with_nms(threshold=NMS_THRESHOLD, class_agnostic=False)

            player_dets = filter_players(dets, player_class_ids)
            tracked     = byte_tracker.update_with_detections(player_dets)

            if frame_count % PRUNE_INTERVAL == 0:
                stable_tracker.prune(frame_count)

            annotated       = frame.copy()
            active_counts   = Counter()
            active_ids      = set()
            used_stable_ids = set()

            all_player_bboxes = [tracked.xyxy[i] for i in range(len(tracked))]

            
            for i in range(len(tracked)):
                raw_tid = tracked.tracker_id[i]
                if raw_tid is None:
                    continue
                raw_tid  = int(raw_tid)
                bbox     = tracked.xyxy[i]
                class_id = tracked.class_id[i]

                
                sid = stable_tracker.get_or_assign(
                    raw_tid, bbox, frame_count, used_stable_ids)
                used_stable_ids.add(sid)
                stable_tracker.commit(raw_tid, sid, bbox, frame_count)

                st = stable_tracker.state(sid)
                st.age += 1

                if class_id not in player_class_ids:
                    continue

                x1, y1, x2, y2 = map(int, bbox)
                overlapping = boxes_overlap(bbox, all_player_bboxes)

                
                if (st.ready_to_vote()
                        and not st.locked
                        and not overlapping):
                    last_f, _ = color_cache.get(sid, (0, None))
                    if (frame_count - last_f) >= COLOR_REFRESH_FRAMES:
                        new_color = extract_jersey_color(frame, bbox)
                        color_cache[sid] = (frame_count, new_color)
                        if new_color is not None:
                            st.observe_color(new_color)
                            vote = predict_team(team_model, new_color)
                            st.add_vote(vote)
                            if st.try_lock():
                                print(f"  Locked #{sid} → Team {st.team + 1} "
                                      f"(age={st.age}, clean={st.clean_count})")

                
                display_team = st.display_team
                confirmed    = st.locked

                if display_team is not None:
                    box_color = team_box_color(display_team)
                    suffix    = "" if confirmed else "?"
                    label     = f"#{sid} T{display_team + 1}{suffix}"
                    active_counts[display_team] += 1
                else:
                    box_color = (160, 160, 160)
                    label     = f"#{sid} ..."

                active_ids.add(sid)
                thickness = 3 if confirmed else 2
                cv2.rectangle(annotated, (x1, y1), (x2, y2), box_color, thickness)
                cv2.putText(annotated, label, (x1, max(25, y1 - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.60, box_color, 2)

            
            for i in range(len(dets)):
                cid  = int(dets.class_id[i])
                name = (class_names[cid] if isinstance(class_names, dict)
                        else class_names[cid])
                if name not in {BALL_CLASS_NAME, REFEREE_CLASS_NAME}:
                    continue
                x1, y1, x2, y2 = map(int, dets.xyxy[i])
                is_ball   = name == BALL_CLASS_NAME
                label     = "Ball" if is_ball else "Ref"
                box_color = (255, 255, 255) if is_ball else (0, 220, 220)
                cv2.rectangle(annotated, (x1, y1), (x2, y2), box_color, 2)
                cv2.putText(annotated, label, (x1, max(25, y1 - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.60, box_color, 2)

            
            locked_count   = sum(1 for s in stable_tracker._states.values()
                                 if s.locked)
            maturing_count = sum(1 for s in stable_tracker._states.values()
                                 if not s.locked
                                 and s.age < MIN_TRACK_AGE_FOR_VOTES)
            hud = (f"Players: {len(active_ids)}  |  "
                   f"T1: {active_counts[0]}  T2: {active_counts[1]}  |  "
                   f"Locked: {locked_count}  Maturing: {maturing_count}")
            cv2.rectangle(annotated, (10, 10), (570, 46), (0, 0, 0), -1)
            cv2.putText(annotated, hud, (20, 36),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 2)

            out.write(annotated)

    finally:
        cap.release()
        out.release()
        print(f"\n✅ Saved: {output_path}")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Football video analysis — tracking + team classification")
    parser.add_argument("--model",  default=DEFAULT_MODEL,
                        help="Path to YOLO .pt weights (default: trained_YOLO.pt)")
    parser.add_argument("--video",  default=DEFAULT_VIDEO,
                        help="Input video path (default: sample.mp4)")
    parser.add_argument("--output", default=DEFAULT_OUTPUT,
                        help="Output video path (default: football_out.mp4)")
    args = parser.parse_args()

    run_pipeline(args.model, args.video, args.output)
