import argparse
import os
os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/matplotlib")

import cv2
import numpy as np
import torch
from ultralytics import YOLO
import supervision as sv
from collections import deque, Counter



PLAYER_CLASS_NAME  = "player"
BALL_CLASS_NAME    = "ball"
REFEREE_CLASS_NAME = "referee"

CONF_THRESHOLD    = 0.40
NMS_THRESHOLD     = 0.45
INFERENCE_IMGSZ   = 1280

MIN_PLAYER_BOX_AREA   = 450
MAX_PLAYERS_PER_FRAME = 26


MIN_TRACK_AGE_FOR_VOTES = 20
MIN_CLEAN_SAMPLES       = 8
MIN_VOTES_TO_LOCK       = 15
CONFIDENCE_TO_LOCK      = 0.82


REID_MAX_MISSING_FRAMES  = 180
REID_MAX_CENTER_DISTANCE = 170
REID_MIN_SCORE           = 0.38
SPATIAL_WEIGHT           = 0.55
IOU_WEIGHT               = 0.30
SIZE_WEIGHT              = 0.15

PRUNE_INTERVAL = 300




def select_device():
    if torch.cuda.is_available():
        return "cuda", True
    return "cpu", False




def bbox_center(bbox):
    x1, y1, x2, y2 = bbox
    return np.array([(x1 + x2) / 2, (y1 + y2) / 2])

def bbox_area(bbox):
    x1, y1, x2, y2 = bbox
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)

def bbox_iou(a, b):
    ix1 = max(a[0], b[0]);  iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2]);  iy2 = min(a[3], b[3])
    inter  = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    area_a = bbox_area(a)
    area_b = bbox_area(b)
    union  = area_a + area_b - inter
    return inter / union if union > 0 else 0.0

def size_similarity(bbox_a, bbox_b):
    a, b = bbox_area(bbox_a), bbox_area(bbox_b)
    if a == 0 and b == 0:
        return 1.0
    return min(a, b) / max(a, b) if max(a, b) > 0 else 0.0




class PlayerState:
    """
    Mutable state for one persistent player ID.

    Phases
    ------
    1  age < MIN_TRACK_AGE_FOR_VOTES  →  identity stabilising, no votes yet
    2  mature, not locked             →  votes accumulating
    3  locked                         →  team frozen, no further sampling
    """

    def __init__(self, stable_id: int):
        self.stable_id   = stable_id
        self.age         = 0
        self.locked      = False
        self.team        = None
        self.votes       = deque(maxlen=40)
        self.clean_count = 0
        self.smooth_color: np.ndarray | None = None

    def observe_color(self, color: np.ndarray, ema_alpha: float = 0.20):
        if color is None:
            return
        if self.smooth_color is None:
            self.smooth_color = color.copy()
        else:
            self.smooth_color = (ema_alpha * color
                                 + (1 - ema_alpha) * self.smooth_color)
        self.clean_count += 1

    def add_vote(self, team_id):
        if team_id is not None:
            self.votes.append(team_id)

    def ready_to_vote(self) -> bool:
        return self.age >= MIN_TRACK_AGE_FOR_VOTES

    def try_lock(self) -> bool:
        if self.locked:
            return False
        if (not self.ready_to_vote()
                or self.clean_count < MIN_CLEAN_SAMPLES
                or len(self.votes) < MIN_VOTES_TO_LOCK):
            return False
        cnt = Counter(self.votes)
        majority, maj_n = cnt.most_common(1)[0]
        if maj_n / len(self.votes) >= CONFIDENCE_TO_LOCK:
            self.team   = majority
            self.locked = True
            return True
        return False

    @property
    def display_team(self):
        if self.locked:
            return self.team
        if self.ready_to_vote() and self.votes:
            return Counter(self.votes).most_common(1)[0][0]
        return None




class StableTracker:
    """
    Maps ByteTrack raw IDs → persistent stable IDs using spatial-only ReID.

    Colour is excluded from ReID to prevent misread colours causing ID swaps.
    """

    def __init__(self):
        self._raw_to_stable: dict[int, int]         = {}
        self._last_bbox:     dict[int, np.ndarray]  = {}
        self._last_seen:     dict[int, int]         = {}
        self._states:        dict[int, PlayerState] = {}
        self._next_id = 1

    def get_or_assign(self, raw_id: int, bbox,
                      frame_count: int, used: set) -> int:
        if raw_id in self._raw_to_stable:
            sid = self._raw_to_stable[raw_id]
            if sid not in used:
                return sid
        return self._reid(raw_id, bbox, frame_count, used)

    def state(self, stable_id: int) -> PlayerState:
        if stable_id not in self._states:
            self._states[stable_id] = PlayerState(stable_id)
        return self._states[stable_id]

    def commit(self, raw_id: int, stable_id: int, bbox, frame_count: int):
        self._raw_to_stable[raw_id] = stable_id
        self._last_bbox[stable_id]  = bbox
        self._last_seen[stable_id]  = frame_count

    def prune(self, frame_count: int):
        stale  = [sid for sid, last in self._last_seen.items()
                  if frame_count - last > REID_MAX_MISSING_FRAMES]
        pruned = set(stale)
        for sid in stale:
            self._last_bbox.pop(sid, None)
            self._last_seen.pop(sid, None)
        dead = [r for r, s in self._raw_to_stable.items() if s in pruned]
        for r in dead:
            del self._raw_to_stable[r]

    def _alloc(self) -> int:
        sid = self._next_id
        self._next_id += 1
        return sid

    def _reid(self, raw_id: int, bbox, frame_count: int, used: set) -> int:
        center     = bbox_center(bbox)
        best_sid   = None
        best_score = 0.0

        for sid, last_bbox in self._last_bbox.items():
            if sid in used:
                continue
            gap = frame_count - self._last_seen.get(sid, 0)
            if gap > REID_MAX_MISSING_FRAMES:
                continue
            dist = np.linalg.norm(center - bbox_center(last_bbox))
            if dist > REID_MAX_CENTER_DISTANCE:
                continue

            spatial_score = 1.0 - dist / REID_MAX_CENTER_DISTANCE
            iou_score     = bbox_iou(bbox, last_bbox)
            size_score    = size_similarity(bbox, last_bbox)

            score = (SPATIAL_WEIGHT * spatial_score
                     + IOU_WEIGHT   * iou_score
                     + SIZE_WEIGHT  * size_score)

            if score > best_score:
                best_score = score
                best_sid   = sid

        if best_sid is not None and best_score >= REID_MIN_SCORE:
            self._raw_to_stable[raw_id] = best_sid
            return best_sid

        sid = self._alloc()
        self._raw_to_stable[raw_id] = sid
        return sid




def build_model(model_path: str):
    device, use_half = select_device()
    model = YOLO(model_path)
    model.to(device)
    if use_half:
        model.half()
    return model, device

def build_bytetrack():
    return sv.ByteTrack(
        track_activation_threshold=0.45,
        lost_track_buffer=180,
        minimum_matching_threshold=0.80,
        frame_rate=30,
    )

def get_class_ids(class_names, names_set):
    src = class_names.items() if isinstance(class_names, dict) \
          else enumerate(class_names)
    return {cid for cid, n in src if n in names_set}

def filter_players(detections, player_class_ids):
    if detections.is_empty():
        return detections
    keep  = np.isin(detections.class_id, list(player_class_ids))
    keep &= detections.box_area >= MIN_PLAYER_BOX_AREA
    det   = detections[keep]
    if det.is_empty():
        return det
    det = det.with_nms(threshold=NMS_THRESHOLD, class_agnostic=True)
    if len(det) > MAX_PLAYERS_PER_FRAME:
        best = np.argsort(det.confidence)[-MAX_PLAYERS_PER_FRAME:]
        det  = det[best]
    return det




def run_tracker_standalone(model_path: str, video_path: str,
                            max_frames: int = 300):
    """
    Run tracking only and print stable ID assignments.
    No team classification — shows raw tracking output.
    """
    model, _ = build_model(model_path)
    byte_tracker = build_bytetrack()
    stable_tracker = StableTracker()

    player_class_ids = get_class_ids(model.names, {PLAYER_CLASS_NAME})
    if not player_class_ids:
        raise RuntimeError(f"Class '{PLAYER_CLASS_NAME}' not found.")

    cap = cv2.VideoCapture(video_path)
    frame_count = 0

    print(f"\n{'Frame':>6}  {'Stable IDs':}")
    print("-" * 50)

    while frame_count < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1

        res  = model(frame, conf=CONF_THRESHOLD, imgsz=INFERENCE_IMGSZ,
                     verbose=False)[0]
        dets = sv.Detections.from_ultralytics(res)
        dets = dets.with_nms(threshold=NMS_THRESHOLD, class_agnostic=False)
        player_dets = filter_players(dets, player_class_ids)
        tracked = byte_tracker.update_with_detections(player_dets)

        if frame_count % PRUNE_INTERVAL == 0:
            stable_tracker.prune(frame_count)

        frame_ids    = []
        used_sids    = set()

        for i in range(len(tracked)):
            raw_tid = tracked.tracker_id[i]
            if raw_tid is None:
                continue
            raw_tid = int(raw_tid)
            bbox    = tracked.xyxy[i]
            sid     = stable_tracker.get_or_assign(
                raw_tid, bbox, frame_count, used_sids)
            used_sids.add(sid)
            stable_tracker.commit(raw_tid, sid, bbox, frame_count)
            st = stable_tracker.state(sid)
            st.age += 1
            phase = ("maturing" if not st.ready_to_vote()
                     else "voting" if not st.locked
                     else "locked")
            frame_ids.append(f"#{sid}({phase})")

        id_str = "  ".join(frame_ids) if frame_ids else "(none)"
        print(f"{frame_count:>6}  {id_str}")

    cap.release()
    print(f"\nDone. Processed {frame_count} frames.")




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tracker standalone test")
    parser.add_argument("--model", default="trained_YOLO.pt")
    parser.add_argument("--video", default="sample.mp4")
    parser.add_argument("--frames", type=int, default=300,
                        help="Max frames to process")
    args = parser.parse_args()

    run_tracker_standalone(args.model, args.video, args.frames)
