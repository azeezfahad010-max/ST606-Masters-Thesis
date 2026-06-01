import argparse
import os
os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/matplotlib")

import cv2
import numpy as np
from ultralytics import YOLO
import supervision as sv
from sklearn.cluster import KMeans



PLAYER_CLASS_NAME = "player"

CONF_THRESHOLD    = 0.40
NMS_THRESHOLD     = 0.45
INFERENCE_IMGSZ   = 1280

MIN_PLAYER_BOX_AREA   = 450
MAX_PLAYERS_PER_FRAME = 26

MIN_COLOR_SAMPLES    = 120
TEAM_DISTANCE_MARGIN = 1.15   

OVERLAP_IOU_THRESHOLD = 0.08
MIN_CROP_PIXELS       = 30



def bbox_area(bbox):
    x1, y1, x2, y2 = bbox
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)

def bbox_iou(a, b):
    ix1 = max(a[0], b[0]);  iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2]);  iy2 = min(a[3], b[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    union = bbox_area(a) + bbox_area(b) - inter
    return inter / union if union > 0 else 0.0

def boxes_overlap(bbox, all_bboxes, threshold=OVERLAP_IOU_THRESHOLD):
    for other in all_bboxes:
        if np.array_equal(other, bbox):
            continue
        if bbox_iou(bbox, other) > threshold:
            return True
    return False




def extract_jersey_color(frame: np.ndarray, bbox) -> np.ndarray | None:
    """
    Extract the dominant jersey colour from the torso region of a bounding box.

    Uses HSV space for illumination robustness; returns RGB float32 [R, G, B].
    Returns None if the crop is too small or lacks sufficient saturated pixels.
    """
    x1, y1, x2, y2 = map(int, bbox)
    x1 = max(0, x1);  y1 = max(0, y1)
    x2 = min(frame.shape[1], x2);  y2 = min(frame.shape[0], y2)

    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return None

    h, w = crop.shape[:2]

    
    upper = crop[int(0.18 * h):int(0.52 * h),
                 int(0.25 * w):int(0.75 * w)]
    if upper.size == 0:
        return None

    hsv    = cv2.cvtColor(upper, cv2.COLOR_BGR2HSV)
    pixels = hsv.reshape(-1, 3).astype(np.float32)

    
    mask   = (pixels[:, 1] > 30) & (pixels[:, 2] > 40) & (pixels[:, 2] < 245)
    pixels = pixels[mask]

    if len(pixels) < MIN_CROP_PIXELS:
        return None

    n  = min(2, len(pixels))
    km = KMeans(n_clusters=n, n_init=10, random_state=42)
    km.fit(pixels)
    counts   = np.bincount(km.labels_)
    best_hsv = km.cluster_centers_[np.argmax(counts)]

    
    hsv_pixel = np.array([[best_hsv]], dtype=np.float32)
    hsv_u8    = np.clip(hsv_pixel, [0, 0, 0], [179, 255, 255]).astype(np.uint8)
    rgb_u8    = cv2.cvtColor(hsv_u8, cv2.COLOR_HSV2RGB)
    return rgb_u8[0, 0].astype(np.float32)



def train_team_model(samples: list) -> KMeans:
    """Fit a 2-cluster KMeans on a list of RGB colour samples."""
    km = KMeans(n_clusters=2, n_init=30, random_state=42)
    km.fit(samples)
    return km

def predict_team(team_model: KMeans, color: np.ndarray) -> int | None:
    """
    Returns 0 or 1 for the predicted team, or None if the prediction is
    too ambiguous (distance ratio below TEAM_DISTANCE_MARGIN).
    """
    dists = team_model.transform([color])[0]
    order = np.argsort(dists)
    if len(order) > 1:
        ratio = dists[order[1]] / max(dists[order[0]], 1e-6)
        if ratio < TEAM_DISTANCE_MARGIN:
            return None
    return int(order[0])

def team_box_color(team_id: int) -> tuple:
    """Returns a BGR draw colour for each team."""
    return {0: (220, 50, 50), 1: (50, 50, 220)}.get(team_id, (0, 200, 0))




def collect_color_samples(model: YOLO, video_path: str,
                           player_class_ids: set,
                           max_frames: int = 800) -> list:
    """
    First-pass colour harvest: run inference over up to max_frames frames,
    collecting one RGB sample per non-occluded player box.
    Stops early once MIN_COLOR_SAMPLES are gathered.
    """
    samples = []
    cap = cv2.VideoCapture(video_path)

    for _ in range(max_frames):
        ret, frame = cap.read()
        if not ret:
            break

        res  = model(frame, conf=CONF_THRESHOLD, imgsz=INFERENCE_IMGSZ,
                     verbose=False)[0]
        dets = sv.Detections.from_ultralytics(res)

        if dets.is_empty():
            continue

        keep  = np.isin(dets.class_id, list(player_class_ids))
        keep &= dets.box_area >= MIN_PLAYER_BOX_AREA
        dets  = dets[keep]
        if dets.is_empty():
            continue

        dets   = dets.with_nms(threshold=NMS_THRESHOLD, class_agnostic=True)
        bboxes = list(dets.xyxy)

        for bbox in bboxes:
            if boxes_overlap(bbox, bboxes):
                continue
            c = extract_jersey_color(frame, bbox)
            if c is not None:
                samples.append(c)

        if len(samples) >= MIN_COLOR_SAMPLES:
            break

    cap.release()
    return samples




def save_cluster_swatch(team_model: KMeans, out_path: str = "team_clusters.png"):
    """Save a simple colour swatch PNG showing the two team colours."""
    swatch_w, swatch_h = 300, 150
    img = np.zeros((swatch_h, swatch_w * 2, 3), dtype=np.uint8)

    for team_id in range(2):
        rgb = team_model.cluster_centers_[team_id].astype(np.uint8)
        bgr = rgb[::-1]  # OpenCV uses BGR
        img[:, team_id * swatch_w:(team_id + 1) * swatch_w] = bgr
        label = f"Team {team_id + 1}"
        cv2.putText(img, label,
                    (team_id * swatch_w + 20, swatch_h // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)

    cv2.imwrite(out_path, img)
    print(f"Swatch saved: {out_path}")




def run_classifier_standalone(model_path: str, video_path: str,
                               swatch_path: str = "team_clusters.png"):
    """
    Standalone run: collect colour samples, fit team model, print centres,
    and save a visual swatch.
    """
    from player_tracking_id import build_model, get_class_ids  # local import to avoid circular

    model, _ = build_model(model_path)
    player_class_ids = get_class_ids(model.names, {PLAYER_CLASS_NAME})
    if not player_class_ids:
        raise RuntimeError(f"Class '{PLAYER_CLASS_NAME}' not found.")

    print("Collecting jersey colour samples...")
    samples = collect_color_samples(model, video_path, player_class_ids)

    if len(samples) < 2:
        raise RuntimeError("Not enough colour samples to build team model.")

    print(f"Collected {len(samples)} samples. Fitting KMeans(k=2)...")
    team_model = train_team_model(samples)

    for i, centre in enumerate(team_model.cluster_centers_):
        r, g, b = centre.astype(int)
        print(f"  Team {i + 1} centre → R={r:3d}  G={g:3d}  B={b:3d}")

    save_cluster_swatch(team_model, swatch_path)
    return team_model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Classifier standalone test")
    parser.add_argument("--model",  default="trained_YOLO.pt")
    parser.add_argument("--video",  default="sample.mp4")
    parser.add_argument("--swatch", default="team_clusters.png")
    args = parser.parse_args()

    run_classifier_standalone(args.model, args.video, args.swatch)
