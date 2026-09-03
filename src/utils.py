"""
Shared utilities for the InHome Physical Therapy Pose Estimation System.

Centralizing these functions ensures that the offline evaluation (MMFi
dataset) and the online/live scoring (webcam) use *exactly* the same
geometry and scoring logic, so the two are actually comparable.

This file lives in <PROJECT_ROOT>/src/utils.py
"""
import os
import numpy as np

# --- Project structure -----------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_ROOT = os.path.join(PROJECT_ROOT, 'dataset', 'MMFi_Dataset')
MODEL_PATH = os.path.join(PROJECT_ROOT, 'yolov8n-pose.pt')
LOG_DIR = os.path.join(PROJECT_ROOT, 'logs')


def get_reference_path(action_name):
    """Build a path for an action-specific reference curve, e.g.
    get_reference_path('lunge') -> PROJECT_ROOT/lunge_reference.npy
    Lets different actions (squat, lunge, ...) keep separate reference
    files without overwriting each other."""
    return os.path.join(PROJECT_ROOT, f'{action_name}_reference.npy')


REFERENCE_PATH = get_reference_path('squat')  # kept for backward compatibility with the squat scripts

# --- COCO keypoint indices (same format used by YOLOv8-Pose and MMFi) ------
LEFT_HIP, LEFT_KNEE, LEFT_ANKLE = 11, 13, 15
RIGHT_HIP, RIGHT_KNEE, RIGHT_ANKLE = 12, 14, 16

# Upper-limb joints (needed for arm movements such as A07/A08 "limb
# extension" -- angle measured at the elbow, between shoulder-elbow-wrist).
LEFT_SHOULDER, LEFT_ELBOW, LEFT_WRIST = 5, 7, 9
RIGHT_SHOULDER, RIGHT_ELBOW, RIGHT_WRIST = 6, 8, 10

# --- Scoring parameters (shared between offline and online evaluation) -----
DEPTH_TOLERANCE = 10.0        # "good zone": within this range, score stays high (80-100%)
DEPTH_FALLOFF_RANGE = 30.0    # beyond the tolerance, score decreases gradually to 0%
KP_CONF_THRESHOLD = 0.5       # minimum keypoint confidence to be considered "detected"


def calculate_angle(a, b, c):
    """Angle in degrees at vertex b, between segments a-b and b-c."""
    a, b, c = np.array(a, dtype=float), np.array(b, dtype=float), np.array(c, dtype=float)
    ba = a - b
    bc = c - b
    cos_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-8)
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    return np.degrees(np.arccos(cos_angle))


def calculate_depth_score(depth_diff, tolerance=DEPTH_TOLERANCE, falloff=DEPTH_FALLOFF_RANGE):
    """Two-zone scoring: high plateau within tolerance, gradual falloff beyond it.

    - Within `tolerance`: score decreases linearly from 100% (perfect) to 80%
      (at the edge of the tolerance zone).
    - Beyond it: score decreases from 80% down to 0% over `falloff` degrees.

    Using the SAME function offline (MMFi evaluation) and online (webcam)
    keeps the two scoring pipelines directly comparable.
    """
    if depth_diff <= tolerance:
        return 100.0 - (depth_diff / tolerance) * 20.0
    else:
        extra = depth_diff - tolerance
        score = 80.0 - (extra / falloff) * 80.0
        return max(0.0, score)


def keypoints_are_valid(kp_xy, kp_conf, indices, threshold=KP_CONF_THRESHOLD):
    """
    Check that all keypoints in `indices` are usable.

    If a per-keypoint confidence array is available (e.g. from YOLO), require
    conf >= threshold for every required keypoint. Otherwise fall back to the
    weaker "not all-zero" check (used e.g. for MMFi data, which may not
    expose a confidence score).
    """
    if kp_conf is not None:
        return all(kp_conf[i] >= threshold for i in indices)
    return all(np.any(kp_xy[i]) for i in indices)


def select_patient_keypoints(results, previous_center=None, conf_threshold=KP_CONF_THRESHOLD):
    """
    Decide which detected person (if several are in frame) is the patient,
    from a single Ultralytics YOLO-Pose `results` object (already run on one
    frame).

    Strategy:
      - If we know where the patient was in the previous frame, pick the
        detected person whose bounding-box center is closest to it
        (temporal continuity -> avoids "jumping" between people frame to
        frame, e.g. when a caregiver enters/leaves the frame).
      - Otherwise (first detection, or track lost) pick the person with the
        largest bounding box, assuming the patient is the one closest to the
        camera in a typical in-home therapy setup.

    Returns (kp_xy, kp_conf, center) for the selected person, or
    (None, None, None) if nobody was detected.
    """
    if results[0].keypoints is None or len(results[0].keypoints.xy) == 0:
        return None, None, None

    boxes = results[0].boxes.xyxy.cpu().numpy()             # (N, 4)
    all_kp = results[0].keypoints.xy.cpu().numpy()           # (N, 17, 2)
    all_conf = (results[0].keypoints.conf.cpu().numpy()
                if results[0].keypoints.conf is not None else None)  # (N, 17) or None

    n_people = len(boxes)
    if n_people == 0:
        return None, None, None

    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    centers = np.stack([(boxes[:, 0] + boxes[:, 2]) / 2.0,
                         (boxes[:, 1] + boxes[:, 3]) / 2.0], axis=1)

    if previous_center is not None:
        dists = np.linalg.norm(centers - previous_center, axis=1)
        idx = int(np.argmin(dists))
    else:
        idx = int(np.argmax(areas))

    selected_kp = all_kp[idx]
    selected_conf = all_conf[idx] if all_conf is not None else None
    return selected_kp, selected_conf, centers[idx]