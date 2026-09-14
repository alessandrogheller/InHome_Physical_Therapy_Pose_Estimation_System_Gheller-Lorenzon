"""
src/evaluate_dataset/extract_keypoints_from_video.py

Extracts a raw (T, 17, 2) keypoint sequence + (T, 17) confidence sequence
from a single video file, using the SAME YOLO model and patient-selection
logic as the live webcam scripts (utils.select_patient_keypoints), so the
extracted data has the same statistical "flavor" as what the network sees
live -- not MMFi's ResNet-48-based keypoints.

Usage:
    python extract_keypoints_from_video.py <video_path> <subject_id> <exercise_name>

exercise_name must be one of the 6 classes used in build_windowed_dataset.py:
    squat, lunge_left, lunge_right, limb_extension_left,
    limb_extension_right, jumping_jacks

Output is written to <PROJECT_ROOT>/eval_dataset/<subject_id>/<exercise_name>/,
regardless of where the source video lives on disk.
"""
import sys
import os

# --- Path anchoring -----------------------------------------------------
# This script lives in src/evaluate_dataset/, two levels below the project
# root, but needs to import utils.py from src/. Adding src/ to sys.path
# explicitly (instead of relying on the current working directory) means
# this script works no matter where it's launched from -- same fix already
# applied to the other cross-folder scripts (see reference_extraction_*.py).
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_SRC_DIR = os.path.dirname(_THIS_DIR)
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

import cv2
import numpy as np
from ultralytics import YOLO

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import MODEL_PATH, PROJECT_ROOT, select_patient_keypoints

EVAL_DATASET_ROOT = os.path.join(PROJECT_ROOT, 'eval_dataset')

# MMFi's rgb sequences are ~297 frames over ~30s -> ~10fps. Videos recorded
# with a phone/webcam are almost always higher fps, so we downsample by
# frame-skipping to roughly match the temporal density the network (and its
# WINDOW_LENGTH/WINDOW_STRIDE, defined in frames not seconds) was trained
# on -- same rationale as FRAME_SKIP in realtime_inference_action_quality.py.
MMFI_APPROX_FPS = 10.0

VALID_EXERCISES = {
    'squat', 'lunge_left', 'lunge_right',
    'limb_extension_left', 'limb_extension_right', 'jumping_jacks',
}


def extract(video_path, subject_id, exercise_name):
    if exercise_name not in VALID_EXERCISES:
        raise ValueError(f"Unknown exercise '{exercise_name}'. Must be one of {sorted(VALID_EXERCISES)}")

    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")

    model = YOLO(MODEL_PATH)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    video_fps = cap.get(cv2.CAP_PROP_FPS) or MMFI_APPROX_FPS
    frame_skip = max(1, round(video_fps / MMFI_APPROX_FPS))
    print(f"Video fps={video_fps:.1f} -> frame_skip={frame_skip} "
          f"(effective extraction rate ~{video_fps / frame_skip:.1f}fps)")

    keypoints_seq = []
    conf_seq = []
    patient_center = None
    frame_idx = 0
    n_missed = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % frame_skip == 0:
            results = model(frame, verbose=False)
            kp, kp_conf, new_center = select_patient_keypoints(results, previous_center=patient_center)
            if kp is not None:
                patient_center = new_center
                keypoints_seq.append(kp)
                # keep a (17,) confidence array even if YOLO didn't return one,
                # so downstream code (keypoints_are_valid) always gets a consistent shape
                conf_seq.append(kp_conf if kp_conf is not None else np.ones(17))
            else:
                n_missed += 1

        frame_idx += 1

    cap.release()

    if len(keypoints_seq) == 0:
        raise RuntimeError(f"No person detected in any frame of {video_path}.")

    keypoints_seq = np.stack(keypoints_seq, axis=0)  # (T, 17, 2)
    conf_seq = np.stack(conf_seq, axis=0)            # (T, 17)

    out_dir = os.path.join(EVAL_DATASET_ROOT, subject_id, exercise_name)
    os.makedirs(out_dir, exist_ok=True)
    np.save(os.path.join(out_dir, 'keypoints.npy'), keypoints_seq)
    np.save(os.path.join(out_dir, 'confidences.npy'), conf_seq)

    print(f"Saved {keypoints_seq.shape[0]} frames (skipped {n_missed} with no detection) to {out_dir}")


if __name__ == '__main__':
    if len(sys.argv) != 4:
        print("Usage: python extract_keypoints_from_video.py <video_path> <subject_id> <exercise_name>")
        sys.exit(1)
    extract(sys.argv[1], sys.argv[2], sys.argv[3])