"""
src/evaluate_dataset/extract_keypoints_from_video.py
Extract the 2D pose sequence and confidence scores from a single video using the
same YOLO-based patient selection logic as the live webcam pipeline. The output
is stored under the evaluation dataset folder and can be used by the trainer or
by the evaluation scripts that expect a consistent per-frame keypoint format.
"""
import sys
import os

# --- Path anchoring -----------------------------------------------------
# This script lives in src/evaluate_dataset/, two levels below the project
# root, but needs to import utils.py from src/. Adding src/ to sys.path
# explicitly (instead of relying on the current working directory) means
# this script works no matter where it's launched from -- 
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

# MMFi sequences are captured at roughly 10 fps, while webcam or phone videos are
# typically much faster. We skip frames to approximate the temporal sampling used
# during model training, so the extracted sequence has a similar motion density.
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

    # Load the detection model once and open the source video for frame-by-frame
    # processing.
    model = YOLO(MODEL_PATH)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    video_fps = cap.get(cv2.CAP_PROP_FPS) or MMFI_APPROX_FPS
    frame_skip = max(1, round(video_fps / MMFI_APPROX_FPS))
    print(f"Video fps={video_fps:.1f} -> frame_skip={frame_skip} "
          f"(effective extraction rate ~{video_fps / frame_skip:.1f}fps)")

    # Store the valid pose detections over time; these will later be saved as the
    # evaluation sequence for this subject and exercise.
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

    # Save the extracted sequence in the same structure expected by the evaluation
    # and training pipelines.
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