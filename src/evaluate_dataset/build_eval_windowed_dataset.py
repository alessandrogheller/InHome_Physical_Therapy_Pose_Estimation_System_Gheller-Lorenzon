"""
src/evaluate_dataset/build_eval_windowed_dataset.py
Same windowing/normalization logic as build_windowed_dataset.py, but reads
from eval_dataset/ (new subjects, extracted via extract_keypoints_from_video.py)
instead of MMFi, and produces a single (unsplit) set since this is
evaluation-only data.
"""
import os
import sys
import csv
import numpy as np

# --- Path anchoring -------------------------------------------------
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_SRC_DIR = os.path.dirname(_THIS_DIR)
_NN_DIR = os.path.join(_SRC_DIR, 'neural_network')
for _p in (_SRC_DIR, _NN_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)
        
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import PROJECT_ROOT
from keypoint_normalize import normalize_sequence, flatten_sequence, FLAT_SIZE

EVAL_DATASET_ROOT = os.path.join(PROJECT_ROOT, 'eval_dataset')
EVAL_LABELS_CSV = os.path.join(PROJECT_ROOT, 'eval_quality_labels.csv')
OUTPUT_NPZ = os.path.join(PROJECT_ROOT, 'eval_action_quality_dataset.npz')

# Must match build_windowed_dataset.py exactly -- same class index mapping
# the trained checkpoints expect.
CLASS_NAMES = ['squat', 'lunge_left', 'lunge_right',
               'limb_extension_left', 'limb_extension_right', 'jumping_jacks']
WINDOW_LENGTH = 30
WINDOW_STRIDE = 10


def load_labels(csv_path):
    labels = {}
    with open(csv_path, newline='') as f:
        for row in csv.DictReader(f):
            labels[(row['subject'], row['action'])] = float(row['score'])
    return labels


def make_windows(seq_flat, length, stride):
    T = seq_flat.shape[0]
    if T < length:
        return np.zeros((0, length, seq_flat.shape[1]), dtype=np.float32)
    windows = [seq_flat[s:s + length] for s in range(0, T - length + 1, stride)]
    return np.stack(windows, axis=0).astype(np.float32)


def main():
    if not os.path.exists(EVAL_LABELS_CSV):
        print(f"ERROR: {EVAL_LABELS_CSV} not found. Run score_eval_subjects.py first.")
        sys.exit(1)
    labels = load_labels(EVAL_LABELS_CSV)

    X_all, y_class_all, y_score_all, subject_all = [], [], [], []

    for subject_id in sorted(os.listdir(EVAL_DATASET_ROOT)):
        subject_dir = os.path.join(EVAL_DATASET_ROOT, subject_id)
        if not os.path.isdir(subject_dir):
            continue
        for exercise_name in sorted(os.listdir(subject_dir)):
            key = (subject_id, exercise_name)
            if key not in labels:
                continue
            kp_path = os.path.join(subject_dir, exercise_name, 'keypoints.npy')
            if not os.path.exists(kp_path):
                continue

            raw_seq = np.load(kp_path)
            normalized = normalize_sequence(raw_seq)  # no per-keypoint conf needed here
            if normalized.shape[0] < WINDOW_LENGTH:
                print(f"  {subject_id}/{exercise_name}: only {normalized.shape[0]} usable frames, skipping.")
                continue

            flat = flatten_sequence(normalized)
            windows = make_windows(flat, WINDOW_LENGTH, WINDOW_STRIDE)
            if windows.shape[0] == 0:
                continue

            class_idx = CLASS_NAMES.index(exercise_name)
            score = labels[key]

            X_all.append(windows)
            y_class_all.append(np.full(windows.shape[0], class_idx, dtype=np.int64))
            y_score_all.append(np.full(windows.shape[0], score, dtype=np.float32))
            subject_all.extend([subject_id] * windows.shape[0])

            print(f"  {subject_id}/{exercise_name}: {windows.shape[0]} windows (score={score:.1f}%)")

    if not X_all:
        print("No eval windows produced. Aborting.")
        sys.exit(1)

    X = np.concatenate(X_all, axis=0)
    y_class = np.concatenate(y_class_all, axis=0)
    y_score = np.concatenate(y_score_all, axis=0)

    np.savez_compressed(
        OUTPUT_NPZ,
        class_names=np.array(CLASS_NAMES),
        window_length=WINDOW_LENGTH,
        X=X, y_class=y_class, y_score=y_score,
        subject=np.array(subject_all),
    )
    print(f"\nSaved eval dataset: X={X.shape}  y_class={y_class.shape}  y_score={y_score.shape}")
    print(f"-> {OUTPUT_NPZ}")


if __name__ == '__main__':
    main()