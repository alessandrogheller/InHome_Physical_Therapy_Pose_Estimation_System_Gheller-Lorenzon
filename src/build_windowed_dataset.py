"""
Builds the training dataset for the action-recognition + quality-scoring
network: sliding windows of normalized keypoints, each labeled with
(a) which of the 6 exercises it is, and (b) the subject-level quality
score from quality_labels.csv (see generate_quality_labels.py -- run that
script FIRST).

Output: a single compressed .npz file (action_quality_dataset.npz)
containing already-split train/val arrays, ready to be loaded by
train_action_quality_net.py.

This file lives in <PROJECT_ROOT>/src/build_windowed_dataset.py.
"""
import os
import sys
import csv
import numpy as np

from utils import PROJECT_ROOT, DATASET_ROOT
from keypoint_normalize import (
    normalize_sequence, flatten_sequence, mirror_normalized_sequence, FLAT_SIZE,
)

sys.path.append(os.path.join(PROJECT_ROOT, 'mmfi_lib'))
from mmfi import MMFi_Database, MMFi_Dataset

# --- The 6 exercises this project currently handles ------------------------
# CLASS index = position in this dict (stable in Python 3.7+ insertion
# order). Keep this dict in sync with generate_quality_labels.py's action
# names if you ever add/remove an exercise.
ACTIONS = {
    'squat': 'A12',
    'lunge_left': 'A15',
    'lunge_right': 'A16',
    'limb_extension_left': 'A07',
    'limb_extension_right': 'A08',
    'jumping_jacks': 'A26',
}
CLASS_NAMES = list(ACTIONS.keys())

# Which class a mirrored (horizontally flipped) window belongs to.
# squat/jumping_jacks are bilaterally symmetric exercises, so their mirror
# is themselves (just extra augmented data for the same class). lunge/
# limb_extension are side-specific, so mirroring a real left-side recording
# produces a synthetic (but anatomically valid) right-side sample and
# vice versa -- this is what lets each side's data help train the other
# side's class, boosting the amount of data available to the two classes
# that were showing lower classification confidence.
MIRROR_MAP = {
    'squat': 'squat',
    'lunge_left': 'lunge_right',
    'lunge_right': 'lunge_left',
    'limb_extension_left': 'limb_extension_right',
    'limb_extension_right': 'limb_extension_left',
    'jumping_jacks': 'jumping_jacks',
}

QUALITY_LABELS_CSV = os.path.join(PROJECT_ROOT, 'quality_labels.csv')
OUTPUT_NPZ = os.path.join(PROJECT_ROOT, 'action_quality_dataset.npz')

# --- Window size rationale --------------------------------------------
# MMFi recordings last a fixed ~30s per action regardless of exercise (see
# the MM-Fi paper), so total sequence LENGTH in frames is roughly
# comparable across actions -- what differs is how many repetitions fit
# into that time (a squat rep is quick; a lunge or limb-extension rep
# tends to be slower). A single fixed window size across all 6 actions is
# a reasonable default given this: the window doesn't need to contain one
# complete repetition to be useful, since classification only needs to
# recognize the pose/motion PATTERN inside it, not a full rep boundary.
#
# 30 frames at MMFi's ~10fps keypoint rate is roughly 3 seconds: long
# enough to capture a meaningful chunk of motion even for the slower
# exercises, while still producing several distinct windows even for a
# fast squat repetition. STRIDE=10 (~1s, i.e. 2/3 overlap between
# consecutive windows) both increases the number of training samples
# (useful since MMFi only has ~40 subjects total) and means live
# inference later (sliding this same window over the webcam feed) would
# update its prediction roughly once per second.
#
# These are starting points, not validated values. If some exercise turns
# out to need a different window to be recognizable (e.g. jumping jacks
# cycling faster than a squat), it's worth experimenting per-action --
# but a single shared value keeps the network's input size fixed and the
# rest of the pipeline (including live inference) simple.
WINDOW_LENGTH = 30
WINDOW_STRIDE = 10

MIN_WINDOWS_WARNING = 3  # print a warning if a sequence yields fewer windows than this
VAL_FRACTION = 0.2
RANDOM_SEED = 42


def list_all_subjects():
    """All subjects across all 4 MMFi environments (E01-E04) -- matches
    generate_quality_labels.py, so every (subject, action) pair that has a
    quality label can also be looked up here."""
    subjects = set()
    for env in sorted(os.listdir(DATASET_ROOT)):
        env_path = os.path.join(DATASET_ROOT, env)
        if not os.path.isdir(env_path):
            continue
        for name in os.listdir(env_path):
            if os.path.isdir(os.path.join(env_path, name)) and name.startswith('S'):
                subjects.add(name)
    return sorted(subjects)


def load_quality_labels(csv_path):
    """Returns {(subject, action_name): score}."""
    labels = {}
    if not os.path.exists(csv_path):
        print(f"ERROR: {csv_path} not found. Run generate_quality_labels.py first.")
        sys.exit(1)
    with open(csv_path, newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            labels[(row['subject'], row['action'])] = float(row['score'])
    return labels


def load_raw_sequence(database, subject, action_code):
    try:
        data_form = {subject: [action_code]}
        dataset = MMFi_Dataset(
            data_base=database, data_unit='sequence', modality='rgb',
            split='reference', data_form=data_form,
        )
        return dataset[0]['input_rgb']
    except Exception as e:
        print(f"  {subject}/{action_code}: could not load ({e}), skipping.")
        return None


def make_windows(sequence_flat, length, stride):
    """sequence_flat: (T, 34) normalized+flattened. Returns (N, length, 34)
    -- an empty (0, length, 34) array if the sequence is shorter than one
    window."""
    T = sequence_flat.shape[0]
    if T < length:
        return np.zeros((0, length, sequence_flat.shape[1]), dtype=np.float32)
    windows = [sequence_flat[start:start + length] for start in range(0, T - length + 1, stride)]
    return np.stack(windows, axis=0).astype(np.float32)


def main():
    quality_labels = load_quality_labels(QUALITY_LABELS_CSV)
    subjects = list_all_subjects()
    print(f"Found {len(subjects)} subjects across all environments.")

    database = MMFi_Database(DATASET_ROOT)

    rng = np.random.RandomState(RANDOM_SEED)
    shuffled_subjects = subjects.copy()
    rng.shuffle(shuffled_subjects)
    n_val = max(1, int(len(shuffled_subjects) * VAL_FRACTION))
    val_subjects = set(shuffled_subjects[:n_val])
    print(f"Train subjects: {len(shuffled_subjects) - len(val_subjects)}  Val subjects: {len(val_subjects)}")
    print("(split is by SUBJECT, not by window -- so the network is validated on "
          "people it never saw during training, not just unseen moments of a "
          "person it already trained on)\n")

    splits = {
        'train': {'X': [], 'y_class': [], 'y_score': []},
        'val':   {'X': [], 'y_class': [], 'y_score': []},
    }

    for class_idx, (action_name, action_code) in enumerate(ACTIONS.items()):
        print(f"=== {action_name} ({action_code}) ===")
        for subject in subjects:
            score = quality_labels.get((subject, action_name))
            if score is None:
                # No quality label for this subject/action -- e.g. it was
                # skipped in generate_quality_labels.py for missing/invalid
                # keypoints. Skip rather than guessing a label.
                continue

            raw_seq = load_raw_sequence(database, subject, action_code)
            if raw_seq is None:
                continue

            normalized = normalize_sequence(raw_seq)
            if normalized.shape[0] < WINDOW_LENGTH:
                print(f"  {subject}: only {normalized.shape[0]} usable frames "
                      f"(< window length {WINDOW_LENGTH}), skipping.")
                continue

            flat = flatten_sequence(normalized)
            windows = make_windows(flat, WINDOW_LENGTH, WINDOW_STRIDE)
            if 0 < windows.shape[0] < MIN_WINDOWS_WARNING:
                print(f"  {subject}: only {windows.shape[0]} window(s) produced.")
            if windows.shape[0] == 0:
                continue

            # Mirror augmentation: same subject/sequence, horizontally
            # flipped. The mirrored version is added as EXTRA windows (not
            # a replacement), and is assigned to whichever class an
            # anatomically-consistent flip actually represents (see
            # MIRROR_MAP -- same class for squat/jumping_jacks, the other
            # side's class for lunge/limb_extension). The subject's own
            # score is reused, since mirroring doesn't change how well the
            # subject executed the movement, only which side of the frame
            # it appears on.
            mirrored_normalized = mirror_normalized_sequence(normalized)
            mirrored_flat = flatten_sequence(mirrored_normalized)
            mirrored_windows = make_windows(mirrored_flat, WINDOW_LENGTH, WINDOW_STRIDE)
            mirror_class_name = MIRROR_MAP[action_name]
            mirror_class_idx = CLASS_NAMES.index(mirror_class_name)

            split_name = 'val' if subject in val_subjects else 'train'

            splits[split_name]['X'].append(windows)
            splits[split_name]['y_class'].append(
                np.full(windows.shape[0], class_idx, dtype=np.int64))
            splits[split_name]['y_score'].append(
                np.full(windows.shape[0], score, dtype=np.float32))

            if mirrored_windows.shape[0] > 0:
                splits[split_name]['X'].append(mirrored_windows)
                splits[split_name]['y_class'].append(
                    np.full(mirrored_windows.shape[0], mirror_class_idx, dtype=np.int64))
                splits[split_name]['y_score'].append(
                    np.full(mirrored_windows.shape[0], score, dtype=np.float32))

            print(f"  {subject}: {windows.shape[0]} windows (score={score:.1f}%) "
                  f"+ {mirrored_windows.shape[0]} mirrored (-> {mirror_class_name}) -> {split_name}")
        print()

    save_kwargs = {}
    for split_name in ('train', 'val'):
        if splits[split_name]['X']:
            X = np.concatenate(splits[split_name]['X'], axis=0)
            y_class = np.concatenate(splits[split_name]['y_class'], axis=0)
            y_score = np.concatenate(splits[split_name]['y_score'], axis=0)
        else:
            X = np.zeros((0, WINDOW_LENGTH, FLAT_SIZE), dtype=np.float32)
            y_class = np.zeros((0,), dtype=np.int64)
            y_score = np.zeros((0,), dtype=np.float32)
        save_kwargs[f'X_{split_name}'] = X
        save_kwargs[f'y_class_{split_name}'] = y_class
        save_kwargs[f'y_score_{split_name}'] = y_score
        print(f"{split_name}: X={X.shape}  y_class={y_class.shape}  y_score={y_score.shape}")

    if save_kwargs['X_train'].shape[0] == 0:
        print("\nNo training windows were produced at all. Aborting without saving.")
        sys.exit(1)

    np.savez_compressed(
        OUTPUT_NPZ,
        class_names=np.array(CLASS_NAMES),
        window_length=WINDOW_LENGTH,
        window_stride=WINDOW_STRIDE,
        **save_kwargs,
    )
    print(f"\nSaved windowed dataset to: {OUTPUT_NPZ}")


if __name__ == '__main__':
    main()