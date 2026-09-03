import sys
import os
import numpy as np

from utils import (
    PROJECT_ROOT, DATASET_ROOT, get_reference_path,
    LEFT_HIP, LEFT_KNEE, LEFT_ANKLE,
    RIGHT_HIP, RIGHT_KNEE, RIGHT_ANKLE,
    calculate_angle, keypoints_are_valid,
)

sys.path.append(os.path.join(PROJECT_ROOT, 'mmfi_lib'))
from mmfi import MMFi_Database, MMFi_Dataset

# --- CONFIGURATION ---
# One entry per side. Each lunge direction is its own MMFi action and uses
# the leg that actually loads/bends on that side (mirrors the two separate
# scripts this replaces: reference_extraction_lunge.py for A15/left and
# reference_extraction_lunge_right.py for A16/right), but both are now built
# in a single run so you don't have to remember to run two scripts and keep
# them in sync.
#
# SUBJECTS is per-side on purpose: the two actions are scored separately
# (see evaluate_dataset_fixed_targets_lunge.py / _right.py), so the subjects
# that performed well on A15 aren't necessarily the same ones that performed
# well on A16. Edit each list independently once you have your
# "good_subjects" results for that side.
SIDES = {
    'left': {
        'action': 'A15',                                   # Lunge (toward left side)
        'joints': (LEFT_HIP, LEFT_KNEE, LEFT_ANKLE),        # left leg loads/bends for A15
        'reference_path': get_reference_path('lunge_left'),       # -> PROJECT_ROOT/lunge_reference.npy
        'subjects': ['S01'],                                 # e.g. ['S01', 'S03', 'S07']
    },
    'right': {
        'action': 'A16',                                    # Lunge (toward right side)
        'joints': (RIGHT_HIP, RIGHT_KNEE, RIGHT_ANKLE),      # right leg loads/bends for A16
        'reference_path': get_reference_path('lunge_right'), # -> PROJECT_ROOT/lunge_right_reference.npy
        'subjects': ['S01'],
    },
}

# Number of points used to resample every subject's sequence to a common
# length before averaging (sequences have different numbers of frames).
REFERENCE_LENGTH = 100

database = MMFi_Database(DATASET_ROOT)


def extract_knee_angle_sequence(subject, action, joints):
    """Load one subject's sequence for `action` and return the (filtered)
    knee angle over time as a 1D numpy array, using whichever hip/knee/ankle
    triplet is passed in `joints` (left or right leg -- whichever one loads
    and bends for that lunge direction)."""
    data_form = {subject: [action]}
    dataset = MMFi_Dataset(
        data_base=database,
        data_unit='sequence',
        modality='rgb',
        split='reference',
        data_form=data_form
    )
    sample = dataset[0]
    keypoints_seq = sample['input_rgb']  # (num_frame, 17, 2)

    hip_i, knee_i, ankle_i = joints
    angles = []
    n_skipped = 0
    for frame_kp in keypoints_seq:
        if keypoints_are_valid(frame_kp, None, [hip_i, knee_i, ankle_i]):
            hip = frame_kp[hip_i]
            knee = frame_kp[knee_i]
            ankle = frame_kp[ankle_i]
            angles.append(calculate_angle(hip, knee, ankle))
        else:
            n_skipped += 1

    if n_skipped > 0:
        print(f"  {subject}: skipped {n_skipped}/{len(keypoints_seq)} frames with missing/invalid keypoints.")

    return np.array(angles)


def resample(sequence, length):
    """Resample a 1D sequence to `length` points using linear interpolation
    over normalized time [0, 1], so sequences of different original lengths
    can be averaged point-by-point."""
    if len(sequence) == length:
        return sequence
    original_t = np.linspace(0.0, 1.0, num=len(sequence))
    target_t = np.linspace(0.0, 1.0, num=length)
    return np.interp(target_t, original_t, sequence)


def build_reference(side_name, config):
    """Build and save the reference curve for one side ('left' or 'right'),
    exactly like the standalone scripts did, just parameterized."""
    action = config['action']
    joints = config['joints']
    subjects = config['subjects']
    reference_path = config['reference_path']

    print(f"\n=== Building {side_name} reference ({action}) ===")

    resampled_sequences = []
    for subject in subjects:
        print(f"Loading {subject}...")
        seq = extract_knee_angle_sequence(subject, action, joints)
        if len(seq) == 0:
            print(f"  {subject}: no valid keypoints found, skipping this subject.")
            continue
        print(f"  {subject}: {len(seq)} valid frames, range=[{seq.min():.1f}°, {seq.max():.1f}°]")
        resampled_sequences.append(resample(seq, REFERENCE_LENGTH))

    if len(resampled_sequences) == 0:
        print(f"No subject produced a valid sequence for {side_name} ({action}). Skipping this side.")
        return False

    reference_curve = np.mean(np.stack(resampled_sequences, axis=0), axis=0)

    print(f"Reference built from {len(resampled_sequences)} subject(s), {REFERENCE_LENGTH} points.")
    print(f"Reference range: min={reference_curve.min():.1f}°, max={reference_curve.max():.1f}°")

    np.save(reference_path, reference_curve)
    print(f"Reference curve saved to: {reference_path}")
    return True


# --- Build both reference curves ---
results = {side: build_reference(side, config) for side, config in SIDES.items()}

if not any(results.values()):
    print("\nNo reference curve could be built for either side. Aborting.")
    sys.exit(1)

failed_sides = [side for side, ok in results.items() if not ok]
if failed_sides:
    print(f"\nDone, but the following side(s) were skipped (no valid data): {', '.join(failed_sides)}. "
          "The realtime comparison script will fall back to the other side's reference for these.")
else:
    print("\nBoth reference curves built successfully.")