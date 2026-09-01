import sys
import os
import numpy as np

from utils import (
    PROJECT_ROOT, DATASET_ROOT, REFERENCE_PATH,
    LEFT_HIP, LEFT_KNEE, LEFT_ANKLE,
    calculate_angle, keypoints_are_valid,
)

sys.path.append(os.path.join(PROJECT_ROOT, 'mmfi_lib'))
from mmfi import MMFi_Database, MMFi_Dataset

# --- CONFIGURATION ---
ACTION = 'A12'  # A12 = Squat

# One or more subjects to build the reference from. Using several subjects
# (instead of a single, arbitrarily chosen one) and averaging their curves
# gives a more robust reference, less dependent on one person's individual
# technique or body proportions.
#
# Ideally pick subjects that scored well in evaluate_dataset_fixed_targets.py
# (see the "good_subjects" list printed at the end of that script).
SUBJECTS = ['S01']  # e.g. ['S01', 'S03', 'S07'] to average multiple subjects

# Number of points used to resample every subject's sequence to a common
# length before averaging (needed because raw sequences have different
# numbers of frames). This is a simple normalized-time alignment, not a full
# Dynamic Time Warping -- good enough for a single-repetition reference.
REFERENCE_LENGTH = 100

database = MMFi_Database(DATASET_ROOT)


def extract_knee_angle_sequence(subject):
    """Load one subject's squat sequence and return the (filtered) left-knee
    angle over time as a 1D numpy array."""
    data_form = {subject: [ACTION]}
    dataset = MMFi_Dataset(
        data_base=database,
        data_unit='sequence',
        modality='rgb',
        split='reference',
        data_form=data_form
    )
    sample = dataset[0]
    keypoints_seq = sample['input_rgb']  # (num_frame, 17, 2)

    angles = []
    n_skipped = 0
    for frame_kp in keypoints_seq:
        # Fix vs. the original script: frames with missing/invalid keypoints
        # are now skipped instead of silently producing a spurious angle.
        if keypoints_are_valid(frame_kp, None, [LEFT_HIP, LEFT_KNEE, LEFT_ANKLE]):
            hip = frame_kp[LEFT_HIP]
            knee = frame_kp[LEFT_KNEE]
            ankle = frame_kp[LEFT_ANKLE]
            angles.append(calculate_angle(hip, knee, ankle))
        else:
            n_skipped += 1

    if n_skipped > 0:
        print(f"  {subject}: skipped {n_skipped}/{len(keypoints_seq)} frames with missing/invalid keypoints.")

    return np.array(angles)


def resample(sequence, length):
    """Resample a 1D sequence to `length` points using linear interpolation
    over normalized time [0, 1]. Lets us average sequences of different
    original lengths point-by-point."""
    if len(sequence) == length:
        return sequence
    original_t = np.linspace(0.0, 1.0, num=len(sequence))
    target_t = np.linspace(0.0, 1.0, num=length)
    return np.interp(target_t, original_t, sequence)


# --- Build the reference curve ---
resampled_sequences = []
for subject in SUBJECTS:
    print(f"Loading {subject}...")
    seq = extract_knee_angle_sequence(subject)
    if len(seq) == 0:
        print(f"  {subject}: no valid keypoints found, skipping this subject.")
        continue
    print(f"  {subject}: {len(seq)} valid frames, range=[{seq.min():.1f}°, {seq.max():.1f}°]")
    resampled_sequences.append(resample(seq, REFERENCE_LENGTH))

if len(resampled_sequences) == 0:
    print("No subject produced a valid sequence. Aborting.")
    sys.exit(1)

# Average across subjects, point-by-point on the normalized timeline.
# With a single subject in SUBJECTS this simply returns that subject's
# (resampled) curve, so the script still works exactly as before if you
# only list one subject.
reference_curve = np.mean(np.stack(resampled_sequences, axis=0), axis=0)

print(f"\nReference built from {len(resampled_sequences)} subject(s), "
      f"{REFERENCE_LENGTH} points.")
print(f"Reference range: min={reference_curve.min():.1f}°, max={reference_curve.max():.1f}°")

# Save the reference curve for use in Phase 3 (realtime_comparison.py),
# now to an absolute, portable path (PROJECT_ROOT/squat_reference.npy)
# instead of a relative path that depended on the current working directory.
np.save(REFERENCE_PATH, reference_curve)
print(f"Reference curve saved to: {REFERENCE_PATH}")