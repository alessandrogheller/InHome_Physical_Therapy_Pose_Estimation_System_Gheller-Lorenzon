import sys
import os
import numpy as np

from utils import (
    PROJECT_ROOT, DATASET_ROOT, get_reference_path,
    RIGHT_HIP, RIGHT_KNEE, RIGHT_ANKLE,
    calculate_angle, keypoints_are_valid,
)

sys.path.append(os.path.join(PROJECT_ROOT, 'mmfi_lib'))
from mmfi import MMFi_Database, MMFi_Dataset

# --- CONFIGURATION ---
ACTION = 'A16'  # A16 = Lunge (toward right side)
REFERENCE_PATH = get_reference_path('lunge_right')  # -> PROJECT_ROOT/lunge_right_reference.npy
# Kept in a SEPARATE file from the left-side reference (lunge_reference.npy,
# built by reference_extraction_lunge.py from A15) so building one does not
# overwrite the other. realtime_comparison_lunge.py loads both and picks
# whichever matches the leg the patient actually loads.

# One or more subjects to build the reference from. Ideally pick subjects
# that scored well in evaluate_dataset_fixed_targets_lunge_right.py (see the
# "good_subjects" list printed at the end of that script).
SUBJECTS = ['S01']  # e.g. ['S01', 'S03', 'S07'] to average multiple subjects

# Number of points used to resample every subject's sequence to a common
# length before averaging (sequences have different numbers of frames).
REFERENCE_LENGTH = 100

database = MMFi_Database(DATASET_ROOT)


def extract_knee_angle_sequence(subject):
    """Load one subject's A16 sequence and return the (filtered) right-knee
    angle over time as a 1D numpy array.

    Right knee is used because A16 is a lunge toward the RIGHT side: the
    right leg is the one stepping out and bending (loaded leg), while the
    left leg stays comparatively extended (support leg). This mirrors
    extract_knee_angle_sequence() in reference_extraction_lunge.py, which
    tracks the LEFT knee for A15.
    """
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
        if keypoints_are_valid(frame_kp, None, [RIGHT_HIP, RIGHT_KNEE, RIGHT_ANKLE]):
            hip = frame_kp[RIGHT_HIP]
            knee = frame_kp[RIGHT_KNEE]
            ankle = frame_kp[RIGHT_ANKLE]
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

reference_curve = np.mean(np.stack(resampled_sequences, axis=0), axis=0)

print(f"\nReference built from {len(resampled_sequences)} subject(s), "
      f"{REFERENCE_LENGTH} points.")
print(f"Reference range: min={reference_curve.min():.1f}°, max={reference_curve.max():.1f}°")

np.save(REFERENCE_PATH, reference_curve)
print(f"Reference curve saved to: {REFERENCE_PATH}")