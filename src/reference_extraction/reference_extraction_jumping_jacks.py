"""Build a synchronized jumping-jack reference curve from MMFi keypoints."""

import sys
import os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import (
    PROJECT_ROOT, DATASET_ROOT, get_reference_path,
    LEFT_SHOULDER, RIGHT_SHOULDER,
    LEFT_HIP, RIGHT_HIP,
    LEFT_WRIST, RIGHT_WRIST,
    LEFT_ANKLE, RIGHT_ANKLE,
    calculate_angle, keypoints_are_valid,
)

sys.path.append(os.path.join(PROJECT_ROOT, 'mmfi_lib'))
from mmfi import MMFi_Database, MMFi_Dataset

# --- Configuration ---
ACTION = 'A26'  # jumping jacks

# Track two synchronized signals: normalized ankle distance for leg spread and
# the average shoulder angle for arm elevation.

# All keypoints needed to compute both signals must be valid.
REQUIRED_KEYPOINTS = [
    LEFT_SHOULDER, RIGHT_SHOULDER,
    LEFT_HIP, RIGHT_HIP,
    LEFT_WRIST, RIGHT_WRIST,
    LEFT_ANKLE, RIGHT_ANKLE,
]

# Average one or more subjects to build the reference curve.
SUBJECTS = ['S01']  # e.g. ['S01', 'S03', 'S07'] to average multiple subjects

# Common sequence length used before averaging subjects.
REFERENCE_LENGTH = 100

# Save both time-aligned signals in one (REFERENCE_LENGTH, 2) array.
REFERENCE_PATH = get_reference_path('jumping_jacks')

database = MMFi_Database(DATASET_ROOT)


def euclidean(a, b):
    a, b = np.array(a, dtype=float), np.array(b, dtype=float)
    return float(np.linalg.norm(a - b))


def extract_jumping_jack_signals(subject):
    """Extract synchronized leg-spread and arm-raise signals for one subject."""
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

    leg_spread = []
    arm_raise = []
    n_skipped = 0

    for frame_kp in keypoints_seq:
        if not keypoints_are_valid(frame_kp, None, REQUIRED_KEYPOINTS):
            n_skipped += 1
            continue

        left_shoulder = frame_kp[LEFT_SHOULDER]
        right_shoulder = frame_kp[RIGHT_SHOULDER]
        left_hip = frame_kp[LEFT_HIP]
        right_hip = frame_kp[RIGHT_HIP]
        left_wrist = frame_kp[LEFT_WRIST]
        right_wrist = frame_kp[RIGHT_WRIST]
        left_ankle = frame_kp[LEFT_ANKLE]
        right_ankle = frame_kp[RIGHT_ANKLE]

        shoulder_width = euclidean(left_shoulder, right_shoulder)
        if shoulder_width < 1e-3:
            # Skip frames with an unusable shoulder-width scale.
            n_skipped += 1
            continue

        ankle_dist = euclidean(left_ankle, right_ankle)
        spread_ratio = ankle_dist / shoulder_width

        left_arm_angle = calculate_angle(left_hip, left_shoulder, left_wrist)
        right_arm_angle = calculate_angle(right_hip, right_shoulder, right_wrist)
        avg_arm_angle = (left_arm_angle + right_arm_angle) / 2.0

        leg_spread.append(spread_ratio)
        arm_raise.append(avg_arm_angle)

    if n_skipped > 0:
        print(f"  {subject}: skipped {n_skipped}/{len(keypoints_seq)} frames with missing/invalid keypoints.")

    return np.array(leg_spread), np.array(arm_raise)


def resample(sequence, length):
    """Resample a sequence to a common length using linear interpolation."""
    if len(sequence) == length:
        return sequence
    original_t = np.linspace(0.0, 1.0, num=len(sequence))
    target_t = np.linspace(0.0, 1.0, num=length)
    return np.interp(target_t, original_t, sequence)


# --- Build the reference curves ---
resampled_spread = []
resampled_arm = []

for subject in SUBJECTS:
    print(f"Loading {subject}...")
    spread_seq, arm_seq = extract_jumping_jack_signals(subject)

    if len(spread_seq) == 0:
        print(f"  {subject}: no valid keypoints found, skipping this subject.")
        continue

    print(f"  {subject}: {len(spread_seq)} valid frames, "
          f"leg_spread range=[{spread_seq.min():.2f}, {spread_seq.max():.2f}], "
          f"arm_raise range=[{arm_seq.min():.1f}°, {arm_seq.max():.1f}°]")

    resampled_spread.append(resample(spread_seq, REFERENCE_LENGTH))
    resampled_arm.append(resample(arm_seq, REFERENCE_LENGTH))

if len(resampled_spread) == 0:
    print("No subject produced a valid sequence. Aborting.")
    sys.exit(1)

# Average both signals across subjects on the normalized timeline.
leg_spread_curve = np.mean(np.stack(resampled_spread, axis=0), axis=0)
arm_raise_curve = np.mean(np.stack(resampled_arm, axis=0), axis=0)

print(f"\nReference built from {len(resampled_spread)} subject(s), {REFERENCE_LENGTH} points.")
print(f"Leg spread ratio range: min={leg_spread_curve.min():.2f}, max={leg_spread_curve.max():.2f}")
print(f"Arm raise angle range:  min={arm_raise_curve.min():.1f}°, max={arm_raise_curve.max():.1f}°")
print("NOTE: 'min' of each curve = legs together / arms down (resting stance); "
      "'max' = legs apart / arms overhead (jack's open position).")

reference_curve = np.stack([leg_spread_curve, arm_raise_curve], axis=1)  # shape (REFERENCE_LENGTH, 2)

np.save(REFERENCE_PATH, reference_curve)
print(f"Reference curve saved to: {REFERENCE_PATH}  (shape={reference_curve.shape}, "
      f"columns=[leg_spread_ratio, arm_raise_angle_deg])")