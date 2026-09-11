import sys
import os
import numpy as np

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

# --- CONFIGURATION ---
# IMPORTANT: A26 was previously used here as a placeholder for "jumping up"
# based on the MMFi README's rehabilitation-action list, WITHOUT verifying
# that it specifically corresponds to a jumping-jack pattern (legs
# splitting sideways + arms raised overhead) rather than a plain vertical
# jump. Please confirm the correct action code against the MMFi paper/
# README before trusting the extracted reference -- if A26 turns out to be
# a vertical jump instead, this script will just produce a flat/meaningless
# curve (a vertical jump barely moves the leg-spread or arm-raise signals
# below).
ACTION = 'A26'  # TODO: confirm this is really "jumping jacks" in MMFi, not a vertical jump

# --- Why the metrics changed from the squat/lunge scripts ---
# A jumping jack barely flexes the knee at all -- almost all of the motion
# is (a) hip ABDUCTION, spreading the legs sideways, and (b) shoulder
# elevation, raising the arms overhead. A hip-knee-ankle angle (used for
# squat/lunge/jump) would stay nearly flat throughout and couldn't
# distinguish "legs together" from "legs apart". So instead we track TWO
# independent, synchronized signals and require both to move together to
# call it a valid repetition (see the note in the realtime-comparison
# script this reference is meant to feed).
#
# 1) leg_spread_ratio: ankle-to-ankle distance, normalized by shoulder
#    width. Using a ratio (not raw pixel distance) makes this roughly
#    scale-invariant to how far the patient stands from the camera, the
#    same way the calibration offset in the squat/lunge scripts corrects
#    for setup differences -- except here we normalize by a body-derived
#    unit instead of an additive angle offset, since spread is a distance,
#    not an angle.
# 2) arm_raise_angle: angle at the shoulder between hip-shoulder-wrist,
#    averaged over both arms. Near a resting/arms-down pose the wrist sits
#    close to the hip (small angle); with arms raised overhead the wrist is
#    on the opposite side of the shoulder from the hip (angle approaches
#    180 deg). This is the same calculate_angle() geometry used for the
#    elbow in the limb-extension scripts, just applied to a different
#    joint triplet to capture whole-arm elevation instead of elbow flexion.

# Keypoints required to be valid, for both metrics, on a given frame.
REQUIRED_KEYPOINTS = [
    LEFT_SHOULDER, RIGHT_SHOULDER,
    LEFT_HIP, RIGHT_HIP,
    LEFT_WRIST, RIGHT_WRIST,
    LEFT_ANKLE, RIGHT_ANKLE,
]

# One or more subjects to build the reference from. Averaging several
# subjects gives a more robust reference than a single, arbitrarily chosen
# one (same rationale as the squat/lunge scripts).
SUBJECTS = ['S01']  # e.g. ['S01', 'S03', 'S07'] to average multiple subjects

# Number of points used to resample every subject's sequence to a common
# length before averaging (sequences have different numbers of frames).
REFERENCE_LENGTH = 100

# Reference is saved as a single (REFERENCE_LENGTH, 2) array:
#   column 0 = leg_spread_ratio curve
#   column 1 = arm_raise_angle curve (degrees)
# so both signals stay bundled together and time-aligned in one file,
# instead of two separate .npy files that could drift out of sync if only
# one gets regenerated later.
REFERENCE_PATH = get_reference_path('jumping_jacks')

database = MMFi_Database(DATASET_ROOT)


def euclidean(a, b):
    a, b = np.array(a, dtype=float), np.array(b, dtype=float)
    return float(np.linalg.norm(a - b))


def extract_jumping_jack_signals(subject):
    """Load one subject's jumping-jack sequence and return two aligned 1D
    arrays: (leg_spread_ratio, arm_raise_angle), one value per valid frame.
    Frames missing any of the 8 required keypoints are skipped entirely
    (both signals need the same frames to stay meaningfully synchronized)."""
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
            # Degenerate frame (shoulders detected on top of each other) --
            # normalizing by it would blow up the ratio, so skip.
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
    """Resample a 1D sequence to `length` points using linear interpolation
    over normalized time [0, 1], so sequences of different original lengths
    can be averaged point-by-point."""
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

# Average across subjects, point-by-point on the normalized timeline.
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