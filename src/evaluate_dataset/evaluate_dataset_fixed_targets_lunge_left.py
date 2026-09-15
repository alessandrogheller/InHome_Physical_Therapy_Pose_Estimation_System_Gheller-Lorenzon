"""
src/evaluate_dataset/evaluate_dataset_fixed_targets_lunge_left.py
This script evaluates the left-sided lunge movement by extracting the knee-angle
trajectory from MM-Fi reference sequences. For each subject, it computes the
minimum knee flexion value (the lunge depth) and the maximum standing angle,
then compares them against empirical dataset-level targets. The score reflects
how closely each motion matches the expected lunge pattern and how consistently
it reaches the target depth without drifting outside the acceptable range.
"""
import sys
import os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import (
    PROJECT_ROOT, DATASET_ROOT, LEFT_HIP, LEFT_KNEE, LEFT_ANKLE,
    calculate_angle, calculate_depth_score, keypoints_are_valid,
    DEPTH_TOLERANCE,
)

sys.path.append(os.path.join(PROJECT_ROOT, 'mmfi_lib'))
from mmfi import MMFi_Database, MMFi_Dataset

# --- CONFIGURATION ---
ACTION = 'A15'  # A15 = Lunge (toward left side)
ENVIRONMENT = 'E01'

# --- Target angles ---
# The standing target is the fully extended knee angle reused from the squat
# pipeline. It reflects the neutral, upright pose between repetitions and is not
# a lunge-specific clinical value.
STANDING_TARGET = 180.0
STANDING_TOLERANCE = 1.0

# Unlike the squat, there is no widely accepted clinical target for the depth of
# a lateral lunge. Therefore, DEPTH_TARGET is estimated empirically from the
# dataset itself as the mean of each subject's minimum knee angle. This reflects
# the average depth reached by the participants in this dataset, not a validated
# physiotherapy standard.
#
# If a clinician or supervisor provides a real reference value for this exercise,
# it should replace this empirical target. DEPTH_TOLERANCE and the falloff curve
# are still taken from utils.py so the scoring remains consistent with the rest
# of the project.

# Auto-detect subject folders inside the environment
environment_path = os.path.join(DATASET_ROOT, ENVIRONMENT)
SUBJECTS = sorted([
    name for name in os.listdir(environment_path)
    if os.path.isdir(os.path.join(environment_path, name)) and name.startswith('S')
])
print(f"Found {len(SUBJECTS)} subjects in {ENVIRONMENT}: {SUBJECTS}\n")

database = MMFi_Database(DATASET_ROOT)


def load_subject_angles(subject):
    # Extract the left knee-angle trajectory from the reference motion sequence.
    # Frames with missing or invalid keypoints are discarded to keep the signal
    # centered on valid pose estimates only.
    try:
        data_form = {subject: [ACTION]}
        dataset = MMFi_Dataset(
            data_base=database,
            data_unit='sequence',
            modality='rgb',
            split='reference',
            data_form=data_form
        )
        sample = dataset[0]
        keypoints_seq = sample['input_rgb']
    except Exception as e:
        print(f"{subject}: could not load ({e}), skipping.")
        return None

    angles = []
    n_skipped = 0
    for frame_kp in keypoints_seq:
        if keypoints_are_valid(frame_kp, None, [LEFT_HIP, LEFT_KNEE, LEFT_ANKLE]):
            hip = frame_kp[LEFT_HIP]
            knee = frame_kp[LEFT_KNEE]
            ankle = frame_kp[LEFT_ANKLE]
            angles.append(calculate_angle(hip, knee, ankle))
        else:
            n_skipped += 1

    if len(angles) == 0:
        print(f"{subject}: no valid keypoints found, skipping.")
        return None
    if n_skipped > 0:
        print(f"{subject}: skipped {n_skipped}/{len(keypoints_seq)} frames with missing/invalid keypoints.")

    return np.array(angles)


# --- Pass 1: load all subjects and estimate the empirical depth target ---
# The depth target is derived from the dataset by taking the average of each
# subject's minimum knee angle, which approximates the typical lunge depth in
# this dataset.
subject_angles = {}
for subject in SUBJECTS:
    angles = load_subject_angles(subject)
    if angles is not None:
        subject_angles[subject] = angles

if len(subject_angles) == 0:
    print("No subjects could be evaluated.")
    sys.exit(1)

subject_depths_raw = [angles.min() for angles in subject_angles.values()]
DEPTH_TARGET = float(np.mean(subject_depths_raw))
print(f"\nEmpirical DEPTH_TARGET (mean of subjects' minimum knee angle) = {DEPTH_TARGET:.1f}°")
print("NOTE: this is a data-driven placeholder, not a validated clinical target "
      "(unlike the squat's manually-set 95°). Replace it if you obtain a clinical reference.\n")

# --- Pass 2: score each subject against the empirical target ---
# The quality metric focuses on how closely the subject reaches the expected
# lunge depth while remaining near a standing knee extension between repetitions.
results = []
for subject, angles in subject_angles.items():
    subject_standing = max(angles)
    subject_depth = min(angles)

    standing_diff = abs(subject_standing - STANDING_TARGET)
    depth_diff = abs(subject_depth - DEPTH_TARGET)

    standing_within_tolerance = standing_diff <= STANDING_TOLERANCE
    depth_within_tolerance = depth_diff <= DEPTH_TOLERANCE

    accuracy_pct = calculate_depth_score(depth_diff)

    results.append({
        'subject': subject,
        'standing': subject_standing,
        'depth': subject_depth,
        'depth_diff': depth_diff,
        'depth_within_tolerance': depth_within_tolerance,
        'accuracy_pct': accuracy_pct,
    })

    flag = "OK" if depth_within_tolerance else "--"
    print(f"{subject}: standing={subject_standing:6.1f}°  depth={subject_depth:6.1f}°  "
          f"diff_from_target={depth_diff:5.1f}°  [{flag}]  score={accuracy_pct:5.1f}%")

depth_diffs = [r['depth_diff'] for r in results]
scores = [r['accuracy_pct'] for r in results]
n_within_tolerance = sum(1 for r in results if r['depth_within_tolerance'])

print("\n--- Summary ---")
print(f"Action: {ACTION} (Lunge, toward left side)")
print(f"Targets: standing={STANDING_TARGET}°±{STANDING_TOLERANCE}°  "
      f"depth={DEPTH_TARGET:.1f}°±{DEPTH_TOLERANCE}° (empirical, see note above)")
print(f"Subjects evaluated: {len(results)}")
print(f"Subjects within depth tolerance: {n_within_tolerance}/{len(results)} "
      f"({100 * n_within_tolerance / len(results):.0f}%)")
print(f"Depth diff from target: mean={np.mean(depth_diffs):.1f}°  std={np.std(depth_diffs):.1f}°")
print(f"Score: mean={np.mean(scores):.1f}%  std={np.std(scores):.1f}%  "
      f"min={np.min(scores):.1f}%  max={np.max(scores):.1f}%")

good_subjects = [r['subject'] for r in results if r['depth_within_tolerance']]
print(f"\nSubjects within depth tolerance (candidates for a 'good' reference): {good_subjects}")