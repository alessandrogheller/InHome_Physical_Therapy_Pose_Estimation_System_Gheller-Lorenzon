import sys
import os
import numpy as np

from utils import (
    PROJECT_ROOT, DATASET_ROOT, RIGHT_HIP, RIGHT_KNEE, RIGHT_ANKLE,
    calculate_angle, calculate_depth_score, keypoints_are_valid,
    DEPTH_TOLERANCE,
)

sys.path.append(os.path.join(PROJECT_ROOT, 'mmfi_lib'))
from mmfi import MMFi_Database, MMFi_Dataset

# --- CONFIGURATION ---
ACTION = 'A16'  # A16 = Lunge (toward right side), rehabilitation activity (MMFi README)
ENVIRONMENT = 'E01'

# --- Target angles ---
# STANDING_TARGET is reused from the squat/left-lunge pipelines: it just
# represents the knee fully extended between repetitions, which is not
# specific to any one exercise. It's a reasonable target for any bipedal
# standing pose.
STANDING_TARGET = 180.0
STANDING_TOLERANCE = 1.0

# IMPORTANT: same reasoning as in evaluate_dataset_fixed_targets_lunge.py
# (the left-side script) applies here -- there is no published clinical
# target angle for a lateral lunge's knee flexion. DEPTH_TARGET below is
# estimated empirically as the mean minimum-knee-angle observed across all
# MMFi subjects performing A16, i.e. "the average depth people in this
# dataset reached" on the RIGHT side, not a validated clinical goal.
#
# Treat this as a data-driven placeholder. If you find (or your
# supervisor/a physiotherapist provides) an actual clinical reference angle
# for a lateral lunge, replace DEPTH_TARGET with that value instead.
# DEPTH_TOLERANCE / falloff are still imported from utils.py so scoring stays
# consistent with the rest of the pipeline.

# Auto-detect subject folders inside the environment
environment_path = os.path.join(DATASET_ROOT, ENVIRONMENT)
SUBJECTS = sorted([
    name for name in os.listdir(environment_path)
    if os.path.isdir(os.path.join(environment_path, name)) and name.startswith('S')
])
print(f"Found {len(SUBJECTS)} subjects in {ENVIRONMENT}: {SUBJECTS}\n")

database = MMFi_Database(DATASET_ROOT)


def load_subject_angles(subject):
    """Load one subject's A16 sequence and return the filtered right-knee
    angle over time, or None if the subject/action could not be loaded."""
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
        if keypoints_are_valid(frame_kp, None, [RIGHT_HIP, RIGHT_KNEE, RIGHT_ANKLE]):
            hip = frame_kp[RIGHT_HIP]
            knee = frame_kp[RIGHT_KNEE]
            ankle = frame_kp[RIGHT_ANKLE]
            angles.append(calculate_angle(hip, knee, ankle))
        else:
            n_skipped += 1

    if len(angles) == 0:
        print(f"{subject}: no valid keypoints found, skipping.")
        return None
    if n_skipped > 0:
        print(f"{subject}: skipped {n_skipped}/{len(keypoints_seq)} frames with missing/invalid keypoints.")

    return np.array(angles)


# --- Pass 1: load all subjects, compute the empirical depth target ---
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
print(f"Action: {ACTION} (Lunge, toward right side)")
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