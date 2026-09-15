"""
src/evaluate_dataset/evaluate_dataset_fixed_targets_squat.py
This script evaluates squat performance by extracting the knee-angle trajectory
from MM-Fi reference sequences. For each subject, it measures the minimum knee
flexion (deepest squat) and the maximum standing angle, then compares them to
fixed dataset targets. The score reflects how closely the execution matches the
expected squat pattern and whether the movement is within the tolerance range for
acceptable depth.
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

# Add the mmfi_lib folder to the path (mmfi_lib is a sibling of src/, at PROJECT_ROOT)
sys.path.append(os.path.join(PROJECT_ROOT, 'mmfi_lib'))
from mmfi import MMFi_Database, MMFi_Dataset

# --- CONFIGURATION ---
ACTION = 'A12'  # A12 = Squat
ENVIRONMENT = 'E01'

# --- Fixed clinical target angles (manually defined, not computed from data) ---
# The standing target represents the fully extended knee angle between repetitions,
# while the squat depth target reflects the clinically accepted knee flexion.
STANDING_TARGET = 180.0
STANDING_TOLERANCE = 1.0

DEPTH_TARGET = 95.0
# DEPTH_TOLERANCE and the falloff behavior are imported from utils.py so this
# script uses the same scoring function as the live webcam pipeline.

# Auto-detect subject folders inside the environment
environment_path = os.path.join(DATASET_ROOT, ENVIRONMENT)
SUBJECTS = sorted([
    name for name in os.listdir(environment_path)
    if os.path.isdir(os.path.join(environment_path, name)) and name.startswith('S')
])
print(f"Found {len(SUBJECTS)} subjects in {ENVIRONMENT}: {SUBJECTS}\n")

database = MMFi_Database(DATASET_ROOT)

# Evaluate each subject independently and collect the resulting quality metrics.
results = []

for subject in SUBJECTS:
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
            continue
        if n_skipped > 0:
            print(f"{subject}: skipped {n_skipped}/{len(keypoints_seq)} frames with missing/invalid keypoints.")

        # The MMFi dataset uses a consistent acquisition setup, so no additional
        # calibration offset is needed here as in the live webcam pipeline.
        subject_standing = max(angles)
        subject_depth = min(angles)

        standing_diff = abs(subject_standing - STANDING_TARGET)
        depth_diff = abs(subject_depth - DEPTH_TARGET)

        standing_within_tolerance = standing_diff <= STANDING_TOLERANCE
        depth_within_tolerance = depth_diff <= DEPTH_TOLERANCE

        # Score based on depth, which is the clinically meaningful metric for the
        # squat. The shared depth-based scoring function from utils.py is used to
        # keep the behavior aligned with the rest of the project.
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

    except Exception as e:
        print(f"{subject}: could not load ({e}), skipping.")

if len(results) == 0:
    print("No subjects could be evaluated.")
    sys.exit(1)

depth_diffs = [r['depth_diff'] for r in results]
scores = [r['accuracy_pct'] for r in results]
n_within_tolerance = sum(1 for r in results if r['depth_within_tolerance'])

print("\n--- Summary ---")
print(f"Fixed targets: standing={STANDING_TARGET}°±{STANDING_TOLERANCE}°  "
      f"depth={DEPTH_TARGET}°±{DEPTH_TOLERANCE}°")
print(f"Subjects evaluated: {len(results)}")
print(f"Subjects within depth tolerance: {n_within_tolerance}/{len(results)} "
      f"({100 * n_within_tolerance / len(results):.0f}%)")
print(f"Depth diff from target: mean={np.mean(depth_diffs):.1f}°  std={np.std(depth_diffs):.1f}°")
print(f"Score: mean={np.mean(scores):.1f}%  std={np.std(scores):.1f}%  "
      f"min={np.min(scores):.1f}%  max={np.max(scores):.1f}%")

# List of subjects that scored well -> useful as candidates for building a
# multi-subject reference curve in reference_extraction.py (see GOOD_SUBJECTS).
good_subjects = [r['subject'] for r in results if r['depth_within_tolerance']]
print(f"\nSubjects within depth tolerance (candidates for a 'good' reference): {good_subjects}")