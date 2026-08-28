import sys
import os
import numpy as np

# Add the mmfi_lib folder to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'mmfi_lib'))
from mmfi import MMFi_Database, MMFi_Dataset

# --- CONFIGURATION ---
DATASET_ROOT = r'D:\Università\2 Magistrale\2025-26\secondo semestre\computer vision\progetto\InHome_Physical_Therapy_Pose_Estimation_System_Gheller-Lorenzon\dataset\MMFi_Dataset'
ACTION = 'A12'  # A12 = Squat
ENVIRONMENT = 'E01'

# --- Fixed clinical target angles (manually defined, not computed from data) ---
STANDING_TARGET = 180.0
STANDING_TOLERANCE = 1.0

DEPTH_TARGET = 95.0
DEPTH_TOLERANCE = 10.0       # "good zone": within this range, score stays high (80-100%)
DEPTH_FALLOFF_RANGE = 30.0   # beyond the tolerance, score decreases gradually over
                              # this extra range, down to 0%

# Auto-detect subject folders inside the environment
environment_path = os.path.join(DATASET_ROOT, ENVIRONMENT)
SUBJECTS = sorted([
    name for name in os.listdir(environment_path)
    if os.path.isdir(os.path.join(environment_path, name)) and name.startswith('S')
])
print(f"Found {len(SUBJECTS)} subjects in {ENVIRONMENT}: {SUBJECTS}\n")

# COCO keypoint indices
LEFT_HIP, LEFT_KNEE, LEFT_ANKLE = 11, 13, 15

def calculate_angle(a, b, c):
    """Calculate the angle (in degrees) at vertex b, between segments a-b and b-c."""
    a, b, c = np.array(a), np.array(b), np.array(c)
    ba = a - b
    bc = c - b
    cos_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-8)
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    return np.degrees(np.arccos(cos_angle))

def calculate_depth_score(depth_diff):
    """Two-zone scoring: high plateau within tolerance, gradual falloff beyond it.
    - Within DEPTH_TOLERANCE: score from 100% (perfect) down to 80% (at the edge)
    - Beyond it: score decreases from 80% down to 0% over DEPTH_FALLOFF_RANGE
    """
    if depth_diff <= DEPTH_TOLERANCE:
        return 100.0 - (depth_diff / DEPTH_TOLERANCE) * 20.0
    else:
        extra = depth_diff - DEPTH_TOLERANCE
        score = 80.0 - (extra / DEPTH_FALLOFF_RANGE) * 80.0
        return max(0.0, score)

database = MMFi_Database(DATASET_ROOT)

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
        for frame_kp in keypoints_seq:
            hip = frame_kp[LEFT_HIP]
            knee = frame_kp[LEFT_KNEE]
            ankle = frame_kp[LEFT_ANKLE]
            if np.any(hip) and np.any(knee) and np.any(ankle):
                angles.append(calculate_angle(hip, knee, ankle))

        if len(angles) == 0:
            print(f"{subject}: no valid keypoints found, skipping.")
            continue

        # No calibration offset here: all MMFi subjects share the same
        # camera setup within the dataset, unlike the live webcam case.
        subject_standing = max(angles)
        subject_depth = min(angles)

        standing_diff = abs(subject_standing - STANDING_TARGET)
        depth_diff = abs(subject_depth - DEPTH_TARGET)

        standing_within_tolerance = standing_diff <= STANDING_TOLERANCE
        depth_within_tolerance = depth_diff <= DEPTH_TOLERANCE

        # Score based on depth (the clinically meaningful metric), using the
        # two-zone scoring function (plateau within tolerance, gradual falloff beyond it)
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