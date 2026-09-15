"""
src/evaluate_dataset/score_eval_subjects.py

Score the extracted evaluation videos against the fixed targets in
quality_targets.json and save one quality label per subject and exercise.
"""
import os
import sys
import csv
import json
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import (
    PROJECT_ROOT,
    LEFT_HIP, LEFT_KNEE, LEFT_ANKLE, RIGHT_HIP, RIGHT_KNEE, RIGHT_ANKLE,
    LEFT_SHOULDER, LEFT_ELBOW, LEFT_WRIST, RIGHT_SHOULDER, RIGHT_ELBOW, RIGHT_WRIST,
    calculate_angle, calculate_depth_score, keypoints_are_valid,
)

EVAL_DATASET_ROOT = os.path.join(PROJECT_ROOT, 'eval_dataset')
TARGETS_JSON = os.path.join(PROJECT_ROOT, 'quality_targets.json')
OUTPUT_CSV = os.path.join(PROJECT_ROOT, 'eval_quality_labels.csv')

LEG_JOINTS = {
    'squat': (LEFT_HIP, LEFT_KNEE, LEFT_ANKLE),
    'lunge_left': (LEFT_HIP, LEFT_KNEE, LEFT_ANKLE),
    'lunge_right': (RIGHT_HIP, RIGHT_KNEE, RIGHT_ANKLE),
}
ARM_JOINTS = {
    'limb_extension_left': (LEFT_SHOULDER, LEFT_ELBOW, LEFT_WRIST),
    'limb_extension_right': (RIGHT_SHOULDER, RIGHT_ELBOW, RIGHT_WRIST),
}
JJ_REQUIRED = [LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_HIP, RIGHT_HIP,
               LEFT_WRIST, RIGHT_WRIST, LEFT_ANKLE, RIGHT_ANKLE]


def euclidean(a, b):
    a, b = np.array(a, dtype=float), np.array(b, dtype=float)
    return float(np.linalg.norm(a - b))


def angle_series(kp_seq, conf_seq, joints):
    # Keep only frames where all joints needed for the angle are reliable.
    hip_i, knee_i, ankle_i = joints
    angles = []
    for kp, conf in zip(kp_seq, conf_seq):
        if keypoints_are_valid(kp, conf, [hip_i, knee_i, ankle_i]):
            angles.append(calculate_angle(kp[hip_i], kp[knee_i], kp[ankle_i]))
    return np.array(angles)


def jj_signals(kp_seq, conf_seq):
    # Normalize ankle distance by shoulder width so the leg-spread signal is
    # less sensitive to the subject's distance from the camera.
    leg_spread, arm_raise = [], []
    for kp, conf in zip(kp_seq, conf_seq):
        if not keypoints_are_valid(kp, conf, JJ_REQUIRED):
            continue
        shoulder_width = euclidean(kp[LEFT_SHOULDER], kp[RIGHT_SHOULDER])
        if shoulder_width < 1e-3:
            continue
        leg_spread.append(euclidean(kp[LEFT_ANKLE], kp[RIGHT_ANKLE]) / shoulder_width)
        left_arm = calculate_angle(kp[LEFT_HIP], kp[LEFT_SHOULDER], kp[LEFT_WRIST])
        right_arm = calculate_angle(kp[RIGHT_HIP], kp[RIGHT_SHOULDER], kp[RIGHT_WRIST])
        arm_raise.append((left_arm + right_arm) / 2.0)
    return np.array(leg_spread), np.array(arm_raise)


def score_video(exercise_name, kp_seq, conf_seq, targets):
    if exercise_name in LEG_JOINTS:
        # For squats and lunges, compare the minimum knee angle with the fixed
        # target depth reached during the movement.
        angles = angle_series(kp_seq, conf_seq, LEG_JOINTS[exercise_name])
        if len(angles) == 0:
            return None
        depth_diff = abs(float(angles.min()) - targets['depth_target'])
        return calculate_depth_score(depth_diff)

    if exercise_name in ARM_JOINTS:
        # Arm extensions are scored at both extremes: the resting position and
        # the most extended position observed in the sequence.
        angles = angle_series(kp_seq, conf_seq, ARM_JOINTS[exercise_name])
        if len(angles) == 0:
            return None
        resting_diff = abs(float(angles.min()) - targets['resting_target'])
        extension_diff = abs(float(angles.max()) - targets['extension_target'])
        return (calculate_depth_score(resting_diff) + calculate_depth_score(extension_diff)) / 2.0

    if exercise_name == 'jumping_jacks':
        # Use high percentiles to represent the open-leg and raised-arm phases
        # while reducing the influence of isolated noisy frames.
        leg, arm = jj_signals(kp_seq, conf_seq)
        if len(leg) == 0:
            return None
        leg_diff = abs(float(np.percentile(leg, 95)) - targets['leg_target'])
        arm_diff = abs(float(np.percentile(arm, 95)) - targets['arm_target'])
        leg_score = calculate_depth_score(leg_diff, tolerance=targets['leg_tolerance'], falloff=targets['leg_falloff'])
        arm_score = calculate_depth_score(arm_diff, tolerance=targets['arm_tolerance'], falloff=targets['arm_falloff'])
        return (leg_score + arm_score) / 2.0

    raise ValueError(f"Unknown exercise '{exercise_name}'")


def main():
    # Targets must be generated from the training data beforehand and remain
    # fixed while the evaluation subjects are being scored.
    if not os.path.exists(TARGETS_JSON):
        print(f"ERROR: {TARGETS_JSON} not found. Run the updated generate_quality_labels.py first.")
        sys.exit(1)
    with open(TARGETS_JSON) as f:
        all_targets = json.load(f)

    rows = []
    # Traverse every extracted subject/exercise pair and score only complete
    # directories containing both keypoints and confidence arrays.
    for subject_id in sorted(os.listdir(EVAL_DATASET_ROOT)):
        subject_dir = os.path.join(EVAL_DATASET_ROOT, subject_id)
        if not os.path.isdir(subject_dir):
            continue
        for exercise_name in sorted(os.listdir(subject_dir)):
            ex_dir = os.path.join(subject_dir, exercise_name)
            kp_path = os.path.join(ex_dir, 'keypoints.npy')
            conf_path = os.path.join(ex_dir, 'confidences.npy')
            if not (os.path.exists(kp_path) and os.path.exists(conf_path)):
                continue
            if exercise_name not in all_targets:
                print(f"  {subject_id}/{exercise_name}: no target found in quality_targets.json, skipping.")
                continue

            kp_seq = np.load(kp_path)
            conf_seq = np.load(conf_path)
            score = score_video(exercise_name, kp_seq, conf_seq, all_targets[exercise_name])
            if score is None:
                print(f"  {subject_id}/{exercise_name}: no valid keypoints, skipping.")
                continue

            rows.append((subject_id, exercise_name, score))
            print(f"  {subject_id}/{exercise_name}: score={score:.1f}%")

    if not rows:
        print("No eval scores computed. Aborting.")
        sys.exit(1)

    # Write the results in the same simple format consumed by the evaluation
    # dataset-building scripts.
    with open(OUTPUT_CSV, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['subject', 'action', 'score'])
        writer.writerows(rows)
    print(f"\nSaved {len(rows)} rows to {OUTPUT_CSV}")


if __name__ == '__main__':
    main()