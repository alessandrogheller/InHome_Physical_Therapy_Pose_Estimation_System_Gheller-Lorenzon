"""
Collects, for every subject and every one of the 6 exercises this project
handles, the SAME "quality" score that evaluate_dataset_fixed_targets_*.py
already computes and prints -- squat's fixed 95° clinical target, the
lunges'/limb-extensions'/jumping-jacks' data-driven empirical targets --
and writes them all into one CSV file:

    subject,action,score

This CSV is the (weak, subject-level) label used later by
build_windowed_dataset.py to train the neural network's quality-scoring
head. It does NOT introduce any new notion of "correct execution" -- it's
exactly the same heuristic, data-driven scoring already used and
documented (with the same caveats) in the individual evaluate_dataset_*.py
scripts, just gathered into a file instead of only being printed to the
terminal.

IMPORTANT DIFFERENCE FROM THE ORIGINAL evaluate_dataset_*.py SCRIPTS: those
scripts only look at ENVIRONMENT = 'E01' (10 subjects). This script looks
at ALL 4 environments (all ~40 subjects), since a neural network benefits
from more subjects/body-types/camera-setups to learn from, whereas the
original scripts were only meant as a quick per-environment sanity check.

Run this BEFORE build_windowed_dataset.py.
"""
import os
import sys
import csv
import numpy as np

from utils import (
    PROJECT_ROOT, DATASET_ROOT,
    LEFT_HIP, LEFT_KNEE, LEFT_ANKLE, RIGHT_HIP, RIGHT_KNEE, RIGHT_ANKLE,
    LEFT_SHOULDER, LEFT_ELBOW, LEFT_WRIST, RIGHT_SHOULDER, RIGHT_ELBOW, RIGHT_WRIST,
    calculate_angle, calculate_depth_score, keypoints_are_valid,
)

sys.path.append(os.path.join(PROJECT_ROOT, 'mmfi_lib'))
from mmfi import MMFi_Database, MMFi_Dataset

OUTPUT_CSV = os.path.join(PROJECT_ROOT, 'quality_labels.csv')

# --- Action definitions -----------------------------------------------
# name -> (MMFi action code, (hip, knee, ankle) OR (shoulder, elbow, wrist)
# joint triplet, scorer function to use). Squat/lunges use the leg
# triplet; limb extensions use the arm triplet. Jumping jacks are handled
# separately below since it needs two signals, not one angle.
def score_lunge(subject_angles):
    """Empirical target = mean of subjects' own minimum knee angle, same
    as evaluate_dataset_fixed_targets_lunge_left.py / _right.py."""
    depths = [float(a.min()) for a in subject_angles.values()]
    depth_target = float(np.mean(depths))
    scores = {}
    for subject, angles in subject_angles.items():
        depth_diff = abs(float(angles.min()) - depth_target)
        scores[subject] = calculate_depth_score(depth_diff)
    return scores


def score_limb_extension(subject_angles):
    """Average of a resting-angle score and an extension-angle score, both
    against empirical (population-mean) targets, same as
    evaluate_dataset_fixed_targets_limb_extension_left.py / _right.py."""
    resting_target = float(np.mean([float(a.min()) for a in subject_angles.values()]))
    extension_target = float(np.mean([float(a.max()) for a in subject_angles.values()]))
    scores = {}
    for subject, angles in subject_angles.items():
        resting_diff = abs(float(angles.min()) - resting_target)
        extension_diff = abs(float(angles.max()) - extension_target)
        resting_score = calculate_depth_score(resting_diff)
        extension_score = calculate_depth_score(extension_diff)
        scores[subject] = (resting_score + extension_score) / 2.0
    return scores


ANGLE_ACTIONS = {
    # action_name: (mmfi_code, joint_triplet, scorer)
    # NOTE: score_lunge is generic (population-mean-of-minimum-angle target,
    # same DEPTH_TOLERANCE/DEPTH_FALLOFF_RANGE scoring shape) -- it isn't
    # actually lunge-specific, so it's reused here for squat too, instead of
    # the old fixed 95-degree clinical target. That fixed target was the
    # only exercise in this file NOT using an empirical/population-relative
    # target, and it was producing systematically low labels whenever the
    # population's actual squat depth didn't happen to be close to 95°.
    'squat':               ('A12', (LEFT_HIP, LEFT_KNEE, LEFT_ANKLE), score_lunge),
    'lunge_left':          ('A15', (LEFT_HIP, LEFT_KNEE, LEFT_ANKLE), score_lunge),
    'lunge_right':         ('A16', (RIGHT_HIP, RIGHT_KNEE, RIGHT_ANKLE), score_lunge),
    'limb_extension_left':  ('A07', (LEFT_SHOULDER, LEFT_ELBOW, LEFT_WRIST), score_limb_extension),
    'limb_extension_right': ('A08', (RIGHT_SHOULDER, RIGHT_ELBOW, RIGHT_WRIST), score_limb_extension),
}

JUMPING_JACKS_ACTION_NAME = 'jumping_jacks'
JUMPING_JACKS_ACTION_CODE = 'A26'
JUMPING_JACKS_REQUIRED_KEYPOINTS = [
    LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_HIP, RIGHT_HIP,
    LEFT_WRIST, RIGHT_WRIST, LEFT_ANKLE, RIGHT_ANKLE,
]


def list_all_subjects():
    """All subjects across all 4 MMFi environments (E01-E04) -- see the
    module docstring for why this differs from the original scripts."""
    subjects = set()
    for env in sorted(os.listdir(DATASET_ROOT)):
        env_path = os.path.join(DATASET_ROOT, env)
        if not os.path.isdir(env_path):
            continue
        for name in os.listdir(env_path):
            if os.path.isdir(os.path.join(env_path, name)) and name.startswith('S'):
                subjects.add(name)
    return sorted(subjects)


def load_sequence(database, subject, action_code):
    """Load the raw (T, 17, 2) keypoint sequence for one subject/action,
    or None if it can't be loaded (subject didn't perform this action,
    missing files, etc.)."""
    try:
        data_form = {subject: [action_code]}
        dataset = MMFi_Dataset(
            data_base=database, data_unit='sequence', modality='rgb',
            split='reference', data_form=data_form,
        )
        return dataset[0]['input_rgb']
    except Exception as e:
        print(f"  {subject}/{action_code}: could not load ({e}), skipping.")
        return None


def angle_series(keypoints_seq, joints):
    hip_i, knee_i, ankle_i = joints
    angles = []
    for frame_kp in keypoints_seq:
        if keypoints_are_valid(frame_kp, None, [hip_i, knee_i, ankle_i]):
            angles.append(calculate_angle(frame_kp[hip_i], frame_kp[knee_i], frame_kp[ankle_i]))
    return np.array(angles)


def euclidean(a, b):
    a, b = np.array(a, dtype=float), np.array(b, dtype=float)
    return float(np.linalg.norm(a - b))


def jumping_jack_signals(keypoints_seq):
    leg_spread, arm_raise = [], []
    for frame_kp in keypoints_seq:
        if not keypoints_are_valid(frame_kp, None, JUMPING_JACKS_REQUIRED_KEYPOINTS):
            continue
        shoulder_width = euclidean(frame_kp[LEFT_SHOULDER], frame_kp[RIGHT_SHOULDER])
        if shoulder_width < 1e-3:
            continue
        leg_spread.append(euclidean(frame_kp[LEFT_ANKLE], frame_kp[RIGHT_ANKLE]) / shoulder_width)
        left_arm = calculate_angle(frame_kp[LEFT_HIP], frame_kp[LEFT_SHOULDER], frame_kp[LEFT_WRIST])
        right_arm = calculate_angle(frame_kp[RIGHT_HIP], frame_kp[RIGHT_SHOULDER], frame_kp[RIGHT_WRIST])
        arm_raise.append((left_arm + right_arm) / 2.0)
    return np.array(leg_spread), np.array(arm_raise)


def score_jumping_jacks(subject_signals):
    """Same dual-signal, population-relative scoring as
    evaluate_dataset_fixed_targets_jumping_jacks.py."""
    leg_peaks = [float(np.percentile(leg, 95)) for leg, arm in subject_signals.values()]
    arm_peaks = [float(np.percentile(arm, 95)) for leg, arm in subject_signals.values()]
    leg_target, arm_target = float(np.mean(leg_peaks)), float(np.mean(arm_peaks))
    leg_rom = max(leg_peaks) - min(leg_peaks)
    arm_rom = max(arm_peaks) - min(arm_peaks)
    leg_tol, leg_fall = max(0.15 * leg_rom, 1e-3), max(0.35 * leg_rom, 1e-3)
    arm_tol, arm_fall = max(0.15 * arm_rom, 1e-3), max(0.35 * arm_rom, 1e-3)

    scores = {}
    for subject, (leg, arm) in subject_signals.items():
        leg_peak, arm_peak = float(np.percentile(leg, 95)), float(np.percentile(arm, 95))
        leg_score = calculate_depth_score(abs(leg_peak - leg_target), tolerance=leg_tol, falloff=leg_fall)
        arm_score = calculate_depth_score(abs(arm_peak - arm_target), tolerance=arm_tol, falloff=arm_fall)
        scores[subject] = (leg_score + arm_score) / 2.0
    return scores


def main():
    subjects = list_all_subjects()
    print(f"Found {len(subjects)} subjects across all environments.\n")

    database = MMFi_Database(DATASET_ROOT)
    rows = []  # (subject, action_name, score)

    for action_name, (action_code, joints, scorer) in ANGLE_ACTIONS.items():
        print(f"=== {action_name} ({action_code}) ===")
        subject_angles = {}
        for subject in subjects:
            seq = load_sequence(database, subject, action_code)
            if seq is None:
                continue
            angles = angle_series(seq, joints)
            if len(angles) == 0:
                print(f"  {subject}: no valid keypoints, skipping.")
                continue
            subject_angles[subject] = angles

        if len(subject_angles) == 0:
            print(f"  No subject usable for {action_name}, skipping this action.\n")
            continue

        scores = scorer(subject_angles)
        for subject, score in scores.items():
            rows.append((subject, action_name, score))
        print(f"  Scored {len(scores)} subjects.\n")

    # Jumping jacks needs its own (dual-signal) extraction path.
    print(f"=== {JUMPING_JACKS_ACTION_NAME} ({JUMPING_JACKS_ACTION_CODE}) ===")
    subject_signals = {}
    for subject in subjects:
        seq = load_sequence(database, subject, JUMPING_JACKS_ACTION_CODE)
        if seq is None:
            continue
        leg, arm = jumping_jack_signals(seq)
        if len(leg) == 0:
            print(f"  {subject}: no valid keypoints, skipping.")
            continue
        subject_signals[subject] = (leg, arm)

    if len(subject_signals) > 0:
        scores = score_jumping_jacks(subject_signals)
        for subject, score in scores.items():
            rows.append((subject, JUMPING_JACKS_ACTION_NAME, score))
        print(f"  Scored {len(scores)} subjects.\n")
    else:
        print("  No subject usable for jumping_jacks, skipping this action.\n")

    if len(rows) == 0:
        print("No scores could be computed for any subject/action. Aborting.")
        sys.exit(1)

    with open(OUTPUT_CSV, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['subject', 'action', 'score'])
        writer.writerows(rows)

    print(f"Saved {len(rows)} (subject, action, score) rows to: {OUTPUT_CSV}")


if __name__ == '__main__':
    main()