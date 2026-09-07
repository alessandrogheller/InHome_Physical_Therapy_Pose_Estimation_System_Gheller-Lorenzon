import sys
import os
import numpy as np

from utils import (
    PROJECT_ROOT, DATASET_ROOT,
    LEFT_SHOULDER, RIGHT_SHOULDER,
    LEFT_HIP, RIGHT_HIP,
    LEFT_WRIST, RIGHT_WRIST,
    LEFT_ANKLE, RIGHT_ANKLE,
    calculate_angle, calculate_depth_score, keypoints_are_valid,
)

sys.path.append(os.path.join(PROJECT_ROOT, 'mmfi_lib'))
from mmfi import MMFi_Database, MMFi_Dataset

# --- CONFIGURATION ---
# IMPORTANT: MM-Fi has no dedicated "jumping jacks" action. A26 "Jumping up"
# (a bilateral vertical jump) is used here as the closest available proxy --
# see the note at the top of reference_extraction_jumping_jacks.py. Treat
# any target/score computed here as a data-driven placeholder, not a
# validated jumping-jack reference.
ACTION = 'A26'  # A26 = Jumping up (proxy for jumping jacks)
ENVIRONMENT = 'E01'

# --- Why two signals, not a knee angle ---
# Same reasoning as reference_extraction_jumping_jacks.py: a jumping jack is
# mostly hip abduction (legs spreading sideways) and shoulder elevation
# (arms raised overhead), not knee flexion. So this script tracks the same
# two signals used everywhere else in the jumping-jack pipeline:
#   - leg_spread_ratio: ankle-to-ankle distance / shoulder width
#   - arm_raise_angle: angle at the shoulder (hip-shoulder-wrist), averaged
#     over both arms
REQUIRED_KEYPOINTS = [
    LEFT_SHOULDER, RIGHT_SHOULDER,
    LEFT_HIP, RIGHT_HIP,
    LEFT_WRIST, RIGHT_WRIST,
    LEFT_ANKLE, RIGHT_ANKLE,
]

# Auto-detect subject folders inside the environment
environment_path = os.path.join(DATASET_ROOT, ENVIRONMENT)
SUBJECTS = sorted([
    name for name in os.listdir(environment_path)
    if os.path.isdir(os.path.join(environment_path, name)) and name.startswith('S')
])
print(f"Found {len(SUBJECTS)} subjects in {ENVIRONMENT}: {SUBJECTS}\n")

database = MMFi_Database(DATASET_ROOT)


def euclidean(a, b):
    a, b = np.array(a, dtype=float), np.array(b, dtype=float)
    return float(np.linalg.norm(a - b))


def load_subject_signals(subject):
    """Load one subject's A26 sequence and return (leg_spread_ratio,
    arm_raise_angle) as two aligned 1D numpy arrays, or None if the
    subject/action could not be loaded or produced no valid frames."""
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
            n_skipped += 1
            continue

        spread_ratio = euclidean(left_ankle, right_ankle) / shoulder_width
        left_arm_angle = calculate_angle(left_hip, left_shoulder, left_wrist)
        right_arm_angle = calculate_angle(right_hip, right_shoulder, right_wrist)
        avg_arm_angle = (left_arm_angle + right_arm_angle) / 2.0

        leg_spread.append(spread_ratio)
        arm_raise.append(avg_arm_angle)

    if len(leg_spread) == 0:
        print(f"{subject}: no valid keypoints found, skipping.")
        return None
    if n_skipped > 0:
        print(f"{subject}: skipped {n_skipped}/{len(keypoints_seq)} frames with missing/invalid keypoints.")

    return np.array(leg_spread), np.array(arm_raise)


# --- Pass 1: load all subjects, compute empirical targets ---
# Like the lunge scripts, there is no published clinical target for either
# signal, so both targets are estimated empirically from the dataset
# itself: "how far people in this dataset actually got", not a validated
# clinical goal.
#
# Each subject's own peak is taken as the 95th percentile (not the raw
# max) of their sequence. This mirrors the fix applied in
# realtime_comparison_jumping_jacks.py: a single noisy keypoint frame
# (motion blur momentarily displacing an ankle or wrist during the fast
# jump) can otherwise produce one extreme spike that raw max() would treat
# as "the" peak.
subject_signals = {}
for subject in SUBJECTS:
    signals = load_subject_signals(subject)
    if signals is not None:
        subject_signals[subject] = signals

if len(subject_signals) == 0:
    print("No subjects could be evaluated.")
    sys.exit(1)

subject_leg_peaks_raw = [float(np.percentile(leg, 95)) for leg, arm in subject_signals.values()]
subject_arm_peaks_raw = [float(np.percentile(arm, 95)) for leg, arm in subject_signals.values()]

LEG_SPREAD_TARGET = float(np.mean(subject_leg_peaks_raw))
ARM_RAISE_TARGET = float(np.mean(subject_arm_peaks_raw))

# Tolerance/falloff expressed as a fraction of the population's own range,
# same 15%/35% shape used elsewhere in this pipeline, since one signal is a
# unitless ratio and the other is in degrees -- a single fixed absolute
# tolerance (like the squat's DEPTH_TOLERANCE=10deg) wouldn't make sense
# applied to both.
_leg_population_rom = max(subject_leg_peaks_raw) - min(subject_leg_peaks_raw)
_arm_population_rom = max(subject_arm_peaks_raw) - min(subject_arm_peaks_raw)
LEG_TOLERANCE = max(0.15 * _leg_population_rom, 1e-3)
LEG_FALLOFF = max(0.35 * _leg_population_rom, 1e-3)
ARM_TOLERANCE = max(0.15 * _arm_population_rom, 1e-3)
ARM_FALLOFF = max(0.35 * _arm_population_rom, 1e-3)

print(f"\nEmpirical LEG_SPREAD_TARGET (mean of subjects' 95th-percentile leg spread) = {LEG_SPREAD_TARGET:.2f}")
print(f"Empirical ARM_RAISE_TARGET (mean of subjects' 95th-percentile arm raise) = {ARM_RAISE_TARGET:.1f}°")
print("NOTE: these are data-driven placeholders derived from A26 'Jumping up' as a proxy action, "
      "not validated jumping-jack targets. Replace them if you obtain real jumping-jack recordings "
      "or a clinical reference.\n")

# --- Pass 2: score each subject against the empirical targets ---
results = []
for subject, (leg_seq, arm_seq) in subject_signals.items():
    subject_leg_peak = float(np.percentile(leg_seq, 95))
    subject_arm_peak = float(np.percentile(arm_seq, 95))

    leg_diff = abs(subject_leg_peak - LEG_SPREAD_TARGET)
    arm_diff = abs(subject_arm_peak - ARM_RAISE_TARGET)

    leg_within_tolerance = leg_diff <= LEG_TOLERANCE
    arm_within_tolerance = arm_diff <= ARM_TOLERANCE
    within_tolerance = leg_within_tolerance and arm_within_tolerance

    leg_score = calculate_depth_score(leg_diff, tolerance=LEG_TOLERANCE, falloff=LEG_FALLOFF)
    arm_score = calculate_depth_score(arm_diff, tolerance=ARM_TOLERANCE, falloff=ARM_FALLOFF)
    accuracy_pct = (leg_score + arm_score) / 2.0

    results.append({
        'subject': subject,
        'leg_peak': subject_leg_peak,
        'arm_peak': subject_arm_peak,
        'leg_diff': leg_diff,
        'arm_diff': arm_diff,
        'within_tolerance': within_tolerance,
        'accuracy_pct': accuracy_pct,
    })

    flag = "OK" if within_tolerance else "--"
    print(f"{subject}: leg_peak={subject_leg_peak:5.2f}  arm_peak={subject_arm_peak:6.1f}°  "
          f"leg_diff={leg_diff:5.2f}  arm_diff={arm_diff:5.1f}°  [{flag}]  score={accuracy_pct:5.1f}%")

scores = [r['accuracy_pct'] for r in results]
n_within_tolerance = sum(1 for r in results if r['within_tolerance'])

print("\n--- Summary ---")
print(f"Action: {ACTION} (Jumping up, proxy for jumping jacks)")
print(f"Empirical targets: leg_spread={LEG_SPREAD_TARGET:.2f}±{LEG_TOLERANCE:.2f}  "
      f"arm_raise={ARM_RAISE_TARGET:.1f}°±{ARM_TOLERANCE:.1f}°")
print(f"Subjects evaluated: {len(results)}")
print(f"Subjects within tolerance (both signals): {n_within_tolerance}/{len(results)} "
      f"({100 * n_within_tolerance / len(results):.0f}%)")
print(f"Score: mean={np.mean(scores):.1f}%  std={np.std(scores):.1f}%  "
      f"min={np.min(scores):.1f}%  max={np.max(scores):.1f}%")

good_subjects = [r['subject'] for r in results if r['within_tolerance']]
print(f"\nSubjects within tolerance (candidates for a 'good' reference): {good_subjects}")