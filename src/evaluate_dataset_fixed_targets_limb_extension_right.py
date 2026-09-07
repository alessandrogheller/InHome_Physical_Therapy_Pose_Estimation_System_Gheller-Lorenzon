import sys
import os
import numpy as np

from utils import (
    PROJECT_ROOT, DATASET_ROOT, RIGHT_SHOULDER, RIGHT_ELBOW, RIGHT_WRIST,
    calculate_angle, calculate_depth_score, keypoints_are_valid,
    DEPTH_TOLERANCE,
)

sys.path.append(os.path.join(PROJECT_ROOT, 'mmfi_lib'))
from mmfi import MMFi_Database, MMFi_Dataset

# --- CONFIGURATION ---
ACTION = 'A08'  # A08 = Limb extension (right arm), rehabilitation activity (MMFi README)
ENVIRONMENT = 'E01'

# --- Target angles ---
# Same reasoning as evaluate_dataset_fixed_targets_limb_extension_left.py
# (the left-arm script) applies here -- there is no published clinical
# target angle for either the resting/flexed elbow position or the
# fully-extended elbow position. Both RESTING_TARGET and EXTENSION_TARGET
# below are estimated empirically as the mean of subjects' own minimum
# (resting) and maximum (extended) elbow angle for the RIGHT arm, i.e.
# "what people in this dataset actually did", not a validated clinical
# goal. Treat these as data-driven placeholders; replace them if a
# physiotherapist/supervisor provides real reference values.
#
# WHY BOTH ARE SCORED (not just the extension peak): if only the peak
# extension were scored, a subject who simply holds their arm near-extended
# for the whole clip -- never actually returning to a resting/flexed
# position between reps -- would score ~100% despite not performing a real
# extension-FROM-rest movement. Scoring the resting angle too, and
# averaging the two scores, means a repetition only scores well if it both
# starts/ends near a real resting flexion AND reaches a real extension.
#
# DEPTH_TOLERANCE / falloff are imported from utils.py so scoring stays
# consistent with the rest of the pipeline, applied here to both signals.

# Auto-detect subject folders inside the environment
environment_path = os.path.join(DATASET_ROOT, ENVIRONMENT)
SUBJECTS = sorted([
    name for name in os.listdir(environment_path)
    if os.path.isdir(os.path.join(environment_path, name)) and name.startswith('S')
])
print(f"Found {len(SUBJECTS)} subjects in {ENVIRONMENT}: {SUBJECTS}\n")

database = MMFi_Database(DATASET_ROOT)


def load_subject_angles(subject):
    """Load one subject's A08 sequence and return the filtered right-elbow
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
        if keypoints_are_valid(frame_kp, None, [RIGHT_SHOULDER, RIGHT_ELBOW, RIGHT_WRIST]):
            shoulder = frame_kp[RIGHT_SHOULDER]
            elbow = frame_kp[RIGHT_ELBOW]
            wrist = frame_kp[RIGHT_WRIST]
            angles.append(calculate_angle(shoulder, elbow, wrist))
        else:
            n_skipped += 1

    if len(angles) == 0:
        print(f"{subject}: no valid keypoints found, skipping.")
        return None
    if n_skipped > 0:
        print(f"{subject}: skipped {n_skipped}/{len(keypoints_seq)} frames with missing/invalid keypoints.")

    return np.array(angles)


# --- Pass 1: load all subjects, compute the empirical resting/extension targets ---
subject_angles = {}
for subject in SUBJECTS:
    angles = load_subject_angles(subject)
    if angles is not None:
        subject_angles[subject] = angles

if len(subject_angles) == 0:
    print("No subjects could be evaluated.")
    sys.exit(1)

subject_resting_raw = [angles.min() for angles in subject_angles.values()]
subject_extension_raw = [angles.max() for angles in subject_angles.values()]

RESTING_TARGET = float(np.mean(subject_resting_raw))
EXTENSION_TARGET = float(np.mean(subject_extension_raw))

print(f"\nEmpirical RESTING_TARGET (mean of subjects' minimum elbow angle) = {RESTING_TARGET:.1f}°")
print(f"Empirical EXTENSION_TARGET (mean of subjects' maximum elbow angle) = {EXTENSION_TARGET:.1f}°")
print("NOTE: these are data-driven placeholders, not validated clinical targets "
      "(unlike the squat's manually-set 95°). Replace them if you obtain a clinical reference.\n")

# --- Pass 2: score each subject against BOTH empirical targets ---
results = []
for subject, angles in subject_angles.items():
    subject_resting = float(angles.min())
    subject_extension = float(angles.max())

    resting_diff = abs(subject_resting - RESTING_TARGET)
    extension_diff = abs(subject_extension - EXTENSION_TARGET)

    resting_within_tolerance = resting_diff <= DEPTH_TOLERANCE
    extension_within_tolerance = extension_diff <= DEPTH_TOLERANCE
    within_tolerance = resting_within_tolerance and extension_within_tolerance

    resting_score = calculate_depth_score(resting_diff)
    extension_score = calculate_depth_score(extension_diff)
    # Average of both scores: a subject can't reach 100% by only nailing
    # the extension while never returning to a real resting position (or
    # vice versa) -- both halves of the movement have to be correct.
    accuracy_pct = (resting_score + extension_score) / 2.0

    results.append({
        'subject': subject,
        'resting': subject_resting,
        'extension': subject_extension,
        'resting_diff': resting_diff,
        'extension_diff': extension_diff,
        'within_tolerance': within_tolerance,
        'accuracy_pct': accuracy_pct,
    })

    flag = "OK" if within_tolerance else "--"
    print(f"{subject}: resting={subject_resting:6.1f}°  extension={subject_extension:6.1f}°  "
          f"resting_diff={resting_diff:5.1f}°  extension_diff={extension_diff:5.1f}°  "
          f"[{flag}]  score={accuracy_pct:5.1f}%")

resting_diffs = [r['resting_diff'] for r in results]
extension_diffs = [r['extension_diff'] for r in results]
scores = [r['accuracy_pct'] for r in results]
n_within_tolerance = sum(1 for r in results if r['within_tolerance'])

print("\n--- Summary ---")
print(f"Action: {ACTION} (Limb extension, right arm)")
print(f"Targets: resting={RESTING_TARGET:.1f}°±{DEPTH_TOLERANCE}°  "
      f"extension={EXTENSION_TARGET:.1f}°±{DEPTH_TOLERANCE}° (empirical, see note above)")
print(f"Subjects evaluated: {len(results)}")
print(f"Subjects within tolerance (both resting AND extension): {n_within_tolerance}/{len(results)} "
      f"({100 * n_within_tolerance / len(results):.0f}%)")
print(f"Resting diff from target: mean={np.mean(resting_diffs):.1f}°  std={np.std(resting_diffs):.1f}°")
print(f"Extension diff from target: mean={np.mean(extension_diffs):.1f}°  std={np.std(extension_diffs):.1f}°")
print(f"Score: mean={np.mean(scores):.1f}%  std={np.std(scores):.1f}%  "
      f"min={np.min(scores):.1f}%  max={np.max(scores):.1f}%")

good_subjects = [r['subject'] for r in results if r['within_tolerance']]
print(f"\nSubjects within tolerance (candidates for a 'good' reference): {good_subjects}")