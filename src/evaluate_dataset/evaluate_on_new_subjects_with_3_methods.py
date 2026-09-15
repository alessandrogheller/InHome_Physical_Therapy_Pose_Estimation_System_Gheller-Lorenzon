"""
src/evaluate_dataset/evaluate_three_methods.py

Runs all THREE quality-assessment methods this project implements on the
SAME set of independent subjects/videos -- the ones you already extracted
into an eval folder with extract_keypoints_from_video.py -- and reports
their scores side by side:

  1. Rule-based / geometric method  (same logic as score_eval_subjects.py,
                                      scored against quality_targets.json)
  2. GRU neural network             (GRU/action_quality_net.pt)
  3. TCN neural network             (TCN/action_quality_tcn.pt)

For every subject/exercise video found in the eval folder, each method
independently produces a 0-100 quality score. The two neural methods also
produce a predicted exercise label, which is checked against the ground
truth implied by the folder name (the exercise sub-folder itself, e.g.
"squat", "lunge_left", ...).

Output:
  - one row per (subject, exercise) printed to the terminal as it's scored
  - a CSV report (three_methods_comparison.csv) with one row per video
  - a summary at the end: per-method score statistics, GRU/TCN
    classification accuracy against the ground-truth exercise label, and
    pairwise agreement between the three methods (Pearson correlation +
    mean absolute difference of their scores), so you can see e.g. whether
    the two neural methods agree with each other more than with the
    rule-based baseline.

Nothing here is recomputed from scratch: targets come from
quality_targets.json (generate_quality_labels.py) and the two checkpoints
come from train_action_quality_net.py / train_action_quality_tcn.py. This
script only ties the three together on the same input.

Prerequisites (run in order, if not already done):
  1. extract_keypoints_from_video.py  (once per video, fills the eval folder)
  2. generate_quality_labels.py       -> quality_targets.json
  3. train_action_quality_net.py      -> GRU/action_quality_net.pt
  4. train_action_quality_tcn.py      -> TCN/action_quality_tcn.pt

Usage:
    python evaluate_three_methods.py [eval_dataset_root]

If eval_dataset_root is omitted, it defaults to <PROJECT_ROOT>/eval_dataset
(the same default extract_keypoints_from_video.py writes to). Point it at
any folder with the same layout:

    <eval_dataset_root>/<subject_id>/<exercise_name>/keypoints.npy
    <eval_dataset_root>/<subject_id>/<exercise_name>/confidences.npy   (optional)

where <exercise_name> is one of: squat, lunge_left, lunge_right,
limb_extension_left, limb_extension_right, jumping_jacks.

This file lives in <PROJECT_ROOT>/src/evaluate_dataset/, alongside
score_eval_subjects.py and evaluate_on_new_subjects.py, which this script
overlaps with in purpose but combines into one direct 3-way comparison.
"""
import os
import sys
import csv
import json
import numpy as np

# --- Path anchoring -------------------------------------------------
# Same pattern used by the other evaluate_dataset/ scripts: add src/ and
# src/neural_network/ to sys.path explicitly so this works regardless of
# the current working directory it's launched from.
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_SRC_DIR = os.path.dirname(_THIS_DIR)
_NN_DIR = os.path.join(_SRC_DIR, 'neural_network')
for _p in (_SRC_DIR, _NN_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import torch

from utils import (
    PROJECT_ROOT,
    LEFT_HIP, LEFT_KNEE, LEFT_ANKLE, RIGHT_HIP, RIGHT_KNEE, RIGHT_ANKLE,
    LEFT_SHOULDER, LEFT_ELBOW, LEFT_WRIST, RIGHT_SHOULDER, RIGHT_ELBOW, RIGHT_WRIST,
    calculate_angle, calculate_depth_score, keypoints_are_valid,
)
from keypoint_normalize import normalize_sequence, flatten_sequence, FLAT_SIZE
from train_action_quality_net import ActionQualityNet
from train_action_quality_tcn import ActionQualityTCN

DEVICE = torch.device('cpu')  # evaluation only, no need for GPU here

# Must match build_windowed_dataset.py / build_eval_windowed_dataset.py
# exactly -- this is the class-index mapping the trained checkpoints
# expect their classification head's outputs to follow.
CLASS_NAMES = ['squat', 'lunge_left', 'lunge_right',
               'limb_extension_left', 'limb_extension_right', 'jumping_jacks']

# Same stride used to build the training windows (build_windowed_dataset.py).
# Not stored in the checkpoint, so it's hardcoded here too, exactly like
# build_eval_windowed_dataset.py already does.
WINDOW_STRIDE = 10

TARGETS_JSON = os.path.join(PROJECT_ROOT, 'quality_targets.json')
DEFAULT_EVAL_ROOT = os.path.join(PROJECT_ROOT, 'eval_dataset')
OUTPUT_CSV = os.path.join(PROJECT_ROOT, 'three_methods_comparison.csv')

# --- Robust checkpoint lookup -----------------------------------------
# A trained checkpoint could plausibly end up in a few different places
# depending on where each training script was launched from -- same
# reasoning/implementation as evaluate_on_new_subjects.py.
_CWD = os.getcwd()
_CANDIDATE_BASE_DIRS = list(dict.fromkeys([
    PROJECT_ROOT,
    os.path.join(PROJECT_ROOT, 'src'),
    _SRC_DIR,
    _THIS_DIR,
    _CWD,
]))


def find_file(filename, extra_subdirs=('',)):
    checked = []
    for base in _CANDIDATE_BASE_DIRS:
        for sub in extra_subdirs:
            candidate = os.path.join(base, sub, filename) if sub else os.path.join(base, filename)
            checked.append(candidate)
            if os.path.exists(candidate):
                return candidate, checked
    return None, checked


def require_file(filename, extra_subdirs=('',), hint=""):
    path, checked = find_file(filename, extra_subdirs)
    if path is None:
        print(f"ERROR: could not find '{filename}'. Looked in:")
        for c in checked:
            print(f"  - {c}")
        if hint:
            print(hint)
        sys.exit(1)
    print(f"Found {filename} -> {path}")
    return path


# --- Rule-based / geometric method --------------------------------------
# Identical logic to score_eval_subjects.py, kept here so this script is
# self-contained and always scores every method against the exact same
# targets used by the rest of the pipeline.
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
    hip_i, knee_i, ankle_i = joints
    angles = []
    for kp, conf in zip(kp_seq, conf_seq):
        if keypoints_are_valid(kp, conf, [hip_i, knee_i, ankle_i]):
            angles.append(calculate_angle(kp[hip_i], kp[knee_i], kp[ankle_i]))
    return np.array(angles)


def jj_signals(kp_seq, conf_seq):
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


def rule_based_score(exercise_name, kp_seq, conf_seq, targets):
    """Returns a 0-100 score, or None if no valid frame was found."""
    if exercise_name in LEG_JOINTS:
        angles = angle_series(kp_seq, conf_seq, LEG_JOINTS[exercise_name])
        if len(angles) == 0:
            return None
        depth_diff = abs(float(angles.min()) - targets['depth_target'])
        return calculate_depth_score(depth_diff)

    if exercise_name in ARM_JOINTS:
        angles = angle_series(kp_seq, conf_seq, ARM_JOINTS[exercise_name])
        if len(angles) == 0:
            return None
        resting_diff = abs(float(angles.min()) - targets['resting_target'])
        extension_diff = abs(float(angles.max()) - targets['extension_target'])
        return (calculate_depth_score(resting_diff) + calculate_depth_score(extension_diff)) / 2.0

    if exercise_name == 'jumping_jacks':
        leg, arm = jj_signals(kp_seq, conf_seq)
        if len(leg) == 0:
            return None
        leg_diff = abs(float(np.percentile(leg, 95)) - targets['leg_target'])
        arm_diff = abs(float(np.percentile(arm, 95)) - targets['arm_target'])
        leg_score = calculate_depth_score(leg_diff, tolerance=targets['leg_tolerance'], falloff=targets['leg_falloff'])
        arm_score = calculate_depth_score(arm_diff, tolerance=targets['arm_tolerance'], falloff=targets['arm_falloff'])
        return (leg_score + arm_score) / 2.0

    raise ValueError(f"Unknown exercise '{exercise_name}'")


# --- Neural methods (GRU / TCN) -----------------------------------------
def load_gru(path):
    ckpt = torch.load(path, map_location=DEVICE, weights_only=False)
    model = ActionQualityNet(
        input_size=ckpt['input_size'], num_classes=len(ckpt['class_names']),
        hidden_size=ckpt['hidden_size'], num_layers=ckpt['num_layers'],
    ).to(DEVICE)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    return model, ckpt


def load_tcn(path):
    ckpt = torch.load(path, map_location=DEVICE, weights_only=False)
    model = ActionQualityTCN(
        input_size=ckpt['input_size'], num_classes=len(ckpt['class_names']),
        num_channels=ckpt['num_channels'], kernel_size=ckpt['kernel_size'],
        dropout=ckpt['dropout'],
    ).to(DEVICE)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    return model, ckpt


def make_windows(seq_flat, length, stride):
    T = seq_flat.shape[0]
    if T < length:
        return np.zeros((0, length, seq_flat.shape[1]), dtype=np.float32)
    windows = [seq_flat[s:s + length] for s in range(0, T - length + 1, stride)]
    return np.stack(windows, axis=0).astype(np.float32)


@torch.no_grad()
def neural_score(model, class_names, window_length, normalized_flat):
    """Slides WINDOW_LENGTH/WINDOW_STRIDE windows over the already
    normalized+flattened (T, 34) sequence, runs every window through
    `model` in a single batched forward pass, and averages the per-window
    predictions into one (score_0_100, predicted_class_name,
    predicted_class_confidence) triple for the whole video. Returns None
    if the sequence is too short to produce even one window."""
    windows = make_windows(normalized_flat, window_length, WINDOW_STRIDE)
    if windows.shape[0] == 0:
        return None

    X = torch.from_numpy(windows).float().to(DEVICE)  # (N, window_length, 34)
    class_logits, score = model(X)
    probs = torch.softmax(class_logits, dim=1).cpu().numpy()  # (N, num_classes)
    scores = score.cpu().numpy()  # (N,), in [0, 1]

    mean_probs = probs.mean(axis=0)
    pred_idx = int(np.argmax(mean_probs))
    return float(scores.mean()) * 100.0, class_names[pred_idx], float(mean_probs[pred_idx])


# --- Correlation / agreement helpers ------------------------------------
def pearson_and_mae(a, b):
    """a, b: same-length lists of paired scores (already filtered to rows
    where both are available). Returns (pearson_r, mean_abs_diff), or
    (None, None) if there are fewer than 2 paired points."""
    if len(a) < 2:
        return None, None
    a_arr, b_arr = np.array(a, dtype=float), np.array(b, dtype=float)
    if np.std(a_arr) < 1e-9 or np.std(b_arr) < 1e-9:
        r = None  # correlation undefined if one side is constant
    else:
        r = float(np.corrcoef(a_arr, b_arr)[0, 1])
    mae = float(np.mean(np.abs(a_arr - b_arr)))
    return r, mae


def main():
    eval_root = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_EVAL_ROOT
    eval_root = os.path.abspath(eval_root)
    if not os.path.isdir(eval_root):
        print(f"ERROR: eval folder not found: {eval_root}")
        print("Pass the folder as an argument, e.g.:")
        print("  python evaluate_three_methods.py /path/to/your/eval_dataset")
        sys.exit(1)
    print(f"Evaluation folder: {eval_root}\n")

    targets_path = require_file(
        os.path.basename(TARGETS_JSON),
        hint="Run generate_quality_labels.py first (writes quality_targets.json)."
    ) if not os.path.exists(TARGETS_JSON) else TARGETS_JSON
    with open(targets_path) as f:
        all_targets = json.load(f)

    gru_path = require_file('action_quality_net.pt', extra_subdirs=('GRU', ''),
                             hint="Run train_action_quality_net.py first.")
    tcn_path = require_file('action_quality_tcn.pt', extra_subdirs=('TCN', ''),
                             hint="Run train_action_quality_tcn.py first.")

    gru_model, gru_ckpt = load_gru(gru_path)
    tcn_model, tcn_ckpt = load_tcn(tcn_path)
    gru_window_length = int(gru_ckpt['window_length'])
    tcn_window_length = int(tcn_ckpt['window_length'])
    gru_class_names = list(gru_ckpt['class_names'])
    tcn_class_names = list(tcn_ckpt['class_names'])
    print(f"GRU checkpoint: window_length={gru_window_length}  classes={gru_class_names}")
    print(f"TCN checkpoint: window_length={tcn_window_length}  classes={tcn_class_names}\n")

    rows = []  # each: dict with subject, exercise, and per-method results

    for subject_id in sorted(os.listdir(eval_root)):
        subject_dir = os.path.join(eval_root, subject_id)
        if not os.path.isdir(subject_dir):
            continue

        for exercise_name in sorted(os.listdir(subject_dir)):
            ex_dir = os.path.join(subject_dir, exercise_name)
            kp_path = os.path.join(ex_dir, 'keypoints.npy')
            conf_path = os.path.join(ex_dir, 'confidences.npy')
            if not os.path.isfile(kp_path):
                continue
            if exercise_name not in CLASS_NAMES:
                print(f"  {subject_id}/{exercise_name}: not a recognized exercise name, skipping.")
                continue
            if exercise_name not in all_targets:
                print(f"  {subject_id}/{exercise_name}: no rule-based target in quality_targets.json, skipping.")
                continue

            kp_seq = np.load(kp_path)  # (T, 17, 2)
            conf_seq = np.load(conf_path) if os.path.isfile(conf_path) else [None] * len(kp_seq)

            row = {'subject': subject_id, 'exercise': exercise_name, 'n_frames': len(kp_seq)}

            # --- Method 1: rule-based ---
            row['rule_based_score'] = rule_based_score(exercise_name, kp_seq, conf_seq, all_targets[exercise_name])

            # --- Normalize once, shared by both neural methods ---
            normalized = normalize_sequence(kp_seq, conf_seq if os.path.isfile(conf_path) else None)
            normalized_flat = flatten_sequence(normalized) if normalized.shape[0] > 0 else None

            # --- Method 2: GRU ---
            gru_result = (neural_score(gru_model, gru_class_names, gru_window_length, normalized_flat)
                          if normalized_flat is not None else None)
            if gru_result is not None:
                row['gru_score'], row['gru_pred_class'], row['gru_pred_conf'] = gru_result
                row['gru_class_correct'] = (row['gru_pred_class'] == exercise_name)
            else:
                row['gru_score'] = row['gru_pred_class'] = row['gru_pred_conf'] = row['gru_class_correct'] = None

            # --- Method 3: TCN ---
            tcn_result = (neural_score(tcn_model, tcn_class_names, tcn_window_length, normalized_flat)
                          if normalized_flat is not None else None)
            if tcn_result is not None:
                row['tcn_score'], row['tcn_pred_class'], row['tcn_pred_conf'] = tcn_result
                row['tcn_class_correct'] = (row['tcn_pred_class'] == exercise_name)
            else:
                row['tcn_score'] = row['tcn_pred_class'] = row['tcn_pred_conf'] = row['tcn_class_correct'] = None

            rows.append(row)

            def fmt(v):
                return f"{v:5.1f}%" if isinstance(v, float) else " N/A "
            print(f"{subject_id}/{exercise_name:22s}  "
                  f"rule={fmt(row['rule_based_score'])}  "
                  f"GRU={fmt(row['gru_score'])} ({row['gru_pred_class']})  "
                  f"TCN={fmt(row['tcn_score'])} ({row['tcn_pred_class']})")

    if not rows:
        print("\nNo scoreable (subject, exercise) videos found. Aborting.")
        sys.exit(1)

    # --- Write the per-video CSV report ---
    fieldnames = ['subject', 'exercise', 'n_frames',
                  'rule_based_score', 'gru_score', 'gru_pred_class', 'gru_pred_conf', 'gru_class_correct',
                  'tcn_score', 'tcn_pred_class', 'tcn_pred_conf', 'tcn_class_correct']
    with open(OUTPUT_CSV, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nSaved per-video comparison to: {OUTPUT_CSV}")

    # --- Summary: per-method score statistics ---
    print("\n=== Score statistics per method (0-100) ===")
    for key, label in [('rule_based_score', 'Rule-based'), ('gru_score', 'GRU'), ('tcn_score', 'TCN')]:
        vals = [r[key] for r in rows if r[key] is not None]
        if vals:
            print(f"{label:<12} n={len(vals):3d}  mean={np.mean(vals):5.1f}%  std={np.std(vals):5.1f}%  "
                  f"min={np.min(vals):5.1f}%  max={np.max(vals):5.1f}%")
        else:
            print(f"{label:<12} no scoreable videos")

    # --- Summary: exercise-classification accuracy (neural methods only) ---
    print("\n=== Exercise classification accuracy (ground truth = folder name) ===")
    for key, label in [('gru_class_correct', 'GRU'), ('tcn_class_correct', 'TCN')]:
        flags = [r[key] for r in rows if r[key] is not None]
        if flags:
            acc = sum(flags) / len(flags)
            print(f"{label:<12} n={len(flags):3d}  accuracy={acc*100:5.1f}%")
        else:
            print(f"{label:<12} no scoreable videos")

    # --- Summary: pairwise agreement between the three methods' scores ---
    print("\n=== Pairwise agreement between methods (paired videos only) ===")
    pairs = [
        ('rule_based_score', 'gru_score', 'Rule-based vs GRU'),
        ('rule_based_score', 'tcn_score', 'Rule-based vs TCN'),
        ('gru_score', 'tcn_score', 'GRU vs TCN'),
    ]
    for key_a, key_b, label in pairs:
        a = [r[key_a] for r in rows if r[key_a] is not None and r[key_b] is not None]
        b = [r[key_b] for r in rows if r[key_a] is not None and r[key_b] is not None]
        r_value, mae = pearson_and_mae(a, b)
        if r_value is None and mae is None:
            print(f"{label:<22} not enough paired videos to compare")
        else:
            r_str = f"{r_value:+.2f}" if r_value is not None else "undefined (constant scores)"
            print(f"{label:<22} n={len(a):3d}  Pearson r={r_str}  mean_abs_diff={mae:5.1f} points")

    print("\nNOTE: agreement here means the three methods land on similar scores for the "
          "same repetition, not that any one of them is 'correct' -- none of the three has "
          "access to a clinically validated ground-truth quality label (see the README's "
          "'Known Limitations' section). Use this comparison to discuss where the "
          "geometric and learned approaches agree or diverge, as required by the "
          "project's Model Comparison guideline.")


if __name__ == '__main__':
    main()