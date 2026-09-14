"""
src/evaluate_dataset/evaluate_on_new_subjects.py

Evaluates the trained GRU and TCN checkpoints (action_quality_net.pt,
action_quality_tcn.pt) on eval_action_quality_dataset.npz (new,
MMFi-independent subjects). Same metrics as src/compare_models.py, just
pointed at the eval set instead of MMFi's validation split.

Run AFTER build_eval_windowed_dataset.py.

CHANGE: instead of assuming every file lives in exactly one hardcoded
location, this script now SEARCHES a list of plausible directories for
each required file (checkpoints, eval npz) and uses whichever copy it
finds first. This means you don't need to manually copy files around if
they ended up in a slightly different (but still sensible) folder -- e.g.
running a script from a different working directory, or GRU/TCN being
created relative to cwd instead of the project root.
"""
import os
import sys

# --- Path anchoring -------------------------------------------------
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_SRC_DIR = os.path.dirname(_THIS_DIR)
_NN_DIR = os.path.join(_SRC_DIR, 'neural_network')
for _p in (_SRC_DIR, _NN_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np
import torch
from torch.utils.data import DataLoader

from utils import PROJECT_ROOT
from train_action_quality_net import ActionQualityNet, WindowDataset as GRUWindowDataset
from train_action_quality_tcn import ActionQualityTCN, WindowDataset as TCNWindowDataset

DEVICE = torch.device('cpu')

# --- Robust file lookup ----------------------------------------------
# Every plausible base directory a file might have ended up in, in order
# of preference. Using a set() first to avoid searching the same
# directory twice if some of these coincide.
_CWD = os.getcwd()
_CANDIDATE_BASE_DIRS = list(dict.fromkeys([
    PROJECT_ROOT,                                   # <root>/
    os.path.join(PROJECT_ROOT, 'src'),              # <root>/src/
    _SRC_DIR,                                        # src/ (same as above, kept for clarity)
    _THIS_DIR,                                        # src/evaluate_dataset/
    _CWD,                                             # wherever the script was launched from
]))


def find_file(filename, extra_subdirs=('',)):
    """Search for `filename` under each candidate base directory, also
    trying each of `extra_subdirs` (e.g. 'GRU', 'TCN') under every base.
    Returns the first existing path found, or None."""
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


def load_gru(path):
    ckpt = torch.load(path, map_location=DEVICE, weights_only=False)
    m = ActionQualityNet(input_size=ckpt['input_size'], num_classes=len(ckpt['class_names']),
                          hidden_size=ckpt['hidden_size'], num_layers=ckpt['num_layers']).to(DEVICE)
    m.load_state_dict(ckpt['model_state_dict'])
    m.eval()
    return m, ckpt


def load_tcn(path):
    ckpt = torch.load(path, map_location=DEVICE, weights_only=False)
    m = ActionQualityTCN(input_size=ckpt['input_size'], num_classes=len(ckpt['class_names']),
                          num_channels=ckpt['num_channels'], kernel_size=ckpt['kernel_size'],
                          dropout=ckpt['dropout']).to(DEVICE)
    m.load_state_dict(ckpt['model_state_dict'])
    m.eval()
    return m, ckpt


@torch.no_grad()
def evaluate(model, loader, num_classes):
    correct, total = 0, 0
    score_abs_error_sum = 0.0
    class_correct = np.zeros(num_classes)
    class_total = np.zeros(num_classes)

    for X, y_class, y_score in loader:
        X, y_class, y_score = X.to(DEVICE), y_class.to(DEVICE), y_score.to(DEVICE)
        class_logits, score_pred = model(X)
        preds = class_logits.argmax(dim=1)

        correct += (preds == y_class).sum().item()
        total += X.size(0)
        score_abs_error_sum += (score_pred - y_score).abs().sum().item() * 100.0

        for c in range(num_classes):
            mask = (y_class == c)
            class_total[c] += mask.sum().item()
            class_correct[c] += (preds[mask] == c).sum().item()

    per_class_acc = np.divide(class_correct, class_total,
                               out=np.zeros_like(class_correct), where=class_total > 0)
    return {'accuracy': correct / total, 'score_mae': score_abs_error_sum / total,
            'per_class_acc': per_class_acc}


def main():
    eval_npz_path = require_file(
        'eval_action_quality_dataset.npz',
        hint="Run build_eval_windowed_dataset.py first."
    )
    gru_ckpt_path = require_file(
        'action_quality_net.pt', extra_subdirs=('GRU', ''),
        hint="Run train_action_quality_net.py first (or check it wasn't saved under a different folder)."
    )
    tcn_ckpt_path = require_file(
        'action_quality_tcn.pt', extra_subdirs=('TCN', ''),
        hint="Run train_action_quality_tcn.py first (or check it wasn't saved under a different folder)."
    )

    data = np.load(eval_npz_path, allow_pickle=True)
    class_names = list(data['class_names'])
    num_classes = len(class_names)
    print(f"\nEval windows: {data['X'].shape[0]}\n")

    gru_model, _ = load_gru(gru_ckpt_path)
    tcn_model, _ = load_tcn(tcn_ckpt_path)

    gru_ds = GRUWindowDataset(data['X'], data['y_class'], data['y_score'])
    tcn_ds = TCNWindowDataset(data['X'], data['y_class'], data['y_score'])
    gru_loader = DataLoader(gru_ds, batch_size=64, shuffle=False)
    tcn_loader = DataLoader(tcn_ds, batch_size=64, shuffle=False)

    gru_metrics = evaluate(gru_model, gru_loader, num_classes)
    tcn_metrics = evaluate(tcn_model, tcn_loader, num_classes)

    print("=== Evaluation on NEW (non-MMFi) subjects ===")
    print(f"{'Metric':<28}{'GRU':>15}{'TCN':>15}")
    print(f"{'Accuracy':<28}{gru_metrics['accuracy']*100:>14.1f}%{tcn_metrics['accuracy']*100:>14.1f}%")
    print(f"{'Score MAE (0-100)':<28}{gru_metrics['score_mae']:>15.2f}{tcn_metrics['score_mae']:>15.2f}")

    print("\n=== Per-class accuracy ===")
    print(f"{'Class':<24}{'GRU':>12}{'TCN':>12}")
    for i, name in enumerate(class_names):
        print(f"{name:<24}{gru_metrics['per_class_acc'][i]*100:>11.1f}%{tcn_metrics['per_class_acc'][i]*100:>11.1f}%")


if __name__ == '__main__':
    main()