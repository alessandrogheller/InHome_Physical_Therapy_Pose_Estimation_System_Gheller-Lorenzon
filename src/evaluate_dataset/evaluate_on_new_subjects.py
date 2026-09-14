"""
src/evaluate_dataset/evaluate_on_new_subjects.py

Evaluates the trained GRU and TCN checkpoints (action_quality_net.pt,
action_quality_tcn.pt -- both in PROJECT_ROOT) on
eval_action_quality_dataset.npz (new, MMFi-independent subjects). Same
metrics as src/compare_models.py, just pointed at the eval set instead of
MMFi's validation split.

Run AFTER build_eval_windowed_dataset.py.
"""
import os
import sys

# --- Path anchoring -------------------------------------------------
# This script lives in src/evaluate_dataset/. It needs:
#   - utils.py                       -> src/
#   - train_action_quality_net.py    -> src/neural_network/
#   - train_action_quality_tcn.py    -> src/neural_network/
# train_action_quality_net.py/tcn.py themselves do `from utils import
# PROJECT_ROOT`, so src/ must ALSO be on the path for those imports to
# resolve once we import them from here -- not just src/neural_network/.
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

EVAL_NPZ = os.path.join(PROJECT_ROOT, 'eval_action_quality_dataset.npz')
GRU_CHECKPOINT = os.path.join(PROJECT_ROOT,'GRU', 'action_quality_net.pt')
TCN_CHECKPOINT = os.path.join(PROJECT_ROOT,'TCN', 'action_quality_tcn.pt')
DEVICE = torch.device('cpu')


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
    if not os.path.exists(EVAL_NPZ):
        print(f"ERROR: {EVAL_NPZ} not found. Run build_eval_windowed_dataset.py first.")
        return
    missing = [p for p in (GRU_CHECKPOINT, TCN_CHECKPOINT) if not os.path.exists(p)]
    if missing:
        print("ERROR: missing checkpoint(s):")
        for p in missing:
            print(f"  {p}")
        return

    data = np.load(EVAL_NPZ, allow_pickle=True)
    class_names = list(data['class_names'])
    num_classes = len(class_names)
    print(f"Eval windows: {data['X'].shape[0]}\n")

    gru_model, _ = load_gru(GRU_CHECKPOINT)
    tcn_model, _ = load_tcn(TCN_CHECKPOINT)

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