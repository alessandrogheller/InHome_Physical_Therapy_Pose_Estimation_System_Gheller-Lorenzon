"""
Live inference: loads a trained action-quality checkpoint (either the GRU
from train_action_quality_net.py or the TCN from train_action_quality_tcn.py)
and runs it on a sliding window of NORMALIZED webcam keypoints, showing
(predicted exercise, quality score) as an overlay on the video feed.

CHANGE VS THE ORIGINAL VERSION: this script now picks its architecture
automatically from the checkpoint's 'arch' field (set by both training
scripts), instead of hardcoding ActionQualityNet. This lets you point
CHECKPOINT_PATH at either action_quality_net.pt (GRU) or
action_quality_tcn.pt (TCN) and compare them live without touching any
other code. Checkpoints saved before this change (no 'arch' key) are
assumed to be GRU, for backward compatibility.

CRITICAL: this script re-uses keypoint_normalize.py's frame_is_usable() /
normalize_frame() -- the EXACT same normalization used to build the
training data in build_windowed_dataset.py. If you ever change the
normalization logic, change it only in keypoint_normalize.py so both the
offline training pipeline and this live script stay in sync.

IMPORTANT CAVEAT -- frame rate mismatch:
MMFi's keypoint sequences are NOT necessarily sampled at the same rate
your webcam captures at. See FRAME_SKIP below.

This file lives in <PROJECT_ROOT>/src/realtime_inference_action_quality.py,
alongside utils.py and keypoint_normalize.py.
"""
import os
import sys
from collections import deque

import cv2
import numpy as np
import torch
from ultralytics import YOLO

_SRC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _SRC_DIR)
sys.path.insert(0, os.path.join(_SRC_DIR, 'neural_network'))

from utils import MODEL_PATH, PROJECT_ROOT, GRU_DIR, TCN_DIR, select_patient_keypoints
from keypoint_normalize import frame_is_usable, normalize_frame, FLAT_SIZE
from train_action_quality_net import ActionQualityNet
from train_action_quality_tcn import ActionQualityTCN

# --- Which checkpoint to run live ---
# Point this at either checkpoint to compare them on the same webcam feed.
#CHECKPOINT_PATH = os.path.join(GRU_DIR, 'action_quality_net.pt')  # GRU
CHECKPOINT_PATH = os.path.join(TCN_DIR, 'action_quality_tcn.pt')  # TCN

# --- Frame-rate mismatch mitigation (see caveat above) ---
FRAME_SKIP = 1
_frame_counter = 0

# --- Smoothing ---
PROB_SMOOTHING = 0.3
SCORE_SMOOTHING = 0.3
smoothed_probs = None
smoothed_score = None

# --- Confidence gating ---
CONFIDENCE_THRESHOLD = 0.5

CAMERA_INDEX = 1

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def load_model(checkpoint_path):
    """Reconstruct whichever architecture the checkpoint was trained with.
    Checkpoints without an 'arch' key predate this change and are assumed
    to be the GRU (the only architecture that existed at the time)."""
    if not os.path.exists(checkpoint_path):
        print(f"Error: checkpoint not found at {checkpoint_path}. "
              "Run train_action_quality_net.py or train_action_quality_tcn.py first.")
        exit()

    # weights_only=False: these checkpoints are generated locally by your
    # own training scripts, not downloaded from an untrusted source.
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)
    arch = checkpoint.get('arch', 'gru')
    class_names = list(checkpoint['class_names'])
    window_length = int(checkpoint['window_length'])
    input_size = int(checkpoint['input_size'])

    if arch == 'gru':
        model = ActionQualityNet(
            input_size=input_size,
            num_classes=len(class_names),
            hidden_size=checkpoint['hidden_size'],
            num_layers=checkpoint['num_layers'],
        ).to(DEVICE)
    elif arch == 'tcn':
        model = ActionQualityTCN(
            input_size=input_size,
            num_classes=len(class_names),
            num_channels=checkpoint['num_channels'],
            kernel_size=checkpoint['kernel_size'],
            dropout=checkpoint['dropout'],
        ).to(DEVICE)
    else:
        raise ValueError(f"Unknown architecture '{arch}' in checkpoint.")

    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    print(f"Loaded checkpoint: arch={arch}  classes={class_names}  "
          f"window_length={window_length}  input_size={input_size}")

    return model, class_names, window_length


@torch.no_grad()
def run_inference(model, window_buffer):
    """window_buffer: list/deque of WINDOW_LENGTH (34,) float arrays.
    Returns (probs: (num_classes,) numpy, score_0_100: float). Works
    identically regardless of which architecture `model` actually is,
    since both expose the same forward(x) -> (class_logits, score)
    interface."""
    X = np.stack(list(window_buffer), axis=0).astype(np.float32)  # (window_length, 34)
    X = torch.from_numpy(X).unsqueeze(0).to(DEVICE)                # (1, window_length, 34)

    class_logits, score = model(X)
    probs = torch.softmax(class_logits, dim=1).squeeze(0).cpu().numpy()
    score_0_100 = float(score.item()) * 100.0
    return probs, score_0_100


def main():
    global smoothed_probs, smoothed_score, _frame_counter

    model, class_names, window_length = load_model(CHECKPOINT_PATH)
    yolo_model = YOLO(MODEL_PATH)

    window_buffer = deque(maxlen=window_length)
    patient_center = None

    last_result_text = "Collecting frames..."
    last_color = (200, 200, 200)

    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print("Error: could not open the webcam.")
        exit()

    cv2.namedWindow('Action Recognition + Quality - Press Q to quit', cv2.WINDOW_NORMAL)
    print("Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        _frame_counter += 1

        results = yolo_model(frame, verbose=False)
        annotated_frame = results[0].plot()

        kp, kp_conf, new_center = select_patient_keypoints(results, previous_center=patient_center)

        if kp is not None and (_frame_counter % FRAME_SKIP == 0):
            patient_center = new_center
            cv2.circle(annotated_frame, (int(patient_center[0]), int(patient_center[1])),
                       8, (255, 0, 255), -1)

            if frame_is_usable(kp, kp_conf):
                normalized_frame = normalize_frame(kp)  # (17, 2) or None
                if normalized_frame is not None:
                    flat = normalized_frame.reshape(FLAT_SIZE)  # (34,)
                    window_buffer.append(flat)

            if len(window_buffer) == window_length:
                probs, score_0_100 = run_inference(model, window_buffer)

                smoothed_probs = probs if smoothed_probs is None else (
                    PROB_SMOOTHING * probs + (1 - PROB_SMOOTHING) * smoothed_probs)
                smoothed_score = score_0_100 if smoothed_score is None else (
                    SCORE_SMOOTHING * score_0_100 + (1 - SCORE_SMOOTHING) * smoothed_score)

                top_idx = int(np.argmax(smoothed_probs))
                top_conf = float(smoothed_probs[top_idx])

                if top_conf < CONFIDENCE_THRESHOLD:
                    last_result_text = f"Uncertain (top guess: {class_names[top_idx]}, {top_conf*100:.0f}%)"
                    last_color = (150, 150, 150)
                else:
                    exercise_label = class_names[top_idx]
                    last_result_text = (f"{exercise_label}: {smoothed_score:.1f}% "
                                         f"(conf {top_conf*100:.0f}%)")
                    if smoothed_score >= 80:
                        last_color = (0, 200, 0)
                    elif smoothed_score >= 50:
                        last_color = (0, 200, 255)
                    else:
                        last_color = (0, 0, 255)
            else:
                last_result_text = f"Collecting frames... ({len(window_buffer)}/{window_length})"
                last_color = (200, 200, 200)
        elif kp is None:
            pass

        cv2.putText(annotated_frame, last_result_text, (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, last_color, 2, cv2.LINE_AA)

        cv2.imshow('Action Recognition + Quality - Press Q to quit', annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()