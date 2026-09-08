"""
Live inference: loads the trained multi-task network (action_quality_net.pt,
produced by train_action_quality_net.py) and runs it on a sliding window of
NORMALIZED webcam keypoints, showing (predicted exercise, quality score) as
an overlay on the video feed.

CRITICAL: this script re-uses keypoint_normalize.py's frame_is_usable() /
normalize_frame() -- the EXACT same normalization used to build the training
data in build_windowed_dataset.py. If you ever change the normalization
logic, change it only in keypoint_normalize.py so both the offline training
pipeline and this live script stay in sync (same rationale as utils.py's
docstring for calculate_depth_score -- one shared implementation, not two
that can drift apart).

IMPORTANT CAVEAT -- frame rate mismatch:
MMFi's keypoint sequences are NOT necessarily sampled at the same rate your
webcam captures at. The network was trained on windows of WINDOW_LENGTH
consecutive MMFi frames, which corresponds to whatever real-world duration
MMFi's own frame rate implies. If your webcam runs at ~30fps and MMFi's
frames are effectively sampled slower (e.g. ~10fps), a WINDOW_LENGTH-frame
window from the webcam covers LESS real-world time / motion than the window
the network learned on, and the "shape" of the motion in the window will
look different (more static). If predictions look unstable or scores look
systematically off compared to the MMFi-based evaluate_dataset_*.py scripts,
try setting FRAME_SKIP below so effectively 1 out of every N webcam frames
feeds the window, approximating MMFi's own frame rate. You'll need to check
what MMFi's actual capture rate is (or measure your own webcam's fps here)
to pick a sensible N -- there's no single correct answer built into this
script.

This file lives in <PROJECT_ROOT>/src/realtime_inference_action_quality.py,
alongside utils.py and keypoint_normalize.py.
"""
import os
import time
from collections import deque

import cv2
import numpy as np
import torch
import torch.nn as nn
from ultralytics import YOLO

from utils import MODEL_PATH, PROJECT_ROOT, select_patient_keypoints
from keypoint_normalize import frame_is_usable, normalize_frame, flatten_sequence, FLAT_SIZE

MODEL_CHECKPOINT = os.path.join(PROJECT_ROOT, 'action_quality_net.pt')

# --- Frame-rate mismatch mitigation (see caveat above) ---
# 1 = use every webcam frame (default, simplest). Set higher (e.g. 3) to
# only feed every Nth captured frame into the window, if you determine your
# webcam fps is roughly N times MMFi's effective sampling rate.
FRAME_SKIP = 1
_frame_counter = 0

# --- Smoothing ---
# Raw per-frame predictions from a 30-frame sliding window can flicker
# between classes/scores frame to frame, especially right as the window
# transitions between two different phases of a movement. An exponential
# moving average over the class-probability vector and the score keeps the
# on-screen readout stable without needing to wait for a full repetition.
PROB_SMOOTHING = 0.3   # lower = smoother/slower to react
SCORE_SMOOTHING = 0.3
smoothed_probs = None
smoothed_score = None

# --- Confidence gating ---
# Below this, the top class is shown as "uncertain" instead of committing
# to a specific exercise -- useful during the moment the patient is still
# getting into position, or transitioning between exercises.
CONFIDENCE_THRESHOLD = 0.5

CAMERA_INDEX = 0

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# --- Must match the architecture in train_action_quality_net.py exactly,
# since we're loading its state_dict. ---
class ActionQualityNet(nn.Module):
    def __init__(self, input_size, num_classes, hidden_size, num_layers):
        super().__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers=num_layers, batch_first=True)
        self.classifier = nn.Linear(hidden_size, num_classes)
        self.scorer = nn.Linear(hidden_size, 1)

    def forward(self, x):
        _, h_n = self.gru(x)
        last_hidden = h_n[-1]
        class_logits = self.classifier(last_hidden)
        score = torch.sigmoid(self.scorer(last_hidden)).squeeze(-1)
        return class_logits, score


def load_model():
    if not os.path.exists(MODEL_CHECKPOINT):
        print(f"Error: checkpoint not found at {MODEL_CHECKPOINT}. "
              "Run train_action_quality_net.py first.")
        exit()

    # weights_only=False: PyTorch 2.6 defaults to a restrictive "safe"
    # unpickler that rejects some numpy objects (e.g. class_names saved as
    # a numpy array of numpy scalars in train_action_quality_net.py). This
    # checkpoint is generated locally by your own training script, not
    # downloaded from an untrusted source, so it's safe to disable here.
    checkpoint = torch.load(MODEL_CHECKPOINT, map_location=DEVICE, weights_only=False)
    class_names = list(checkpoint['class_names'])
    window_length = int(checkpoint['window_length'])
    input_size = int(checkpoint['input_size'])
    hidden_size = int(checkpoint['hidden_size'])
    num_layers = int(checkpoint['num_layers'])

    model = ActionQualityNet(input_size, len(class_names), hidden_size, num_layers).to(DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    print(f"Loaded checkpoint: classes={class_names}  window_length={window_length}  "
          f"input_size={input_size}  hidden_size={hidden_size}  num_layers={num_layers}")

    return model, class_names, window_length


@torch.no_grad()
def run_inference(model, window_buffer):
    """window_buffer: list/deque of WINDOW_LENGTH (34,) float arrays.
    Returns (probs: (num_classes,) numpy, score_0_100: float)."""
    X = np.stack(list(window_buffer), axis=0).astype(np.float32)  # (window_length, 34)
    X = torch.from_numpy(X).unsqueeze(0).to(DEVICE)                # (1, window_length, 34)

    class_logits, score = model(X)
    probs = torch.softmax(class_logits, dim=1).squeeze(0).cpu().numpy()
    score_0_100 = float(score.item()) * 100.0
    return probs, score_0_100


def main():
    global smoothed_probs, smoothed_score, _frame_counter

    model, class_names, window_length = load_model()
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

            # Same "skip rather than fabricate" policy used everywhere else
            # in this pipeline: a frame that fails the hip/shoulder check
            # (needed for normalization itself) is dropped, not zero-filled.
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
            # Nobody detected: keep showing the last known result, don't
            # reset the buffer -- a brief detection gap (person stepping
            # slightly out of frame) shouldn't force starting over.
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