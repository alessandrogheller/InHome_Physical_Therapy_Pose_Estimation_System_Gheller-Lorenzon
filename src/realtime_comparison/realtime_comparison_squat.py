"""Compare real-time squat repetitions with a reference movement."""

import cv2
import time
import sys
import os
import numpy as np

from ultralytics import YOLO

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import (
    MODEL_PATH, REFERENCE_PATH,
    LEFT_HIP, LEFT_KNEE, LEFT_ANKLE,
    calculate_angle, calculate_depth_score, keypoints_are_valid,
    select_patient_keypoints,
)

# --- Load model and reference ---
model = YOLO(MODEL_PATH)
reference = np.load(REFERENCE_PATH)

print(f"Reference loaded: {len(reference)} frames.")
print(f"Reference range: min={reference.min():.1f}°, max={reference.max():.1f}°")

# --- Thresholds for the repetition detector ---
# Initialize thresholds from the reference and refine them during calibration.
HIGH_THRESHOLD = 165.0   # above this angle = considered "standing"
LOW_THRESHOLD = 145.0    # below this angle = considered "moving/down"
MIN_REP_FRAMES = 10      # discard repetitions that are too short (likely noise)

# Use the same scoring function as the offline evaluation pipeline.

# Adapt thresholds to the patient's observed range of motion.
USE_ADAPTIVE_THRESHOLDS = True
CALIBRATION_DURATION = 5.0  # seconds; ask the patient to stand still at first

# Discard repetitions that do not return to the standing range in time.
MAX_MOVING_DURATION = 8.0  # seconds

# --- Smoothing filter for the angle signal ---
SMOOTHING_FACTOR = 0.3  # lower = smoother/slower to react, higher = more responsive
smoothed_angle = None

# --- Detector state ---
state = "calibrating" if USE_ADAPTIVE_THRESHOLDS else "standing"
rep_buffer = []
last_result_text = "Waiting for movement..."
last_color = (200, 200, 200)
moving_start_time = None

# Store recent standing angles to align the patient's pose with the reference.
standing_angle_history = []
MAX_STANDING_HISTORY = 10

# Calibration-phase buffer (only used if USE_ADAPTIVE_THRESHOLDS)
calibration_angles = []
calibration_start_time = time.time()

# Keep tracking the same detected patient across frames.
patient_center = None

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: could not open the webcam.")
    exit()

cv2.namedWindow('Movement Comparison - Press Q to quit', cv2.WINDOW_NORMAL)
print("Press 'q' to quit.")
if USE_ADAPTIVE_THRESHOLDS:
    print(f"Calibrating for {CALIBRATION_DURATION:.0f}s -- please stand still and face the camera.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame, verbose=False)
    annotated_frame = results[0].plot()

    kp, kp_conf, new_center = select_patient_keypoints(results, previous_center=patient_center)

    if kp is not None:
        patient_center = new_center
        # Mark the person selected as the patient.
        cv2.circle(annotated_frame, (int(patient_center[0]), int(patient_center[1])),
                   8, (255, 0, 255), -1)

        if keypoints_are_valid(kp, kp_conf, [LEFT_HIP, LEFT_KNEE, LEFT_ANKLE]):
            raw_angle = calculate_angle(kp[LEFT_HIP], kp[LEFT_KNEE], kp[LEFT_ANKLE])

            # Smooth the angle to reduce frame-to-frame jitter.
            if smoothed_angle is None:
                smoothed_angle = raw_angle
            else:
                smoothed_angle = SMOOTHING_FACTOR * raw_angle + (1 - SMOOTHING_FACTOR) * smoothed_angle

            angle = smoothed_angle

            # --- Adaptive threshold calibration ---
            if state == "calibrating":
                calibration_angles.append(angle)
                if time.time() - calibration_start_time > CALIBRATION_DURATION:
                    obs_min = min(calibration_angles)
                    obs_max = max(calibration_angles)
                    obs_range = max(obs_max - obs_min, 1e-3)
                    HIGH_THRESHOLD = obs_max - 0.1 * obs_range
                    LOW_THRESHOLD = obs_max - 0.3 * obs_range
                    state = "standing"
                    print(f"Calibration done: standing baseline={obs_max:.1f}°  "
                          f"HIGH_THRESHOLD={HIGH_THRESHOLD:.1f}°  LOW_THRESHOLD={LOW_THRESHOLD:.1f}°")

            # --- Repetition state machine ---
            elif state == "standing":
                standing_angle_history.append(angle)
                if len(standing_angle_history) > MAX_STANDING_HISTORY:
                    standing_angle_history.pop(0)

                if angle < LOW_THRESHOLD:
                    state = "moving"
                    rep_buffer = [angle]
                    moving_start_time = time.time()

            elif state == "moving":
                rep_buffer.append(angle)

                timed_out = (time.time() - moving_start_time) > MAX_MOVING_DURATION

                if timed_out:
                    # Prevent the detector from remaining in "moving" forever.
                    last_result_text = "Movement timeout, discarded"
                    last_color = (150, 150, 150)
                    state = "standing"
                    rep_buffer = []

                elif angle > HIGH_THRESHOLD:
                    state = "standing"

                    if len(rep_buffer) >= MIN_REP_FRAMES and len(standing_angle_history) > 0:
                        # Align the standing angle without changing movement depth.
                        user_standing_baseline = np.mean(standing_angle_history)
                        calibration_offset = reference.max() - user_standing_baseline
                        corrected_rep = [a + calibration_offset for a in rep_buffer]

                        depth_achieved = min(corrected_rep)
                        depth_target = reference.min()
                        depth_diff = abs(depth_achieved - depth_target)

                        # Use the same two-zone scoring function as offline evaluation.
                        accuracy_pct = calculate_depth_score(depth_diff)

                        last_result_text = f"Repetition: {accuracy_pct:.1f}% correct"
                        if accuracy_pct >= 80:
                            last_color = (0, 200, 0)
                        elif accuracy_pct >= 50:
                            last_color = (0, 200, 255)
                        else:
                            last_color = (0, 0, 255)

                        print(f"Rep: depth_achieved={depth_achieved:.1f}° depth_target={depth_target:.1f}° "
                              f"diff={depth_diff:.1f}° (offset: {calibration_offset:.1f}°) "
                              f"-> score={accuracy_pct:.1f}%")
                    else:
                        last_result_text = "Movement too short, discarded"
                        last_color = (150, 150, 150)

                    rep_buffer = []
    else:
        # Nobody detected this frame: don't update angle/state, just show the frame.
        pass

    # --- On-screen status text ---
    state_text = {"calibrating": "CALIBRATING...", "standing": "STANDING", "moving": "MOVING..."}[state]
    cv2.putText(annotated_frame, state_text, (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2, cv2.LINE_AA)
    cv2.putText(annotated_frame, last_result_text, (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, last_color, 2, cv2.LINE_AA)

    cv2.imshow('Movement Comparison - Press Q to quit', annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()