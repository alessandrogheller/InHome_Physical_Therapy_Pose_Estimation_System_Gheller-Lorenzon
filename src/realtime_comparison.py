import cv2
import time
import numpy as np
from ultralytics import YOLO

from utils import (
    MODEL_PATH, REFERENCE_PATH,
    LEFT_HIP, LEFT_KNEE, LEFT_ANKLE,
    calculate_angle, calculate_depth_score, keypoints_are_valid,
    select_patient_keypoints,
)

# --- Load model and reference (now via portable paths from utils.py) ---
model = YOLO(MODEL_PATH)
reference = np.load(REFERENCE_PATH)

print(f"Reference loaded: {len(reference)} frames.")
print(f"Reference range: min={reference.min():.1f}°, max={reference.max():.1f}°")

# --- Thresholds for the repetition detector ---
# These are derived from the reference range by default. If a patient's own
# range of motion never reaches HIGH_THRESHOLD (e.g. reduced mobility), an
# initial adaptive calibration phase (below) re-estimates them from the
# patient's own observed range instead.
HIGH_THRESHOLD = 165.0   # above this angle = considered "standing"
LOW_THRESHOLD = 145.0    # below this angle = considered "moving/down"
MIN_REP_FRAMES = 10      # discard repetitions that are too short (likely noise)

# Fix: previously the live score used a different (linear) formula than the
# offline MMFi evaluation. Both now use calculate_depth_score() from utils.py
# so the two pipelines are directly comparable.

# --- Adaptive threshold calibration ---
# If True, the first CALIBRATION_DURATION seconds are used to observe the
# patient's own range of motion and set HIGH/LOW_THRESHOLD relative to it,
# instead of using the fixed values above (which assume a healthy-range
# reference and may never be reached by a patient with reduced mobility).
USE_ADAPTIVE_THRESHOLDS = True
CALIBRATION_DURATION = 5.0  # seconds; ask the patient to stand still at first

# --- Timeout for an in-progress repetition ---
# If the angle never comes back above HIGH_THRESHOLD within this many
# seconds, the repetition is discarded instead of leaving the state machine
# stuck in "moving" forever.
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

# Tracks the patient's recent "standing" angle, used to calibrate the offset
# between the patient's setup (camera angle, body proportions) and the
# reference sequence, without distorting the depth/amplitude of the squat.
standing_angle_history = []
MAX_STANDING_HISTORY = 10

# Calibration-phase buffer (only used if USE_ADAPTIVE_THRESHOLDS)
calibration_angles = []
calibration_start_time = time.time()

# Multi-person tracking: remembers where the patient was last seen so we
# keep following the same person frame-to-frame instead of "jumping" to
# someone else who enters the frame (e.g. a caregiver).
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
        # Visual feedback: mark which detected person is being tracked as the patient.
        cv2.circle(annotated_frame, (int(patient_center[0]), int(patient_center[1])),
                   8, (255, 0, 255), -1)

        if keypoints_are_valid(kp, kp_conf, [LEFT_HIP, LEFT_KNEE, LEFT_ANKLE]):
            raw_angle = calculate_angle(kp[LEFT_HIP], kp[LEFT_KNEE], kp[LEFT_ANKLE])

            # Exponential smoothing to reduce frame-to-frame jitter
            if smoothed_angle is None:
                smoothed_angle = raw_angle
            else:
                smoothed_angle = SMOOTHING_FACTOR * raw_angle + (1 - SMOOTHING_FACTOR) * smoothed_angle

            angle = smoothed_angle

            # --- Adaptive threshold calibration phase ---
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

            # --- State machine logic ---
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
                    # Fix: without this, a patient who never reaches
                    # HIGH_THRESHOLD (e.g. limited mobility, or briefly
                    # leaving the frame) would leave the detector stuck in
                    # "moving" indefinitely.
                    last_result_text = "Movement timeout, discarded"
                    last_color = (150, 150, 150)
                    state = "standing"
                    rep_buffer = []

                elif angle > HIGH_THRESHOLD:
                    state = "standing"

                    if len(rep_buffer) >= MIN_REP_FRAMES and len(standing_angle_history) > 0:
                        # Calibration: align the patient's standing angle to
                        # the reference's standing angle (additive shift
                        # only, so the actual depth/amplitude is preserved).
                        user_standing_baseline = np.mean(standing_angle_history)
                        calibration_offset = reference.max() - user_standing_baseline
                        corrected_rep = [a + calibration_offset for a in rep_buffer]

                        depth_achieved = min(corrected_rep)
                        depth_target = reference.min()
                        depth_diff = abs(depth_achieved - depth_target)

                        # Unified scoring: same two-zone function used offline
                        # on the MMFi dataset (evaluate_dataset_fixed_targets.py).
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