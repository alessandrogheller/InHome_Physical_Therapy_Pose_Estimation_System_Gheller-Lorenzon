import cv2
import os
import numpy as np
from ultralytics import YOLO

# Project root directory
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Load the YOLO-Pose model
model = YOLO(os.path.join(PROJECT_ROOT, 'yolov8n-pose.pt'))

# Load the reference curve
reference = np.load(os.path.join(PROJECT_ROOT, 'squat_reference.npy'))

print(f"Reference loaded: {len(reference)} frames.")
print(f"Reference range: min={reference.min():.1f}°, max={reference.max():.1f}°")

# COCO keypoint indices
LEFT_HIP, LEFT_KNEE, LEFT_ANKLE = 11, 13, 15

def calculate_angle(a, b, c):
    """Calculate the angle (in degrees) at vertex b, between segments a-b and b-c."""
    a, b, c = np.array(a), np.array(b), np.array(c)
    ba = a - b
    bc = c - b
    cos_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-8)
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    return np.degrees(np.arccos(cos_angle))


# --- Thresholds for the repetition detector ---
# Based on the reference range (min ~85°, max ~180°): tune if needed
HIGH_THRESHOLD = 165.0   # above this angle = considered "standing"
LOW_THRESHOLD = 145.0    # below this angle = considered "moving/down"
MIN_REP_FRAMES = 10      # discard repetitions that are too short (likely noise)

MAX_ERROR_THRESHOLD = 45.0  # depth difference in degrees = 100% error (tune this)

# --- Smoothing filter for the angle signal ---
# Reduces frame-to-frame jitter from keypoint detection noise (see discussion:
# monocular 2D pose estimation has no temporal memory between frames).
SMOOTHING_FACTOR = 0.3  # lower = smoother/slower to react, higher = more responsive
smoothed_angle = None

# --- Detector state ---
state = "standing"  # or "moving"
rep_buffer = []
last_result_text = "Waiting for movement..."
last_color = (200, 200, 200)

# Tracks the user's recent "standing" angle, used to calibrate the offset
# between the user's setup (camera angle, body proportions) and the
# reference sequence, without distorting the depth/amplitude of the squat.
standing_angle_history = []
MAX_STANDING_HISTORY = 10

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: could not open the webcam.")
    exit()

cv2.namedWindow('Movement Comparison - Press Q to quit', cv2.WINDOW_NORMAL)
print("Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame, verbose=False)
    annotated_frame = results[0].plot()

    if results[0].keypoints is not None and len(results[0].keypoints.xy) > 0:
        kp = results[0].keypoints.xy[0].cpu().numpy()

        if np.any(kp[LEFT_HIP]) and np.any(kp[LEFT_KNEE]) and np.any(kp[LEFT_ANKLE]):
            raw_angle = calculate_angle(kp[LEFT_HIP], kp[LEFT_KNEE], kp[LEFT_ANKLE])

            # Apply exponential smoothing to reduce jitter
            if smoothed_angle is None:
                smoothed_angle = raw_angle
            else:
                smoothed_angle = SMOOTHING_FACTOR * raw_angle + (1 - SMOOTHING_FACTOR) * smoothed_angle

            angle = smoothed_angle  # use the smoothed value everywhere below
            # --- State machine logic ---
            if state == "standing":
                # Keep a rolling history of the standing angle, used later
                # to calibrate the offset against the reference sequence.
                standing_angle_history.append(angle)
                if len(standing_angle_history) > MAX_STANDING_HISTORY:
                    standing_angle_history.pop(0)

                if angle < LOW_THRESHOLD:
                    # Start of a new repetition
                    state = "moving"
                    rep_buffer = [angle]
            elif state == "moving":
                rep_buffer.append(angle)
                if angle > HIGH_THRESHOLD:
                    # End of the repetition: the patient is back standing
                    state = "standing"

                    if len(rep_buffer) >= MIN_REP_FRAMES and len(standing_angle_history) > 0:
                        # Calibration: align the user's standing angle to the
                        # reference's standing angle. This shifts the curve
                        # up/down to remove systematic bias (camera angle,
                        # body proportions) WITHOUT stretching it, so the
                        # actual depth/amplitude of the squat is preserved.
                        user_standing_baseline = np.mean(standing_angle_history)
                        calibration_offset = reference.max() - user_standing_baseline

                        corrected_rep = [a + calibration_offset for a in rep_buffer]

                        # Depth-based scoring: how close did the user get to
                        # the target depth (minimum angle) of the reference?
                        # This reflects what matters clinically: how deep the
                        # squat went, rather than the shape of the whole curve.
                        depth_achieved = min(corrected_rep)
                        depth_target = reference.min()
                        depth_diff = abs(depth_achieved - depth_target)

                        print(f"DEBUG depth_achieved={depth_achieved:.1f}° depth_target={depth_target:.1f}° diff={depth_diff:.1f}° (offset: {calibration_offset:.1f}°)")
                        error_pct = min(100.0, (depth_diff / MAX_ERROR_THRESHOLD) * 100)
                        accuracy_pct = 100.0 - error_pct

                        last_result_text = f"Repetition: {accuracy_pct:.1f}% correct"
                        if accuracy_pct >= 80:
                            last_color = (0, 200, 0)
                        elif accuracy_pct >= 50:
                            last_color = (0, 200, 255)
                        else:
                            last_color = (0, 0, 255)
                    else:
                        last_result_text = "Movement too short, discarded"
                        last_color = (150, 150, 150)

                    print(f"Your rep range: min={min(rep_buffer):.1f}°, max={max(rep_buffer):.1f}° | Reference range: min={reference.min():.1f}°, max={reference.max():.1f}°")
                    rep_buffer = []

    # --- On-screen status text ---
    state_text = "STANDING" if state == "standing" else "MOVING..."
    cv2.putText(annotated_frame, state_text, (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2, cv2.LINE_AA)
    cv2.putText(annotated_frame, last_result_text, (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, last_color, 2, cv2.LINE_AA)

    cv2.imshow('Movement Comparison - Press Q to quit', annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()