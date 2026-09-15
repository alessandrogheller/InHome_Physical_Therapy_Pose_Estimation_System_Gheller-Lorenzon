"""Compare real-time arm-extension repetitions with reference movements."""

import cv2
import time
import sys
import os
import numpy as np
from ultralytics import YOLO

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import (
    MODEL_PATH, get_reference_path,
    LEFT_SHOULDER, LEFT_ELBOW, LEFT_WRIST,
    RIGHT_SHOULDER, RIGHT_ELBOW, RIGHT_WRIST,
    calculate_angle, calculate_depth_score, keypoints_are_valid,
    select_patient_keypoints, KP_CONF_THRESHOLD,
)

# Track the elbow angle and automatically select the arm that moves during
# calibration, unless a specific side is configured.
REFERENCE_PATH_LEFT = get_reference_path('limb_extension_left')    
REFERENCE_PATH_RIGHT = get_reference_path('limb_extension_right')  

# --- Debug logging ---
DEBUG = False

# --- On-screen messaging ---
calibration_instruction = ""
transient_message = None

# --- Which arm to track ---
# Use 'auto' to detect the moving arm or set 'left'/'right' manually.
TRACKED_SIDE = 'auto'  # 'auto', 'left', or 'right'

ARM_JOINTS = {
    'left': (LEFT_SHOULDER, LEFT_ELBOW, LEFT_WRIST),
    'right': (RIGHT_SHOULDER, RIGHT_ELBOW, RIGHT_WRIST),
}

# Use a lower confidence threshold for the fast-moving wrist.
WRIST_CONF_THRESHOLD = 0.35


def arm_keypoints_valid(kp_xy, kp_conf, joints):
    """Like keypoints_are_valid, but applies WRIST_CONF_THRESHOLD to the
    wrist and the normal KP_CONF_THRESHOLD to shoulder/elbow, instead of one
    threshold for all three. `joints` is (shoulder_idx, elbow_idx, wrist_idx)."""
    shoulder_i, elbow_i, wrist_i = joints
    if kp_conf is not None:
        return (kp_conf[shoulder_i] >= KP_CONF_THRESHOLD
                and kp_conf[elbow_i] >= KP_CONF_THRESHOLD
                and kp_conf[wrist_i] >= WRIST_CONF_THRESHOLD)
    return keypoints_are_valid(kp_xy, kp_conf, list(joints))


# Temporarily reuse the last valid angle when wrist confidence drops.
MAX_HOLD_FRAMES = 20

# Require several frames below the resting threshold before ending a repetition.
RESTING_CONFIRM_FRAMES = 3
resting_confirm_count = 0

# --- Load model and both reference curves ---
model = YOLO(MODEL_PATH)

references = {}
for _side, _path in (('left', REFERENCE_PATH_LEFT), ('right', REFERENCE_PATH_RIGHT)):
    try:
        references[_side] = np.load(_path)
    except FileNotFoundError:
        references[_side] = None

if references['left'] is None and references['right'] is None:
    print("Error: no reference curve found. Run reference_extraction_limb_extension.py first.")
    exit()


def resolve_reference(side):
    """Return the requested reference or a fallback from the other side."""
    other = 'right' if side == 'left' else 'left'
    if references.get(side) is not None:
        return references[side], side, None
    note = (f"No {side} reference found -- using the {other} reference "
            "as a mirrored approximation.")
    return references[other], other, note


for _side, _ref in references.items():
    if _ref is not None:
        print(f"Reference loaded ({_side}): resting(min)={_ref.min():.1f}°  "
              f"extended(max)={_ref.max():.1f}°  "
              f"(target ROM: {_ref.max() - _ref.min():.1f}°)")

# Use a reference curve to initialize thresholds before calibration completes.
active_reference = references['left'] if references['left'] is not None else references['right']

# --- Thresholds for the repetition detector ---
# A repetition starts when the angle rises above HIGH_THRESHOLD and ends
# after it falls below LOW_THRESHOLD.
_ref_range = active_reference.max() - active_reference.min()
HIGH_THRESHOLD = active_reference.min() + 0.3 * _ref_range   # crossing above -> extension has started
LOW_THRESHOLD = active_reference.min() + 0.1 * _ref_range    # crossing below (while extending) -> back to rest
MIN_REP_FRAMES = 10  # discard repetitions that are too short (likely noise)

# --- Adaptive threshold calibration ---
# Estimate thresholds from the patient's own range; perform one full extension
# during calibration to capture both resting and extended positions.
CALIBRATION_DURATION = 12.0  # seconds; long enough to rest briefly, then do one full rep

# --- Timeout for an in-progress repetition ---
MAX_MOVING_DURATION = 10.0  # seconds

# --- Smoothing filter for the angle signal ---
# Smooth the angle signal to reduce noise during fast movements.
SMOOTHING_FACTOR = 0.5
smoothed_angle = None
last_valid_raw_angle = None
hold_frames_left = 0  # counts down while reusing the last valid raw angle

# --- Detector state ---
# "calibrating": tracking both arms to find the working one and its thresholds
# "resting"    : arm flexed at the baseline, waiting for the movement to start
# "extending"  : movement in progress, tracking the angle up to its peak and
#                back down
state = "calibrating"
rep_buffer = []
last_result_text = "Waiting for movement..."
last_color = (200, 200, 200)
moving_start_time = None

resting_angle_history = []
MAX_RESTING_HISTORY = 10

# Track both arms during automatic calibration and select the one with the
# largest observed range of motion.
calibration_angles = {'left': [], 'right': []} if TRACKED_SIDE == 'auto' else {TRACKED_SIDE: []}
working_side = None if TRACKED_SIDE == 'auto' else TRACKED_SIDE

if working_side is not None:
    # Side was forced (not 'auto') -- resolve its reference right away
    # instead of waiting for calibration to finish.
    active_reference, working_side, _fallback_note = resolve_reference(working_side)
    if _fallback_note:
        transient_message = (_fallback_note, time.time() + 5.0, (0, 165, 255))

# --- Which camera to use ---
CAMERA_INDEX = 0

patient_center = None

cap = cv2.VideoCapture(CAMERA_INDEX)
if not cap.isOpened():
    print("Error: could not open the webcam.")
    exit()

cv2.namedWindow('Limb Extension Comparison - Press Q to quit', cv2.WINDOW_NORMAL)

# Start calibration after the webcam and display are ready.
calibration_start_time = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame, verbose=False)
    annotated_frame = results[0].plot()

    kp, kp_conf, new_center = select_patient_keypoints(results, previous_center=patient_center)

    if kp is not None:
        patient_center = new_center
        cv2.circle(annotated_frame, (int(patient_center[0]), int(patient_center[1])),
                   8, (255, 0, 255), -1)

        if state == "calibrating":
            elapsed = time.time() - calibration_start_time
            remaining = max(0.0, CALIBRATION_DURATION - elapsed)
            if elapsed < 2.0:
                calibration_instruction = "Rest with your arm flexed and face the camera..."
            else:
                calibration_instruction = "Perform ONE full arm extension to calibrate the system"

            # Collect valid angles for each candidate arm.
            for side, joints in ARM_JOINTS.items():
                if side not in calibration_angles:
                    continue
                if arm_keypoints_valid(kp, kp_conf, joints):
                    shoulder_i, elbow_i, wrist_i = joints
                    a = calculate_angle(kp[shoulder_i], kp[elbow_i], kp[wrist_i])
                    calibration_angles[side].append(a)
                    if DEBUG:
                        conf_str = (f"{kp_conf[[shoulder_i, elbow_i, wrist_i]]}" if kp_conf is not None else "N/A")
                        print(f"[calib][{side}] raw={a:6.1f}  conf={conf_str}")
                elif DEBUG:
                    print(f"[calib][{side}] INVALID keypoints, skipped")

            if elapsed > CALIBRATION_DURATION:
                # Select the arm with the largest observed range of motion.
                best_side, best_range, best_angles = None, -1.0, None
                for side, angles in calibration_angles.items():
                    if len(angles) < 2:
                        continue
                    rng = max(angles) - min(angles)
                    if DEBUG:
                        print(f"Calibration [{side}]: {len(angles)} valid frames, "
                              f"range=[{min(angles):.1f}°, {max(angles):.1f}°] (ROM={rng:.1f}°)")
                    if rng > best_range:
                        best_side, best_range, best_angles = side, rng, angles

                if best_side is None:
                    transient_message = (
                        "Calibration failed: no arm detected reliably. Please retry.",
                        time.time() + 4.0, (0, 0, 255))
                    state = "resting"
                    working_side = 'left'
                    active_reference, working_side, _fallback_note = resolve_reference(working_side)
                    _ref_range = active_reference.max() - active_reference.min()
                    HIGH_THRESHOLD = active_reference.min() + 0.3 * _ref_range
                    LOW_THRESHOLD = active_reference.min() + 0.1 * _ref_range
                else:
                    working_side = best_side
                    active_reference, working_side, _fallback_note = resolve_reference(working_side)
                    obs_min = min(best_angles)
                    obs_max = max(best_angles)
                    obs_range = max(obs_max - obs_min, 1e-3)
                    HIGH_THRESHOLD = obs_min + 0.3 * obs_range
                    LOW_THRESHOLD = obs_min + 0.1 * obs_range
                    state = "resting"
                    if best_range < 15.0:
                        transient_message = (
                            f"Warning: detected range of motion is very small ({best_range:.0f} deg). "
                            "Results may be unreliable.",
                            time.time() + 5.0, (0, 165, 255))
                    elif _fallback_note:
                        transient_message = (_fallback_note, time.time() + 5.0, (0, 165, 255))
                    else:
                        arm_label = "left" if working_side == "left" else "right"
                        transient_message = (
                            f"Calibration complete (arm: {arm_label})",
                            time.time() + 3.0, (0, 200, 0))

        elif working_side is not None:
            joints_valid = arm_keypoints_valid(kp, kp_conf, ARM_JOINTS[working_side])

            if joints_valid:
                shoulder_i, elbow_i, wrist_i = ARM_JOINTS[working_side]
                raw_angle = calculate_angle(kp[shoulder_i], kp[elbow_i], kp[wrist_i])
                last_valid_raw_angle = raw_angle
                hold_frames_left = MAX_HOLD_FRAMES
            elif last_valid_raw_angle is not None and hold_frames_left > 0:
                # Reuse the last valid angle during a brief confidence dip.
                raw_angle = last_valid_raw_angle
                hold_frames_left -= 1
                if DEBUG:
                    print(f"[{state}][{working_side}] keypoints invalid, "
                          f"holding last angle={raw_angle:.1f} ({hold_frames_left} frames left)")
            else:
                raw_angle = None
                if DEBUG:
                    print(f"[{state}][{working_side}] keypoints invalid, no hold left -- frame dropped")

            if raw_angle is not None:
                if smoothed_angle is None:
                    smoothed_angle = raw_angle
                else:
                    smoothed_angle = SMOOTHING_FACTOR * raw_angle + (1 - SMOOTHING_FACTOR) * smoothed_angle

                angle = smoothed_angle

                if DEBUG and state in ("resting", "extending") and joints_valid:
                    print(f"[{state}][{working_side}] raw={raw_angle:6.1f}  smoothed={angle:6.1f}  "
                          f"HIGH={HIGH_THRESHOLD:.1f}  LOW={LOW_THRESHOLD:.1f}")

            if raw_angle is None:
                pass  # no usable angle this frame; state machine stays as-is

            elif state == "resting":
                resting_angle_history.append(angle)
                if len(resting_angle_history) > MAX_RESTING_HISTORY:
                    resting_angle_history.pop(0)

                if angle > HIGH_THRESHOLD:
                    state = "extending"
                    rep_buffer = [angle]
                    moving_start_time = time.time()
                    resting_confirm_count = 0

            elif state == "extending":
                rep_buffer.append(angle)

                timed_out = (time.time() - moving_start_time) > MAX_MOVING_DURATION

                if timed_out:
                    last_result_text = "Movement timeout, discarded"
                    last_color = (150, 150, 150)
                    state = "resting"
                    rep_buffer = []
                    resting_confirm_count = 0

                elif angle < LOW_THRESHOLD:
                    resting_confirm_count += 1
                    if DEBUG:
                        print(f"[extending][{working_side}] below LOW_THRESHOLD "
                              f"({resting_confirm_count}/{RESTING_CONFIRM_FRAMES} to confirm rest)")

                    if resting_confirm_count < RESTING_CONFIRM_FRAMES:
                        # Wait for consecutive confirmations before ending the rep.
                        pass
                    else:
                        state = "resting"
                        resting_confirm_count = 0

                        if len(rep_buffer) >= MIN_REP_FRAMES:
                            # Compare the achieved extension directly with the
                            # reference target without applying an angle offset.
                            extension_achieved = max(rep_buffer)
                            extension_target = active_reference.max()
                            extension_diff = abs(extension_achieved - extension_target)

                            # Use the shared two-zone scoring function.
                            accuracy_pct = calculate_depth_score(extension_diff)

                            arm_label = "left" if working_side == "left" else "right"
                            last_result_text = f"Repetition ({arm_label} arm): {accuracy_pct:.1f}% correct"
                            if accuracy_pct >= 80:
                                last_color = (0, 200, 0)
                            elif accuracy_pct >= 50:
                                last_color = (0, 200, 255)
                            else:
                                last_color = (0, 0, 255)

                            print(f"Rep ({arm_label}): extension_achieved={extension_achieved:.1f}° "
                                  f"extension_target={extension_target:.1f}° "
                                  f"diff={extension_diff:.1f}° -> score={accuracy_pct:.1f}%")
                        else:
                            last_result_text = "Movement too short, discarded"
                            last_color = (150, 150, 150)

                        rep_buffer = []
                else:
                    # Angle rose back above LOW_THRESHOLD before we
                    # confirmed rest -- reset the debounce counter.
                    resting_confirm_count = 0
    else:
        pass

    state_text = {"calibrating": "CALIBRATING...", "resting": "RESTING", "extending": "EXTENDING..."}[state]
    cv2.putText(annotated_frame, state_text, (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2, cv2.LINE_AA)
    cv2.putText(annotated_frame, last_result_text, (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, last_color, 2, cv2.LINE_AA)

    if state == "calibrating":
        cv2.putText(annotated_frame, calibration_instruction, (20, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
        remaining = max(0.0, CALIBRATION_DURATION - (time.time() - calibration_start_time))
        cv2.putText(annotated_frame, f"Time remaining: {remaining:0.0f}s", (20, 150),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
    elif working_side is not None:
        arm_label = "LEFT" if working_side == "left" else "RIGHT"
        cv2.putText(annotated_frame, f"Tracking arm: {arm_label}", (20, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 200, 0), 2, cv2.LINE_AA)

    if transient_message is not None:
        text, expire_at, color = transient_message
        if time.time() < expire_at:
            cv2.putText(annotated_frame, text, (20, 180),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)
        else:
            transient_message = None

    cv2.imshow('Limb Extension Comparison - Press Q to quit', annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()