import cv2
import time
import numpy as np
from ultralytics import YOLO

from utils import (
    MODEL_PATH, get_reference_path,
    LEFT_SHOULDER, LEFT_ELBOW, LEFT_WRIST,
    calculate_angle, calculate_depth_score, keypoints_are_valid,
    select_patient_keypoints, KP_CONF_THRESHOLD,
)

# A07 = "limb extension, left arm". Angle measured at the elbow, between
# shoulder-elbow-wrist. Unlike the lunge (which can be loaded on either
# leg and therefore needs auto side-detection), A07 is explicitly the LEFT
# arm, so the side is fixed here. A mirrored A08 ("limb extension right")
# script would be a straight copy of this file with the RIGHT_* joints from
# utils.py swapped in.
REFERENCE_PATH = get_reference_path('limb_extension_left')  # A07 -- build with your reference-extraction step first
ARM_JOINTS = (LEFT_SHOULDER, LEFT_ELBOW, LEFT_WRIST)

# --- Debug logging ---
# Set to True to print, every frame during "resting" and "extending", the
# raw/smoothed angle and whether keypoints were valid (with their
# confidences). Noisy -- leave False for normal sessions. When False, the
# only thing printed to the terminal is the one-line "Rep: ..." result per
# repetition; everything else (calibration instructions, status, warnings)
# is shown as an overlay on the video window instead.
DEBUG = False

# --- On-screen messaging ---
calibration_instruction = ""
transient_message = None

# --- Asymmetric confidence threshold for the wrist ---
# Same reasoning as the ankle in the lunge script: hip/knee-equivalents
# (shoulder, elbow) tend to stay well-tracked, but the wrist is a small,
# fast-moving extremity that foreshortens/self-occludes at full extension
# (arm pointed toward or across the camera) -- exactly when we most need a
# reading. Since the wrist only defines the far end of the segment used for
# the angle (small positional errors barely change the elbow angle itself),
# we accept a lower confidence for it specifically instead of applying
# KP_CONF_THRESHOLD to all three joints.
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


# --- Fallback for missing/low-confidence keypoints ---
# The wrist can drop below confidence threshold for several consecutive
# frames right around full extension (self-occlusion / motion blur at the
# fastest part of the movement). Rather than silently dropping those frames
# (which could erase the true peak extension angle), we hold the last valid
# raw angle for a limited number of frames. If keypoints stay invalid longer
# than this, we stop updating -- better to lose a frame than fabricate data
# from a stale pose. Tune this against your own logs (DEBUG=True) the same
# way it was tuned for the lunge's ankle.
MAX_HOLD_FRAMES = 20

# --- Debounce before ending a repetition ---
# A single noisy frame that dips back toward the resting angle right after
# an occlusion gap (angle rebounding sharply) shouldn't be enough to close
# out the repetition before the patient has actually returned to rest.
# Requiring several consecutive confirming frames filters out that kind of
# spike.
RESTING_CONFIRM_FRAMES = 3
resting_confirm_count = 0

# --- Load model and reference curve ---
model = YOLO(MODEL_PATH)

try:
    active_reference = np.load(REFERENCE_PATH)
except FileNotFoundError:
    print("Error: no reference curve found. Run the reference-extraction step "
          "for A07 (limb_extension_left) first.")
    exit()

# --- Thresholds for the repetition detector ---
# IMPORTANT DIRECTION NOTE: this mirrors the squat/lunge state machine, but
# flipped. In a squat/lunge, "standing" is the HIGH-angle baseline and a
# repetition is detected by the angle DROPPING down and coming back up. In a
# limb extension, the baseline is the FLEXED (LOW-angle) resting position,
# and a repetition is detected by the angle RISING toward full extension and
# then coming back down. So here HIGH_THRESHOLD is the trigger to *enter*
# the movement (crossed while resting) and LOW_THRESHOLD is the trigger to
# *confirm return to rest* (crossed while extending) -- the opposite roles
# to their names in the lunge script.
_ref_range = active_reference.max() - active_reference.min()
HIGH_THRESHOLD = active_reference.min() + 0.3 * _ref_range   # crossing above -> extension has started
LOW_THRESHOLD = active_reference.min() + 0.1 * _ref_range    # crossing below (while extending) -> back to rest
MIN_REP_FRAMES = 10  # discard repetitions that are too short (likely noise)

# --- Adaptive threshold calibration ---
# As with the lunge, we don't assume the reference curve's absolute values
# transfer directly to an arbitrary patient/camera setup, so by default we
# re-estimate HIGH/LOW threshold from the patient's own observed range
# during a short calibration window.
#
# IMPORTANT: the patient must perform ONE FULL REPETITION during this
# window (rest with the arm flexed -> extend the left arm fully -> return
# to rest), not just hold still -- holding still only shows the resting
# extreme, never the extended extreme, and the two thresholds would end up
# almost identical and useless for detecting repetitions.
USE_ADAPTIVE_THRESHOLDS = True
CALIBRATION_DURATION = 8.0  # seconds; long enough to rest briefly, then do one full rep

# --- Timeout for an in-progress repetition ---
MAX_MOVING_DURATION = 8.0  # seconds

# --- Smoothing filter for the angle signal ---
# Elbow extension can be a fast movement; if the smoothed signal looks too
# laggy/compressed on your setup, lower this, but validate with DEBUG=True
# that the smoothed signal still reaches the true peak extension angle.
SMOOTHING_FACTOR = 0.5
smoothed_angle = None
last_valid_raw_angle = None
hold_frames_left = 0  # counts down while reusing the last valid raw angle

# --- Detector state ---
# "resting"   : arm flexed at the baseline, waiting for the movement to start
# "extending" : movement in progress, tracking the angle up to its peak and
#               back down
state = "calibrating" if USE_ADAPTIVE_THRESHOLDS else "resting"
rep_buffer = []
last_result_text = "Waiting for movement..."
last_color = (200, 200, 200)
moving_start_time = None

resting_angle_history = []
MAX_RESTING_HISTORY = 10

calibration_angles = []
calibration_start_time = time.time()

# --- Which camera to use ---
CAMERA_INDEX = 0

patient_center = None

cap = cv2.VideoCapture(CAMERA_INDEX)
if not cap.isOpened():
    print("Error: could not open the webcam.")
    exit()

cv2.namedWindow('Limb Extension (Left) Comparison - Press Q to quit', cv2.WINDOW_NORMAL)

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
                calibration_instruction = "Rest with your left arm flexed and face the camera..."
            else:
                calibration_instruction = "Perform ONE full left-arm extension to calibrate the system"

            if arm_keypoints_valid(kp, kp_conf, ARM_JOINTS):
                shoulder_i, elbow_i, wrist_i = ARM_JOINTS
                a = calculate_angle(kp[shoulder_i], kp[elbow_i], kp[wrist_i])
                calibration_angles.append(a)
                if DEBUG:
                    conf_str = (f"{kp_conf[[shoulder_i, elbow_i, wrist_i]]}" if kp_conf is not None else "N/A")
                    print(f"[calib] raw={a:6.1f}  conf={conf_str}")
            elif DEBUG:
                print("[calib] INVALID keypoints, skipped")

            if elapsed > CALIBRATION_DURATION:
                if len(calibration_angles) < 2:
                    transient_message = (
                        "Calibration failed: arm not detected reliably. Please retry.",
                        time.time() + 4.0, (0, 0, 255))
                    state = "resting"
                    _ref_range = active_reference.max() - active_reference.min()
                    HIGH_THRESHOLD = active_reference.min() + 0.3 * _ref_range
                    LOW_THRESHOLD = active_reference.min() + 0.1 * _ref_range
                else:
                    obs_min = min(calibration_angles)
                    obs_max = max(calibration_angles)
                    obs_range = max(obs_max - obs_min, 1e-3)
                    HIGH_THRESHOLD = obs_min + 0.3 * obs_range
                    LOW_THRESHOLD = obs_min + 0.1 * obs_range
                    state = "resting"
                    if DEBUG:
                        print(f"Calibration: {len(calibration_angles)} valid frames, "
                              f"range=[{obs_min:.1f}°, {obs_max:.1f}°] (ROM={obs_range:.1f}°)")
                    if obs_range < 15.0:
                        transient_message = (
                            f"Warning: detected range of motion is very small ({obs_range:.0f} deg). "
                            "Results may be unreliable.",
                            time.time() + 5.0, (0, 165, 255))
                    else:
                        transient_message = (
                            "Calibration complete (left arm)",
                            time.time() + 3.0, (0, 200, 0))

        elif state is not None:
            joints_valid = arm_keypoints_valid(kp, kp_conf, ARM_JOINTS)

            if joints_valid:
                shoulder_i, elbow_i, wrist_i = ARM_JOINTS
                raw_angle = calculate_angle(kp[shoulder_i], kp[elbow_i], kp[wrist_i])
                last_valid_raw_angle = raw_angle
                hold_frames_left = MAX_HOLD_FRAMES
            elif last_valid_raw_angle is not None and hold_frames_left > 0:
                # Brief confidence dip (common right at full extension, due
                # to wrist foreshortening/self-occlusion): reuse the last
                # valid angle instead of dropping the frame entirely, so a
                # momentary dip doesn't erase the true peak.
                raw_angle = last_valid_raw_angle
                hold_frames_left -= 1
                if DEBUG:
                    print(f"[{state}] keypoints invalid, "
                          f"holding last angle={raw_angle:.1f} ({hold_frames_left} frames left)")
            else:
                raw_angle = None
                if DEBUG:
                    print(f"[{state}] keypoints invalid, no hold left -- frame dropped")

            if raw_angle is not None:
                if smoothed_angle is None:
                    smoothed_angle = raw_angle
                else:
                    smoothed_angle = SMOOTHING_FACTOR * raw_angle + (1 - SMOOTHING_FACTOR) * smoothed_angle

                angle = smoothed_angle

                if DEBUG and state in ("resting", "extending") and joints_valid:
                    print(f"[{state}] raw={raw_angle:6.1f}  smoothed={angle:6.1f}  "
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
                        print(f"[extending] below LOW_THRESHOLD "
                              f"({resting_confirm_count}/{RESTING_CONFIRM_FRAMES} to confirm rest)")

                    if resting_confirm_count < RESTING_CONFIRM_FRAMES:
                        # Not confirmed yet -- could be a noise dip right
                        # after an occlusion gap. Stay in "extending" and
                        # keep appending to rep_buffer so we don't lose real
                        # data if the patient really is still extending.
                        pass
                    else:
                        state = "resting"
                        resting_confirm_count = 0

                        if len(rep_buffer) >= MIN_REP_FRAMES and len(resting_angle_history) > 0:
                            user_resting_baseline = np.mean(resting_angle_history)
                            calibration_offset = active_reference.min() - user_resting_baseline
                            corrected_rep = [a + calibration_offset for a in rep_buffer]

                            extension_achieved = max(corrected_rep)
                            extension_target = active_reference.max()
                            extension_diff = abs(extension_achieved - extension_target)

                            # Same two-zone scoring function used for the
                            # squat/lunge, so results stay comparable in
                            # structure (score is still 0-100%).
                            accuracy_pct = calculate_depth_score(extension_diff)

                            last_result_text = f"Repetition: {accuracy_pct:.1f}% correct"
                            if accuracy_pct >= 80:
                                last_color = (0, 200, 0)
                            elif accuracy_pct >= 50:
                                last_color = (0, 200, 255)
                            else:
                                last_color = (0, 0, 255)

                            print(f"Rep: extension_achieved={extension_achieved:.1f}° "
                                  f"extension_target={extension_target:.1f}° "
                                  f"diff={extension_diff:.1f}° (offset: {calibration_offset:.1f}°) "
                                  f"-> score={accuracy_pct:.1f}%")
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

    if transient_message is not None:
        text, expire_at, color = transient_message
        if time.time() < expire_at:
            cv2.putText(annotated_frame, text, (20, 180),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)
        else:
            transient_message = None

    cv2.imshow('Limb Extension (Left) Comparison - Press Q to quit', annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()