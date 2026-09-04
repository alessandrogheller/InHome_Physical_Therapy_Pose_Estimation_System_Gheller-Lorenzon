import cv2
import time
import numpy as np
from ultralytics import YOLO

from utils import (
    MODEL_PATH, get_reference_path,
    LEFT_SHOULDER, RIGHT_SHOULDER,
    LEFT_HIP, RIGHT_HIP,
    LEFT_WRIST, RIGHT_WRIST,
    LEFT_ANKLE, RIGHT_ANKLE,
    calculate_angle, calculate_depth_score, keypoints_are_valid,
    select_patient_keypoints, KP_CONF_THRESHOLD, DEPTH_TOLERANCE, DEPTH_FALLOFF_RANGE,
)

# Reference built by reference_extraction_jumping_jacks.py: a single
# (REFERENCE_LENGTH, 2) array, column 0 = leg_spread_ratio curve,
# column 1 = arm_raise_angle curve (degrees). See that script for why a
# jumping jack needs two signals instead of one knee angle.
REFERENCE_PATH = get_reference_path('jumping_jacks')

# --- Debug logging ---
# Set to True to print, every frame, the raw/smoothed values for both
# signals and the current thresholds. Noisy -- leave False for normal
# sessions. When False, the only thing printed to the terminal is the
# one-line "Rep: ..." result per repetition; everything else (calibration
# instructions, status, warnings) is shown as an overlay on the video
# window instead.
DEBUG = False

# --- On-screen messaging ---
calibration_instruction = ""
transient_message = None

# --- Keypoints required every frame ---
# Jumping jacks are bilateral/symmetric (unlike the lunge or single-arm
# limb extension), so there's no "auto-detect the working side" step here
# -- we always need both shoulders, both hips, both wrists and both ankles
# to compute the two signals.
REQUIRED_KEYPOINTS = [
    LEFT_SHOULDER, RIGHT_SHOULDER,
    LEFT_HIP, RIGHT_HIP,
    LEFT_WRIST, RIGHT_WRIST,
    LEFT_ANKLE, RIGHT_ANKLE,
]


def compute_signals(kp_xy):
    """Compute (leg_spread_ratio, arm_raise_angle) for one frame's
    keypoints. Assumes REQUIRED_KEYPOINTS have already been validated by
    the caller. Mirrors extract_jumping_jack_signals() in
    reference_extraction_jumping_jacks.py so the live and offline pipelines
    use exactly the same geometry."""
    left_shoulder = kp_xy[LEFT_SHOULDER]
    right_shoulder = kp_xy[RIGHT_SHOULDER]
    left_hip = kp_xy[LEFT_HIP]
    right_hip = kp_xy[RIGHT_HIP]
    left_wrist = kp_xy[LEFT_WRIST]
    right_wrist = kp_xy[RIGHT_WRIST]
    left_ankle = kp_xy[LEFT_ANKLE]
    right_ankle = kp_xy[RIGHT_ANKLE]

    shoulder_width = float(np.linalg.norm(np.array(left_shoulder, dtype=float)
                                           - np.array(right_shoulder, dtype=float)))
    if shoulder_width < 1e-3:
        return None  # degenerate frame, caller should treat as invalid

    ankle_dist = float(np.linalg.norm(np.array(left_ankle, dtype=float)
                                       - np.array(right_ankle, dtype=float)))
    leg_spread_ratio = ankle_dist / shoulder_width

    left_arm_angle = calculate_angle(left_hip, left_shoulder, left_wrist)
    right_arm_angle = calculate_angle(right_hip, right_shoulder, right_wrist)
    arm_raise_angle = (left_arm_angle + right_arm_angle) / 2.0

    return leg_spread_ratio, arm_raise_angle


# --- Fallback for missing/low-confidence keypoints ---
# At the fully-open point of a jack, wrists (overhead, often motion-blurred)
# and ankles (wide stance, near frame edges) are the keypoints most likely
# to briefly drop below confidence threshold -- exactly the extremes we
# need to measure. Rather than dropping those frames outright (which could
# erase the true peak spread/raise), we hold the last valid pair of signals
# for a limited number of frames. If keypoints stay invalid longer than
# this, we stop updating -- better to lose a frame than fabricate data from
# a stale pose.
MAX_HOLD_FRAMES = 20
last_valid_signals = None
hold_frames_left = 0

# --- Debounce before ending a repetition ---
# A single noisy frame that dips back toward the closed pose right after a
# confidence drop (signal rebounding sharply) shouldn't be enough to close
# out the repetition before the patient has actually returned to the
# starting stance. Requiring several consecutive confirming frames (on
# BOTH signals) filters out that kind of spike.
CLOSED_CONFIRM_FRAMES = 3
closed_confirm_count = 0

# --- Load model and reference ---
model = YOLO(MODEL_PATH)
try:
    reference = np.load(REFERENCE_PATH)
except FileNotFoundError:
    print(f"Error: no reference curve found at {REFERENCE_PATH}. "
          "Run reference_extraction_jumping_jacks.py first.")
    exit()

reference_spread_curve = reference[:, 0]
reference_arm_curve = reference[:, 1]

# Scoring targets: how far the movement should open, taken from the
# population reference -- NOT from the patient's own calibration. Unlike
# the squat's knee angle (which needs a per-patient standing-baseline
# offset to correct for camera/body differences), both of these signals
# are already scale-invariant by construction: leg_spread_ratio is
# normalized by the patient's own shoulder width, and arm_raise_angle is
# an angle. So, exactly like the limb-extension script's reasoning for a
# fully-extended arm, we score the raw achieved peak directly against the
# reference peak, with no additive correction.
SPREAD_TARGET = float(reference_spread_curve.max())
ARM_TARGET = float(reference_arm_curve.max())

print(f"Reference loaded: leg_spread target={SPREAD_TARGET:.2f}  "
      f"arm_raise target={ARM_TARGET:.1f}°")

# --- Scoring tolerance for the leg-spread ratio ---
# DEPTH_TOLERANCE/DEPTH_FALLOFF_RANGE (imported from utils.py) are tuned in
# DEGREES for the squat/lunge/limb-extension angle metrics, so they're
# reused as-is for arm_raise_angle. But leg_spread_ratio is a unitless
# distance ratio on a very different scale (typically ~0.5-3.0), so reusing
# the same numbers would be meaningless. There is no established clinical
# tolerance for this ratio metric (it's a metric we introduced for this
# script, not a validated clinical measurement) -- the values below are a
# reasonable-looking placeholder, not a validated clinical target. Adjust
# them if you get real feedback on what a meaningful vs. negligible
# difference in stance width looks like.
SPREAD_TOLERANCE = 0.3
SPREAD_FALLOFF_RANGE = 1.0

# --- Thresholds for the repetition detector (per signal) ---
# Same reasoning as the limb-extension script: baseline is the CLOSED
# (low-value) pose for both signals -- legs together, arms down -- and a
# repetition is detected by the signals RISING toward the open extreme and
# then coming back down. HIGH_THRESHOLD is the trigger to *enter* the
# movement (crossed while closed) and LOW_THRESHOLD is the trigger to
# *confirm return to closed* (crossed back while opening).
#
# These fixed, reference-derived values only bootstrap state before
# calibration finishes; adaptive calibration (always on, below) immediately
# overwrites them from the patient's own observed range, since we can't
# assume a patient's natural stance width or arm-raise range matches
# whichever subject(s) built the reference curve.
_spread_range = reference_spread_curve.max() - reference_spread_curve.min()
_arm_range = reference_arm_curve.max() - reference_arm_curve.min()
HIGH_THRESHOLD_SPREAD = reference_spread_curve.min() + 0.3 * _spread_range
LOW_THRESHOLD_SPREAD = reference_spread_curve.min() + 0.1 * _spread_range
HIGH_THRESHOLD_ARM = reference_arm_curve.min() + 0.3 * _arm_range
LOW_THRESHOLD_ARM = reference_arm_curve.min() + 0.1 * _arm_range

MIN_REP_FRAMES = 10  # discard repetitions that are too short (likely noise)

# --- Adaptive threshold calibration ---
# Always on: we don't know in advance how wide a stance or how high an arm
# raise is "full" for this particular patient, so thresholds are always
# re-estimated from their own observed range during the calibration window
# rather than assumed from the reference curve.
#
# IMPORTANT: the patient must perform ONE FULL REPETITION during this
# window (stand at rest -> open into a jack -> return to rest), not just
# hold still -- holding still only shows the closed extreme, never the open
# extreme, and the thresholds would end up nearly identical and useless for
# detecting repetitions.
CALIBRATION_DURATION = 8.0  # seconds; long enough to rest briefly, then do one full rep

# --- Timeout for an in-progress repetition ---
MAX_MOVING_DURATION = 6.0  # seconds; a jumping jack is fast, no need for a long window

# --- Smoothing filter for each signal ---
# A jack is a fast, ballistic movement -- if the smoothed signal looks too
# laggy/compressed on your setup, lower this, but validate with DEBUG=True
# that the smoothed signal still reaches the true peak spread/raise.
SMOOTHING_FACTOR = 0.5
smoothed_spread = None
smoothed_arm = None

# --- Detector state ---
# "calibrating": observing one full rep to set per-signal thresholds
# "closed"     : baseline (legs together, arms down), waiting for a jack to start
# "open"       : movement in progress, tracking both signals up to their
#                peak and back down
state = "calibrating"
rep_buffer_spread = []
rep_buffer_arm = []
last_result_text = "Waiting for movement..."
last_color = (200, 200, 200)
moving_start_time = None

closed_spread_history = []
closed_arm_history = []
MAX_CLOSED_HISTORY = 10

calibration_spread = []
calibration_arm = []

# --- Which camera to use ---
CAMERA_INDEX = 1

patient_center = None

cap = cv2.VideoCapture(CAMERA_INDEX)
if not cap.isOpened():
    print("Error: could not open the webcam.")
    exit()

cv2.namedWindow('Jumping Jacks Comparison - Press Q to quit', cv2.WINDOW_NORMAL)

# Calibration timer starts here (after webcam init), so driver/autofocus
# warm-up doesn't silently eat into the patient's "get into position" time.
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

        frame_valid = keypoints_are_valid(kp, kp_conf, REQUIRED_KEYPOINTS)
        raw_signals = compute_signals(kp) if frame_valid else None

        if raw_signals is not None:
            last_valid_signals = raw_signals
            hold_frames_left = MAX_HOLD_FRAMES
        elif last_valid_signals is not None and hold_frames_left > 0:
            raw_signals = last_valid_signals
            hold_frames_left -= 1
            if DEBUG:
                print(f"[{state}] keypoints invalid, holding last signals={raw_signals} "
                      f"({hold_frames_left} frames left)")
        else:
            raw_signals = None
            if DEBUG:
                print(f"[{state}] keypoints invalid, no hold left -- frame dropped")

        if raw_signals is not None:
            raw_spread, raw_arm = raw_signals

            if smoothed_spread is None:
                smoothed_spread = raw_spread
                smoothed_arm = raw_arm
            else:
                smoothed_spread = SMOOTHING_FACTOR * raw_spread + (1 - SMOOTHING_FACTOR) * smoothed_spread
                smoothed_arm = SMOOTHING_FACTOR * raw_arm + (1 - SMOOTHING_FACTOR) * smoothed_arm

            spread = smoothed_spread
            arm = smoothed_arm

            if DEBUG and state in ("closed", "open"):
                print(f"[{state}] spread={spread:.2f} (HIGH={HIGH_THRESHOLD_SPREAD:.2f} "
                      f"LOW={LOW_THRESHOLD_SPREAD:.2f})  arm={arm:6.1f}° "
                      f"(HIGH={HIGH_THRESHOLD_ARM:.1f} LOW={LOW_THRESHOLD_ARM:.1f})")

            # --- Calibration phase ---
            if state == "calibrating":
                elapsed = time.time() - calibration_start_time
                if elapsed < 2.0:
                    calibration_instruction = "Stand at rest, arms down, feet together..."
                else:
                    calibration_instruction = "Perform ONE full jumping jack to calibrate the system"

                calibration_spread.append(spread)
                calibration_arm.append(arm)

                if elapsed > CALIBRATION_DURATION:
                    if len(calibration_spread) < 2:
                        transient_message = (
                            "Calibration failed: person not detected reliably. Please retry.",
                            time.time() + 4.0, (0, 0, 255))
                        state = "closed"
                    else:
                        obs_spread_min, obs_spread_max = min(calibration_spread), max(calibration_spread)
                        obs_arm_min, obs_arm_max = min(calibration_arm), max(calibration_arm)
                        obs_spread_range = max(obs_spread_max - obs_spread_min, 1e-3)
                        obs_arm_range = max(obs_arm_max - obs_arm_min, 1e-3)

                        HIGH_THRESHOLD_SPREAD = obs_spread_min + 0.3 * obs_spread_range
                        LOW_THRESHOLD_SPREAD = obs_spread_min + 0.1 * obs_spread_range
                        HIGH_THRESHOLD_ARM = obs_arm_min + 0.3 * obs_arm_range
                        LOW_THRESHOLD_ARM = obs_arm_min + 0.1 * obs_arm_range

                        state = "closed"
                        if obs_spread_range < 0.15 or obs_arm_range < 15.0:
                            transient_message = (
                                "Warning: detected range of motion is very small. "
                                "Results may be unreliable.",
                                time.time() + 5.0, (0, 165, 255))
                        else:
                            transient_message = (
                                "Calibration complete", time.time() + 3.0, (0, 200, 0))

            # --- State machine ---
            elif state == "closed":
                closed_spread_history.append(spread)
                closed_arm_history.append(arm)
                if len(closed_spread_history) > MAX_CLOSED_HISTORY:
                    closed_spread_history.pop(0)
                    closed_arm_history.pop(0)

                # Enter "open" as soon as EITHER signal starts rising --
                # more responsive than requiring both at once, since one
                # limb's confidence dip shouldn't delay detecting that the
                # movement has begun.
                if spread > HIGH_THRESHOLD_SPREAD or arm > HIGH_THRESHOLD_ARM:
                    state = "open"
                    rep_buffer_spread = [spread]
                    rep_buffer_arm = [arm]
                    moving_start_time = time.time()
                    closed_confirm_count = 0

            elif state == "open":
                rep_buffer_spread.append(spread)
                rep_buffer_arm.append(arm)

                timed_out = (time.time() - moving_start_time) > MAX_MOVING_DURATION

                if timed_out:
                    last_result_text = "Movement timeout, discarded"
                    last_color = (150, 150, 150)
                    state = "closed"
                    rep_buffer_spread, rep_buffer_arm = [], []
                    closed_confirm_count = 0

                elif spread < LOW_THRESHOLD_SPREAD and arm < LOW_THRESHOLD_ARM:
                    # Require BOTH signals back near baseline to confirm
                    # the patient has actually returned to the closed
                    # stance, not just that one limb dipped momentarily.
                    closed_confirm_count += 1
                    if DEBUG:
                        print(f"[open] below LOW thresholds "
                              f"({closed_confirm_count}/{CLOSED_CONFIRM_FRAMES} to confirm closed)")

                    if closed_confirm_count < CLOSED_CONFIRM_FRAMES:
                        pass
                    else:
                        state = "closed"
                        closed_confirm_count = 0

                        if len(rep_buffer_spread) >= MIN_REP_FRAMES:
                            peak_spread = max(rep_buffer_spread)
                            peak_arm = max(rep_buffer_arm)

                            spread_diff = abs(peak_spread - SPREAD_TARGET)
                            arm_diff = abs(peak_arm - ARM_TARGET)

                            spread_score = calculate_depth_score(
                                spread_diff, tolerance=SPREAD_TOLERANCE, falloff=SPREAD_FALLOFF_RANGE)
                            arm_score = calculate_depth_score(
                                arm_diff, tolerance=DEPTH_TOLERANCE, falloff=DEPTH_FALLOFF_RANGE)

                            # Combined score: simple average of the two
                            # component scores. Both are weighted equally
                            # here for lack of a clinical reason to prefer
                            # one over the other -- adjust the weighting if
                            # one aspect (e.g. leg spread for hip mobility)
                            # matters more for your specific patient.
                            accuracy_pct = (spread_score + arm_score) / 2.0

                            reached_spread = peak_spread > HIGH_THRESHOLD_SPREAD
                            reached_arm = peak_arm > HIGH_THRESHOLD_ARM
                            if not (reached_spread and reached_arm):
                                weak_part = []
                                if not reached_spread:
                                    weak_part.append("legs")
                                if not reached_arm:
                                    weak_part.append("arms")
                                last_result_text = (f"Partial jack ({'/'.join(weak_part)} didn't open fully): "
                                                     f"{accuracy_pct:.1f}%")
                            else:
                                last_result_text = f"Repetition: {accuracy_pct:.1f}% correct"

                            if accuracy_pct >= 80:
                                last_color = (0, 200, 0)
                            elif accuracy_pct >= 50:
                                last_color = (0, 200, 255)
                            else:
                                last_color = (0, 0, 255)

                            print(f"Rep: peak_spread={peak_spread:.2f} (target={SPREAD_TARGET:.2f}, "
                                  f"diff={spread_diff:.2f}, score={spread_score:.1f}%)  "
                                  f"peak_arm={peak_arm:.1f}° (target={ARM_TARGET:.1f}°, "
                                  f"diff={arm_diff:.1f}°, score={arm_score:.1f}%)  "
                                  f"-> combined={accuracy_pct:.1f}%")
                        else:
                            last_result_text = "Movement too short, discarded"
                            last_color = (150, 150, 150)

                        rep_buffer_spread, rep_buffer_arm = [], []
                else:
                    # At least one signal is still elevated -- not back to
                    # baseline yet, reset the debounce counter.
                    closed_confirm_count = 0
    else:
        pass

    state_label = {"calibrating": "CALIBRATING...", "closed": "CLOSED (rest)", "open": "OPEN..."}[state]
    cv2.putText(annotated_frame, state_label, (20, 40),
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

    cv2.imshow('Jumping Jacks Comparison - Press Q to quit', annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()