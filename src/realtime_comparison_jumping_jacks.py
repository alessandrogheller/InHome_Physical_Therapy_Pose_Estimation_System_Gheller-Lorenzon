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
    select_patient_keypoints, KP_CONF_THRESHOLD,
)

# Reference built by reference_extraction_jumping_jacks.py: a single
# (REFERENCE_LENGTH, 2) array, column 0 = leg_spread_ratio curve,
# column 1 = arm_raise_angle curve (degrees). See that script for why a
# jumping jack needs these two signals instead of a knee angle.
REFERENCE_PATH = get_reference_path('jumping_jacks')

# --- Debug logging ---
# Set to True for extra terminal output while tuning thresholds. Unlike
# before, this NO LONGER prints one line per frame -- that produced
# thousands of lines per session and made it impossible to actually read.
# Instead:
#   - during "calibrating": one summary line every DEBUG_FRAME_STRIDE frames
#   - during "waiting"/"open": one line every DEBUG_FRAME_STRIDE frames,
#     PLUS an immediate line on every state change (rep start / rep end),
#     since those are the moments you actually want to see.
# "Rep: ..." results are always printed regardless of DEBUG -- that line is
# the actual output of the program, not a debug aid.
DEBUG = True
DEBUG_FRAME_STRIDE = 5  # print at most 1 out of every N frames when DEBUG is on
_debug_frame_counter = 0


def dbg(msg, force=False):
    """Print a debug line, throttled to 1/DEBUG_FRAME_STRIDE frames unless
    force=True (used for state transitions / rep events, which should always
    show up immediately)."""
    if not DEBUG:
        return
    if force or (_debug_frame_counter % DEBUG_FRAME_STRIDE == 0):
        print(msg)


# --- On-screen messaging ---
calibration_instruction = ""
transient_message = None

# --- Load model and reference ---
model = YOLO(MODEL_PATH)
try:
    reference = np.load(REFERENCE_PATH)
except FileNotFoundError:
    print(f"Error: reference file not found at {REFERENCE_PATH}. "
          "Run reference_extraction_jumping_jacks.py first.")
    exit()

ref_leg_spread = reference[:, 0]
ref_arm_raise = reference[:, 1]
print(f"Reference loaded: leg_spread range=[{ref_leg_spread.min():.2f}, {ref_leg_spread.max():.2f}]  "
      f"arm_raise range=[{ref_arm_raise.min():.1f}°, {ref_arm_raise.max():.1f}°]")

# Targets = the "fully open" extreme of each signal in the reference
# (jack's open position: legs apart, arms overhead).
#
# NOTE: using the raw max() here made leg-spread scores consistently very
# low (~25-40%) even on visually solid reps, while arm scores stayed high.
# That pattern points to the target itself being off, not the patient's
# form: the reference was built from MM-Fi action A26 "Jumping up" as a
# proxy (see reference_extraction_jumping_jacks.py -- MM-Fi has no real
# jumping-jack action), and a single noisy keypoint frame (e.g. motion-blur
# briefly displacing an ankle during the jump) can produce one extreme
# spike that raw max() would lock onto as "the" target. The 95th percentile
# still represents "close to the top of the observed range" but is far less
# sensitive to one outlier frame.
LEG_SPREAD_TARGET = float(np.percentile(ref_leg_spread, 95))
ARM_RAISE_TARGET = float(np.percentile(ref_arm_raise, 95))

# Tolerance/falloff for scoring, expressed as a FRACTION of the reference's
# own range of motion for each signal (rather than fixed absolute units,
# since one signal is a unitless ratio and the other is in degrees -- a
# single absolute tolerance like the squat's DEPTH_TOLERANCE=10deg would
# make no sense applied to the leg-spread ratio). 15%/35% mirrors the
# shape of the squat/lunge two-zone scoring (DEPTH_TOLERANCE / DEPTH_
# FALLOFF_RANGE), just rescaled per-signal.
_leg_rom = ref_leg_spread.max() - ref_leg_spread.min()
_arm_rom = ref_arm_raise.max() - ref_arm_raise.min()
LEG_TOLERANCE = 0.15 * _leg_rom
LEG_FALLOFF = 0.35 * _leg_rom
ARM_TOLERANCE = 0.15 * _arm_rom
ARM_FALLOFF = 0.35 * _arm_rom

# --- Required keypoints and per-joint confidence thresholds ---
# Shoulders/hips/ankles are usually tracked reliably throughout. Wrists,
# however, move fast and can motion-blur or self-occlude right at the top
# of the jack (arms overhead, near the edge of frame) -- same issue seen
# with the elbow-extension script's wrist tracking. We accept a lower
# confidence for wrists specifically instead of applying KP_CONF_THRESHOLD
# to all eight joints.
WRIST_CONF_THRESHOLD = 0.35
CORE_JOINTS = [LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_HIP, RIGHT_HIP, LEFT_ANKLE, RIGHT_ANKLE]
ALL_JOINTS = CORE_JOINTS + [LEFT_WRIST, RIGHT_WRIST]


def signals_valid(kp_xy, kp_conf):
    if kp_conf is not None:
        core_ok = all(kp_conf[i] >= KP_CONF_THRESHOLD for i in CORE_JOINTS)
        wrists_ok = (kp_conf[LEFT_WRIST] >= WRIST_CONF_THRESHOLD
                     and kp_conf[RIGHT_WRIST] >= WRIST_CONF_THRESHOLD)
        return core_ok and wrists_ok
    return keypoints_are_valid(kp_xy, None, ALL_JOINTS)


def euclidean(a, b):
    a, b = np.array(a, dtype=float), np.array(b, dtype=float)
    return float(np.linalg.norm(a - b))


def compute_signals(kp):
    """Return (leg_spread_ratio, arm_raise_angle) for one frame's keypoints,
    or None if the frame is degenerate (near-zero shoulder width)."""
    left_shoulder = kp[LEFT_SHOULDER]
    right_shoulder = kp[RIGHT_SHOULDER]
    left_hip = kp[LEFT_HIP]
    right_hip = kp[RIGHT_HIP]
    left_wrist = kp[LEFT_WRIST]
    right_wrist = kp[RIGHT_WRIST]
    left_ankle = kp[LEFT_ANKLE]
    right_ankle = kp[RIGHT_ANKLE]

    shoulder_width = euclidean(left_shoulder, right_shoulder)
    if shoulder_width < 1e-3:
        return None

    leg_spread_ratio = euclidean(left_ankle, right_ankle) / shoulder_width
    left_arm_angle = calculate_angle(left_hip, left_shoulder, left_wrist)
    right_arm_angle = calculate_angle(right_hip, right_shoulder, right_wrist)
    arm_raise_angle = (left_arm_angle + right_arm_angle) / 2.0

    return leg_spread_ratio, arm_raise_angle


# --- Fallback for missing/low-confidence keypoints ---
# Same rationale as the lunge/limb-extension scripts: a confidence dip
# right at the extremes of the movement (arms overhead here) shouldn't
# silently drop the frame and risk missing the true peak. We hold the last
# valid signals for a limited number of frames before giving up.
MAX_HOLD_FRAMES = 20
last_valid_signals = None
hold_frames_left = 0

MIN_REP_FRAMES = 8         # discard repetitions that are too short (likely noise)
MAX_MOVING_DURATION = 6.0  # seconds; safety timeout in case a rep never resolves

# --- Smoothing filter ---
# Jumping jacks are a fast, rhythmic movement; a moderate smoothing factor
# balances jitter reduction against not lagging behind the true peak.
SMOOTHING_FACTOR = 0.5
smoothed_leg = None
smoothed_arm = None

# --- Adaptive calibration ---
# We don't assume any patient's natural standing leg-spread or arm-down
# angle matches the MMFi reference subject's setup, so thresholds for
# detecting "open" are always estimated from the patient's own observed
# range during calibration, exactly like the other realtime scripts.
# Unlike the lunge (which needs a left/right choice), a jumping jack is
# bilateral and symmetric, so there is no "working side" to select -- both
# signals are computed directly from both sides every frame.
#
# IMPORTANT: perform a couple of FULL, natural-paced repetitions during
# this window (stand with legs together/arms down -> jump legs apart with
# arms overhead -> back down -> repeat once or twice), ideally at the same
# continuous rhythm you intend to use for the real set. Standing still
# alone only shows the "closed" extreme, never the "open" extreme.
CALIBRATION_DURATION = 8.0  # seconds
calibration_leg = []
calibration_arm = []
calibration_start_time = time.time()

# openness(t) combines both signals into a single normalized 0-1 scalar
# once calibration has established each signal's observed min/max.
OPENNESS_HIGH_THRESHOLD = 0.3  # crossing above (while "waiting") -> a rep has started opening
leg_min = leg_max = arm_min = arm_max = None  # set once calibration completes


def openness(leg_val, arm_val):
    leg_component = np.clip((leg_val - leg_min) / max(leg_max - leg_min, 1e-3), 0.0, 1.0)
    arm_component = np.clip((arm_val - arm_min) / max(arm_max - arm_min, 1e-3), 0.0, 1.0)
    return 0.5 * leg_component + 0.5 * arm_component


# --- Continuous rep detection (peak-prominence based) ---
# WHY THIS CHANGED: the previous version only finalized a rep once
# `openness` dropped below a near-zero absolute threshold for several
# consecutive frames. That works if the patient pauses fully closed
# between reps, but during CONTINUOUS jumping jacks (no stop) the body's
# momentum means it often never returns anywhere near "fully closed"
# between jumps -- so the rep buffer just kept growing across multiple
# real repetitions until the 6s safety timeout discarded the whole thing.
#
# Instead, a rep now ends as soon as `openness` has declined by
# PEAK_DROP_MARGIN from the highest value seen since the rep started --
# i.e. we detect that the peak has been passed, regardless of how low the
# signal goes afterwards. This is a standard "peak with prominence"
# detector and works whether the patient pauses at the bottom or not.
#
# REFRACTORY_PERIOD prevents a second rep from being registered
# immediately after the first due to jitter right around the peak (e.g.
# the smoothed signal wobbling up/down by a few points at the top of the
# jump) -- no new rep can start within this many seconds of the previous
# one ending.
PEAK_DROP_MARGIN = 0.15   # normalized openness units (0-1 scale)
REFRACTORY_PERIOD = 0.25  # seconds
last_rep_end_time = 0.0

# --- Detector state ---
# "calibrating": observing the patient's own closed/open range
# "waiting"    : ready to detect the start of the next rep (rising edge)
# "open"       : a rep is in progress, tracking its peak
state = "calibrating"
rep_buffer = []       # list of (leg_spread, arm_raise) tuples during "open"
peak_openness = 0.0    # running peak of `openness` for the rep in progress
last_result_text = "Waiting for movement..."
last_color = (200, 200, 200)
moving_start_time = None
rep_count = 0

CAMERA_INDEX = 1
patient_center = None

cap = cv2.VideoCapture(CAMERA_INDEX)
if not cap.isOpened():
    print("Error: could not open the webcam.")
    exit()

cv2.namedWindow('Jumping Jack Comparison - Press Q to quit', cv2.WINDOW_NORMAL)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    _debug_frame_counter += 1

    results = model(frame, verbose=False)
    annotated_frame = results[0].plot()

    kp, kp_conf, new_center = select_patient_keypoints(results, previous_center=patient_center)

    if kp is not None:
        patient_center = new_center
        cv2.circle(annotated_frame, (int(patient_center[0]), int(patient_center[1])),
                   8, (255, 0, 255), -1)

        valid = signals_valid(kp, kp_conf)
        raw_signals = compute_signals(kp) if valid else None

        if raw_signals is not None:
            last_valid_signals = raw_signals
            hold_frames_left = MAX_HOLD_FRAMES
        elif last_valid_signals is not None and hold_frames_left > 0:
            raw_signals = last_valid_signals
            hold_frames_left -= 1
            dbg(f"[{state}] keypoints invalid, holding last signals "
                f"({hold_frames_left} frames left)")
        else:
            raw_signals = None
            dbg(f"[{state}] keypoints invalid, no hold left -- frame dropped")

        if raw_signals is not None:
            raw_leg, raw_arm = raw_signals
            smoothed_leg = raw_leg if smoothed_leg is None else (
                SMOOTHING_FACTOR * raw_leg + (1 - SMOOTHING_FACTOR) * smoothed_leg)
            smoothed_arm = raw_arm if smoothed_arm is None else (
                SMOOTHING_FACTOR * raw_arm + (1 - SMOOTHING_FACTOR) * smoothed_arm)

        if state == "calibrating":
            elapsed = time.time() - calibration_start_time
            if elapsed < 2.0:
                calibration_instruction = "Stand still, legs together, arms down..."
            else:
                calibration_instruction = "Perform 1-2 full jumping jacks, at your normal pace, to calibrate"

            if raw_signals is not None:
                calibration_leg.append(smoothed_leg)
                calibration_arm.append(smoothed_arm)
                dbg(f"[calib] leg={smoothed_leg:.2f}  arm={smoothed_arm:.1f}")

            if elapsed > CALIBRATION_DURATION:
                if len(calibration_leg) < 2:
                    transient_message = (
                        "Calibration failed: patient not detected reliably. Please retry.",
                        time.time() + 4.0, (0, 0, 255))
                    leg_min, leg_max = float(ref_leg_spread.min()), float(ref_leg_spread.max())
                    arm_min, arm_max = float(ref_arm_raise.min()), float(ref_arm_raise.max())
                else:
                    leg_min, leg_max = min(calibration_leg), max(calibration_leg)
                    arm_min, arm_max = min(calibration_arm), max(calibration_arm)
                    leg_rom = leg_max - leg_min
                    arm_rom = arm_max - arm_min
                    dbg(f"Calibration: leg_rom={leg_rom:.2f}  arm_rom={arm_rom:.1f}°", force=True)

                    if leg_rom < 0.15 or arm_rom < 20.0:
                        transient_message = (
                            "Warning: small range of motion detected during calibration "
                            "(legs and/or arms may be out of frame). Results may be unreliable.",
                            time.time() + 5.0, (0, 165, 255))
                    else:
                        transient_message = (
                            "Calibration complete", time.time() + 3.0, (0, 200, 0))
                state = "waiting"

        elif leg_min is not None:  # calibration has completed
            if raw_signals is not None:
                current_openness = openness(smoothed_leg, smoothed_arm)

                dbg(f"[{state}] leg={smoothed_leg:.2f}  arm={smoothed_arm:.1f}  "
                    f"openness={current_openness:.2f}")

                if state == "waiting":
                    can_start = (time.time() - last_rep_end_time) > REFRACTORY_PERIOD
                    if can_start and current_openness > OPENNESS_HIGH_THRESHOLD:
                        state = "open"
                        rep_buffer = [(smoothed_leg, smoothed_arm)]
                        peak_openness = current_openness
                        moving_start_time = time.time()
                        dbg(f"[open] rep started (openness={current_openness:.2f})", force=True)

                elif state == "open":
                    rep_buffer.append((smoothed_leg, smoothed_arm))
                    peak_openness = max(peak_openness, current_openness)

                    timed_out = (time.time() - moving_start_time) > MAX_MOVING_DURATION
                    peak_passed = current_openness < (peak_openness - PEAK_DROP_MARGIN)

                    if timed_out:
                        last_result_text = "Movement timeout, discarded"
                        last_color = (150, 150, 150)
                        dbg("[open] timed out, discarding rep", force=True)
                        state = "waiting"
                        rep_buffer = []
                        last_rep_end_time = time.time()

                    elif peak_passed:
                        dbg(f"[open] peak passed (peak={peak_openness:.2f}, "
                            f"now={current_openness:.2f}) -- finalizing rep", force=True)
                        state = "waiting"
                        last_rep_end_time = time.time()

                        if len(rep_buffer) >= MIN_REP_FRAMES:
                            leg_values = [p[0] for p in rep_buffer]
                            arm_values = [p[1] for p in rep_buffer]
                            leg_peak = max(leg_values)
                            arm_peak = max(arm_values)

                            leg_diff = abs(leg_peak - LEG_SPREAD_TARGET)
                            arm_diff = abs(arm_peak - ARM_RAISE_TARGET)

                            leg_score = calculate_depth_score(
                                leg_diff, tolerance=LEG_TOLERANCE, falloff=LEG_FALLOFF)
                            arm_score = calculate_depth_score(
                                arm_diff, tolerance=ARM_TOLERANCE, falloff=ARM_FALLOFF)
                            accuracy_pct = (leg_score + arm_score) / 2.0

                            rep_count += 1
                            last_result_text = f"Rep {rep_count}: {accuracy_pct:.1f}% correct"
                            if accuracy_pct >= 80:
                                last_color = (0, 200, 0)
                            elif accuracy_pct >= 50:
                                last_color = (0, 200, 255)
                            else:
                                last_color = (0, 0, 255)

                            print(f"Rep {rep_count}: leg_peak={leg_peak:.2f} (target={LEG_SPREAD_TARGET:.2f}, "
                                  f"diff={leg_diff:.2f}, score={leg_score:.1f}%)  "
                                  f"arm_peak={arm_peak:.1f}° (target={ARM_RAISE_TARGET:.1f}°, "
                                  f"diff={arm_diff:.1f}°, score={arm_score:.1f}%)  "
                                  f"-> combined score={accuracy_pct:.1f}%")
                        else:
                            last_result_text = "Movement too short, discarded"
                            last_color = (150, 150, 150)

                        rep_buffer = []
    else:
        pass

    state_text = {"calibrating": "CALIBRATING...", "waiting": "READY", "open": "JUMPING..."}[state]
    cv2.putText(annotated_frame, state_text, (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2, cv2.LINE_AA)
    cv2.putText(annotated_frame, f"Reps: {rep_count}", (20, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(annotated_frame, last_result_text, (20, 105),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, last_color, 2, cv2.LINE_AA)

    if state == "calibrating":
        cv2.putText(annotated_frame, calibration_instruction, (20, 140),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
        remaining = max(0.0, CALIBRATION_DURATION - (time.time() - calibration_start_time))
        cv2.putText(annotated_frame, f"Time remaining: {remaining:0.0f}s", (20, 170),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)

    if transient_message is not None:
        text, expire_at, color = transient_message
        if time.time() < expire_at:
            cv2.putText(annotated_frame, text, (20, 200),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)
        else:
            transient_message = None

    cv2.imshow('Jumping Jack Comparison - Press Q to quit', annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()