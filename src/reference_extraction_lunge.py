import cv2
import time
import numpy as np
from ultralytics import YOLO

from utils import (
    MODEL_PATH, get_reference_path,
    LEFT_HIP, LEFT_KNEE, LEFT_ANKLE,
    RIGHT_HIP, RIGHT_KNEE, RIGHT_ANKLE,
    calculate_angle, calculate_depth_score, keypoints_are_valid,
    select_patient_keypoints, KP_CONF_THRESHOLD,
)

REFERENCE_PATH_LEFT = get_reference_path('lunge')        # A15 (left side) -- built by reference_extraction_lunge.py
REFERENCE_PATH_RIGHT = get_reference_path('lunge_right')  # A16 (right side) -- built by reference_extraction_lunge_right.py
# Run whichever of the two extraction scripts you need before using this
# script. If only one of the two reference files exists, the other side
# falls back to it automatically (see resolve_reference() below).

# --- Debug logging ---
# Set to True to print, every frame during "calibrating" and "moving",
# the raw/smoothed angle and whether keypoints were valid (with their
# confidences). Useful for diagnosing detection problems, but noisy for
# normal use -- leave False for day-to-day sessions. When False, the only
# thing printed to the terminal is the one-line "Rep: ..." result per
# repetition; everything else (calibration instructions, status, warnings)
# is shown as an overlay on the video window instead.
DEBUG = False

# --- On-screen messaging (replaces the old terminal prints) ---
# calibration_instruction: shown continuously while state == "calibrating".
# transient_message: (text, expire_timestamp, bgr_color) shown for a few
# seconds after calibration ends (success, failure, or a low-ROM warning),
# then cleared automatically. Declared early because resolve_reference()
# below can also populate it (e.g. when falling back from a missing
# reference file), before the rest of the setup runs.
calibration_instruction = ""
transient_message = None

# --- Which leg to track ---
# A real patient may naturally lunge to either side. If you hardcoded
# LEFT_KNEE while they load the RIGHT leg (or vice versa), the other knee
# barely moves: calibration collapses to a tiny obs_range and HIGH/LOW
# thresholds end up almost identical, so no repetition ever fires.
#
# 'auto': track both legs during calibration and automatically pick
#         whichever one shows the larger range of motion as the "working"
#         (loaded) leg for the rest of the session. This is the recommended
#         default. The matching reference (left -> A15, right -> A16) is
#         then used automatically for scoring.
# 'left' / 'right': force a specific side (use if auto-detection misfires,
#         or if you know in advance which side the patient will lunge to).
TRACKED_SIDE = 'auto'  # 'auto', 'left', or 'right'

LEG_JOINTS = {
    'left': (LEFT_HIP, LEFT_KNEE, LEFT_ANKLE),
    'right': (RIGHT_HIP, RIGHT_KNEE, RIGHT_ANKLE),
}

# --- Asymmetric confidence threshold for the ankle ---
# In the calibration logs, hip and knee confidence were almost always >0.9,
# but ankle confidence hovered right around KP_CONF_THRESHOLD (0.5) --
# sometimes 0.51, sometimes 0.49 -- and kept invalidating the whole
# hip-knee-ankle triplet even though the knee itself (what we actually
# measure) was tracked perfectly. This is expected for a lateral lunge: the
# loaded ankle rotates/foreshortens relative to the camera as you go down,
# which is exactly when the model is least confident about it. Since we
# only need the ankle to define the lower segment of the angle (small
# errors in its exact position barely change the knee angle), we accept a
# lower confidence for it specifically instead of using KP_CONF_THRESHOLD
# for all three joints.
ANKLE_CONF_THRESHOLD = 0.35


def leg_keypoints_valid(kp_xy, kp_conf, joints):
    """Like keypoints_are_valid, but applies ANKLE_CONF_THRESHOLD to the
    ankle and the normal KP_CONF_THRESHOLD to hip/knee, instead of one
    threshold for all three. `joints` is (hip_idx, knee_idx, ankle_idx)."""
    hip_i, knee_i, ankle_i = joints
    if kp_conf is not None:
        return (kp_conf[hip_i] >= KP_CONF_THRESHOLD
                and kp_conf[knee_i] >= KP_CONF_THRESHOLD
                and kp_conf[ankle_i] >= ANKLE_CONF_THRESHOLD)
    return keypoints_are_valid(kp_xy, kp_conf, list(joints))


# --- Fallback for missing/low-confidence keypoints ---
# Lateral lunges rotate/partially occlude the loaded leg right at the
# deepest point of the movement -- exactly where the pose model is most
# likely to drop below confidence threshold. Previously, an invalid frame
# was silently skipped (no angle update at all), which meant the true
# minimum angle could be missed entirely if it coincided with a
# low-confidence frame. Instead, we hold the last valid raw angle for a
# number of frames so a confidence dip doesn't erase the deepest part of
# the rep. Raised from 5 to 20: the logs showed occlusion gaps of up to
# ~15-20 consecutive invalid frames right at the bottom of the movement,
# and 5 frames of hold was nowhere near enough to bridge that. If keypoints
# stay invalid longer than this, we stop updating (better to lose a frame
# than to fabricate data from a stale, no-longer-true pose).
MAX_HOLD_FRAMES = 20

# --- Debounce before ending a repetition ---
# Previously, a single noisy frame above HIGH_THRESHOLD (e.g. right after
# recovering from an occlusion gap, with the angle rebounding sharply) was
# enough to end the "moving" state and score the rep -- even though the
# patient hadn't actually returned to standing yet. Requiring several
# consecutive frames above threshold filters out that kind of spike.
STANDING_CONFIRM_FRAMES = 3
standing_confirm_count = 0

# --- Load model and both reference curves ---
model = YOLO(MODEL_PATH)

references = {}
for _side, _path in (('left', REFERENCE_PATH_LEFT), ('right', REFERENCE_PATH_RIGHT)):
    try:
        references[_side] = np.load(_path)
    except FileNotFoundError:
        references[_side] = None

if references['left'] is None and references['right'] is None:
    print("Error: no reference curve found. Run reference_extraction_lunge.py "
          "(left, A15) and/or reference_extraction_lunge_right.py (right, A16) first.")
    exit()


def resolve_reference(side):
    """Return (array, actual_side, note) for the requested side. If that
    side's reference file is missing, fall back to the other side's
    reference -- knee flexion angle at a given depth is ~symmetric
    left/right, so this is a reasonable approximation -- and return a short
    note describing the fallback so it can be shown on screen (never
    printed to the terminal, to keep stdout limited to "Rep: ..." lines)."""
    other = 'right' if side == 'left' else 'left'
    if references.get(side) is not None:
        return references[side], side, None
    note = (f"No {side} reference found -- using the {other} reference "
            "as a mirrored approximation.")
    return references[other], other, note


# Reference used only to bootstrap the initial (pre-calibration) thresholds
# below -- it doesn't matter much which side, since USE_ADAPTIVE_THRESHOLDS
# overwrites HIGH_THRESHOLD/LOW_THRESHOLD as soon as calibration completes.
active_reference = references['left'] if references['left'] is not None else references['right']

# --- Thresholds for the repetition detector ---
# IMPORTANT: the 165/145 values used in the squat pipeline were tuned for a
# squat's range of motion and are NOT assumed valid here. Since we don't have
# a validated range for a lateral lunge either, adaptive calibration is kept
# ON by default (see USE_ADAPTIVE_THRESHOLDS below) so the thresholds are
# estimated from the reference curve's own range instead of hardcoded.
_ref_range = active_reference.max() - active_reference.min()
HIGH_THRESHOLD = active_reference.max() - 0.1 * _ref_range   # "standing" / leg extended
LOW_THRESHOLD = active_reference.max() - 0.3 * _ref_range    # "moving/down" into the lunge
MIN_REP_FRAMES = 10  # discard repetitions that are too short (likely noise)

# --- Adaptive threshold calibration ---
# If True, the first CALIBRATION_DURATION seconds re-estimate HIGH/LOW
# threshold from the patient's own observed range instead of the
# reference-derived values above -- recommended here even more than for the
# squat script, since we have less certainty about what a "normal" A15
# range of motion looks like for an arbitrary patient/camera setup.
#
# IMPORTANT: unlike just "standing still", the patient must perform ONE
# FULL REPETITION of the lunge during this window (stand -> step out to the
# side and bend the left knee -> return to standing). Standing still alone
# only shows the "standing" extreme, never the "depth" extreme, which would
# make HIGH_THRESHOLD and LOW_THRESHOLD end up almost identical and useless
# for detecting repetitions.
USE_ADAPTIVE_THRESHOLDS = True
CALIBRATION_DURATION = 8.0  # seconds; long enough to stand still briefly, then do one full rep

# --- Timeout for an in-progress repetition ---
MAX_MOVING_DURATION = 8.0  # seconds

# --- Smoothing filter for the angle signal ---
# Raised from 0.3 to 0.5: with a fast movement like a lateral lunge (the
# low point of the rep can last under a second), a heavy filter (0.3)
# significantly lags behind and compresses the true range of motion, which
# was likely contributing to thresholds that were never actually reached.
# If the signal now looks too jittery on your setup, lower it again, but
# validate with DEBUG=True that the true minimum angle is still being
# reached by the smoothed signal.
SMOOTHING_FACTOR = 0.5
smoothed_angle = None
last_valid_raw_angle = None
hold_frames_left = 0  # counts down while reusing the last valid raw angle

# --- Detector state ---
state = "calibrating" if USE_ADAPTIVE_THRESHOLDS else "standing"
rep_buffer = []
last_result_text = "Waiting for movement..."
last_color = (200, 200, 200)
moving_start_time = None

standing_angle_history = []
MAX_STANDING_HISTORY = 10

# During calibration we track BOTH legs (unless TRACKED_SIDE forces one),
# so we can compare their observed range of motion at the end and pick the
# one that actually moved -- that's the leg the patient loaded.
calibration_angles = {'left': [], 'right': []} if TRACKED_SIDE == 'auto' else {TRACKED_SIDE: []}
calibration_start_time = time.time()
working_side = None if TRACKED_SIDE == 'auto' else TRACKED_SIDE

if working_side is not None:
    # Side was forced (not 'auto') -- resolve its reference right away
    # instead of waiting for calibration to finish.
    active_reference, working_side, _fallback_note = resolve_reference(working_side)
    if _fallback_note:
        transient_message = (_fallback_note, time.time() + 5.0, (0, 165, 255))

# --- Which camera to use ---
# Index passed to cv2.VideoCapture. 0 is normally the default/built-in
# webcam. If you have more than one camera (an external USB webcam plus a
# laptop's built-in one, for example), try 1, 2, etc. See the chat message
# for how to find out which index corresponds to which physical camera.
CAMERA_INDEX = 1

patient_center = None

cap = cv2.VideoCapture(CAMERA_INDEX)
if not cap.isOpened():
    print("Error: could not open the webcam.")
    exit()

cv2.namedWindow('Lunge Comparison - Press Q to quit', cv2.WINDOW_NORMAL)

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
                calibration_instruction = "Stand still and face the camera..."
            else:
                calibration_instruction = "Perform ONE full lunge repetition to calibrate the system"

            # Track every candidate leg so we can pick the one that actually
            # moves. keypoints_are_valid / calculate_angle are evaluated
            # per-leg since one leg may be valid while the other is occluded.
            for side, joints in LEG_JOINTS.items():
                if side not in calibration_angles:
                    continue
                if leg_keypoints_valid(kp, kp_conf, joints):
                    hip_i, knee_i, ankle_i = joints
                    a = calculate_angle(kp[hip_i], kp[knee_i], kp[ankle_i])
                    calibration_angles[side].append(a)
                    if DEBUG:
                        conf_str = (f"{kp_conf[[hip_i, knee_i, ankle_i]]}" if kp_conf is not None else "N/A")
                        print(f"[calib][{side}] raw={a:6.1f}  conf={conf_str}")
                elif DEBUG:
                    print(f"[calib][{side}] INVALID keypoints, skipped")

            if elapsed > CALIBRATION_DURATION:
                # Pick the working leg: whichever has the larger observed
                # range of motion. If TRACKED_SIDE forced a side, that's the
                # only key present and this just uses it directly.
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
                        "Calibration failed: no leg detected reliably. Please retry.",
                        time.time() + 4.0, (0, 0, 255))
                    state = "standing"
                    working_side = 'left'
                    active_reference, working_side, _fallback_note = resolve_reference(working_side)
                    _ref_range = active_reference.max() - active_reference.min()
                    HIGH_THRESHOLD = active_reference.max() - 0.1 * _ref_range
                    LOW_THRESHOLD = active_reference.max() - 0.3 * _ref_range
                else:
                    working_side = best_side
                    active_reference, working_side, _fallback_note = resolve_reference(working_side)
                    obs_min = min(best_angles)
                    obs_max = max(best_angles)
                    obs_range = max(obs_max - obs_min, 1e-3)
                    HIGH_THRESHOLD = obs_max - 0.1 * obs_range
                    LOW_THRESHOLD = obs_max - 0.3 * obs_range
                    state = "standing"
                    if best_range < 15.0:
                        transient_message = (
                            f"Warning: detected range of motion is very small ({best_range:.0f} deg). "
                            "Results may be unreliable.",
                            time.time() + 5.0, (0, 165, 255))
                    elif _fallback_note:
                        transient_message = (_fallback_note, time.time() + 5.0, (0, 165, 255))
                    else:
                        leg_label = "left" if working_side == "left" else "right"
                        transient_message = (
                            f"Calibration complete (leg: {leg_label})",
                            time.time() + 3.0, (0, 200, 0))

        elif working_side is not None:
            joints_valid = leg_keypoints_valid(kp, kp_conf, LEG_JOINTS[working_side])

            if joints_valid:
                hip_i, knee_i, ankle_i = LEG_JOINTS[working_side]
                raw_angle = calculate_angle(kp[hip_i], kp[knee_i], kp[ankle_i])
                last_valid_raw_angle = raw_angle
                hold_frames_left = MAX_HOLD_FRAMES
            elif last_valid_raw_angle is not None and hold_frames_left > 0:
                # Brief confidence dip (common right at the deepest point of
                # a lateral lunge, due to partial self-occlusion): reuse the
                # last valid angle instead of dropping the frame entirely,
                # so a momentary dip doesn't erase the true minimum.
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

                if DEBUG and state in ("standing", "moving") and joints_valid:
                    print(f"[{state}][{working_side}] raw={raw_angle:6.1f}  smoothed={angle:6.1f}  "
                          f"HIGH={HIGH_THRESHOLD:.1f}  LOW={LOW_THRESHOLD:.1f}")

            if raw_angle is None:
                pass  # no usable angle this frame; state machine stays as-is

            elif state == "standing":
                standing_angle_history.append(angle)
                if len(standing_angle_history) > MAX_STANDING_HISTORY:
                    standing_angle_history.pop(0)

                if angle < LOW_THRESHOLD:
                    state = "moving"
                    rep_buffer = [angle]
                    moving_start_time = time.time()
                    standing_confirm_count = 0

            elif state == "moving":
                rep_buffer.append(angle)

                timed_out = (time.time() - moving_start_time) > MAX_MOVING_DURATION

                if timed_out:
                    last_result_text = "Movement timeout, discarded"
                    last_color = (150, 150, 150)
                    state = "standing"
                    rep_buffer = []
                    standing_confirm_count = 0

                elif angle > HIGH_THRESHOLD:
                    standing_confirm_count += 1
                    if DEBUG:
                        print(f"[moving][{working_side}] above HIGH_THRESHOLD "
                              f"({standing_confirm_count}/{STANDING_CONFIRM_FRAMES} to confirm standing)")

                    if standing_confirm_count < STANDING_CONFIRM_FRAMES:
                        # Not confirmed yet -- could be a noise spike right
                        # after an occlusion gap. Stay in "moving" and keep
                        # appending to rep_buffer so we don't lose real data
                        # if it turns out the patient really is still going.
                        pass
                    else:
                        state = "standing"
                        standing_confirm_count = 0

                        if len(rep_buffer) >= MIN_REP_FRAMES and len(standing_angle_history) > 0:
                            user_standing_baseline = np.mean(standing_angle_history)
                            calibration_offset = active_reference.max() - user_standing_baseline
                            corrected_rep = [a + calibration_offset for a in rep_buffer]

                            depth_achieved = min(corrected_rep)
                            depth_target = active_reference.min()
                            depth_diff = abs(depth_achieved - depth_target)

                            # Same two-zone scoring function used for the squat,
                            # so results stay comparable in structure (score is
                            # still 0-100%). The clinical meaning of the target
                            # itself is weaker here, see the note in
                            # evaluate_dataset_fixed_targets_lunge.py.
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
                    # Angle dropped back below HIGH_THRESHOLD before we
                    # confirmed standing -- reset the debounce counter.
                    standing_confirm_count = 0
    else:
        pass

    state_text = {"calibrating": "CALIBRATING...", "standing": "STANDING", "moving": "MOVING..."}[state]
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

    cv2.imshow('Lunge Comparison - Press Q to quit', annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()