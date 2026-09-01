import cv2
import time
import numpy as np
from ultralytics import YOLO

from utils import (
    MODEL_PATH, get_reference_path,
    LEFT_HIP, LEFT_KNEE, LEFT_ANKLE,
    RIGHT_HIP, RIGHT_KNEE, RIGHT_ANKLE,
    calculate_angle, calculate_depth_score, keypoints_are_valid,
    select_patient_keypoints,
)

REFERENCE_PATH = get_reference_path('lunge')  # -> PROJECT_ROOT/lunge_reference.npy
# Make sure you ran reference_extraction_lunge.py first to generate this file.

# --- Debug logging ---
# Set to True to print, every frame during "calibrating" and "moving",
# the raw/smoothed angle and whether keypoints were valid (with their
# confidences). Extremely useful to diagnose "calibration finishes but the
# rep is never detected" style problems: if you see many valid=False during
# the lunge's deepest point, or a raw angle that dips low but the smoothed
# one barely moves, that tells you exactly which stage is failing.
DEBUG = True

# --- Which leg to track ---
# The reference curve (lunge_reference.npy) was built from the LEFT knee,
# because MMFi action A15 is "lunge toward the left side" (see
# reference_extraction_lunge.py). But a real patient may naturally lunge to
# either side, and if you hardcode LEFT_KNEE while they load the RIGHT leg,
# the left knee barely moves: calibration collapses to a tiny obs_range and
# HIGH/LOW thresholds end up almost identical, so no repetition ever fires.
#
# 'auto': track both legs during calibration and automatically pick
#         whichever one shows the larger range of motion as the "working"
#         (loaded) leg for the rest of the session. This is the recommended
#         default. Knee flexion angle at a given depth is ~symmetric
#         left/right, so comparing an auto-selected working leg against a
#         reference built from the left leg is still valid.
# 'left' / 'right': force a specific side (use if auto-detection misfires,
#         or if you know in advance which side the patient will lunge to).
TRACKED_SIDE = 'auto'  # 'auto', 'left', or 'right'

LEG_JOINTS = {
    'left': (LEFT_HIP, LEFT_KNEE, LEFT_ANKLE),
    'right': (RIGHT_HIP, RIGHT_KNEE, RIGHT_ANKLE),
}

# --- Fallback for missing/low-confidence keypoints ---
# Lateral lunges rotate/partially occlude the loaded leg right at the
# deepest point of the movement -- exactly where the pose model is most
# likely to drop below KP_CONF_THRESHOLD. Previously, an invalid frame was
# silently skipped (no angle update at all), which meant the true minimum
# angle could be missed entirely if it coincided with a low-confidence
# frame. Instead, we hold the last valid raw angle for a short number of
# frames so a brief confidence dip doesn't erase the deepest part of the
# rep. If keypoints stay invalid longer than this, we stop updating (better
# to lose a frame than to fabricate data from a stale, no-longer-true pose).
MAX_HOLD_FRAMES = 5

# --- Load model and reference ---
model = YOLO(MODEL_PATH)
reference = np.load(REFERENCE_PATH)

print(f"Reference loaded: {len(reference)} frames.")
print(f"Reference range: min={reference.min():.1f}°, max={reference.max():.1f}°")

# --- Thresholds for the repetition detector ---
# IMPORTANT: the 165/145 values used in the squat pipeline were tuned for a
# squat's range of motion and are NOT assumed valid here. Since we don't have
# a validated range for a lateral lunge either, adaptive calibration is kept
# ON by default (see USE_ADAPTIVE_THRESHOLDS below) so the thresholds are
# estimated from the reference curve's own range instead of hardcoded.
_ref_range = reference.max() - reference.min()
HIGH_THRESHOLD = reference.max() - 0.1 * _ref_range   # "standing" / leg extended
LOW_THRESHOLD = reference.max() - 0.3 * _ref_range    # "moving/down" into the lunge
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

patient_center = None

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: could not open the webcam.")
    exit()

cv2.namedWindow('Lunge Comparison - Press Q to quit', cv2.WINDOW_NORMAL)
print("Press 'q' to quit.")
if USE_ADAPTIVE_THRESHOLDS:
    print(f"Calibrating for {CALIBRATION_DURATION:.0f}s -- stand still and face the camera, "
          f"then perform ONE full lunge repetition (step out and back) before it ends.")

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
            # Track every candidate leg so we can pick the one that actually
            # moves. keypoints_are_valid / calculate_angle are evaluated
            # per-leg since one leg may be valid while the other is occluded.
            for side, joints in LEG_JOINTS.items():
                if side not in calibration_angles:
                    continue
                if keypoints_are_valid(kp, kp_conf, list(joints)):
                    hip_i, knee_i, ankle_i = joints
                    a = calculate_angle(kp[hip_i], kp[knee_i], kp[ankle_i])
                    calibration_angles[side].append(a)
                    if DEBUG:
                        conf_str = (f"{kp_conf[[hip_i, knee_i, ankle_i]]}" if kp_conf is not None else "N/A")
                        print(f"[calib][{side}] raw={a:6.1f}  conf={conf_str}")
                elif DEBUG:
                    print(f"[calib][{side}] INVALID keypoints, skipped")

            if time.time() - calibration_start_time > CALIBRATION_DURATION:
                # Pick the working leg: whichever has the larger observed
                # range of motion. If TRACKED_SIDE forced a side, that's the
                # only key present and this just uses it directly.
                best_side, best_range, best_angles = None, -1.0, None
                for side, angles in calibration_angles.items():
                    if len(angles) < 2:
                        continue
                    rng = max(angles) - min(angles)
                    print(f"Calibration [{side}]: {len(angles)} valid frames, "
                          f"range=[{min(angles):.1f}°, {max(angles):.1f}°] (ROM={rng:.1f}°)")
                    if rng > best_range:
                        best_side, best_range, best_angles = side, rng, angles

                if best_side is None:
                    print("Calibration FAILED: no leg produced enough valid keypoints. "
                          "Check lighting/framing and restart.")
                    state = "standing"
                    HIGH_THRESHOLD = reference.max() - 0.1 * _ref_range
                    LOW_THRESHOLD = reference.max() - 0.3 * _ref_range
                    working_side = 'left'
                else:
                    working_side = best_side
                    obs_min = min(best_angles)
                    obs_max = max(best_angles)
                    obs_range = max(obs_max - obs_min, 1e-3)
                    HIGH_THRESHOLD = obs_max - 0.1 * obs_range
                    LOW_THRESHOLD = obs_max - 0.3 * obs_range
                    state = "standing"
                    print(f"Calibration done: working_side={working_side}  "
                          f"standing baseline={obs_max:.1f}°  "
                          f"HIGH_THRESHOLD={HIGH_THRESHOLD:.1f}°  LOW_THRESHOLD={LOW_THRESHOLD:.1f}°")
                    if best_range < 15.0:
                        print(f"WARNING: observed range of motion is only {best_range:.1f}°. "
                              "This is suspiciously small for a full lunge repetition -- "
                              "thresholds may be unreliable. Make sure you actually performed "
                              "a full rep during calibration and that the working leg stayed "
                              "in frame and well-lit.")

        elif working_side is not None:
            joints_valid = keypoints_are_valid(kp, kp_conf, list(LEG_JOINTS[working_side]))

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

            elif state == "moving":
                rep_buffer.append(angle)

                timed_out = (time.time() - moving_start_time) > MAX_MOVING_DURATION

                if timed_out:
                    last_result_text = "Movement timeout, discarded"
                    last_color = (150, 150, 150)
                    state = "standing"
                    rep_buffer = []

                elif angle > HIGH_THRESHOLD:
                    state = "standing"

                    if len(rep_buffer) >= MIN_REP_FRAMES and len(standing_angle_history) > 0:
                        user_standing_baseline = np.mean(standing_angle_history)
                        calibration_offset = reference.max() - user_standing_baseline
                        corrected_rep = [a + calibration_offset for a in rep_buffer]

                        depth_achieved = min(corrected_rep)
                        depth_target = reference.min()
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
        pass

    state_text = {"calibrating": "CALIBRATING...", "standing": "STANDING", "moving": "MOVING..."}[state]
    cv2.putText(annotated_frame, state_text, (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2, cv2.LINE_AA)
    cv2.putText(annotated_frame, last_result_text, (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, last_color, 2, cv2.LINE_AA)

    if state == "calibrating":
        cv2.putText(annotated_frame, "Perform ONE full lunge rep now", (20, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)

    cv2.imshow('Lunge Comparison - Press Q to quit', annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()