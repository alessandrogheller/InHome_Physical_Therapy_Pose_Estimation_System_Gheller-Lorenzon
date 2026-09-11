"""
Normalization utilities shared between offline training (MMFi dataset) and
live webcam inference for the action-recognition + quality-scoring network.

WHY NORMALIZE AT ALL:
Feeding raw (pixel) keypoint coordinates into the network would let it
"cheat" by learning things it shouldn't -- where in the frame the person
happens to stand, how far they are from the camera, how tall they are --
instead of the actual SHAPE of the pose, which is what we want it to
classify/score. Normalizing removes translation and scale before the
network ever sees the data.

WHY THIS HAS TO BE ITS OWN SHARED MODULE:
The whole point of training on MMFi and then running live on a webcam is
that the network sees the SAME kind of input in both cases. If training
used one normalization and the live script used another (or none), the
network would face a distribution it never saw during training and
predictions would be unreliable. So this module is meant to be imported
by build_windowed_dataset.py (offline) AND by whatever live webcam script
uses the trained model later -- never re-implemented in either place.

Normalization steps, per frame:
  1. Center every keypoint on the mid-hip point (average of LEFT_HIP and
     RIGHT_HIP). Removes translation.
  2. Scale every keypoint by the shoulder width (distance between
     LEFT_SHOULDER and RIGHT_SHOULDER). Removes scale (distance from
     camera, body-size differences between people).

This file lives in <PROJECT_ROOT>/src/keypoint_normalize.py, alongside
utils.py.
"""
import numpy as np

from utils import (
    LEFT_HIP, RIGHT_HIP, LEFT_SHOULDER, RIGHT_SHOULDER, keypoints_are_valid,
)

# All 17 COCO keypoints are kept as network input (not just the 3-4 joints
# the angle-based scripts use) -- the network can potentially pick up on
# cues a single joint-angle can't (e.g. torso lean, arm swing during a
# squat), so there's no reason to throw information away here.
NUM_KEYPOINTS = 17
NUM_COORDS = 2  # (x, y)
FLAT_SIZE = NUM_KEYPOINTS * NUM_COORDS  # 34 -- the network's per-timestep input size

MIN_SCALE = 1e-3  # guards against a degenerate (near-zero) shoulder width


def frame_is_usable(frame_kp, frame_conf=None):
    """A frame is usable for normalization only if the 4 reference joints
    (both hips, both shoulders) are valid -- those are the only ones
    normalization itself depends on. Other joints being missing/invalid is
    not checked here; this function only guards the normalization step."""
    return keypoints_are_valid(
        frame_kp, frame_conf, [LEFT_HIP, RIGHT_HIP, LEFT_SHOULDER, RIGHT_SHOULDER]
    )


def normalize_frame(frame_kp):
    """Center on mid-hip and scale by shoulder width.

    `frame_kp` is a (17, 2) array-like. Returns a (17, 2) numpy array, or
    None if the frame is degenerate (near-zero shoulder width -- dividing
    by it would blow up the coordinates instead of normalizing them)."""
    frame_kp = np.asarray(frame_kp, dtype=float)

    mid_hip = (frame_kp[LEFT_HIP] + frame_kp[RIGHT_HIP]) / 2.0
    shoulder_width = np.linalg.norm(frame_kp[LEFT_SHOULDER] - frame_kp[RIGHT_SHOULDER])

    if shoulder_width < MIN_SCALE:
        return None

    centered = frame_kp - mid_hip
    scaled = centered / shoulder_width
    return scaled


def normalize_sequence(keypoints_seq, kp_conf_seq=None):
    """Normalize a whole (T, 17, 2) sequence.

    Frames that fail the hip/shoulder validity check, or are degenerate,
    are DROPPED (not zero-filled) -- same "skip rather than fabricate"
    policy used throughout the rest of this codebase (see
    keypoints_are_valid usage in the evaluate_dataset_*.py scripts). This
    means the returned array can be shorter than the input.

    Returns a (T', 17, 2) float array, T' <= T. T' can be 0 if no frame in
    the sequence was usable.
    """
    normalized = []
    for i, frame_kp in enumerate(keypoints_seq):
        frame_conf = kp_conf_seq[i] if kp_conf_seq is not None else None
        if not frame_is_usable(frame_kp, frame_conf):
            continue
        normalized_frame = normalize_frame(frame_kp)
        if normalized_frame is not None:
            normalized.append(normalized_frame)

    if len(normalized) == 0:
        return np.zeros((0, NUM_KEYPOINTS, NUM_COORDS), dtype=float)
    return np.stack(normalized, axis=0)


def flatten_sequence(normalized_seq):
    """(T, 17, 2) -> (T, 34). This flat, per-timestep vector is what
    actually gets fed into the GRU (one 34-dim vector per frame)."""
    T = normalized_seq.shape[0]
    return normalized_seq.reshape(T, FLAT_SIZE)


# --- Left/right mirroring (used as a training-time augmentation) ---------
# COCO 17-keypoint index pairs that are anatomical left/right counterparts.
# Index 0 (nose) has no counterpart and is left untouched.
MIRROR_PAIRS = [(1, 2), (3, 4), (5, 6), (7, 8), (9, 10), (11, 12), (13, 14), (15, 16)]


def mirror_normalized_sequence(normalized_seq):
    """Horizontally mirror an already-normalized (T, 17, 2) sequence.

    Since normalize_frame() already centers each frame on mid-hip, a clean
    left-right flip is just negating x. But that alone isn't enough: after
    negating x, the keypoint that used to sit at index 5 (anatomical LEFT
    shoulder) now sits on the right side of the (flipped) image, while the
    network still expects index 5 to mean "left shoulder". So each
    left/right pair also needs to be swapped, or the mirrored pose would be
    anatomically inconsistent (e.g. a mirrored right-lunge would show its
    "left ankle" channel where a real left-lunge's left ankle would be, but
    built from a right lunge's geometry -- silently wrong, not just visually
    flipped).

    Used as a training-time augmentation in build_windowed_dataset.py: it
    both increases the amount of data per exercise, and, for the
    left/right-specific exercises (lunge, limb extension), lets each side's
    recordings contribute training data to the *other* side's class too.
    """
    mirrored = np.array(normalized_seq, dtype=float, copy=True)
    mirrored[:, :, 0] *= -1.0
    for i, j in MIRROR_PAIRS:
        mirrored[:, [i, j], :] = mirrored[:, [j, i], :]
    return mirrored