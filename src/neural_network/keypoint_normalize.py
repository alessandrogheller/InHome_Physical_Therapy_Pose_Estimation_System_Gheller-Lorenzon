"""
Shared keypoint normalization utilities for training and live inference.

Each frame is centered on the midpoint of the hips and scaled by shoulder
width, making the network focus on pose shape rather than position or size.
"""
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import (
    LEFT_HIP, RIGHT_HIP, LEFT_SHOULDER, RIGHT_SHOULDER, keypoints_are_valid,
)


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
    """Center keypoints on the hips and scale them by shoulder width."""
    frame_kp = np.asarray(frame_kp, dtype=float)

    # Remove translation by centering the pose on the midpoint of the hips.
    mid_hip = (frame_kp[LEFT_HIP] + frame_kp[RIGHT_HIP]) / 2.0
    shoulder_width = np.linalg.norm(frame_kp[LEFT_SHOULDER] - frame_kp[RIGHT_SHOULDER])

    if shoulder_width < MIN_SCALE:
        return None

    # Remove scale differences by dividing all coordinates by shoulder width.
    centered = frame_kp - mid_hip
    scaled = centered / shoulder_width
    return scaled


def normalize_sequence(keypoints_seq, kp_conf_seq=None):
    # Keep only frames with the reference joints needed for reliable scaling.
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
    # Flatten each frame while preserving the temporal order of the sequence.
    T = normalized_seq.shape[0]
    return normalized_seq.reshape(T, FLAT_SIZE)


# --- Left/right mirroring (used as a training-time augmentation) ---------
# COCO 17-keypoint index pairs that are anatomical left/right counterparts.
# Index 0 (nose) has no counterpart and is left untouched.
MIRROR_PAIRS = [(1, 2), (3, 4), (5, 6), (7, 8), (9, 10), (11, 12), (13, 14), (15, 16)]


def mirror_normalized_sequence(normalized_seq):
    # Flip the horizontal coordinate and swap anatomical left/right channels.
    mirrored = np.array(normalized_seq, dtype=float, copy=True)
    mirrored[:, :, 0] *= -1.0
    for i, j in MIRROR_PAIRS:
        mirrored[:, [i, j], :] = mirrored[:, [j, i], :]
    return mirrored

def augment_window(window_flat, noise_std=0.02, scale_range=0.10, rotate_deg=10.0):
    """window_flat: (T, 34) normalizzato e appiattito -> versione augmentata."""
    T = window_flat.shape[0]
    kp = window_flat.reshape(T, NUM_KEYPOINTS, NUM_COORDS).copy()

    # scala casuale
    scale = 1.0 + np.random.uniform(-scale_range, scale_range)
    kp *= scale

    # piccola rotazione casuale (invarianza a leggeri disallineamenti di camera)
    theta = np.radians(np.random.uniform(-rotate_deg, rotate_deg))
    c, s = np.cos(theta), np.sin(theta)
    rot = np.array([[c, -s], [s, c]])
    kp = kp @ rot.T

    # jitter gaussiano
    kp += np.random.normal(0.0, noise_std, size=kp.shape)

    return kp.reshape(T, FLAT_SIZE).astype(np.float32)