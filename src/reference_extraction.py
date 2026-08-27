import sys
import os
import numpy as np

# Add the mmfi_lib folder to the project path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'mmfi_lib'))
from mmfi import MMFi_Database, MMFi_Dataset

# --- CONFIGURATION ---
DATASET_ROOT = 'D:\\Università\\2 Magistrale\\2025-26\\secondo semestre\\computer vision\\progetto\\InHome_Physical_Therapy_Pose_Estimation_System_Gheller-Lorenzon\\dataset\\MMFi_Dataset'  # <-- replace with your actual path
SUBJECT = 'S01'
ACTION = 'A12'  # A12 = Squat

# --- Load the database and the specific sequence ---
database = MMFi_Database(DATASET_ROOT)
data_form = {SUBJECT: [ACTION]}

dataset = MMFi_Dataset(
    data_base=database,
    data_unit='sequence',
    modality='rgb',
    split='reference',
    data_form=data_form
)

sample = dataset[0]
keypoints_seq = sample['input_rgb']  # shape of the action: (num_frame, 17, 2)
print("Shape of the input_rgb sequence:", keypoints_seq.shape)

# --- CALCULATION OF ANGLES ---
# 

# COCO keypoint indices (same format as YOLOv8-Pose)
LEFT_HIP, LEFT_KNEE, LEFT_ANKLE = 11, 13, 15
RIGHT_HIP, RIGHT_KNEE, RIGHT_ANKLE = 12, 14, 16

def calcola_angolo(a, b, c):
    """Calculate the angle in degrees at vertex b, between segments a-b and b-c."""
    a, b, c = np.array(a), np.array(b), np.array(c)
    ba = a - b
    bc = c - b
    cos_angolo = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-8)
    cos_angolo = np.clip(cos_angolo, -1.0, 1.0)
    return np.degrees(np.arccos(cos_angolo))

# Calculate the left knee angle for each frame in the sequence
knee_angles_left = []
for frame_kp in keypoints_seq:
    anca = frame_kp[LEFT_HIP]
    ginocchio = frame_kp[LEFT_KNEE]
    caviglia = frame_kp[LEFT_ANKLE]
    angolo = calcola_angolo(anca, ginocchio, caviglia)
    knee_angles_left.append(angolo)

knee_angles_left = np.array(knee_angles_left)
print("Left knee angles over time (first 10 frames):", knee_angles_left[:10])

# Save the reference curve for use in Phase 3
np.save('squat_reference.npy', knee_angles_left)
print("Reference curve saved.")