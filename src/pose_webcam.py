import cv2
from ultralytics import YOLO

from utils import MODEL_PATH

# Load the model YOLOv8 pose (version "nano", lightweight and fast for CPU)
# Path now comes from utils.py (PROJECT_ROOT/yolov8n-pose.pt) instead of a
# bare relative filename, so this works regardless of the current working
# directory the script is launched from.
model = YOLO(MODEL_PATH)

# Open the webcam (0 = default PC webcam)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Unable to open the webcam.")
    exit()

print("Press 'q' to exit.")

# Create the window in resizable mode
cv2.namedWindow('Pose Recognition - Press Q to exit', cv2.WINDOW_NORMAL)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error in reading frame.")
        break

    # Execute pose estimation on the current frame
    # Note: this demo script draws ALL detected people (no patient
    # selection). See realtime_comparison.py for the multi-person-aware
    # version used for actual scoring.
    results = model(frame, verbose=False)

    # Draw the skeleton automatically on the frame
    annotated_frame = results[0].plot()

    # Display the result on screen
    cv2.imshow('Pose Recognition - Press Q to exit', annotated_frame)

    # Exit by pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()