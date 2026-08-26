import cv2
from ultralytics import YOLO

# Load the model YOLOv8 pose (version "nano", lightweight and fast for CPU)
# On first run, it automatically downloads (~6 MB)
model = YOLO('yolov8n-pose.pt')

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