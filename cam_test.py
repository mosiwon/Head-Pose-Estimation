
import cv2

cap = cv2.VideoCapture(1)

if not cap.isOpened():
    print("Camera not detected.")
else:
    while True:
        ret, frame = cap.read()
        if ret:
            cv2.imshow("Camera Test", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    