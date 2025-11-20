import cv2
import time
import imutils

# Initialize camera
cam = cv2.VideoCapture(0)
time.sleep(1)

first_frame = None
min_area = 500

while True:
    ret, frame = cam.read()
    if not ret:
        print("Failed to grab frame.")
        break

    # Resize for speed and consistency
    frame = imutils.resize(frame, width=600)

    # Convert to grayscale and blur
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray_blur = cv2.GaussianBlur(gray, (21, 21), 0)

    # Initialize the first frame
    if first_frame is None:
        first_frame = gray_blur
        continue

    # Compute absolute difference between current frame and first frame
    frame_delta = cv2.absdiff(first_frame, gray_blur)
    thresh = cv2.threshold(frame_delta, 25, 255, cv2.THRESH_BINARY)[1]
    thresh = cv2.dilate(thresh, None, iterations=2)

    # Find contours
    contours = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(contours)

    # Default status
    status_text = "Normal"

    for contour in contours:
        if cv2.contourArea(contour) < min_area:
            continue
        (x, y, w, h) = cv2.boundingRect(contour)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        status_text = "Motion Detected"

    # Show status on the screen
    color = (0, 0, 255) if status_text == "Motion Detected" else (255, 255, 255)
    cv2.putText(frame, f"Status: {status_text}", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    cv2.imshow("Motion Detection", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

# Cleanup
cam.release()
cv2.destroyAllWindows()
