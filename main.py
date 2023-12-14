import cv2

hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

# Open the webcam
webcam = cv2.VideoCapture(0)

while True:
    # Read a frame from the webcam
    ret, frame = webcam.read()

    # If reading the frame was successful
    if ret:
        # Detect humans in the frame
        humans, _ = hog.detectMultiScale(frame)

        # Draw rectangles around detected humans
        for (x, y, w, h) in humans:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Display the frame
        cv2.imshow("Human Detection", frame)

        # Check for the 'q' key to quit the loop
        key = cv2.waitKey(1)
        if key == ord("q"):
            break

# Release the webcam and close all windows
webcam.release()
cv2.destroyAllWindows()