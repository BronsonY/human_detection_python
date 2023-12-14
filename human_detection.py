import cv2
import pygame

# Open video file
# cap = cv2.VideoCapture('in.avi')

# Open video file
cap = cv2.VideoCapture(0)

# Load pre-trained Haar Cascades for full and upper body detection
fullbody_cascade = cv2.CascadeClassifier('haarcascade_fullbody.xml')
upperbody_cascade = cv2.CascadeClassifier('haarcascade_upperbody.xml')
lowerbody_cascade = cv2.CascadeClassifier('haarcascade_lowerbody.xml')


# Create a zone rectangle
zone_x, zone_y, zone_width, zone_height = 100, 100, 400, 300
zone_color = (0, 255, 0)  # Green color

# Set the window name
window_name = 'frame'

# Initialize Pygame for sound
pygame.init()

# Load the alarm sound
alarm_sound = pygame.mixer.Sound('beep-warning-6387.mp3')  # Replace 'alarm.wav' with the path to your sound file

# Create a window and set its initial size
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
cv2.resizeWindow(window_name, 800, 600)  # Set your desired width and height

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    if not ret:
        break

    # Set a slight Gaussian blur to the frame
    blurred_frame = cv2.GaussianBlur(frame, (5, 5), 0)

    # Convert the frame to grayscale for Haar Cascades
    gray = cv2.cvtColor(blurred_frame, cv2.COLOR_BGR2GRAY)

    # Detect full bodies
    fullbodies = fullbody_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3, minSize=(40, 40))

    # Detect upper bodies
    upperbodies = upperbody_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3, minSize=(30, 30))

    # Detect upper bodies
    lowerbodies = lowerbody_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3, minSize=(30, 30))

    # Combine the detected bodies
    all_bodies = list(fullbodies) + list(upperbodies) + list(lowerbodies)

    # Draw zone rectangle
    cv2.rectangle(frame, (zone_x, zone_y), (zone_x + zone_width, zone_y + zone_height), zone_color, 2)

    # Check if any bodies are inside the zone rectangle
    for (x, y, w, h) in all_bodies:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        # Check if the center of the bounding box is inside the zone rectangle
        if zone_x < x + w/2 < zone_x + zone_width and zone_y < y + h/2 < zone_y + zone_height:
            print("Inside the Zone")
            alarm_sound.play()

    # Display the resulting frame
    cv2.imshow(window_name, frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object, stop the sound, and close all windows
cap.release()
pygame.mixer.stop()
cv2.destroyAllWindows()