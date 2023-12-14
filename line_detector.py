# import cv2
# import numpy as np
# import matplotlib.pyplot as plt

# #read image 
# image = cv2.imread( "parking_area.png")

# #convert to gray
# gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# #performing binary thresholding
# kernel_size = 3
# ret,thresh = cv2.threshold(gray,200,255,cv2.THRESH_BINARY)  

# #finding contours 
# cnts = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
# cnts = cnts[0] if len(cnts) == 2 else cnts[1]

# #drawing Contours
# radius =2
# color = (0,0,255)
# cv2.drawContours(image, cnts, -1,color , radius)
# # cv2.imshow(image) commented as colab don't support cv2.imshow()
# # cv2.imshow("Contours", image)
# # cv2.waitKey()

# plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
# plt.title("Contours")
# plt.show()



import cv2
import numpy as np

# Capture video from webcam (change 0 to the desired camera index)
cap = cv2.VideoCapture(0)

while True:
    # Read a frame from the webcam
    ret, frame = cap.read()

    # Convert the frame to the HSV color space
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Define a range for red color (you may need to adjust these values)
    lower_red = np.array([0, 100, 100])
    upper_red = np.array([10, 255, 255])

    # Create a mask using the inRange function
    mask = cv2.inRange(hsv, lower_red, upper_red)

    # Bitwise AND to isolate the red color in the original frame
    red_isolated = cv2.bitwise_and(frame, frame, mask=mask)

    # Convert the isolated frame to grayscale
    gray = cv2.cvtColor(red_isolated, cv2.COLOR_BGR2GRAY)

    # Performing binary thresholding
    _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)

    # Finding contours
    cnts, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    # Drawing contours
    radius = 2
    color = (0, 0, 255)
    cv2.drawContours(frame, cnts, -1, color, radius)

    # Display the frame with contours
    cv2.imshow("Contours", frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close all windows
cap.release()
cv2.destroyAllWindows()