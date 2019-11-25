# import the packages
import cv2
import numpy as np
import argparse
import imutils

# argument parser construction
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", default="shapes_and_colors.jpg", help="Add image path")
args = vars(ap.parse_args())

# loading the image
image = cv2.imread(args['image'])
# cv2.imshow("Image", image)
# cv2.waitKey(0)

# convert the image to gray image
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# cv2.imshow("Gray Image", gray)
# cv2.waitKey(0)

# Apply Gaussian Blur
blur = cv2.GaussianBlur(gray, (5,5), 0)
# cv2.imshow("Blurred Image", blur)
threshold = cv2.threshold(blur, 60, 255, cv2.THRESH_BINARY)[1]
# cv2.imshow("Threshold Image", threshold)
# cv2.waitKey(0)

# find contours
contours = cv2.findContours(threshold.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
contours = imutils.grab_contours(contours)

# loop over the contours
for cnt in contours:
    if cv2.contourArea(cnt) > 60: # 60 is min threshold
        # center of the contour using cv2.moments(cnt)
        M = cv2.moments(cnt)
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
        
        # draw the contour and center of the shape on the image
        cv2.drawContours(image, [cnt], -1, (0,255,0), 2)
        cv2.circle(image, (cX, cY), 7, (255, 255, 255), -1)
        cv2.putText(image, "center", (cX - 20, cY - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    
        # show the image
        cv2.imshow("Image", image)
        cv2.waitKey(0)

# edge detection using Canny method
# edged = cv2.Canny(blur, 30, 150)
# cv2.imshow("Canny/Edged Image", edged)
# cv2.waitKey(0)