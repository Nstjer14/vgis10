# -*- coding: utf-8 -*-
"""
Created on Mon Apr 22 13:52:49 2019

@author: Niclas
"""
import numpy as np
import imutils
import cv2
from skimage.morphology import skeletonize_3d



# load the image, convert it to grayscale, blur it slightly,
# and threshold it
image = cv2.imread('moment.jpg')
image = image[167:972, 0:1920]
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (5, 5), 0)
thresh = cv2.threshold(blurred, 60, 255, cv2.THRESH_BINARY)[1]
#image = thresh

#roi = thresh[167:972, 0:1920]
roi = cv2.bitwise_not(thresh)
#cv2.rectangle(image,(0,166),(1920,972),(0,255,0),1)

skeleton = skeletonize_3d(roi)



    
#print(np.where(skeleton == 1))
#print(skeleton[626][750])

# find contours in the thresholded image
cnts = cv2.findContours(roi.copy(), cv2.RETR_EXTERNAL,
	cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)

# loop over the contours
for (i, c) in enumerate(cnts):
	# compute the center of the contour
    M = cv2.moments(c)
    cX = int(M["m10"] / M["m00"])
    cY = int(M["m01"] / M["m00"])
 
	# draw the contour and center of the shape on the image
    cv2.drawContours(image, [c], -1, (0, 255, 0), 1)
	#cv2.circle(image, (cX, cY), 2, (255, 255, 255), -2)
    cv2.putText(image, "ID#{}".format(i + 1) , (cX + 10, cY + 30),
		cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)



rows = len(skeleton[1])
cols = len(skeleton)
for i in range(cols):
    for j in range(rows):
        if skeleton[i][j] == 255:
            cv2.circle(image, (j,i), 0, (0, 255, 0), -2)


# show the image
cv2.imshow("Image", image)
cv2.waitKey(0) & 0xFF
cv2.destroyAllWindows()
