# -*- coding: utf-8 -*-
"""
Created on Tue Apr 30 09:37:06 2019

@author: Niclas
"""

import cv2
import numpy as np
from skimage.morphology import skeletonize_3d
import skvideo.io
import imutils

cap = cv2.VideoCapture('../../videos/Narrow-3f-bg-50fps_Trim.mp4')


#out = cv2.VideoWriter('outpy.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 30, (1920,805))
#fourcc = cv2.VideoWriter_fourcc(*'MJPG')
#out = cv2.VideoWriter('output.mp4',fourcc, 50.0, (1920,805))
writer = skvideo.io.FFmpegWriter("outputvideo_short.mp4")
#writer.open()

while True:
    ret, frame = cap.read()
    if frame is None:
        break
    image = frame[167:972, 0:1920]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.threshold(blurred, 60, 255, cv2.THRESH_BINARY)[1]

    roi = cv2.bitwise_not(thresh)

    skeleton = skeletonize_3d(image)

    cnts = cv2.findContours(roi.copy(), cv2.RETR_EXTERNAL,
    	cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)

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
                cv2.circle(frame, (j,i), 0, (0, 255, 0), -2)



    #cv2.imshow('video',image)
    writer.writeFrame(image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
writer.close()
#out.release()
cv2.destroyAllWindows()
