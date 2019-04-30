# -*- coding: utf-8 -*-
"""
Created on Tue Apr 30 09:37:06 2019

@author: Niclas
"""

import cv2
import numpy as np
from skimage.morphology import skeletonize

cap = cv2.VideoCapture('../videos/Narrow-3f-bg-50fps.avi')


#out = cv2.VideoWriter('outpy.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 30, (1920,805))
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output.avi',fourcc, 25, (805,1920))

while True:
    ret, frame = cap.read()
    if frame is None:
        break
    image = frame[167:972, 0:1920]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.threshold(blurred, 60, 255, cv2.THRESH_BINARY)[1]
    
    image = cv2.bitwise_not(thresh)
    '''
    skeleton = skeletonize(image)
    
    rows = len(skeleton[1])
    cols = len(skeleton)
    for i in range(cols):
        for j in range(rows):
            if skeleton[i][j] == 255:
                cv2.circle(frame, (j,i), 0, (0, 255, 0), -2)
    '''            
    out.write(image)
    
    cv2.imshow('video',image)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
cap.release()
out.release()
cv2.destroyAllWindows()