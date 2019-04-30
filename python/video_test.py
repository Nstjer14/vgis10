# -*- coding: utf-8 -*-
"""
Created on Wed Apr 10 10:15:43 2019

@author: Niclas
"""

import numpy as np
import cv2
from skimage.morphology import skeletonize, medial_axis
from skimage import color
from skimage import data
from skimage.util import invert
import matplotlib.pyplot as plt
import matplotlib.image as im


#cap = cv2.VideoCapture(-1)
cap = cv2.VideoCapture('../videos/Narrow-3f-bg-50fps.avi')

kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
fgbg = cv2.bgsegm.createBackgroundSubtractorMOG()

image = im.imread('screen.png') # read in the image
image = color.rgb2gray(image) # convert the image from rgb to grayscale, as only black and white is present.
image = np.sign(image) # 

skel, distance = medial_axis(image, return_distance=True)
skeleton = skeletonize(image)

# Distance to the background for pixels of the skeleton
dist_on_skel = distance * skel


fig1 = plt.imshow(dist_on_skel)

fig2 = plt.savefig('skeleton.png')

yes = np.matrix.nonzero(skeleton)
count_skel = np.count_nonzero(skeleton)
count_img = np.count_nonzero(image)
print(count_skel, count_img)


'''
while True:
    ret, frame=cap.read()
    
    fgmask = fgbg.apply(frame)
    #fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)
    
    #fgmask = skeletonize(frame)
    
    cv2.imshow('frame',fgmask)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break  

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
'''