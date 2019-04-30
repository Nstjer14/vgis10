# -*- coding: utf-8 -*-
"""
Created on Mon Apr 29 11:14:37 2019

@author: Niclas
"""
import numpy as np
import imutils
import cv2
from skimage.morphology import skeletonize_3d
import sknw
from matplotlib import pyplot as plt


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


graph = sknw.build_sknw(skeleton)
plt.imshow(image, cmap='gray')

for (s,e) in graph.edges():
    ps = graph[s][e]['pts']
    plt.plot(ps[:,1], ps[:,0], 'green')
    
# draw node by o
node, nodes = graph.node, graph.nodes()
ps = np.array([node[i]['o'] for i in nodes])
plt.plot(ps[:,1], ps[:,0], 'r.')

# title and show
plt.title('Build Graph')
plt.show()