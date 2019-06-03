# -*- coding: utf-8 -*-
"""
Created on Fri May 31 15:55:06 2019

@author: Niclas
"""

import numpy as np
import imutils
import cv2
from skimage.morphology import skeletonize_3d, skeletonize
#import sknw
from matplotlib import pyplot as plt
import csv

vidPath = '../videos/Narrow-3f-bg-50fps_Trim.avi'
contPath = 'contours/contours_3f.csv'
intPath = 'intersections/intersections_3f.csv'
vidObj = cv2.VideoCapture(vidPath)

def preProcess(image):
    img = image[167:972, 0:1920]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.threshold(blurred, 35, 255, cv2.THRESH_BINARY)[1]
    img = cv2.bitwise_not(thresh)
    binImg = img/255
    return img, binImg

def getContours(img):
    cnts = cv2.findContours(img.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    return cnts

def findNeighbours(x,y,img):
    """Return 8-neighbours of image point P1(x,y), in a clockwise order"""
    n_img = img
    x_1, y_1, x1, y1 = x-1, y-1, x+1, y+1;
    return [ n_img[x_1][y], n_img[x_1][y1], n_img[x][y1], n_img[x1][y1], n_img[x1][y], n_img[x1][y_1], n_img[x][y_1], img[x_1][y_1] ]

def getSkeletonIntersection(skeleton):
    """ Given a skeletonised image, it will give the coordinates of the intersections of the skeleton.
    
    Keyword arguments:
    skeleton -- the skeletonised image to detect the intersections of
    
    Returns: 
    List of 2-tuples (x,y) containing the intersection coordinates
    """
    # A biiiiiig list of valid intersections             2 3 4
    # These are in the format shown to the right         1 C 5
    #                                                    8 7 6 
    validIntersection = [[0,1,0,1,0,0,1,0],[0,0,1,0,1,0,0,1],[1,0,0,1,0,1,0,0],
                         [0,1,0,0,1,0,1,0],[0,0,1,0,0,1,0,1],[1,0,0,1,0,0,1,0],
                         [0,1,0,0,1,0,0,1],[1,0,1,0,0,1,0,0],[0,1,0,0,0,1,0,1],
                         [0,1,0,1,0,0,0,1],[0,1,0,1,0,1,0,0],[0,0,0,1,0,1,0,1],
                         [1,0,1,0,0,0,1,0],[1,0,1,0,1,0,0,0],[0,0,1,0,1,0,1,0],
                         [1,0,0,0,1,0,1,0],[1,0,0,1,1,1,0,0],[0,0,1,0,0,1,1,1],
                         [1,1,0,0,1,0,0,1],[0,1,1,1,0,0,1,0],[1,0,1,1,0,0,1,0],
                         [1,0,1,0,0,1,1,0],[1,0,1,1,0,1,1,0],[0,1,1,0,1,0,1,1],
                         [1,1,0,1,1,0,1,0],[1,1,0,0,1,0,1,0],[0,1,1,0,1,0,1,0],
                         [0,0,1,0,1,0,1,1],[1,0,0,1,1,0,1,0],[1,0,1,0,1,1,0,1],
                         [1,0,1,0,1,1,0,0],[1,0,1,0,1,0,0,1],[0,1,0,0,1,0,1,1],
                         [0,1,1,0,1,0,0,1],[1,1,0,1,0,0,1,0],[0,1,0,1,1,0,1,0],
                         [0,0,1,0,1,1,0,1],[1,0,1,0,0,1,0,1],[1,0,0,1,0,1,1,0],
                         [1,0,1,1,0,1,0,0]];
    skeImg = skeleton
    #skeImg = skeImg/255;
    intersections = list();
    for x in range(1,len(skeImg)-1):
        for y in range(1,len(skeImg[x])-1):
            # If we have a white pixel
            if skeImg[x][y] == 1:
                neighbours = findNeighbours(x,y,skeImg);
                valid = True;
                if neighbours in validIntersection:
                    intersections.append((y,x));
    # Filter intersections to make sure we don't count them twice or ones that are very close together
    for point1 in intersections:
        for point2 in intersections:
            if (((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2) < 10**2) and (point1 != point2):
                intersections.remove(point2);
    # Remove duplicates
    intersections = list(set(intersections));
    return intersections;

#Video reading loop - also main loop

vidObj = cv2.VideoCapture(vidPath)


count = 0
positions = []
countInts = 0
for i in range(100): positions.append((0,0))

while True:
    ret, frame = vidObj.read()
    if frame is None:
        break
    #Using preProcess to extract greyscale and binary images
    img, binImg = preProcess(frame)
    
    
    #getting skeleton from the binary image
    skeleton = skeletonize(binImg)
    rows = len(skeleton[1])
    cols = len(skeleton)
    for i in range(cols):
        for j in range(rows):
            if skeleton[i][j] == 1:
                cv2.circle(img, (j,i), 0, (0, 255, 0), -2)
  
    #finding intersections from the skeleton. Looking 
    #for any valid intersection between two fish spines.
    intersections = getSkeletonIntersection(skeleton)
    for h in intersections:
        print(h[0],h[1])
    #append intersection coordinates 
    #together with frame number
    if len(intersections) > 0:
        countInts += 1
        for k in intersections:
            positions[count] = (k[0],k[1])
    
    count += 1
'''   

    cv2.imshow('video',img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
'''
    
vidObj.release()
cv2.destroyAllWindows() 
print('Finished')
posCount = 0
g = 0
with open(intPath, 'w', newline="") as csvfile:
    #fieldnames = ['x_position', 'y_position']
    writer = csv.writer(csvfile)
    #writer.writeheader()
    for position in positions:
        posCount += 1
        writer.writerow(position)
        #x, y = position[0], position[1]
        #writer.writerow({'x_position': x, 'y_position': y})
        #g += 1
                    
print('finished writing data')
print(intPath)