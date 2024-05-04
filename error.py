import numpy
import cv2
import os
import sys
import math

image_list = []
for i in range(1,len(sys.argv)):
    image_list.append(sys.argv[i])

img = []
for n in range(len(image_list)):
    img.append(cv2.imread(image_list[n], cv2.IMREAD_COLOR))

rows, cols,_ = img[0].shape

double l2_1
double l2_2
l2_1 = 0.0
l2_2 = 0.0
for i in range(rows):
    for j in range(cols):
        for k in range(3):
            l2_1+= img[0][i,j,k]*img[0][i,j,k]
            l2_2+= (img[1][i,j,k]-img[0][i,j,k])*(img[1][i,j,k]-img[0][i,j,k])
l2_1 = math.sqrt(l2_1)
l2_2 = math.sqrt(l2_2)
error = l2_2/l2_1

print("Relative Error: "+str(error))

