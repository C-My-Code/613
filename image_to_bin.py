import numpy
import cv2
import matplotlib.pyplot as plt
import os
import sys



image_list = []

for i in range(1,len(sys.argv)):
    image_list.append(sys.argv[i])

size_total = 0
file = open("image.bin", "ab")
for n in range(len(image_list)):
    img = cv2.imread(image_list[n], cv2.IMREAD_COLOR)
    rows, cols,_ = img.shape
    print("Adding:"+image_list[n]+" Rows: "+str(rows)+" Cols:"+str(cols))
    for i in range(rows):
        for j in range(cols):
            for k in range(3):
                file.write(img[i,j,k])
    size_total+=rows*cols
file.close()
print("Total Pixels:"+str(size_total))
