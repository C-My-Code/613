import numpy
import cv2
import matplotlib.pyplot as plt
import sys

image_list = []

for i in range(1,len(sys.argv), 2):
    image_list.append((int(sys.argv[i]),int(sys.argv[i+1])))

file = open("image_out.bin", "rb")
for i in range(len(image_list)):
    img_ar = image_list[i]
    img = numpy.zeros([img_ar[0],img_ar[1],3],dtype=numpy.uint8)

    for l in range(img_ar[0]):
        for j in range(img_ar[1]):
            for k in range(3):
                img[l,j,k] = int.from_bytes(file.read(1), byteorder="big", signed=False)
    filen = "processed"+str(i)+".jpg"
    cv2.imwrite(filen, img)