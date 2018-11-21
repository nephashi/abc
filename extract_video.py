import cv2
import numpy as np

vidcap = cv2.VideoCapture('C:/Users/Lenovo/Desktop/videos1/001-nm-01-090.avi')
count = 0
imgs = []
while True:
    #cv2.imwrite("tmp/frame%d.jpg" % count, image)     # save frame as JPEG file
    success,image = vidcap.read()
    if success:
        #image = cv2.resize(image, (299, 299))
        imgs.append(image)
        count += 1
    else:
        break

cv2.imwrite('./bkgd.png', imgs[0])
for i in range(40, 50):
    cv2.imwrite('./' + str(i) + '.png', imgs[i])