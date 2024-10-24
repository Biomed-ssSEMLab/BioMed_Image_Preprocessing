import cv2
import numpy as np

img = cv2.imread('/media/liuxz/3EA0B4CEA0B48E41/output/001_S1R1/000001/001_000001_009_2021-03-26T0913128371167.bmp',0)
#dinimish 'white-line'
print('\n白线消除')
img0= 255 - img
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
img1 = clahe.apply(img0)
cv2.imwrite('/media/liuxz/3EA0B4CEA0B48E41/output/test2/image1.bmp', img1)
klist = [1807, 1808, 1809]
for k in klist:
    for i in range(3876):
        img1[k, i] = int(img1[k - 2, i] / 2 + img1[k + 2, i] / 2)

cv2.imwrite('/media/liuxz/3EA0B4CEA0B48E41/output/test2/img2.bmp', img1)
