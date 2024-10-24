# -*- coding: utf-8 -*-
import numpy as np
import os
import cv2

gridsize1 = 128
gs1 = (gridsize1, gridsize1)
gridsize2 = 64
gs2 = (gridsize2, gridsize2)
limit = 3.0
clahe = cv2.createCLAHE(clipLimit=limit, tileGridSize=gs1)
clahe2 = cv2.createCLAHE(clipLimit=limit, tileGridSize=gs2)

path = '/home/liuxz/clahe/000005org/'
files = os.listdir(path)
for i in files:
    image = cv2.imread(os.path.join(path,i), 0)
    image1 =  clahe.apply(image)
    image2 =  clahe2.apply(image1)
    print('/home/liuxz/clahe/000005/'+i)
    cv2.imwrite('/home/liuxz/clahe/000005/'+i,image2)
print('all finished ...')