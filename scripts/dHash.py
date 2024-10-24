# -*- coding : UTF-8 -*-

'''
基于感知哈希的图像相似度检测
'''

import cv2 as cv
from PIL import Image
import os
import numpy as np
import copy

def averageGray(image):
    image = image.astype(int)
    for y in range(image.shape[1]): # y is width
        for x in range(image.shape[0]): # x is height
            gray = (image[x,y,0] + image[x,y,1] + image[x,y,2]) // 3
            image[x,y] = gray
    return image.astype(np.uint8)

def averageGrayWithWeighted(image):
    image = image.astype(int)
    for y in range(image.shape[1]): # y is width
        for x in range(image.shape[0]): # x is height
            gray = image[x,y,0] * 0.3 + image[x,y,1] * 0.59 + image[x,y,2] * 0.11
            image[x,y] = int(gray)
    return image.astype(np.uint8)

def maxGray(image):
    for y in range(image.shape[1]): # y is width
        for x in range(image.shape[0]):
            gray = max(image[x,y]) # x is height
            image[x,y] = gray
    return image

def resize_opencv(image,weight = 8,height = 8):
    smallImage = cv.resize(image,(weight,height),interpolation=cv.INTER_LANCZOS4)
    return smallImage

def calculateDifference(image,weight = 8,height = 8):
    differenceBuffer = []
    for x in range(weight):
        for y in range(height - 1):
            differenceBuffer.append(image[x,y] > image[x,y + 1])
    return differenceBuffer

def makeHash(differ):
    hashOrdString = "0b"
    for value in differ:
        hashOrdString += str(int(value))
    hashString = hex(int(hashOrdString,2))
    return hashString

def stringToHash(image1):
    # grayImage1 = averageGrayWithWeighted(copy.deepcopy(image1))
    # smallImage1 = resize_opencv(copy.deepcopy(grayImage1))
    grayImage1 = copy.deepcopy(image1)
    smallImage1 = resize_opencv(copy.deepcopy(grayImage1))
    differ = calculateDifference(copy.deepcopy(smallImage1))
    return makeHash(differ)

def calculateHammingDistance(differ1,differ2):
    difference = (int(differ1, 16)) ^ (int(differ2, 16))
    return bin(difference).count("1")

def compute(img1,img2):
    pic1 = stringToHash(img1)
    pic2 = stringToHash(img2)
    return (8 * 8 - calculateHammingDistance(pic1,pic2)) / (8 * 8), np.mean(np.multiply((img1-np.mean(img1)),(img2-np.mean(img2))))/(np.std(img1)*np.std(img2))