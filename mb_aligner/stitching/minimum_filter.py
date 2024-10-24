from skimage import color
from skimage.morphology import disk
import skimage.filters.rank as sfr
import cv2

def minimum(image,k):
    img = color.rgb2gray(image)
    # print('rgb2gray finished')
    dst = sfr.windowed_histogram(img,disk(k))
    print('minimum finished')
    return dst

img0 = cv2.imread('/media/liuxz/3EA0B4CEA0B48E41/shiyan/SIFT_0.75.png')
# img0 = 255-img0
# cv2.imwrite('/media/liuxz/3EA0B4CEA0B48E41/shiyan/SIFT_0.75.png',img0)
k = 1
type = 'shihsihsihsisi'
dst = minimum(img0,k)
cv2.imwrite('/media/liuxz/3EA0B4CEA0B48E41/shiyan/SIFT_'+type+'_'+str(k)+'.png',dst)













# import cv2
#
# # 最大最小值滤波算子
# def max_min_value_filter(image, ksize=3, mode=1):
#     """
#     :param image: 原始图像
#     :param ksize: 卷积核大小
#     :param mode:  最大值：1 或最小值：2
#     :return:
#     """
#     img = image.copy()
#     rows, cols, channels = img.shape
#     if channels == 3:
#         img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#     padding = (ksize-1) // 2
#     new_img = cv2.copyMakeBorder(img, padding, padding, padding, padding, cv2.BORDER_CONSTANT, value=255)
#     for i in range(rows):
#         for j in range(cols):
#             roi_img = new_img[i:i+ksize, j:j+ksize].copy()
#             min_val, max_val, min_index, max_index = cv2.minMaxLoc(roi_img)
#             if mode == 1:
#                 img[i, j] = max_val
#             elif mode == 2:
#                 img[i, j] = min_val
#             else:
#                 raise Exception("please Select a Mode: max(1) or min(2)")
#     return img
#
# if __name__ == "__main__":
#     img0 = cv2.imread("/media/liuxz/3EA0B4CEA0B48E41/shiyan/001_000001_003_2021-03-26T0913129636621.bmp")
#
#     # mid_img = mid_functin(3, 3, copy.copy(img0))
#     # max_img = max_functin(3, 3, copy.copy(img0))
#     min_img = max_min_value_filter(img0, 3, 1)
#
#     # cv.imshow("original", img0)
#     # cv.imshow("max_img", max_img)
#     cv2.imwrite("/media/liuxz/3EA0B4CEA0B48E41/shiyan/min_img.bmp", min_img)
#     print('main ending')
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
# # import cv2 as cv
# # import numpy as np
# # import math
# # import copy
# #
# #
# # def spilt(a):
# #     if a / 2 == 0:
# #         x1 = x2 = a / 2
# #     else:
# #         x1 = math.floor(a / 2)
# #         x2 = a - x1
# #     return -x1, x2
# #
# #
# # def original(i, j, k, a, b, img):
# #     x1, x2 = spilt(a)
# #     y1, y2 = spilt(b)
# #     temp = np.zeros(a * b)
# #     count = 0
# #     for m in range(x1, x2):
# #         for n in range(y1, y2):
# #             print('original  running')
# #             if i + m < 0 or i + m > img.shape[0] - 1 or j + n < 0 or j + n > img.shape[1] - 1:
# #                 temp[count] = img[i, j, k]
# #             else:
# #                 temp[count] = img[i + m, j + n, k]
# #             count += 1
# #     return temp
# #
# #
# # # 中值滤波
# # def mid_functin(a, b, img):
# #     img0 = copy.copy(img)
# #     for i in range(0, img.shape[0]):
# #         for j in range(2, img.shape[1]):
# #             for k in range(img.shape[2]):
# #                 temp = original(i, j, k, a, b, img0)
# #                 img[i, j, k] = np.median(temp)
# #     return img
# #
# #
# # # 最大值滤波
# # def max_functin(a, b, img):
# #     img0 = copy.copy(img)
# #     for i in range(0, img.shape[0]):
# #         for j in range(2, img.shape[1]):
# #             for k in range(img.shape[2]):
# #                 temp = original(i, j, k, a, b, img0)
# #                 img[i, j, k] = np.max(temp)
# #     return img
# #
# #
# # # 最小值滤波
# # def min_functin(a, b, img):
# #     img0 = copy.copy(img)
# #     for i in range(0, img.shape[0]):
# #         for j in range(2, img.shape[1]):
# #             for k in range(img.shape[2]):
# #                 print('minimum  running')
# #                 temp = original(i, j, k, a, b, img0)
# #                 img[i, j, k] = np.min(temp)
# #     return img
# #
# #
# #
# #
# #
# # if __name__ == "__main__":
# #     img0 = cv.imread("/media/liuxz/3EA0B4CEA0B48E41/shiyan/001_000001_003_2021-03-26T0913129636621.bmp")
# #
# #     # mid_img = mid_functin(3, 3, copy.copy(img0))
# #     # max_img = max_functin(3, 3, copy.copy(img0))
# #     min_img = min_functin(3, 3, copy.copy(img0))
# #
# #     # cv.imshow("original", img0)
# #     # cv.imshow("max_img", max_img)
# #     cv.imwrite("/media/liuxz/3EA0B4CEA0B48E41/shiyan/min_img.bmp", min_img)
# #     print('main ending')
# #     # cv.imshow("mid_img", mid_img)
# #
# #     # cv.waitKey(0)
# #     # cv.destroyAllWindows()