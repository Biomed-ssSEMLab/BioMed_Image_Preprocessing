from PIL import Image
import cv2
import numpy as np

from skimage import io
Image.MAX_IMAGE_PIXELS = None

img1 = Image.open("/braindat/lab/xzliu/output/output/render/test/0310_W01_Sec310_tr1-tc1.png")
img2 = Image.open("/braindat/lab/xzliu/output/output/render/test/0311_W01_Sec311_tr1-tc1.png")

img1 = np.array(img1)
img2 = np.array(img2)

img1 = 255 - img1
img2 = 255 - img2


io.imsave('/braindat/lab/xzliu/output/output/render/test/0310_W01_Sec310.png', img1)
io.imsave('/braindat/lab/xzliu/output/output/render/test/0311_W01_Sec311.png', img2)

