import cv2
import numpy as np
from .fs_access import FSAccess
import os

def read_image_file(fname_url):
    if "://" in fname_url:
        with FSAccess(fname_url, True) as image_f:
            img_buf = image_f.read()
            np_arr = np.frombuffer(img_buf, np.uint8)
            img = cv2.imdecode(np_arr, 0)
            if "Aligned" in fname_url:
                img = 255 - img
    else:

        img = cv2.imread(fname_url, cv2.IMREAD_GRAYSCALE)

    return img

def write_image_file(fname_url, img):
    if "://" in fname_url:
        retval, img_buf = cv2.imencode(os.path.splitext(fname_url)[1], img)
        assert(retval == True)
        with FSAccess(fname_url, True, read=False) as image_f:
            image_f.write(img_buf)
    else:
        cv2.imwrite(fname_url, img)
        img = cv2.imread(fname_url, cv2.IMREAD_ANYDEPTH)
    return img
