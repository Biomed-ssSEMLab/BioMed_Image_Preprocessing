import os
import cv2
import json
import math
import numpy as np
import skimage
import file_utils
import img_utils
import argparse
import shutil
import ujson
from PIL import Image  # 用cv2读取大图可能会失败，所以需要PIL库，具体用法：src_img = np.array(Image.open(src_img_path))
Image.MAX_IMAGE_PIXELS = None

class my_class:
    def __init__(self):
	    self._embedding_type='base'

    @property
    def is_bert(self):
        return self._embedding_type == 'bert'

    @property
    def embedding_type(self):
        return self._embedding_type
    
def test():
    pass

def main():
    p=my_class()
    print(p.embedding_type) # 用户进行属性调用的时候，直接调用embedding_type即可，而不用知道属性名_embedding_type，因此用户无法更改属性，从而保护了类的属性。
    print(p.is_bert)

if __name__ == '__main__':
    main()