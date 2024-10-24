import cv2
import numpy as np
import file_utils

def convert_img_to_uint8(src_img):
    """将非uint8的图像转成uint8"""
    if src_img.dtype == "uint8":
        return src_img
    else:
        dst_img = src_img - src_img.min()
        dst_img = dst_img / (dst_img.max() - dst_img.min())
        dst_img *= 255
        dst_img = dst_img.astype(np.uint8)
        return dst_img

def get_central_symmetry_img(src_img):
    """根据图像获取其中心对称图像"""
    dst_img = np.zeros(src_img.shape, src_img.dtype)
    for i in range(src_img.shape[0]):
        for j in range(src_img.shape[1]):
            dst_img[i][j] = src_img[src_img.shape[0] - 1 - i][src_img.shape[1] - 1 - j]
    return dst_img

def img_pixel_value_invert(src_img):
    """图像像素值反转, 比如uint8的图像, 像素为0变为255, 像素255变为0. 暂时只支持uint8和uint16"""
    if src_img.dtype == "uint8":
        max_v = np.power(2, 8) - 1
    elif src_img.dtype == "uint16":
        max_v = np.power(2, 16) - 1
    else:
        raise Exception("Error! Unsupport format {}, only support uint8 and uint16!".format(src_img.dtype))

    dst_img = max_v - src_img
    return dst_img

def main():
    # /media/hqjin/Elements/om_data2/Image-1-(2).png
    # /home/hqjin/tmp/imgs/002/S2M1_tr1-tc1.png
    # /home/hqjin/Pictures/2022-09-07_14-12.png
    src_img_path = "/home/hqjin/Pictures/2022-09-07_14-12.png"
    src_img = cv2.imread(src_img_path, cv2.IMREAD_UNCHANGED)
    dst_img = img_pixel_value_invert(src_img)
    tmp_img_path = file_utils.create_tmp_file_path(src_img_path)
    cv2.imwrite(tmp_img_path, dst_img)    

if __name__ == '__main__':
    main()