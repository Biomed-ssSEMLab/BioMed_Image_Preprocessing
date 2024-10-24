# -*- encoding: utf-8 -*-
'''
@Filename    : auxiliary_tools.py
@Description : 
@Datatime    : 2022/07/06 13:33:47
@Author      : hqjin
'''

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
from multiprocessing import Pool
from PIL import Image  # 用cv2读取大图可能会失败，所以需要PIL库，具体用法：src_img = np.array(Image.open(src_img_path))
Image.MAX_IMAGE_PIXELS = None

def invert_imgs(src_dir, dst_dir):
    """原始的电镜数据图片, 里面的像素值是255-的关系, 对所有图像255-, 保存，以便观察"""
    src_dir = file_utils.get_abs_dir(src_dir)
    dst_dir = file_utils.create_dir(dst_dir)

    for fname in os.listdir(src_dir):
        abs_fname = os.path.join(src_dir, fname)
        if not os.path.isfile(abs_fname):
            continue

        suffix = fname.split(".")[-1]
        if suffix != "bmp" or "thumbnail" in fname:
            continue

        img = cv2.imread(abs_fname, 0)
        img = img_utils.img_pixel_value_invert(img)
        cv2.imwrite(os.path.join(dst_dir, fname), img)

def get_tile_layout_by_json(json_file, save_img_file):
    """根据json获取61个tile的排列方式, 并以图片方式保存"""
    json_file = file_utils.get_abs_file_path(json_file)
    file_utils.create_save_dir_from_file(save_img_file)

    with open(json_file, "r") as f:
        json_list = json.load(f)

    if len(json_list) < 1 or len(json_list) % 61 != 0:
        raise Exception("Error! json_list less than 1 or not a multiple of 61.")

    # Only 61 tiles' bbox of the same section and the same MFOV are required
    sec_idx = json_list[0]["layer"]
    mFov_idx = json_list[0]["mfov"]

    idx_bbox_dict = {}
    for tile in json_list:
        if tile["layer"] == sec_idx and tile["mfov"] == mFov_idx:
            img_path = tile["mipmapLevels"]["0"]["imageUrl"]  # like: */001_000001_001_2022-05-08T1237143142689.bmp
            img_fname = img_path.split("/")[-1]
            tmp_list = img_fname.split("_")
            if len(tmp_list) != 4 or not tmp_list[2].isdigit():
                raise Exception("Error! Image name {} should be like 001_000001_001_2022-05-08T1237143142689.bmp.".format(img_fname))

            img_idx = int(tmp_list[2])
            if img_idx in idx_bbox_dict.keys():
                raise Exception("Error! Repetitive section index {}".format(img_idx))

            idx_bbox_dict[img_idx] = tile["bbox"]

    # Reduce the size of bbox by 20 times
    for idx, bbox in idx_bbox_dict.items():
        bbox[0] = int(math.floor(bbox[0] / 20))
        bbox[1] = int(math.ceil(bbox[1] / 20))
        bbox[2] = int(math.floor(bbox[2] / 20))
        bbox[3] = int(math.ceil(bbox[3] / 20))
        idx_bbox_dict[idx] = bbox

    min_x, min_y = np.iinfo(np.int32).max, np.iinfo(np.int32).max
    max_x, max_y = 0, 0
    for idx, bbox in idx_bbox_dict.items():
        if bbox[0] < min_x:
            min_x = bbox[0]
        if bbox[2] < min_y:
            min_y = bbox[2]
        if bbox[1] > max_x:
            max_x = bbox[1]
        if bbox[3] > max_y:
            max_y = bbox[3]

    for idx, bbox in idx_bbox_dict.items():
        bbox[0] -= min_x
        bbox[1] -= min_x
        bbox[2] -= min_y
        bbox[3] -= min_y
        idx_bbox_dict[idx] = bbox

    whole_img_w = max_x - min_x
    whole_img_h = max_y - min_y
    whole_img = np.ones([whole_img_h, whole_img_w, 3], np.uint8) * 255
    for idx, bbox in idx_bbox_dict.items():
        color = np.random.randint(0, 255, 3, dtype=np.uint8)
        cv2.rectangle(whole_img, (bbox[0], bbox[2]), (bbox[1], bbox[3]), (int(color[0]), int(color[1]), int(color[2])), 2)
        if idx < 10:
            txt_center = (int((bbox[0] + bbox[1]) / 2) - 12, int((bbox[2] + bbox[3]) / 2) + 15)
        else:
            txt_center = (int((bbox[0] + bbox[1]) / 2) - 32, int((bbox[2] + bbox[3]) / 2) + 15)
        cv2.putText(whole_img, str(idx), txt_center, cv2.FONT_HERSHEY_SIMPLEX, 2, (int(color[0]), int(color[1]), int(color[2])), 2)

    cv2.imwrite(save_img_file, whole_img)

def get_mFov_layout_by_json(json_file, save_img_file):
    """根据json获取mFov的排列方式, 并以图片方式保存"""
    json_file = file_utils.get_abs_file_path(json_file)
    file_utils.create_save_dir_from_file(save_img_file)

    with open(json_file, "r") as f:
        json_list = json.load(f)

    if len(json_list) < 1 or len(json_list) % 61 != 0:
        raise Exception("Error! json_list less than 1 or not a multiple of 61.")

    sec_idx = json_list[0]["layer"]
    for tile in json_list:
        if tile["layer"] != sec_idx:
            raise Exception("Error! Find different layer in the json_file.")

    idx_bbox_dict = {}
    cur_mFov_idx = json_list[0]["mfov"]
    cur_mFov_bbox = []
    for tile in json_list:
        mFov_idx = tile["mfov"]
        if mFov_idx == cur_mFov_idx:
            cur_mFov_bbox.append(tile["bbox"])
        else:
            idx_bbox_dict[cur_mFov_idx] = cur_mFov_bbox
            cur_mFov_idx = mFov_idx
            cur_mFov_bbox = []
            cur_mFov_bbox.append(tile["bbox"])
    idx_bbox_dict[cur_mFov_idx] = cur_mFov_bbox

    for idx, bbox in idx_bbox_dict.items():
        bbox = np.array(bbox)
        bbox = [np.min(bbox[:, 0]), np.max(bbox[:, 1]), np.min(bbox[:, 2]), np.max(bbox[:, 3])]
        bbox[0] = int(math.floor(bbox[0] / 200))  # Reduce the size of bbox by 200 times
        bbox[1] = int(math.ceil(bbox[1] / 200))
        bbox[2] = int(math.floor(bbox[2] / 200))
        bbox[3] = int(math.ceil(bbox[3] / 200))
        idx_bbox_dict[idx] = bbox

    min_x, min_y = np.iinfo(np.int32).max, np.iinfo(np.int32).max
    max_x, max_y = 0, 0
    for idx, bbox in idx_bbox_dict.items():
        if bbox[0] < min_x:
            min_x = bbox[0]
        if bbox[2] < min_y:
            min_y = bbox[2]
        if bbox[1] > max_x:
            max_x = bbox[1]
        if bbox[3] > max_y:
            max_y = bbox[3]

    for idx, bbox in idx_bbox_dict.items():
        bbox[0] -= min_x
        bbox[1] -= min_x
        bbox[2] -= min_y
        bbox[3] -= min_y
        idx_bbox_dict[idx] = bbox

    whole_img_w = max_x - min_x
    whole_img_h = max_y - min_y
    whole_img = np.ones([whole_img_h, whole_img_w, 3], np.uint8) * 255
    for idx, bbox in idx_bbox_dict.items():
        color = np.random.randint(0, 255, 3, dtype=np.uint8)
        cv2.rectangle(whole_img, (bbox[0], bbox[2]), (bbox[1], bbox[3]), (int(color[0]), int(color[1]), int(color[2])), 2)
        if idx < 10:
            txt_center = (int((bbox[0] + bbox[1]) / 2) - 18, int((bbox[2] + bbox[3]) / 2) + 18)
        else:
            txt_center = (int((bbox[0] + bbox[1]) / 2) - 38, int((bbox[2] + bbox[3]) / 2) + 18)
        cv2.putText(whole_img, str(idx), txt_center, cv2.FONT_HERSHEY_SIMPLEX, 2, (int(color[0]), int(color[1]), int(color[2])), 2)

    cv2.imwrite(save_img_file, whole_img)

def get_mFov_layout_by_txt(txt_file, save_img_file):
    """根据txt获取mFov的排列方式, 并以图片方式保存"""
    txt_file = file_utils.get_abs_file_path(txt_file)
    file_utils.create_save_dir_from_file(save_img_file)

    mFov_idx = []
    start_x, start_y = [], []
    with open(txt_file, "r") as f:
        lines = f.readlines()
        lines = sorted(lines)
        for line in lines:
            line_list = line.strip().split("\t")
            img_fname = line_list[0].replace("\\", "/")
            if not (img_fname.split("/")[0]).isdigit():
                continue

            mFov_idx.append(int(img_fname.split("/")[0]))
            start_x.append(float(line_list[1]))
            start_y.append(float(line_list[2]))
            
    start_x = np.array(start_x)
    start_y = np.array(start_y)
    start_x -= np.min(start_x)
    start_y -= np.min(start_y)

    idx_bbox_dict = {}
    cur_mFov_idx = mFov_idx[0]
    cur_mFov_bbox = []
    for idx, x, y in zip(mFov_idx, start_x, start_y):
        if idx == cur_mFov_idx:
            cur_mFov_bbox.append([int(math.floor(x)), int(math.ceil(x + 3876)), int(math.floor(y)), int(math.ceil(y + 3376))]) # left, right, top, down
        else:
            idx_bbox_dict[cur_mFov_idx] = cur_mFov_bbox
            cur_mFov_idx = idx
            cur_mFov_bbox = []
            cur_mFov_bbox.append([int(math.floor(x)), int(math.ceil(x + 3876)), int(math.floor(y)), int(math.ceil(y + 3376))])
    idx_bbox_dict[cur_mFov_idx] = cur_mFov_bbox

    for idx, bbox in idx_bbox_dict.items():
        bbox = np.array(bbox)
        bbox = [np.min(bbox[:, 0]), np.max(bbox[:, 1]), np.min(bbox[:, 2]), np.max(bbox[:, 3])]
        bbox[0] = int(math.floor(bbox[0] / 200))  # Reduce the size of bbox by 200 times
        bbox[1] = int(math.ceil(bbox[1] / 200))
        bbox[2] = int(math.floor(bbox[2] / 200))
        bbox[3] = int(math.ceil(bbox[3] / 200))
        idx_bbox_dict[idx] = bbox

    max_x, max_y = 0, 0
    for idx, bbox in idx_bbox_dict.items():
        if bbox[1] > max_x:
            max_x = bbox[1]
        if bbox[3] > max_y:
            max_y = bbox[3]

    whole_img = np.ones([max_y, max_x, 3], np.uint8) * 255
    for idx, bbox in idx_bbox_dict.items():
        color = np.random.randint(0, 255, 3, dtype=np.uint8)
        cv2.rectangle(whole_img, (bbox[0], bbox[2]), (bbox[1], bbox[3]), (int(color[0]), int(color[1]), int(color[2])), 2)
        if idx < 10:
            txt_center = (int((bbox[0] + bbox[1]) / 2) - 18, int((bbox[2] + bbox[3]) / 2) + 18)
        else:
            txt_center = (int((bbox[0] + bbox[1]) / 2) - 38, int((bbox[2] + bbox[3]) / 2) + 18)
        cv2.putText(whole_img, str(idx), txt_center, cv2.FONT_HERSHEY_SIMPLEX, 2, (int(color[0]), int(color[1]), int(color[2])), 2)

    cv2.imwrite(save_img_file, whole_img)

def check_rect_in_img_max_contour(img_path, rect):
    """对每一个mFov图片提取最大轮廓, 并判断指定的rect是否在轮廓内, 若在, 返回True; 否则, 返回False.
    其中, rect:[left, top, width, height]"""
    img_path = file_utils.get_abs_file_path(img_path)

    if rect[0] < 0:
        rect[0] = 0
    if rect[1] < 0:
        rect[1] = 0
    if rect[2] <= 0 or rect[3] <= 0:
        raise Exception("Error! {} is illegal!".format(rect))
        
    img = cv2.imread(img_path, 0)
    # 阈值化，有效区域为255，无效区域为0
    ret, thres_img = cv2.threshold(img, 254, 255, cv2.THRESH_BINARY_INV)

    tmp_str_list = img_path.rsplit(".", 1)

    # # 在灰度图上会框查看位置
    # cv2.rectangle(img, (rect[0], rect[1]), (rect[0] + rect[2], rect[1] + rect[3]), 0, 10)
    # cv2.imwrite(tmp_str_list[0] + "_tmp." + tmp_str_list[1], img)

    contours, hierarchy = cv2.findContours(thres_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = list(contours)
    contours.sort(key=lambda c: cv2.contourArea(c), reverse=True)

    # # 画最大轮廓
    # thres_img2 = np.zeros(thres_img.shape, np.uint8)
    # cv2.drawContours(thres_img2, contours, 0, 255, cv2.FILLED)
    # cv2.imwrite(tmp_str_list[0] + "_tmp2." + tmp_str_list[1], thres_img2)

    if img.shape[0] < rect[1] + rect[3] or img.shape[1] < rect[0] + rect[2]:
        print("Warning! rect {} exceed img size {} of {}.".format(rect, img.shape, img_path))
        return False

    for x in range(int(rect[0]), int(rect[0] + rect[2])):
        ret = cv2.pointPolygonTest(contours[0], (x, rect[1]), False)
        if ret < 0:
            print("Warning! rect {} not in the contour of img {}.".format(rect, img_path))
            return False

    for x in range(int(rect[0]), int(rect[0] + rect[2])):
        ret = cv2.pointPolygonTest(contours[0], (x, rect[1] + rect[3]), False)
        if ret < 0:
            print("Warning! rect {} not in the contour of img {}.".format(rect, img_path))
            return False

    for y in range(int(rect[1]), int(rect[1] + rect[3])):
        ret = cv2.pointPolygonTest(contours[0], (rect[0], y), False)
        if ret < 0:
            print("Warning! rect {} not in the contour of img {}.".format(rect, img_path))
            return False

    for y in range(int(rect[1]), int(rect[1] + rect[3])):
        ret = cv2.pointPolygonTest(contours[0], (rect[0] + rect[2], y), False)
        if ret < 0:
            print("Warning! rect {} not in the contour of img {}.".format(rect, img_path))
            return False

    return True

def loop_check_rect_in_img_max_contour(img_dir, rect_w, rect_h):
    """轮询文件夹下的所有图片, 对每一个图片提取最大轮廓, 并判断指定的rect是否在轮廓内, 若在, 返回True; 否则, 返回False.
    其中, rect只给定宽和高, 中心是图片中心, 所以可以算出对应每张图片其相应的rect是什么"""
    img_dir = file_utils.get_abs_dir(img_dir)

    rect_w = int(rect_w)
    rect_h = int(rect_h)
    total_img_num, dissatisfy_img_num = 0, 0
    for sub_dir in os.listdir(img_dir):
        abs_sub_dir = os.path.join(img_dir, sub_dir)
        if not os.path.isdir(abs_sub_dir):
            continue

        for file in os.listdir(abs_sub_dir):
            abs_file_path = os.path.join(abs_sub_dir, file)
            if not os.path.isfile(abs_file_path):
                continue
            if file.split(".")[-1].lower() not in ["jpg", "jpeg", "png", "bmp"]:
                continue

            img = cv2.imread(abs_file_path, 0)
            rect_x = int(img.shape[1] / 2 - rect_w / 2)
            rect_y = int(img.shape[0] / 2 - rect_h / 2)
            rect = [rect_x, rect_y, rect_w, rect_h]
            ret = check_rect_in_img_max_contour(abs_file_path, rect)
            if not ret:
                dissatisfy_img_num += 1
            total_img_num += 1

    print("total img num: {}, dissatisfy img num: {}, ratio: {}%".format(total_img_num, dissatisfy_img_num, round(dissatisfy_img_num / total_img_num * 100, 2)))

def crop_img_by_rect(rect, src_img_path, dst_img_path):
    """根据指定的rect对每一个mFov图片crop出相应的图片并保存.
    其中, rect:[left, top, width, height]"""
    src_img_path = file_utils.get_abs_file_path(src_img_path)
    file_utils.create_save_dir_from_file(dst_img_path)

    if rect[0] < 0:
        rect[0] = 0
    if rect[1] < 0:
        rect[1] = 0
    if rect[2] <= 0 or rect[3] <= 0:
        raise Exception("Error! {} is illegal!".format(rect))
        
    src_img = cv2.imread(src_img_path, 0)
    if src_img.shape[0] < rect[1] + rect[3] or src_img.shape[1] < rect[0] + rect[2]:
        print("Warning! rect {} exceed img size {} of {}.".format(rect, src_img.shape[:2], src_img_path))

    dst_img = src_img[rect[1]:rect[1] + rect[3], rect[0]:rect[0] + rect[2]]
    cv2.imwrite(dst_img_path, dst_img)

def loop_crop_img_by_rect(rect_w, rect_h, src_img_dir, dst_img_dir, dst_img_name_prefix=""):
    """轮询文件夹下的所有图片, 对每一个图片根据rect crop出相应的图片并保存.
    其中, rect只给定宽和高, 中心是图片中心, 所以可以算出对应每张图片其相应的rect是什么"""
    src_img_dir = file_utils.get_abs_dir(src_img_dir)
    dst_img_dir = file_utils.create_dir(dst_img_dir)

    for sub_dir in os.listdir(src_img_dir):
        abs_sub_dir = os.path.join(src_img_dir, sub_dir)
        if not os.path.isdir(abs_sub_dir):
            continue

        for file in os.listdir(abs_sub_dir):
            abs_file_path = os.path.join(abs_sub_dir, file)
            if not os.path.isfile(abs_file_path):
                continue
            if file.split(".")[-1].lower() not in ["jpg", "jpeg", "png", "bmp"]:
                continue

            img = cv2.imread(abs_file_path, 0)
            rect_x = int(img.shape[1] / 2 - rect_w / 2)
            rect_y = int(img.shape[0] / 2 - rect_h / 2)
            rect = [rect_x, rect_y, rect_w, rect_h]
            dst_img_path = os.path.join(dst_img_dir, dst_img_name_prefix + os.path.basename(abs_file_path))
            crop_img_by_rect(rect, abs_file_path, dst_img_path)

def convert_tif(src_tif_path, dst_tif_path):
    """转换tif, 原来的光镜图像全是tif文件, type是uint16的, 和电镜采集的图像是中心对称关系，所以需要把光镜图像全转换一下"""
    src_tif_path = file_utils.get_abs_file_path(src_tif_path)
    file_utils.create_save_dir_from_file(dst_tif_path)

    src_img = skimage.io.imread(src_tif_path) # 注意，读光镜tif图片用skimage.io.imread(), 不要用cv2.imread()
    dst_img = img_utils.img_pixel_value_invert(src_img) # 先像素值反转
    dst_img = img_utils.get_central_symmetry_img(dst_img) # 再中心对称
    dst_img = img_utils.convert_img_to_uint8(dst_img) # 再uint16转uint8
    cv2.imwrite(dst_tif_path, dst_img)

def loop_convert_tif(src_tif_dir, dst_tif_dir):
    """轮询转换tif, 并保存成png图片"""
    src_tif_dir = file_utils.get_abs_dir(src_tif_dir)
    dst_tif_dir = file_utils.create_dir(dst_tif_dir)

    for file in os.listdir(src_tif_dir):
        abs_file_path = os.path.join(src_tif_dir, file)
        if not os.path.isfile(abs_file_path):
            continue
        tmp_list = file.split(".")
        if tmp_list[-1].lower() != "tif":
            continue
        dst_tif_name = file.replace(".tif", ".png")
        dst_tif_path = os.path.join(dst_tif_dir, dst_tif_name)
        convert_tif(abs_file_path, dst_tif_path)

def draw_img_center(src_img_path, dst_img_path):
    src_img_path = file_utils.get_abs_file_path(src_img_path)
    file_utils.create_save_dir_from_file(dst_img_path)

    img = cv2.imread(src_img_path, 0)
    cv2.circle(img, (int(img.shape[1] / 2), int(img.shape[0] / 2)), 30, 255, 10)
    cv2.imwrite(dst_img_path, img)

def crop_optics_img_sec_to_mFovs(optics_img_path, full_image_coordinates_txt_file, save_img_dir, first_mFov_center):
    """将光镜图像一整张section图像切成多个mFov图像, 根据原始电镜图像中的坐标关系以及手动找到的光镜图像中第一个mFov中心位置, 来推出其他mFov位置并裁剪。"""
    optics_img_path = file_utils.get_abs_file_path(optics_img_path)
    full_image_coordinates_txt_file = file_utils.get_abs_file_path(full_image_coordinates_txt_file)
    save_img_dir = file_utils.create_dir(save_img_dir)

    # 根据原始的full_image_coordinates_txt_file获取每个mFov框位置
    mFov_idx = []
    start_x, start_y = [], []
    with open(full_image_coordinates_txt_file, "r") as f:
        lines = f.readlines()
        lines = sorted(lines)
        for line in lines:
            line_list = line.strip().split("\t")
            img_fname = line_list[0].replace("\\", "/")
            if not (img_fname.split("/")[0]).isdigit():
                continue

            mFov_idx.append(int(img_fname.split("/")[0]))
            start_x.append(float(line_list[1]))
            start_y.append(float(line_list[2]))
            
    start_x = np.array(start_x)
    start_y = np.array(start_y)
    start_x -= np.min(start_x)
    start_y -= np.min(start_y)

    em_idx_bbox_dict = {}
    cur_mFov_idx = mFov_idx[0]
    cur_mFov_bbox = []
    for idx, x, y in zip(mFov_idx, start_x, start_y):
        if idx == cur_mFov_idx:
            cur_mFov_bbox.append([int(math.floor(x)), int(math.ceil(x + 3876)), int(math.floor(y)), int(math.ceil(y + 3376))]) # left, right, top, down
        else:
            em_idx_bbox_dict[cur_mFov_idx] = cur_mFov_bbox
            cur_mFov_idx = idx
            cur_mFov_bbox = []
            cur_mFov_bbox.append([int(math.floor(x)), int(math.ceil(x + 3876)), int(math.floor(y)), int(math.ceil(y + 3376))])
    em_idx_bbox_dict[cur_mFov_idx] = cur_mFov_bbox

    for idx, bbox in em_idx_bbox_dict.items():
        bbox = np.array(bbox)
        bbox = [np.min(bbox[:, 0]), np.max(bbox[:, 1]), np.min(bbox[:, 2]), np.max(bbox[:, 3])]
        em_idx_bbox_dict[idx] = bbox

    # 拼接结果是降采样1倍，所以坐标位置除以2
    for idx, bbox in em_idx_bbox_dict.items():
        bbox[0] = int(math.floor(bbox[0] / 2))
        bbox[1] = int(math.ceil(bbox[1] / 2))
        bbox[2] = int(math.floor(bbox[2] / 2))
        bbox[3] = int(math.ceil(bbox[3] / 2))
        em_idx_bbox_dict[idx] = bbox

    # 根据原始的每个mFov框位置推断出光镜图像中每个mFov的位置
    # 这里原始mFov框的大小不完全等于拼接好的mFov图片大小，但是近似
    # 从拼接好的mFov图像中接近中心位置(暂时方案)裁了10800*10800区域大小(每个像素8nm)保存下来，对应光镜图就是250*250大小
    # 但是，光镜图裁剪区域稍微放大一点，1.5倍，即375*375大小
    # 那么可以根据光镜图第一个mFov中心位置，再根据原始的每个mFov框位置的对应关系可以大概判断出光镜中其余mFov位置
    om_idx_center_dict = {}
    for idx, bbox in em_idx_bbox_dict.items():
        if not om_idx_center_dict: # 光镜图像第一个mFov的中心是指定的first_mFov_center
            om_idx_center_dict[idx] = (int(first_mFov_center[0]), int(first_mFov_center[1]))
        else:
            pre_em_mFov_bbox = em_idx_bbox_dict[idx - 1]
            pre_em_mFov_center = (int((pre_em_mFov_bbox[0] + pre_em_mFov_bbox[1]) / 2), int((pre_em_mFov_bbox[2] + pre_em_mFov_bbox[3]) / 2))
            cur_em_mFov_bbox = em_idx_bbox_dict[idx]
            cur_em_mFov_center = (int((cur_em_mFov_bbox[0] + cur_em_mFov_bbox[1]) / 2), int((cur_em_mFov_bbox[2] + cur_em_mFov_bbox[3]) / 2))

            pre_om_mFov_center = om_idx_center_dict[idx - 1]
            cur_om_mFov_center_x = int((cur_em_mFov_center[0] - pre_em_mFov_center[0]) / 43.125 + pre_om_mFov_center[0])
            cur_om_mFov_center_y = int((cur_em_mFov_center[1] - pre_em_mFov_center[1]) / 43.125 + pre_om_mFov_center[1])
            om_idx_center_dict[idx] = (cur_om_mFov_center_x, cur_om_mFov_center_y)


    optics_img = cv2.imread(optics_img_path, cv2.IMREAD_UNCHANGED)
    om_idx_bbox_dict = {}
    crop_size = 375
    for idx, center in om_idx_center_dict.items():
        # 确保bbox不会超出图像边界
        bbox_x0 = max(0, center[0] - int(crop_size / 2))
        bbox_x1 = min(bbox_x0 + crop_size, optics_img.shape[1])
        bbox_y0 = max(0, center[1] - int(crop_size / 2))
        bbox_y1 = min(bbox_y0 + crop_size, optics_img.shape[0])
        # 确保bbox大小为375*375
        if bbox_x1 - bbox_x0 < crop_size:
            if bbox_x0 == 0:
                bbox_x1 = crop_size
            else:
                bbox_x0 = bbox_x1 - crop_size
        if bbox_y1 - bbox_y0 < crop_size:
            if bbox_y0 == 0:
                bbox_y1 = crop_size
            else:
                bbox_y0 = bbox_y1 - crop_size
        om_idx_bbox_dict[idx] = [bbox_x0, bbox_x1, bbox_y0, bbox_y1]

    # # 画框查看是否正确
    # for idx, bbox in om_idx_bbox_dict.items():
    #     cv2.rectangle(optics_img, (bbox[0], bbox[2]), (bbox[1], bbox[3]), 0, 2)
    #     if idx < 10:
    #         txt_center = (int((bbox[0] + bbox[1]) / 2) - 18, int((bbox[2] + bbox[3]) / 2) + 18)
    #     else:
    #         txt_center = (int((bbox[0] + bbox[1]) / 2) - 38, int((bbox[2] + bbox[3]) / 2) + 18)
    #     cv2.putText(optics_img, str(idx), txt_center, cv2.FONT_HERSHEY_SIMPLEX, 2, 0, 2)
    # cv2.imwrite("/media/hqjin/Elements/OEunion_data/Image-1-(2)_tmp.png", optics_img)

    # 原来的光镜图像名称像Image-1-0001.png，需要改成Image-1-S1M*.png
    optics_img_name = os.path.basename(optics_img_path) # optics_img_name like: Image-1-0001.png
    tmp_list = optics_img_name.split("-")
    sec_idx = int(tmp_list[-1].split(".")[0])
    crop_img_name_prefix = "-".join(tmp_list[:-1]) + "-S" + str(sec_idx)
    for idx, bbox in om_idx_bbox_dict.items():
        crop_img = optics_img[bbox[2]:bbox[3], bbox[0]:bbox[1]]
        crop_img_name = os.path.join(save_img_dir, crop_img_name_prefix + "M" + str(idx) + "." + optics_img_name.split(".")[-1])
        cv2.imwrite(crop_img_name, crop_img)

def crop_optics_img_sec_to_mFovs2(optics_img_path, full_image_coordinates_txt_file, save_img_dir, first_three_mFov_center):
    """将光镜图像一整张section图像切成多个mFov图像, 根据原始电镜图像中的坐标关系以及手动找到的光镜图像中前3个mFov中心位置, 来求出一个仿射变换矩阵, 其余的mFov中心可以根据该矩阵求出位置, 再裁剪。
       实践证明该方法不行，因为有的计算出来会超出图像边界。"""
    optics_img_path = file_utils.get_abs_file_path(optics_img_path)
    full_image_coordinates_txt_file = file_utils.get_abs_file_path(full_image_coordinates_txt_file)
    save_img_dir = file_utils.create_dir(save_img_dir)

    # 根据原始的full_image_coordinates_txt_file获取每个mFov框位置
    mFov_idx = []
    start_x, start_y = [], []
    with open(full_image_coordinates_txt_file, "r") as f:
        lines = f.readlines()
        lines = sorted(lines)
        for line in lines:
            line_list = line.strip().split("\t")
            img_fname = line_list[0].replace("\\", "/")
            if not (img_fname.split("/")[0]).isdigit():
                continue

            mFov_idx.append(int(img_fname.split("/")[0]))
            start_x.append(float(line_list[1]))
            start_y.append(float(line_list[2]))
            
    start_x = np.array(start_x)
    start_y = np.array(start_y)
    start_x -= np.min(start_x)
    start_y -= np.min(start_y)

    em_idx_bbox_dict = {}
    cur_mFov_idx = mFov_idx[0]
    cur_mFov_bbox = []
    for idx, x, y in zip(mFov_idx, start_x, start_y):
        if idx == cur_mFov_idx:
            cur_mFov_bbox.append([int(math.floor(x)), int(math.ceil(x + 3876)), int(math.floor(y)), int(math.ceil(y + 3376))]) # left, right, top, down
        else:
            em_idx_bbox_dict[cur_mFov_idx] = cur_mFov_bbox
            cur_mFov_idx = idx
            cur_mFov_bbox = []
            cur_mFov_bbox.append([int(math.floor(x)), int(math.ceil(x + 3876)), int(math.floor(y)), int(math.ceil(y + 3376))])
    em_idx_bbox_dict[cur_mFov_idx] = cur_mFov_bbox

    for idx, bbox in em_idx_bbox_dict.items():
        bbox = np.array(bbox)
        bbox = [np.min(bbox[:, 0]), np.max(bbox[:, 1]), np.min(bbox[:, 2]), np.max(bbox[:, 3])]
        em_idx_bbox_dict[idx] = bbox

    # 拼接结果是降采样1倍，所以坐标位置除以2
    for idx, bbox in em_idx_bbox_dict.items():
        bbox[0] = int(math.floor(bbox[0] / 2))
        bbox[1] = int(math.ceil(bbox[1] / 2))
        bbox[2] = int(math.floor(bbox[2] / 2))
        bbox[3] = int(math.ceil(bbox[3] / 2))
        em_idx_bbox_dict[idx] = bbox

    # 根据原始的每个mFov框位置推断出光镜图像中每个mFov的位置
    # 这里原始mFov框的大小不完全等于拼接好的mFov图片大小，但是近似
    # 从拼接好的mFov图像中接近中心位置(暂时方案)裁了10800*10800区域大小(每个像素8nm)保存下来，对应光镜图就是250*250大小
    # 但是，光镜图裁剪区域稍微放大一点，1.5倍，即375*375大小
    # 那么可以根据光镜图第一个mFov中心位置，再根据原始的每个mFov框位置的对应关系可以大概判断出光镜中其余mFov位置
    if 1 not in em_idx_bbox_dict.keys() or 2 not in em_idx_bbox_dict.keys():
        raise Exception("Error! 1 or 2 not in em_idx_bbox_dict.keys().")

    em_mFov1_center_x = (em_idx_bbox_dict[1][0] + em_idx_bbox_dict[1][1]) / 2
    em_mFov1_center_y = (em_idx_bbox_dict[1][2] + em_idx_bbox_dict[1][3]) / 2
    em_mFov2_center_x = (em_idx_bbox_dict[2][0] + em_idx_bbox_dict[2][1]) / 2
    em_mFov2_center_y = (em_idx_bbox_dict[2][2] + em_idx_bbox_dict[2][3]) / 2
    em_mFov3_center_x = (em_idx_bbox_dict[3][0] + em_idx_bbox_dict[3][1]) / 2
    em_mFov3_center_y = (em_idx_bbox_dict[3][2] + em_idx_bbox_dict[3][3]) / 2

    src_pts = np.float32([[em_mFov1_center_x, em_mFov1_center_y], [em_mFov2_center_x, em_mFov2_center_y], [em_mFov3_center_x, em_mFov3_center_y]])
    dst_pts = np.float32(first_three_mFov_center)
    matrix = cv2.getAffineTransform(src_pts, dst_pts)

    om_idx_center_dict = {}
    for idx, bbox in em_idx_bbox_dict.items():
        if idx == 1:
            om_idx_center_dict[idx] = [int(first_three_mFov_center[0][0]), int(first_three_mFov_center[0][1])]
        elif idx == 2:
            om_idx_center_dict[idx] = [int(first_three_mFov_center[1][0]), int(first_three_mFov_center[1][1])]
        elif idx == 3:
            om_idx_center_dict[idx] = [int(first_three_mFov_center[2][0]), int(first_three_mFov_center[2][1])]
        else:
            cur_em_mFov_center_x = (em_idx_bbox_dict[idx][0] + em_idx_bbox_dict[idx][1]) / 2
            cur_em_mFov_center_y = (em_idx_bbox_dict[idx][2] + em_idx_bbox_dict[idx][3]) / 2
            cur_om_mFov_center_x = int(matrix[0][0] * cur_em_mFov_center_x + matrix[0][1] * cur_em_mFov_center_y + matrix[0][2])
            cur_om_mFov_center_y = int(matrix[1][0] * cur_em_mFov_center_x + matrix[1][1] * cur_em_mFov_center_y + matrix[1][2])
            om_idx_center_dict[idx] = [cur_om_mFov_center_x, cur_om_mFov_center_y]

    optics_img = cv2.imread(optics_img_path, cv2.IMREAD_UNCHANGED)
    om_idx_bbox_dict = {}
    crop_size = 375
    for idx, center in om_idx_center_dict.items():
        # 确保bbox不会超出图像边界
        bbox_x0 = max(0, center[0] - int(crop_size / 2))
        bbox_x1 = min(bbox_x0 + crop_size, optics_img.shape[1])
        bbox_y0 = max(0, center[1] - int(crop_size / 2))
        bbox_y1 = min(bbox_y0 + crop_size, optics_img.shape[0])
        # 确保bbox大小为375*375
        if bbox_x1 - bbox_x0 < crop_size:
            if bbox_x0 == 0:
                bbox_x1 = crop_size
            else:
                bbox_x0 = bbox_x1 - crop_size
        if bbox_y1 - bbox_y0 < crop_size:
            if bbox_y0 == 0:
                bbox_y1 = crop_size
            else:
                bbox_y0 = bbox_y1 - crop_size
        om_idx_bbox_dict[idx] = [bbox_x0, bbox_x1, bbox_y0, bbox_y1]

    # # 画框查看是否正确
    # for idx, bbox in om_idx_bbox_dict.items():
    #     cv2.rectangle(optics_img, (bbox[0], bbox[2]), (bbox[1], bbox[3]), 0, 2)
    #     if idx < 10:
    #         txt_center = (int((bbox[0] + bbox[1]) / 2) - 18, int((bbox[2] + bbox[3]) / 2) + 18)
    #     else:
    #         txt_center = (int((bbox[0] + bbox[1]) / 2) - 38, int((bbox[2] + bbox[3]) / 2) + 18)
    #     cv2.putText(optics_img, str(idx), txt_center, cv2.FONT_HERSHEY_SIMPLEX, 2, 0, 2)
    # cv2.imwrite("/media/hqjin/Elements/OEunion_data/Image-1-(2)_tmp.png", optics_img)

    # 原来的光镜图像名称像Image-1-0001.png，需要改成Image-1-S1M*.png
    optics_img_name = os.path.basename(optics_img_path) # optics_img_name like: Image-1-0001.png
    tmp_list = optics_img_name.split("-")
    sec_idx = int(tmp_list[-1].split(".")[0])
    crop_img_name_prefix = "-".join(tmp_list[:-1]) + "-S" + str(sec_idx)
    for idx, bbox in om_idx_bbox_dict.items():
        crop_img = optics_img[bbox[2]:bbox[3], bbox[0]:bbox[1]]
        crop_img_name = os.path.join(save_img_dir, crop_img_name_prefix + "M" + str(idx) + "." + optics_img_name.split(".")[-1])
        cv2.imwrite(crop_img_name, crop_img)

def crop_optics_img_sec(em_img_path, om_img_path, em_resolution, em_zip_scale, om_resolution, om_img_center, om_img_save_dir):
    """这是根据电镜图拼成的一整个section图片, 根据中心以及图片大小, 通过手动在光镜图中寻找中心，并算出对应光镜图中的范围"""
    om_img_save_dir = file_utils.create_dir(om_img_save_dir)
    em_img = cv2.imread(em_img_path, 0)
    em_h, em_w = em_img.shape
    em_real_h, em_real_w = round(em_h / em_zip_scale), round(em_w / em_zip_scale)

    em_block_size = 5120
    # # 画电镜图像网格图，电镜图像每一小块大小是5120*5120
    # block_size = int(em_block_size * em_zip_scale)
    # row_block_num, col_block_num = int(em_h / block_size), int(em_w / block_size)
    # for i in range(1, row_block_num + 1):
    #     cv2.line(em_img, (0, i * block_size), (em_w, i * block_size), 0, 2)
    # for i in range(1, col_block_num + 1):
    #     cv2.line(em_img, (i * block_size, 0), (i * block_size, em_h), 0, 2)
    # em_block_img_path = os.path.join(om_img_save_dir, os.path.basename(em_img_path))
    # cv2.imwrite(em_block_img_path, em_img)

    cor_om_h = round(em_real_h * em_resolution / om_resolution)
    cor_om_w = round(em_real_w * em_resolution / om_resolution)
    left = int(om_img_center[0] - cor_om_w / 2)
    top = int(om_img_center[1] - cor_om_h / 2)
    cor_om_bbox = [left, left + cor_om_w, top, top + cor_om_h]  # left, right, top, down

    om_img = cv2.imread(om_img_path, 0)
    om_img_save_img = om_img[cor_om_bbox[2]:cor_om_bbox[3], cor_om_bbox[0]:cor_om_bbox[1]]
    # om_img_save_path = os.path.join(om_img_save_dir, os.path.basename(om_img_path))
    # cv2.imwrite(om_img_save_path, om_img_save_img)  # 将光镜图像对应区域扣出保存

    block_size = int(em_block_size * em_resolution / om_resolution)
    row_block_num, col_block_num = int(om_img_save_img.shape[0] / block_size), int(om_img_save_img.shape[1] / block_size)

    # # 画光镜图像网格图
    # for i in range(1, row_block_num + 1):
    #     cv2.line(om_img_save_img, (0, i * block_size), (om_img_save_img.shape[1], i * block_size), 0, 2)
    # for i in range(1, col_block_num + 1):
    #     cv2.line(om_img_save_img, (i * block_size, 0), (i * block_size, om_img_save_img.shape[0]), 0, 2)
    # om_block_img_name_list = os.path.basename(om_img_path).rsplit(".", 1)
    # om_block_img_name = om_block_img_name_list[0] + "_grid." + om_block_img_name_list[1]
    # om_block_img_path = os.path.join(om_img_save_dir, om_block_img_name)
    # cv2.imwrite(om_block_img_path, om_img_save_img)

    om_img_name_list = os.path.basename(om_img_path).rsplit(".", 1)
    tmp_om_img_idx_str = om_img_name_list[0].rsplit("-", 1)[1]
    tmp_om_img_dir = os.path.join(om_img_save_dir, tmp_om_img_idx_str)
    tmp_om_img_dir = file_utils.create_dir(tmp_om_img_dir)
    for i in range(0, row_block_num + 1):
        for j in range(0, col_block_num + 1):
            left_x = j * block_size
            right_x = min((j + 1) * block_size, om_img_save_img.shape[1])
            top_y = i * block_size
            down_y = min((i + 1) * block_size, om_img_save_img.shape[0])
            if left_x == right_x or top_y == down_y:
                continue
            tmp_om_img = om_img_save_img[top_y:down_y, left_x:right_x]
            tmp_om_img_name = om_img_name_list[0] + "_tr" + str(i + 1) + "-tc" + str(j + 1) + "." + om_img_name_list[1]
            tmp_om_img_path = os.path.join(tmp_om_img_dir, tmp_om_img_name)
            cv2.imwrite(tmp_om_img_path, tmp_om_img)

def crop_optics_img_sec2(em_img_path, om_img_path, em_resolution, em_zip_scale, om_resolution, om_img_center, om_img_save_dir):
    """这是根据电镜图拼成的一整个section图片, 根据中心以及图片大小, 通过手动在光镜图中寻找中心，并算出对应光镜图中的范围,
    电镜中图像假如最后一行, 最后一列不是5120倍数, 会丢掉；
    找到对应的光镜图像是以当前中心为中心, 3倍实际长宽范围保存, 超出范围pad 0"""
    om_img_save_dir = file_utils.create_dir(om_img_save_dir)
    em_img = cv2.imread(em_img_path, 0)
    em_h, em_w = em_img.shape
    em_real_h, em_real_w = round(em_h / em_zip_scale), round(em_w / em_zip_scale)

    em_block_size = 5120
    row_block_num, col_block_num = int(em_real_h / em_block_size), int(em_real_w / em_block_size)

    # 创建对应光镜crop文件夹
    om_img_name_list = os.path.basename(om_img_path).rsplit(".", 1)
    tmp_om_img_idx_str = om_img_name_list[0].rsplit("-", 1)[1]
    tmp_om_img_dir = os.path.join(om_img_save_dir, tmp_om_img_idx_str)
    tmp_om_img_dir = file_utils.create_dir(tmp_om_img_dir)

    om_img = cv2.imread(om_img_path, 0)

    for i in range(0, row_block_num):
        for j in range(0, col_block_num):
            cur_em_block_center_x = int(j * em_block_size + em_block_size / 2)
            cur_em_block_center_y = int(i * em_block_size + em_block_size / 2)
            cur_em_dis_x = cur_em_block_center_x - em_real_w / 2
            cur_em_dis_y = cur_em_block_center_y - em_real_h / 2

            cur_om_dis_x = cur_em_dis_x * em_resolution / om_resolution
            cur_om_dis_y = cur_em_dis_y * em_resolution / om_resolution
            cur_om_block_center_x = int(cur_om_dis_x + om_img_center[0])
            cur_om_block_center_y = int(cur_om_dis_y + om_img_center[1])
            
            om_block_size = int(em_block_size * em_resolution / om_resolution)
            cur_om_block_left_x = int(cur_om_block_center_x - om_block_size * 3 / 2)  # 裁的图像是实际的3倍大小
            cur_om_block_right_x = cur_om_block_left_x + om_block_size * 3
            cur_om_block_top_y = int(cur_om_block_center_y - om_block_size * 3 / 2)
            cur_om_block_down_y = cur_om_block_top_y + om_block_size * 3
            
            if cur_om_block_left_x >= 0 and cur_om_block_right_x <= om_img.shape[1] and cur_om_block_top_y >= 0 and cur_om_block_down_y <= om_img.shape[0]:
                # 不超出边界的情况
                tmp_om_img = om_img[cur_om_block_top_y:cur_om_block_down_y, cur_om_block_left_x:cur_om_block_right_x]
            else:
                # 超出边界的情况，先将om_img扩大，然后再从中间取
                big_om_img = np.zeros((om_img.shape[0] + om_block_size * 6, om_img.shape[1] + om_block_size * 6), np.uint8)
                big_om_img[om_block_size * 3:om_block_size * 3 + om_img.shape[0], om_block_size * 3:om_block_size * 3 + om_img.shape[1]] = om_img
                cur_om_block_left_x += om_block_size * 3
                cur_om_block_right_x += om_block_size * 3
                cur_om_block_top_y += om_block_size * 3
                cur_om_block_down_y += om_block_size * 3
                tmp_om_img = big_om_img[cur_om_block_top_y:cur_om_block_down_y, cur_om_block_left_x:cur_om_block_right_x]

            tmp_om_img_name = om_img_name_list[0] + "_tr" + str(i + 1) + "-tc" + str(j + 1) + "." + om_img_name_list[1]
            tmp_om_img_path = os.path.join(tmp_om_img_dir, tmp_om_img_name)
            cv2.imwrite(tmp_om_img_path, tmp_om_img)

def manual_check_electronic_optical_imgs(electronic_img_dir, optical_img_dir):
    electronic_img_dir = file_utils.get_abs_dir(electronic_img_dir)
    optical_img_dir = file_utils.get_abs_dir(optical_img_dir)
    files = os.listdir(electronic_img_dir)
    files = sorted(files)
    # optical_img_names = os.listdir(optical_img_dir)

    # 根据电镜图像名字找光镜图像名字
    # 电镜图像名字，例如：sample1_S2M1_tr1-tc1.png
    # 光镜图像名字，例如：Image-1-S2M1.png
    # cv2.namedWindow("电镜 VS 光镜", cv2.WINDOW_NORMAL)
    for file in files:
        #======== 有些数据从服务器传到本地出问题 =======#
        if os.path.exists(os.path.join(os.path.dirname(electronic_img_dir), "test2", file)):
            continue
        #===============#
        abs_file_path = os.path.join(electronic_img_dir, file)
        if not os.path.isfile(abs_file_path):
            continue
        if file.split(".")[-1].lower() not in ["jpg", "jpeg", "png", "bmp"]:
            continue

        img_name_prefix, img_name_suffix = file.rsplit(".", 1)
        tmp_list = img_name_prefix.split("_")
        if len(tmp_list) < 3 or "sample" not in tmp_list[0] or "S" not in tmp_list[1] or "M" not in tmp_list[1]:
            raise Exception("Error! Nonstandard name {}, it should be like sample1_S2M1_tr1-tc1.png".format(file))

        sample_idx = int(tmp_list[0].split("sample")[-1])
        sec_mFov_info = tmp_list[1]

        corres_optical_img_name = "Image-" + str(sample_idx) + "-" + sec_mFov_info + ".png"
        abs_optical_img_path = os.path.join(optical_img_dir, corres_optical_img_name)
        if not os.path.exists(abs_optical_img_path):
            print("Error! Current electronic img is {}, but cannot find corresponding optical img {}".format(file, corres_optical_img_name))
            continue

        # 读取两张图片并显示
        electronic_img = cv2.imread(abs_file_path, 0)
        electronic_img = cv2.resize(electronic_img, (0, 0), fx=0.25, fy=0.25) # 电镜图太大，resize到1/2

        optical_img = cv2.imread(abs_optical_img_path, 0)
        optical_img = cv2.resize(optical_img, (0, 0), fx=5, fy=5)

        total_img_h = max(electronic_img.shape[0], optical_img.shape[0])
        total_img_w = electronic_img.shape[1] + optical_img.shape[1] + 20
        total_img = np.ones((total_img_h, total_img_w), np.uint8) * 255
        if electronic_img.shape[0] >= optical_img.shape[0]:
            total_img[:, :electronic_img.shape[1]] = electronic_img
            optical_img_put_y = int(electronic_img.shape[0] / 2 - optical_img.shape[0] / 2)
            total_img[optical_img_put_y:optical_img_put_y + optical_img.shape[0], total_img_w - optical_img.shape[1]:total_img_w] = optical_img
        else:
            electronic_img_put_y = int(optical_img.shape[0] / 2 - electronic_img.shape[0] / 2)
            total_img[electronic_img_put_y:electronic_img_put_y + electronic_img.shape[0], :electronic_img.shape[1]] = electronic_img
            total_img[:,total_img_w - optical_img.shape[1]:total_img_w] = optical_img
        # cv2.imshow("电镜 VS 光镜", total_img)
        print("Current electronic img is: {}, optical img is: {}".format(file, corres_optical_img_name))
        cv2.imwrite(os.path.join(os.path.dirname(electronic_img_dir), "test2", file), total_img)
    #     key = cv2.waitKey()
    #     if key == 119: # 对应"w"键，表示当前组合有误，记录下有误的这一组信息
    #         print("Error! Current electronic img {} and optical img {} are unqualified or not matched.".format(file, corres_optical_img_name))
    #         shutil.move(abs_file_path, "/media/hqjin/Elements/OEunion_data/unqualified_data/em_data")
    #         shutil.move(abs_optical_img_path, "/media/hqjin/Elements/OEunion_data/unqualified_data/om_data")
    #     elif key == 27: # 对应"Esc"键，表示退出
    #         break
    #     else: # 其他的key均自动调到下一张
    #         pass

    # cv2.destroyAllWindows()

def get_all_block_img_name_row_col_range(mdir):
    """参数文件夹存放的必须是一张大图或全是block图片, 将大图或所有block图片名称保存在列表中, 将列表从小到大排序；
       大图或每个block图片的名称都要遵循S*M*_tr*-tc*.png命名标准, 第一个*表示section信息, 第二个*表示mFov信息, 第三个*表示行索引，第四个*表示列索引;
       根据图片名获取行列范围，返回图片名列表及行列范围"""
    mdir = file_utils.get_abs_dir(mdir)
    
    all_img_names = []
    min_row, min_col = np.iinfo(np.int32).max, np.iinfo(np.int32).max
    max_row, max_col = 0, 0
    for file_name in os.listdir(mdir):  # file_name如S2M1-10_tr1-tc2.png，表示section 2的第1到第10个mFov图像的第1行第2列图像
        abs_file_name = os.path.join(mdir, file_name)
        if not os.path.isfile(abs_file_name):
            raise Exception("Error! {} is not a file!".format(abs_file_name))

        if "S" not in file_name or "M" not in file_name or "tr" not in file_name or "tc" not in file_name:
            raise Exception("Error! {} may be not a block img! Block img file name should be like: S*M*_tr*-tc*.png".format(abs_file_name))

        tmp_split_list = file_name.split("tr")
        try:
            row_idx = int(tmp_split_list[1].split("-")[0])
            col_idx = int(tmp_split_list[1].split("tc")[1].split(".")[0])
        except Exception as err:
            raise Exception("Error! Unknown block img file -> {}. Block img file name should be like: S*M*_tr1-tc2.png".format(abs_file_name))

        all_img_names.append(abs_file_name)

        if row_idx < min_row:
            min_row = row_idx

        if col_idx < min_col:
            min_col = col_idx

        if row_idx > max_row:
            max_row = row_idx

        if col_idx > max_col:
            max_col = col_idx

    all_img_names = sorted(all_img_names)
    row_col_dict = {"min_row": min_row, "max_row": max_row, "min_col": min_col, "max_col": max_col}
    return all_img_names, row_col_dict

def check_block_imgs_integral(all_img_names, row_col_dict):
    theoretical_num = (row_col_dict["max_row"] - row_col_dict["min_row"] + 1) * (row_col_dict["max_col"] - row_col_dict["min_col"] + 1)
    actual_num = len(all_img_names)
    if theoretical_num != actual_num:
        raise Exception("Error!! Theoretical block images num is {}, but actual block images num is {}.".format(theoretical_num, actual_num))

def merge_block_imgs(all_img_names, row_col_dict, scale, merge_img_path):
    file_utils.create_save_dir_from_file(merge_img_path)

    img_prefix = all_img_names[0].split("_tr")[0]
    img_type = all_img_names[0].rsplit(".", 1)[1]

    # 获取第一行第一列图片的shape
    row1_col1_img_path = img_prefix + "_tr" + str(row_col_dict["min_row"]) + "-tc" + str(row_col_dict["min_col"]) + "." + img_type
    row1_col1_img = cv2.imread(row1_col1_img_path, 0)
    row1_col1_img_h, row1_col1_img_w = row1_col1_img.shape
    row1_col1_img = None

    # 获取最后一行最后一列图片的shape
    last_img_path = img_prefix + "_tr" + str(row_col_dict["max_row"]) + "-tc" + str(row_col_dict["max_col"]) + "." + img_type
    last_img = cv2.imread(last_img_path, 0)
    last_img_h, last_img_w = last_img.shape
    last_img = None

    # 计算合并大图的size，因为大图是有小图resize相应scale得到，防止小数影响总的大小，所以应该这样计算
    merge_img_h = (row_col_dict["max_row"] - row_col_dict["min_row"]) * round(row1_col1_img_h * scale) + round(last_img_h * scale)
    merge_img_w = (row_col_dict["max_col"] - row_col_dict["min_col"]) * round(row1_col1_img_w * scale) + round(last_img_w * scale)
    merge_img = np.zeros((merge_img_h, merge_img_w), np.uint8)

    # 按行列依次读取图片，并放置大图中
    cur_pos_x, cur_pos_y = 0, 0
    for i in range(row_col_dict["min_row"], row_col_dict["max_row"] + 1):
        for j in range(row_col_dict["min_col"], row_col_dict["max_col"] + 1):
            cur_img_path = img_prefix + "_tr" + str(i) + "-tc" + str(j) + "." + img_type
            if not os.path.exists(cur_img_path):
                raise Exception("Error! {} is not exist!".format(cur_img_path))

            cur_img = cv2.imread(cur_img_path, 0)
            cur_img = cv2.resize(cur_img, (0, 0), fx=scale, fy=scale)
            merge_img[cur_pos_y:cur_pos_y + cur_img.shape[0], cur_pos_x:cur_pos_x + cur_img.shape[1]] = cur_img
            cur_pos_x += cur_img.shape[1]
        cur_pos_x = 0
        cur_pos_y += cur_img.shape[0]
    cv2.imwrite(merge_img_path, merge_img)

def loop_merge_block_imgs(block_imgs_dir, merge_imgs_dir, scale=0.05, process_num=1):
    block_imgs_dir = file_utils.get_abs_dir(block_imgs_dir)
    merge_imgs_dir = file_utils.create_dir(merge_imgs_dir)

    pool = Pool(processes=process_num)
    res_l = []
    for sub_dir in os.listdir(block_imgs_dir):  # sub_dir like: 001
        abs_sub_dir = os.path.join(block_imgs_dir, sub_dir)
        if not os.path.isdir(abs_sub_dir):
            continue

        all_img_names, row_col_dict = get_all_block_img_name_row_col_range(abs_sub_dir)
        if not all_img_names:
            continue

        check_block_imgs_integral(all_img_names, row_col_dict)
        merge_img_name = os.path.basename(all_img_names[0]).split("_tr")[0] + "." + os.path.basename(all_img_names[0]).rsplit(".", 1)[1]
        merge_img_name = os.path.join(merge_imgs_dir, merge_img_name)
        res = pool.apply_async(merge_block_imgs, (all_img_names, row_col_dict, scale, merge_img_name))
        res_l.append(res)

    pool.close()
    pool.join()

    for i in res_l:
        res = i.get()
    print("Done!")

def convert_src_section_imgs(src_dir, dst_dir):
    """原始采集的电镜图像, 有可能会出现当前section和其他section图片看起来是中心对称的关系, 需要把当前section转换一下: tile图像中心对称, 坐标也要进程处理"""
    src_dir = file_utils.get_abs_dir(src_dir)
    dst_dir = file_utils.create_dir(dst_dir)

    src_coord_txt_path = os.path.join(src_dir, "full_image_coordinates.txt")
    src_coord_txt_path = file_utils.get_abs_file_path(src_coord_txt_path)
    dst_coord_txt_path = os.path.join(dst_dir, "full_image_coordinates.txt")

    src_thumb_coord_txt_path = os.path.join(src_dir, "full_thumbnail_coordinates.txt")
    src_thumb_coord_txt_path = file_utils.get_abs_file_path(src_thumb_coord_txt_path)
    dst_thumb_coord_txt_path = os.path.join(dst_dir, "full_thumbnail_coordinates.txt")

    # 先将所有图片做中心对称后存储下来
    for sub_dir in os.listdir(src_dir):  # sub_dir like: 001
        abs_sub_dir = os.path.join(src_dir, sub_dir)
        if not os.path.isdir(abs_sub_dir) or not sub_dir.isdigit():
            continue

        abs_dst_sub_dir = os.path.join(dst_dir, sub_dir)
        abs_dst_sub_dir = file_utils.create_dir(abs_dst_sub_dir)

        for file_name in os.listdir(abs_sub_dir):
            abs_file_name = os.path.join(abs_sub_dir, file_name)
            suffix = file_name.rsplit(".", 1)[1]
            if suffix != "bmp":
                continue

            img = cv2.imread(abs_file_name, 0)
            dst_img = img_utils.get_central_symmetry_img(img)
            cv2.imwrite(os.path.join(abs_dst_sub_dir, file_name), dst_img)

    # 转换坐标
    img_paths, img_start_x, img_start_y, img_label = [], [], [], []
    with open(src_coord_txt_path, "r") as f:
        lines = f.readlines()
        for line in lines:
            line_list = line.strip().split("\t")
            img_paths.append(line_list[0])
            img_start_x.append(float(line_list[1]))
            img_start_y.append(float(line_list[2]))
            img_label.append(line_list[3])
            
    img_start_x_min, img_start_x_max = np.array(img_start_x).min(), np.array(img_start_x).max()
    img_start_y_min, img_start_y_max = np.array(img_start_y).min(), np.array(img_start_y).max()

    with open(dst_coord_txt_path, "w") as f:
        for i in range(len(img_paths)):
            cur_start_x = img_start_x_max - img_start_x[i] + img_start_x_min
            cur_start_y = img_start_y_max - img_start_y[i] + img_start_y_min
            f.writelines(img_paths[i] + "\t" + str(cur_start_x) + "\t" + str(cur_start_y) + "\t" + img_label[i] + "\r\n")  # Windows下面是以"\r\n"换行，linux是以"\n"换行，但是后面解析的代码是以Windows下为标准的，所以改成Windows下格式
    
    # 转换缩略图坐标
    thumb_img_paths, thumb_img_start_x, thumb_img_start_y, thumb_img_label = [], [], [], []
    with open(src_thumb_coord_txt_path, "r") as f:
        lines = f.readlines()
        for line in lines:
            line_list = line.strip().split("\t")
            thumb_img_paths.append(line_list[0])
            thumb_img_start_x.append(float(line_list[1]))
            thumb_img_start_y.append(float(line_list[2]))
            thumb_img_label.append(line_list[3])
            
    thumb_img_start_x_min, thumb_img_start_x_max = np.array(thumb_img_start_x).min(), np.array(thumb_img_start_x).max()
    thumb_img_start_y_min, thumb_img_start_y_max = np.array(thumb_img_start_y).min(), np.array(thumb_img_start_y).max()

    with open(dst_thumb_coord_txt_path, "w") as f:
        for i in range(len(thumb_img_paths)):
            cur_thumb_start_x = thumb_img_start_x_max - thumb_img_start_x[i] + thumb_img_start_x_min
            cur_thumb_start_y = thumb_img_start_y_max - thumb_img_start_y[i] + thumb_img_start_y_min
            f.writelines(thumb_img_paths[i] + "\t" + str(cur_thumb_start_x) + "\t" + str(cur_thumb_start_y) + "\t" + thumb_img_label[i] + "\r\n")  # Windows下面是以"\r\n"换行，linux是以"\n"换行，但是后面解析的代码是以Windows下为标准的，所以改成Windows下格式

def main():
    parser = argparse.ArgumentParser(description="This is a auxiliary tool.")
    parser.add_argument("-i", "--in_dir", type=str, default="/media/hqjin/Elements/em_data", help="source dir")
    parser.add_argument("-o", "--out_dir", type=str, default="/media/hqjin/Elements/em_data", help="output dir")
    parser.add_argument("-p", "--process_num", type=int, default=16, help="the number of processes to use(default: 16)")
    args = parser.parse_args()

    # invert_imgs("/media/hqjin/Elements/em_data/Experiment_20220508_18-33-18/001_S1R1/000001", 
    #             "/media/hqjin/Elements/em_data/Experiment_20220508_18-33-18/out/stitch/tmp/001_S1R1/000001")

    # get_tile_layout_by_json("/media/hqjin/Elements/em_data/Experiment_20220508_18-33-18/out/stitch/json/ECS_test9_cropped_010_S001R1-1.json", 
    #                 "/media/hqjin/Elements/em_data/Experiment_20220508_18-33-18/out/stitch/tmp/tile_layout.png")

    # get_mFov_layout_by_json("/media/hqjin/Elements/em_data/Experiment_20220508_18-33-18/out/stitch/json/ECS_test9_cropped_010_S001R1-1.json", 
    #                 "/media/hqjin/Elements/em_data/Experiment_20220508_18-33-18/out/stitch/tmp/mFov_layout.png")

    # get_mFov_layout_by_txt("/home/hqjin/tmp/full_image_coordinates.txt", "/home/hqjin/tmp/full_image_coordinates.png")

    # ret = check_rect_in_img_max_contour("/home/hqjin/tmp/imgs/002/S2M1_tr1-tc1.png", [3100, 2000, 10800, 10800])
    # print("ret: ", ret)

    # loop_check_rect_in_img_max_contour("/home/hqjin/tmp/imgs", 10800, 10800)

    # crop_img_by_rect([3100, 2000, 10800, 10800], "/home/hqjin/tmp/imgs/002/S2M8_tr1-tc1.png", "/home/hqjin/tmp/imgs/002/S2M8_tr1-tc1_tmp.png")

    # loop_crop_img_by_rect(10800, 10800, "/home/hqjin/tmp/imgs/", "/home/hqjin/tmp/crop_imgs", dst_img_name_prefix="sample1_")

    # convert_tif("/home/hqjin/tmp/Image.tif", "/home/hqjin/tmp/Image2.png")

    # loop_convert_tif("/media/hqjin/Elements/OEunion_data/om_data_wafer12_1", "/media/hqjin/Elements/OEunion_data/om_data_wafer12_2")

    # draw_img_center("/home/hqjin/tmp/imgs/002/S2M1_tr1-tc1.png", "/home/hqjin/tmp/imgs/002/S2M1_tr1-tc1_tmp.png")

    # crop_optics_img_sec_to_mFovs("/media/hqjin/Elements/OEunion_data/om_data/Image-1-(2).png", 
    # "/home/hqjin/tmp/full_image_coordinates.txt", "/media/hqjin/Elements/OEunion_data/om_data_crop", (640, 1246))

    # manual_check_electronic_optical_imgs("/media/hqjin/Elements/OEunion_data/em_data/crop_imgs", "/media/hqjin/Elements/OEunion_data/om_data_crop")

    # all_img_names, row_col_dict = get_all_block_img_name_row_col_range("/media/hqjin/Elements/OEunion_data/em_data/sample1/out/stitch/imgs/009")
    # check_block_imgs_integral(all_img_names, row_col_dict)
    # merge_block_imgs(all_img_names, row_col_dict, 0.05, "/media/hqjin/Elements/OEunion_data/em_data/sample1/out/stitch/thumbnail_imgs/S9M1-32.png")

    loop_merge_block_imgs("/media/hqjin/Elements/OEunion_data/em_data/sample1/out/stitch/img_8nm", "/media/hqjin/Elements/OEunion_data/em_data/sample1/out/stitch/merge_imgs_32nm", scale=0.05, process_num=args.process_num)

    # crop_optics_img_sec("/media/hqjin/Elements/OEunion_data/em_data/sample1/out/stitch/thumbnail_imgs_draw/S2M1-33.png", 
    # "/media/hqjin/Elements/OEunion_data/om_data/Image-1-002.png", 8, 0.05, 345, (1618, 1764), "/media/hqjin/Elements/OEunion_data/om_data_sec_crop")

    # crop_optics_img_sec2("/media/hqjin/Elements/OEunion_data/em_data/sample1/out/align/merge_img_128nm_draw/S2M1-33.png", 
    # "/media/hqjin/Elements/OEunion_data/om_data_align/Image-1-002.png", 8, 0.0625, 345, (2876, 2470), "/media/hqjin/Elements/OEunion_data/om_data_align_sec_crop")

    # convert_src_section_imgs("/media/hqjin/Elements/OEunion_data/em_data/sample1/126_S127R1", "/media/hqjin/Elements/OEunion_data/em_data/sample1/convert_test/126_S127R1")

if __name__ == '__main__':
    main()
    