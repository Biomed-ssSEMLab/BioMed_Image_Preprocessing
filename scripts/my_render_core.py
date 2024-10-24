from rh_renderer.tilespec_renderer import TilespecRenderer
from rh_renderer.multiple_tiles_renderer import BlendType
from rh_renderer import models
# from rh_renderer.hist_matcher import HistMatcher
# import rh_renderer.normalization.hist_adjuster
# from rh_aligner.common.bounding_box import BoundingBox
import cv2
import argparse
import numpy as np
import time
import ujson
import os
import sys
import math
# from rh_renderer.normalization.histogram_diff_minimization import HistogramDiffMinimization
from rh_renderer.normalization.histogram_clahe import HistogramCLAHE, HistogramGB11CLAHE
import common
import rh_img_access_layer
from mb_aligner.stitching.stitcher import Stitcher
from mb_aligner.dal.section import Section

import xlwt
#设置表格样式
def set_style(name,height,bold=False):
    style = xlwt.XFStyle()
    font = xlwt.Font()
    font.name = name
    font.bold = bold
    font.color_index = 4
    font.height = height
    style.font = font
    return style

def pad_image(img, from_x, from_y, start_point):
    """Pads the image (zeros) that starts from start_point (returned from the renderer), to (from_x, from_y)"""
    # Note that start_point is (y, x)
    if start_point[0] == from_y and start_point[1] == from_x:
        # Nothing to pad, return the image as is
        return img

    full_height_width = (img.shape + np.array([start_point[1] - from_y, start_point[0] - from_x])).astype(int)
    full_img = np.zeros(full_height_width, dtype=img.dtype)
    full_img[int(start_point[1] - from_y):int(start_point[1] - from_y + img.shape[0]),
    int(start_point[0] - from_x):int(start_point[0] - from_x + img.shape[1])] = img
    return full_img

# def render_tilespec(tile_fname, output, scale, output_type, in_bbox, tile_size, invert_image, threads_num=1, empty_placeholder=False, hist_adjuster_fname=None, hist_adjuster_alg_type=None, from_to_cols_rows=None):
def render_tilespec(tile_fname, works_dir, scale, in_bbox, threads_num=1, hist_adjuster_alg_type=None, 
                    from_to_cols_rows=None, blend_type=BlendType.NO_BLENDING):
                    
    blend_type=BlendType.NO_BLENDING
    
    # Determine the output shape
    if in_bbox[1] == -1 or in_bbox[3] == -1:
        image_bbox = common.read_bboxes_grep(tile_fname)
        image_bbox[0] = max(image_bbox[0], in_bbox[0])
        image_bbox[2] = max(image_bbox[2], in_bbox[2])
        if in_bbox[1] > 0:
            image_bbox[1] = in_bbox[1]
        if in_bbox[3] > 0:
            image_bbox[3] = in_bbox[3]
    else:
        image_bbox = in_bbox

    scaled_bbox = [
        int(math.floor(image_bbox[0] * scale)),
        int(math.ceil(image_bbox[1] * scale)),
        int(math.floor(image_bbox[2] * scale)),
        int(math.ceil(image_bbox[3] * scale))
    ]

    # hist_adjuster = HistogramCLAHE()
    hist_adjuster = None

    # with rh_img_access_layer.FSAccess(tile_fname, False) as data:
    with open(tile_fname, 'r') as data:
        tilespec = ujson.load(data)

    renderer = TilespecRenderer(tilespec, hist_adjuster=hist_adjuster, dynamic=(scale != 1.0), blend_type=blend_type)

    # Add the downsampling transformation
    if scale != 1.0:
        downsample = models.AffineModel(np.array([
            [scale, 0., 0.],
            [0., scale, 0.],
            [0., 0., 1.]
        ]))
        renderer.add_transformation(downsample)

    # Render the image
    out = works_dir + '/mask/mask_'+ os.path.basename(tile_fname).split('.')[0] +'.png'
    img, start_point = renderer.crop(scaled_bbox[0], scaled_bbox[2], scaled_bbox[1] - 1, scaled_bbox[3] - 1)
    img[img > 0] = 255
    cv2.imwrite(out,img)
    return img