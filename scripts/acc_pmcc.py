from rh_renderer.tilespec_renderer import TilespecRenderer
from rh_renderer.multiple_tiles_renderer import BlendType
from rh_renderer import models
import cv2
import argparse
import numpy as np
import time
import ujson
import os
import sys
import math
from rh_renderer.normalization.histogram_clahe import HistogramCLAHE, HistogramGB11CLAHE
import common
import rh_img_access_layer
from mb_aligner.dal.section import Section
from mb_aligner.alignment.fine_matchers.PMCC_filter import PMCC_match
import dHash


def mask_image(image1,image2):
    mask1 = (image1 > 0).astype(np.uint8)
    mask2 = (image2 > 0).astype(np.uint8)
    mask = np.logical_and(mask1, mask2).astype(np.uint8)
    image1 = cv2.add(image1, np.zeros(image1.shape, image1.dtype), mask = mask)
    image2 = cv2.add(image2, np.zeros(image2.shape, image2.dtype), mask = mask)
    return image1,image2

def ncc(img1, img2):
    return np.mean(np.multiply((img1-np.mean(img1)),(img2-np.mean(img2))))/(np.std(img1)*np.std(img2))


# def render_tilespec(entire_image_bbox, sec1, sec2, infname, output_dir, scale, mesh_spacing, compare_radius, out_image, invert_image):
def render_tilespec(scaled_bbox, entire_image_bbox, sec1, sec2, infname, output_dir, scale, variance, out_image, invert_image, 
                    threads_num=1, empty_placeholder=False, hist_adjuster_alg_type=None, from_to_cols_rows=None,
                    blend_type=BlendType.MULTI_BAND_SEAM):
    
    infname1 = infname[0]
    infname2 = infname[1]
    scale = 1

    
    with open(infname1, 'r') as data:
        tilespec1 = ujson.load(data)
    with open(infname2, 'r') as data:
        tilespec2 = ujson.load(data)
    hist_adjuster = HistogramCLAHE()

    renderer1 = TilespecRenderer(tilespec1, hist_adjuster=hist_adjuster, dynamic=(scale != 1.0), blend_type=blend_type)
    renderer2 = TilespecRenderer(tilespec2, hist_adjuster=hist_adjuster, dynamic=(scale != 1.0), blend_type=blend_type)

    # if scale != 1.0:
    #     scale_float = round(1/scale, 3)
    #     downsample = models.AffineModel(np.array([
    #         [scale_float, 0., 0.],
    #         [0., scale_float, 0.],
    #         [0., 0., 1.]
    #     ]))
    #     renderer1.add_transformation(downsample)
    #     renderer2.add_transformation(downsample)
  


    # Render the image
    img1, start_point1 = renderer1.crop_acc(scaled_bbox[0]-200, scaled_bbox[2]-200, scaled_bbox[1] - 1+200, scaled_bbox[3] - 1+200)
    # img2, start_point2 = renderer2.crop_acc(scaled_bbox[0], scaled_bbox[2], scaled_bbox[1] - 1, scaled_bbox[3] - 1)
    img2, start_point2 = renderer2.crop_acc(scaled_bbox[0]-200, scaled_bbox[2]-200, scaled_bbox[1] - 1+200, scaled_bbox[3] - 1+200)
    

    # img1 = cv2.resize(img1, dsize = (img1.shape[1]//scale,img1.shape[0]//scale))
    # img2 = cv2.resize(img2, dsize = (img2.shape[1]//scale,img2.shape[0]//scale))
    print('\nimg1 size:{}\nimg2 size:{}\n'.format(img1.shape,img2.shape))
    # img1, img2 = mask_image(img1, img2)
    if not (img1[0,:] == 0).all() and not (img1[-1,:] == 0).all() and not (img1[:,0] == 0).all() and not (img1[:,-1] == 0).all():
        if invert_image:
            print("inverting image")
            img1 = 255 - img1
            img2 = 255 - img2
        
        if np.var(img1) > variance and np.var(img2) > variance:

        
            # pmcc_result, reason, match_val = PMCC_match(img1, img2, min_correlation=0.2, maximal_curvature_ratio=10, maximal_ROD=0.9)
            dhash, ncc = dHash.compute(img1, img2)

            if out_image:
                out_fname1 = output_dir+'/image_'+str(scaled_bbox[0])+'-'+str(scaled_bbox[2])+'_'+sec1.canonical_section_name+'.png'
                out_fname2 = output_dir+'/image_'+str(scaled_bbox[0])+'-'+str(scaled_bbox[2])+'_'+sec2.canonical_section_name+'.png'
                print('writing output to {}'.format(sec1.canonical_section_name))
                cv2.imwrite(out_fname1, img1)
                print('writing output to {}'.format(sec2.canonical_section_name))
                cv2.imwrite(out_fname2, img2)
        
            return True, dhash,ncc
        else:
            return False, None, None
    else:
        return False, None, None