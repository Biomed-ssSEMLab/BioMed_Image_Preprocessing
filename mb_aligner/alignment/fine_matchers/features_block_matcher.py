# Setup
from __future__ import print_function
import os
import sys
import time
import numpy as np
import cv2
import argparse
import rh_logger
from rh_logger.api import logger
import logging
from collections import defaultdict
from scipy import spatial
from scipy.spatial import cKDTree as KDTree
import multiprocessing as mp
from rh_renderer import models
from mb_aligner.common import utils
from mb_aligner.common.thread_local_storage_lru import ThreadLocalStorageLRU
from mb_aligner.common.detector import FeaturesDetector
from mb_aligner.common.matcher import FeaturesMatcher
#from mb_aligner.common.grid_dict import GridDict
import tinyr
# from pathos.multiprocessing import ProcessPool

# import pyximport
# pyximport.install()
# from ..common import cv_wrap_module

class FeaturesBlockMatcherDispatcher(object):
    DETECTOR_KEY = "features_matcher_detector"
    BLOCK_FEATURES_KEY = "block_features"

    class FeaturesBlockMatcher(object):
        def __init__(self, sec1, sec2, sec1_to_sec2_transform, sec1_cache_features, sec2_cache_features, **kwargs):
            self._scaling = kwargs.get("scaling", 1.0)
            self._template_size = kwargs.get("template_size", 200)
            self._search_window_size = kwargs.get("search_window_size", 8 * self._template_size)

            # Parameters for PMCC filtering
            # self._min_corr = kwargs.get("min_correlation", 0.2)
            # self._max_curvature = kwargs.get("maximal_curvature_ratio", 10)
            # self._max_rod = kwargs.get("maximal_ROD", 0.9)
            # self._use_clahe = kwargs.get("use_clahe", False)
            # if self._use_clahe:
            #     self._clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))

            #self._debug_dir = kwargs.get("debug_dir", None)
            self._debug_save_matches = None
            
            detector_type = kwargs.get("detector_type", FeaturesDetector.Type.ORB.name)
            #self._detector = FeaturesDetector(detector_type, **kwargs.get("detector_params", {}))
            # scale the epsilon of the matcher
            if self._scaling != 1.0 and "max_epsilon" in kwargs.get("matcher_params", {}):
                kwargs["matcher_params"]["max_epsilon"] *= self._scaling
            self._matcher = FeaturesMatcher(FeaturesDetector.get_matcher_init_fn(detector_type), **kwargs.get("matcher_params", {}))

            self._template_side = self._template_size / 2
            self._search_window_side = self._search_window_size / 2
            #self._template_scaled_side = self._template_size * self._scaling / 2
            #self._search_window_scaled_side = self._search_window_size * self._scaling / 2

            self._sec1 = sec1
            self._sec2 = sec2
            self._sec1_to_sec2_transform = sec1_to_sec2_transform
            self._inverse_model = FeaturesBlockMatcherDispatcher.FeaturesBlockMatcher.inverse_transform(self._sec1_to_sec2_transform)

            self._sec1_cache_features = sec1_cache_features
            self._sec2_cache_features = sec2_cache_features

            # Create an rtree for each section's tiles, to quickly find the relevant tiles
            self._sec1_tiles_rtree = FeaturesBlockMatcherDispatcher.FeaturesBlockMatcher._create_tiles_bbox_rtree(sec1, self._scaling)
            self._sec2_tiles_rtree = FeaturesBlockMatcherDispatcher.FeaturesBlockMatcher._create_tiles_bbox_rtree(sec2, self._scaling)

        @staticmethod
        def _create_tiles_bbox_rtree(sec, scaling):
            sec_tiles_rtree = tinyr.RTree(interleaved=False, max_cap=5, min_cap=2)
            for t_idx, t in enumerate(sec.tiles()):
                bbox = tuple(np.array(t.bbox) * scaling)
                sec_tiles_rtree.insert(t_idx, bbox)
            return sec_tiles_rtree
            
        def set_debug_dir(self, debug_dir):
            self._debug_save_matches = True
            self._debug_dir = debug_dir

        @staticmethod
        def inverse_transform(model):
            mat = model.get_matrix()
            new_model = models.AffineModel(np.linalg.inv(mat))
            return new_model

        @staticmethod
        def _fetch_sec_features(sec, sec_tiles_rtree, sec_cache_features, bbox, detector_type, detector_kwargs, scaling, use_clahe):
            # Assumes the rtree, the sec_cache_features, and the bbox are all after scaling
            relevant_features = [[], []]
            
            # pool = ProcessPool(8)
            # # from pathos.multiprocessing import ProcessPool
            # # apipe 就是 apply_async
            # pool_results = []
            # rect_res = sec_tiles_rtree.search(bbox)
            # for t_idx, t in enumerate(sec.tiles()):
            #     k = "{}_t{}".format(sec.canonical_section_name, t_idx)
            #     if t_idx in rect_res and k not in sec_cache_features:
            #         # res = pool.apipe(FeaturesBlockMatcherDispatcher.compute_features, (t, sec_cache_features, k, detector_type, detector_kwargs, scaling, use_clahe))
            #         # pool_results.append(res)
            #         FeaturesBlockMatcherDispatcher.compute_features(t, sec_cache_features, k, detector_type, detector_kwargs, scaling, use_clahe)
            # # for res in pool_results:
            # #     res.get()
            
            rect_res = sec_tiles_rtree.search(bbox)
            for t_idx in rect_res:
                k = "{}_t{}".format(sec.canonical_section_name, t_idx)
                assert(k in sec_cache_features)
                tile_features_kps, tile_features_descs = sec_cache_features[k]

                # find all the features that are overlap with the bbox
                bbox_mask = (tile_features_kps[:, 0] >= bbox[0]) & (tile_features_kps[:, 0] <= bbox[1]) &\
                            (tile_features_kps[:, 1] >= bbox[2]) & (tile_features_kps[:, 1] <= bbox[3])
                if np.any(bbox_mask):
                    relevant_features[0].append(tile_features_kps[bbox_mask])
                    relevant_features[1].append(tile_features_descs[bbox_mask])
            if len(relevant_features[0]) == 0:
                return relevant_features
            return np.vstack(relevant_features[0]), np.vstack(relevant_features[1])

        def match_sec1_to_sec2_mfov(self, sec1_pts, detector_type, detector_kwargs, scaling, use_clahe):
            """
            sec1_pts will be in the original space (before scaling)
            """
            valid_matches = [[], [], []]
            invalid_matches = [[], []]
            if len(sec1_pts) == 0:
                return valid_matches, invalid_matches

            sec1_pts = np.atleast_2d(sec1_pts)
                
            # Apply the mfov transformation to compute estimated location on sec2
            sec1_mfov_pts_on_sec2 = self._sec1_to_sec2_transform.apply(sec1_pts)

            for sec1_pt, sec2_pt_estimated in zip(sec1_pts, sec1_mfov_pts_on_sec2):

                # Fetch the template around sec1_point (before transformation)
                from_x1, from_y1 = sec1_pt - self._template_side
                to_x1, to_y1 = sec1_pt + self._template_side
                sec1_pt_features_kps, sec1_pt_features_descs = FeaturesBlockMatcherDispatcher.FeaturesBlockMatcher._fetch_sec_features(
                    self._sec1, self._sec1_tiles_rtree, self._sec1_cache_features, tuple(np.array([from_x1, to_x1, from_y1, to_y1]) * self._scaling), \
                    detector_type, detector_kwargs, scaling, use_clahe)

                if len(sec1_pt_features_kps) <= 1:
                    continue
            
                # Fetch a large sub-image around img2_point (using search_window_scaled_size)
                from_x2, from_y2 = sec2_pt_estimated - self._search_window_side
                to_x2, to_y2 = sec2_pt_estimated + self._search_window_side
                sec2_pt_est_features_kps, sec2_pt_est_features_descs = FeaturesBlockMatcherDispatcher.FeaturesBlockMatcher._fetch_sec_features(
                    self._sec2, self._sec2_tiles_rtree, self._sec2_cache_features, tuple(np.array([from_x2, to_x2, from_y2, to_y2]) * self._scaling), \
                    detector_type, detector_kwargs, scaling, use_clahe)
        
                if len(sec2_pt_est_features_kps) <= 1:
                    continue

                # apply the transformation on sec1 feature points locations (after upscaling and then downscaling again)
                sec1_pt_features_kps = self._sec1_to_sec2_transform.apply(sec1_pt_features_kps / self._scaling) * self._scaling
                # Match the features
                transform_model, filtered_matches = self._matcher.match_and_filter(sec1_pt_features_kps, sec1_pt_features_descs, \
                    sec2_pt_est_features_kps, sec2_pt_est_features_descs)
                if transform_model is None:
                    invalid_matches[0].append(sec1_pt)
                    invalid_matches[1].append(1)
                else:
                    # the transform model need to be scaled
                    transform_matrix = transform_model.get_matrix()
                    if transform_matrix.shape!=(3,3):
                        print("transform_model in block_matcher is wrong!", transform_matrix)
                        raise RuntimeError("transform_matrix wrong, should be of size 3x3")

                    # translation model
                    transform_model.set(transform_matrix[:2, 2].T / self._scaling)  # .T 运算为取转置
                    #transform_model.set(transform_matrix.T / self._scaling)  # .T 运算为取转置
                    
                    if (transform_model.get_matrix()).shape!=(3,3):
                        print("transform_model in block_matcher after T is wrong!", transform_model.get_matrix)
                        raise RuntimeError("transform_matrix after T is wrong, should be of size 3x3")

                    # Compute the location of the matched point on sec2
                    sec2_pt = transform_model.apply(sec2_pt_estimated)  # + np.array([from_x1, from_y1]) + self._template_side

                    # # affine model
                    # m = transform_matrix
                    # m = np.array(m)
                    # sec2_pt = ((np.dot(m[:2,:2], sec2_pt_estimated / self._scaling) + np.asarray(m.T[2][:2]).reshape((1, 2))) * self._scaling)[0]
                    
                    # # ragid model
                    # transform_model.set(transform_matrix[:2, 2].T / self._scaling)  # .T 运算为取转置
                    # # Compute the location of the matched point on sec2
                    # sec2_pt = transform_model.apply(sec2_pt_estimated)  # + np.array([from_x1, from_y1]) + self._template_side
                    
                    logger.report_event("{}: match found: {} and {} (orig assumption: {})".format(os.getpid(), sec1_pt, sec2_pt, sec2_pt_estimated), \
                        log_level=logging.DEBUG)
                    valid_matches[0].append(sec1_pt)
                    valid_matches[1].append(sec2_pt)
                    valid_matches[2].append(len(filtered_matches[0]) / len(sec1_pt_features_kps))
            return valid_matches, invalid_matches

        def match_sec2_to_sec1_mfov(self, sec2_pts, detector_type, detector_kwargs, scaling, use_clahe):
            """
            sec2_pts will be in the original space (before scaling)
            """
            valid_matches = [[], [], []]
            invalid_matches = [[], []]
            if len(sec2_pts) == 0:
                return valid_matches, invalid_matches

            # Assume that only sec1 renderer was transformed and not sec2 (and both scaled)
            sec2_pts = np.atleast_2d(sec2_pts)
 
            sec2_pts_on_sec1 = self._inverse_model.apply(sec2_pts)

            for sec2_pt, sec1_pt_estimated in zip(sec2_pts, sec2_pts_on_sec1):
                # Fetch the template around sec2_pt
                from_x2, from_y2 = sec2_pt - self._template_side
                to_x2, to_y2 = sec2_pt + self._template_side
                sec2_pt_features_kps, sec2_pt_features_descs = FeaturesBlockMatcherDispatcher.FeaturesBlockMatcher._fetch_sec_features(
                    self._sec2, self._sec2_tiles_rtree, self._sec2_cache_features, tuple(np.array([from_x2, to_x2, from_y2, to_y2]) * self._scaling), \
                    detector_type, detector_kwargs, scaling, use_clahe)

                if len(sec2_pt_features_kps) <= 1:
                    continue

                # Fetch a large sub-image around sec1_pt_estimated (after transformation, using search_window_scaled_size)
                from_x1, from_y1 = sec1_pt_estimated - self._search_window_side
                to_x1, to_y1 = sec1_pt_estimated + self._search_window_side
                sec1_pt_est_features_kps, sec1_pt_est_features_descs = FeaturesBlockMatcherDispatcher.FeaturesBlockMatcher._fetch_sec_features(
                    self._sec1, self._sec1_tiles_rtree, self._sec1_cache_features, tuple(np.array([from_x1, to_x1, from_y1, to_y1]) * self._scaling), \
                    detector_type, detector_kwargs, scaling, use_clahe)

                if len(sec1_pt_est_features_kps) <= 1:
                    continue

                # apply the inverse transformation on sec2 feature points locations (after upscaling and then downscaling again)
                sec2_pt_features_kps = self._inverse_model.apply(sec2_pt_features_kps / self._scaling) * self._scaling
                # Match the features
                transform_model, filtered_matches = self._matcher.match_and_filter(sec2_pt_features_kps, sec2_pt_features_descs, \
                    sec1_pt_est_features_kps, sec1_pt_est_features_descs)
                if transform_model is None:
                    invalid_matches[0].append(sec2_pt)
                    invalid_matches[1].append(1)
                else:
                    # the transform model need to be scaled
                    transform_matrix = transform_model.get_matrix()
                    
                    # translation model
                    transform_model.set(transform_matrix[:2, 2].T / self._scaling)  # .T 运算为取转置
                    # Compute the location of the matched point on sec2
                    sec1_pt = transform_model.apply(sec1_pt_estimated)  # + np.array([from_x1, from_y1]) + self._template_side

                    # # affine model
                    # m = transform_matrix
                    # m = np.array(m)
                    # sec1_pt = (((np.dot(m[:2,:2], sec1_pt_estimated / self._scaling) )+ np.asarray(m.T[2][:2]).reshape((1, 2))) * self._scaling)[0]

                    logger.report_event("{}: match found: {} and {} (orig assumption: {})".format(os.getpid(), sec2_pt, sec1_pt, sec1_pt_estimated), \
                        log_level=logging.DEBUG)
                    valid_matches[0].append(sec2_pt)
                    valid_matches[1].append(sec1_pt)
                    valid_matches[2].append(len(filtered_matches[0]) / len(sec2_pt_features_kps))
                    # 每个正六边形顶点周围的块匹配结果的匹配通过率，分母为模板图像中的特征点的数量，一个值为一个顶点周围一群特征点的匹配率
            return valid_matches, invalid_matches

        def get_params_1(self):
            sec1_to_sec2_transform = self._sec1_to_sec2_transform
            template_side = self._template_side
            search_window_side = self._search_window_side
            scaling = self._scaling
            sec1_tiles_rtree = self._sec1_tiles_rtree
            sec1 = self._sec1
            return sec1_to_sec2_transform, template_side, search_window_side, scaling, sec1_tiles_rtree, sec1

        def get_params_2(self):
            inverse_model = self._inverse_model
            template_side = self._template_side
            scaling = self._scaling
            sec2_tiles_rtree = self._sec2_tiles_rtree
            sec2 = self._sec2
            return inverse_model, template_side, scaling, sec2_tiles_rtree, sec2

        def compute_tile_features_1(self, sec1_region_mesh_pts, sec2_mesh_pts_cur_sec1_region, sec1_cache, sec2_cache, detector_type, \
            detector_kwargs, scaling, use_clahe):
            if not len(sec1_region_mesh_pts) == 0:
                sec1_pts = np.atleast_2d(sec1_region_mesh_pts)
                sec1_mfov_pts_on_sec2 = self._sec1_to_sec2_transform.apply(sec1_pts)
                for sec1_pt, sec2_pt_estimated in zip(sec1_pts, sec1_mfov_pts_on_sec2):
                    from_x1, from_y1 = sec1_pt - self._template_side
                    to_x1, to_y1 = sec1_pt + self._template_side
                    bbox1 = tuple(np.array([from_x1, to_x1, from_y1, to_y1]) * self._scaling)
                    rect_res1 = self._sec1_tiles_rtree.search(bbox1)
                    for t_idx, t in enumerate(self._sec1.tiles()):
                        k = "{}_t{}".format(self._sec1.canonical_section_name, t_idx)
                        if t_idx in rect_res1 and k not in sec1_cache:
                            FeaturesBlockMatcherDispatcher.compute_features(t, sec1_cache, k, detector_type, detector_kwargs, scaling, use_clahe)

    def __init__(self, **kwargs):
        self._kwargs = kwargs
        self._mesh_spacing = kwargs.get("mesh_spacing", 1500)

        self._scaling = kwargs.get("scaling", 1.0)
        self._template_size = kwargs.get("template_size", 200)
        self._search_window_size = kwargs.get("search_window_size", 8 * self._template_size)
        logger.report_event("Actual template size: {} and window search size: {}".format(self._template_size * self._scaling, \
            self._search_window_size * self._scaling), log_level=logging.INFO)

        self._use_clahe = kwargs.get("use_clahe", False)
        self._detector_type = kwargs.get("detector_type", FeaturesDetector.Type.ORB.name)
        self._detector_kwargs = kwargs.get("detector_params", {})
        self._matcher_kwargs = kwargs.get("matcher_params", {})

#         # Parameters for PMCC filtering
#         self._min_corr = kwargs.get("min_correlation", 0.2)
#         self._max_curvature = kwargs.get("maximal_curvature_ratio", 10)
#         self._max_rod = kwargs.get("maximal_ROD", 0.9)
#         self._use_clahe = kwargs.get("use_clahe", False)

        self._debug_dir = kwargs.get("debug_dir", None)
        if self._debug_dir is not None:
            logger.report_event("Debug mode - on", log_level=logging.INFO)
            # Create a debug directory
            import datetime
            self._debug_dir = os.path.join(self._debug_dir, 'debug_matches_{}'.format(datetime.datetime.now().isoformat()))
            os.mkdirs(self._debug_dir)

    def detect_tile_blobs_1(region1_key, sec1, sec2, sec1_to_sec2_mfov_transform, debug_dir, matcher_args, sec1_region_mesh_pts, \
        sec2_mesh_pts_cur_sec1_region, sec1_cache, sec2_cache, detector_type, detector_kwargs, scaling, use_clahe):
        fine_matcher_key = "block_matcher_{},{},{}".format(sec1.canonical_section_name, sec2.canonical_section_name, region1_key[0])
        thread_local_store = ThreadLocalStorageLRU()
        if fine_matcher_key in thread_local_store.keys():
            fine_matcher = thread_local_store[fine_matcher_key]
        else:
            fine_matcher = FeaturesBlockMatcherDispatcher.FeaturesBlockMatcher(sec1, sec2, sec1_to_sec2_mfov_transform[region1_key[0]], \
                sec1_cache, sec2_cache, **matcher_args)
            if debug_dir is not None:
                fine_matcher.set_debug_dir(debug_dir)
            thread_local_store[fine_matcher_key] = fine_matcher
        fine_matcher.compute_tile_features_1(sec1_region_mesh_pts, sec2_mesh_pts_cur_sec1_region, sec1_cache, sec2_cache, \
            detector_type, detector_kwargs, scaling, use_clahe)

    @staticmethod
    def _is_point_in_img(img_bbox, point):
        """Returns True if the given point lies inside the image as denoted by the given tile_tilespec"""
        # TODO - instead of checking inside the bbox, need to check inside the polygon after transformation
        if point[0] > img_bbox[0] and point[1] > img_bbox[2] and point[0] < img_bbox[1] and point[1] < img_bbox[3]:
            return True
        return False

    @staticmethod
    def sum_invalid_matches(invalid_matches):
        if len(invalid_matches[1]) == 0:
            return [0] * 5
        hist, _ = np.histogram(invalid_matches[1], bins=5)
        return hist

    @staticmethod
    def _perform_matching(sec1_mfov_tile_idx, sec1, sec2, sec1_to_sec2_mfov_transform, sec1_cache_features, sec2_cache_features, \
        sec1_mfov_mesh_pts, sec2_mfov_mesh_pts, debug_dir, matcher_args, detector_type, detector_kwargs, scaling, use_clahe):
#         fine_matcher_key = "block_matcher_{},{},{}".format(sec1.canonical_section_name, sec2.canonical_section_name, sec1_mfov_tile_idx[0])
#         fine_matcher = getattr(threadLocal, fine_matcher_key, None)
#         if fine_matcher is None:
#             fine_matcher = BlockMatcherPMCCDispatcher.BlockMatcherPMCC(sec1, sec2, sec1_to_sec2_mfov_transform, **matcher_args)
#             if debug_dir is not None:
#                 fine_matcher.set_debug_dir(debug_dir)
# 
#             setattr(threadLocal, fine_matcher_key, fine_matcher)

        fine_matcher_key = "block_matcher_{},{},{}".format(sec1.canonical_section_name, sec2.canonical_section_name, sec1_mfov_tile_idx[0])
        thread_local_store = ThreadLocalStorageLRU()
        if fine_matcher_key in thread_local_store.keys():
            fine_matcher = thread_local_store[fine_matcher_key]
        else:
            fine_matcher = FeaturesBlockMatcherDispatcher.FeaturesBlockMatcher(sec1, sec2, sec1_to_sec2_mfov_transform, \
                sec1_cache_features, sec2_cache_features, **matcher_args)
            if debug_dir is not None:
                fine_matcher.set_debug_dir(debug_dir)
            thread_local_store[fine_matcher_key] = fine_matcher

#         fine_matcher = getattr(threadLocal, fine_matcher_key, None)
#         if fine_matcher is None:
#             fine_matcher = FeaturesBlockMatcherDispatcher.FeaturesBlockMatcher(sec1, sec2, sec1_to_sec2_mfov_transform, sec1_cache_features, sec2_cache_features, **matcher_args)
#             if debug_dir is not None:
#                 fine_matcher.set_debug_dir(debug_dir)
#             setattr(threadLocal, fine_matcher_key, fine_matcher)

        logger.report_event("Features-Matching+PMCC layers: {} with {} (mfov1 {}) {} mesh points1, {} mesh points2".format(
            sec1.canonical_section_name, sec2.canonical_section_name, sec1_mfov_tile_idx, len(sec1_mfov_mesh_pts), len(sec2_mfov_mesh_pts)), \
            log_level=logging.INFO)
       
        logger.report_event("Features-Matching+PMCC layers: {} -> {}".format(sec1.canonical_section_name, sec2.canonical_section_name), log_level=logging.INFO)
        valid_matches1, invalid_matches1 = fine_matcher.match_sec1_to_sec2_mfov(sec1_mfov_mesh_pts, detector_type, detector_kwargs, scaling, use_clahe)
        logger.report_event("Features-Matching+PMCC layers: {} -> {} valid matches: {}, invalid_matches: {} {}".format(
            sec1.canonical_section_name, sec2.canonical_section_name, len(valid_matches1[0]), len(invalid_matches1[0]), \
            FeaturesBlockMatcherDispatcher.sum_invalid_matches(invalid_matches1)), log_level=logging.INFO)

        logger.report_event("Features-Matching+PMCC layers: {} <- {}".format(sec1.canonical_section_name, sec2.canonical_section_name), log_level=logging.INFO)
        valid_matches2, invalid_matches2 = fine_matcher.match_sec2_to_sec1_mfov(sec2_mfov_mesh_pts, detector_type, detector_kwargs, scaling, use_clahe)
        logger.report_event("Features-Matching+PMCC layers: {} <- {} valid matches: {}, invalid_matches: {} {}".format(
            sec1.canonical_section_name, sec2.canonical_section_name, len(valid_matches2[0]), len(invalid_matches2[0]), \
            FeaturesBlockMatcherDispatcher.sum_invalid_matches(invalid_matches2)), log_level=logging.INFO)

        return sec1_mfov_tile_idx, valid_matches1, valid_matches2

    @staticmethod
    def compute_features(tile, out_dict, out_dict_key, detector_type, detector_kwargs, scaling, use_clahe):
        thread_local_store = ThreadLocalStorageLRU()
        if FeaturesBlockMatcherDispatcher.DETECTOR_KEY in thread_local_store.keys():
            detector = thread_local_store[FeaturesBlockMatcherDispatcher.DETECTOR_KEY]
        else:
            detector = FeaturesDetector(detector_type, **detector_kwargs)
            thread_local_store[FeaturesBlockMatcherDispatcher.DETECTOR_KEY] = detector

#         detector = getattr(threadLocal, FeaturesBlockMatcherDispatcher.DETECTOR_KEY, None)
#         if detector is None:
#             #detector_type = FeaturesDetector.Type.ORB.name
#             detector = FeaturesDetector(detector_type, **detector_kwargs)
# 
#             setattr(threadLocal, FeaturesBlockMatcherDispatcher.DETECTOR_KEY, detector)
        
        # Load the image
        img = tile.image

        if use_clahe:
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            img = clahe.apply(img)

        # scale the image
        if scaling != 1.0:
            img = cv2.resize(img, None, fx=scaling, fy=scaling, interpolation=cv2.INTER_AREA)

        # compute features
        kps, descs = detector.detect(img)

        # Replace the keypoints with a numpy array
        kps_pts = np.empty((len(kps), 2), dtype=np.float64)
        for kp_i, kp in enumerate(kps):
            kps_pts[kp_i][:] = kp.pt

        # Apply tile transformations to each point
        for transform in tile.transforms:
            # Make sure wwe upscale and then downscale the kps
            kps_pts = transform.apply(kps_pts / scaling) * scaling

        out_dict[out_dict_key] = [kps_pts, np.array(descs)]

#     @staticmethod
#     def create_grid_from_pts(pts, grid_size):
#         """
#         Given 2D points and a grid_size, gathers all points into a dictionary that maps between
#         pt[0] // grid_size, pt[1] // grid_size    to    a numpy array of the points that land in that bucket
#         """
#         grid = defaultdict(list)
# 
#         locations = (pts / grid_size).astype(int)
#         for loc, pt in zip(locations, pts):
#             grid[loc].append(pt)
# 
#         for k, pts_list in grid.items():
#             grid[k] = np.array(pts_list)
# 
#         return grid

    def match_layers_fine_matching(self, sec1, sec2, sec1_cache, sec2_cache, sec1_to_sec2_mfovs_transforms, pool):
        print('\n<================fine matching================>')
        logger.report_event("Features-Matching+PMCC layers: {} with {} (bidirectional)".format(
            sec1.canonical_section_name, sec2.canonical_section_name), log_level=logging.INFO)

        # take just the models (w/o the filtered match points)
        sec1_to_sec2_mfovs_transforms = {k:v[0] for k, v in sec1_to_sec2_mfovs_transforms.items()}
        # create a new dictionary {k:v[0]}, notice the difference between [] and {}

        if FeaturesBlockMatcherDispatcher.BLOCK_FEATURES_KEY not in sec1_cache:
            sec1_cache[FeaturesBlockMatcherDispatcher.BLOCK_FEATURES_KEY] = {}
        if FeaturesBlockMatcherDispatcher.BLOCK_FEATURES_KEY not in sec2_cache:
            sec2_cache[FeaturesBlockMatcherDispatcher.BLOCK_FEATURES_KEY] = {}

        # For each section, detect the features for each tile, and transform them to their stitched location (store in cache for future comparisons)
        # logger.report_event("Computing per tile block features", log_level=logging.INFO)

        # pool_results = []
        # for sec1_t_idx, t in enumerate(sec1.tiles()):
        #     if sec1_t_idx == 610:
        #         break
        #     k = "{}_t{}".format(sec1.canonical_section_name, sec1_t_idx)
        #     if k not in sec1_cache[FeaturesBlockMatcherDispatcher.BLOCK_FEATURES_KEY]:
        #         res = pool.apply_async(FeaturesBlockMatcherDispatcher.compute_features, (t, sec1_cache[FeaturesBlockMatcherDispatcher.BLOCK_FEATURES_KEY], k, self._detector_type, self._detector_kwargs, self._scaling, self._use_clahe))
        #         pool_results.append(res)
        # print('\n','section{} ==> tile features detection finished(tiles num: {})'.format(sec1.layer,sec1_t_idx),'\n')
        
        # for sec2_t_idx, t in enumerate(sec2.tiles()):
        #     if sec1_t_idx == 610:
        #         break
        #     k = "{}_t{}".format(sec2.canonical_section_name, sec2_t_idx)
        #     if k not in sec2_cache[FeaturesBlockMatcherDispatcher.BLOCK_FEATURES_KEY]:
        #         res = pool.apply_async(FeaturesBlockMatcherDispatcher.compute_features, (t, sec2_cache[FeaturesBlockMatcherDispatcher.BLOCK_FEATURES_KEY], k, self._detector_type, self._detector_kwargs, self._scaling, self._use_clahe))
        #         pool_results.append(res)
        # print('\n','section{} ==> tile features detection finished(tiles num: {})'.format(sec2.layer,sec1_t_idx),'\n')
        
        # for res in pool_results:
        #     res.get()
        
        logger.report_event("Computing missing mfovs transformations", log_level=logging.INFO)
        # find the nearest transformations for mfovs1 that are missing in sec1_to_sec2_mfovs_transforms and for sec2 to sec1
        mfovs1_centers_sec2centers = [[], [], []] # lists of mfovs indexes, mfovs centers, and mfovs centers after transformation to sec2
        missing_mfovs1_transforms_centers = [[], []] # lists of missing mfovs in sec1 and their centers

        for mfov1 in sec1.mfovs():
            mfov1_center = np.array([(mfov1.bbox[0] + mfov1.bbox[1])/2, (mfov1.bbox[2] + mfov1.bbox[3])/2])
            if mfov1.mfov_index in sec1_to_sec2_mfovs_transforms and sec1_to_sec2_mfovs_transforms[mfov1.mfov_index] is not None:
                mfovs1_centers_sec2centers[0].append(mfov1.mfov_index)
                mfovs1_centers_sec2centers[1].append(mfov1_center)
                sec1_mfov_model = sec1_to_sec2_mfovs_transforms[mfov1.mfov_index]
                mfovs1_centers_sec2centers[2].append(sec1_mfov_model.apply(mfov1_center)[0])  #model(section1'mfov to section2).apply(cordinate in section1)  [0]:  返回值为二维，即[[cordinate in section2]],加上[0]后表示坐标值
            else:
                missing_mfovs1_transforms_centers[0].append(mfov1.mfov_index)
                missing_mfovs1_transforms_centers[1].append(mfov1_center)

        # estimate the transformation for mfovs in sec1 that do not have one (look at closest neighbor)
        if len(missing_mfovs1_transforms_centers[0]) > 0:
            mfovs1_centers_sec1_kdtree = KDTree(mfovs1_centers_sec2centers[1])
            mfovs1_missing_closest_centers_mfovs1_idxs = mfovs1_centers_sec1_kdtree.query(missing_mfovs1_transforms_centers[1])[1]
            missing_mfovs1_sec2_centers = []
            for i, (mfov1_index, mfov1_closest_mfov_idx) in enumerate(zip(missing_mfovs1_transforms_centers[0], mfovs1_missing_closest_centers_mfovs1_idxs)):
                model = sec1_to_sec2_mfovs_transforms[mfovs1_centers_sec2centers[0][mfov1_closest_mfov_idx]]
                sec1_to_sec2_mfovs_transforms[mfov1_index] = model
                missing_mfovs1_sec2_centers.append(model.apply(np.atleast_2d(missing_mfovs1_transforms_centers[1][i]))[0])

            # update the mfovs1_centers_sec2centers lists to include the missing mfovs and their corresponding values
            mfovs1_centers_sec2centers[0] = np.concatenate((mfovs1_centers_sec2centers[0], missing_mfovs1_transforms_centers[0]))
            mfovs1_centers_sec2centers[1] = np.concatenate((mfovs1_centers_sec2centers[1], missing_mfovs1_transforms_centers[1]))
            mfovs1_centers_sec2centers[2] = np.concatenate((mfovs1_centers_sec2centers[2], missing_mfovs1_sec2_centers))

#         # Put all features of each section in an rtree
#         #sec1_features_rtree = tinyr.RTree(interleaved=False, max_cap=5, min_cap=2)
#         sec1_features_grid = GridDict(self._template_size)
#         #sec2_features_rtree = tinyr.RTree(interleaved=False, max_cap=5, min_cap=2)
#         sec2_features_grid = GridDict(self._template_size)
#         # using the (x_min, x_max, y_min, y_max) notation
#         for sec1_t_idx, t in enumerate(sec1.tiles()):
#             k = "{}_t{}".format(sec1.canonical_section_name, sec1_t_idx)
#             t_kps, t_descs = sec1_cache["block_features"][k]
#             #t_mfov_index = t.mfov_index
#             #t_features_kps_sec1_on_sec2 = sec1_to_sec2_mfovs_transforms[t_mfov_index].apply(t_kps)
# #             for feature_idx, t_feature_kp_sec1 in enumerate(t_kps):
# #                 sec1_features_rtree.insert([k, feature_idx], [t_feature_kp_sec1[0], t_feature_kp_sec1[0]+0.5, t_feature_kp_sec1[1], t_feature_kp_sec1[1]+0.5])
#             for t_feature_kp_sec1, t_feature_desc_sec1 in zip(t_kps, t_descs):
#                 sec1_features_grid.add(t_feature_kp_sec1, t_feature_desc_sec1)
#                 
#          for sec2_t_idx, t in enumerate(sec2.tiles()):
#             k = "{}_t{}".format(sec2.canonical_section_name, sec2_t_idx)
#             t_kps, t_descs = sec2_cache["block_features"][k]
# #             for feature_idx, t_feature_kp_sec2 in enumerate(t_kps):
# #                 sec2_features_rtree.insert([k, feature_idx], [t_feature_kp_sec2[0], t_feature_kp_sec2[0]+0.5, t_feature_kp_sec2[1], t_feature_kp_sec2[1]+0.5])
#             for t_feature_kp_sec2, t_feature_desc_sec2 in zip(t_kps, t_descs):
#                 sec2_features_grid.add(t_feature_kp_sec2, t_feature_desc_sec2)


        logger.report_event("Computing grid points and distributing work", log_level=logging.INFO)
        # Lay a grid on top of each section， 将sec1 ,sec2正六边形网格化
        sec1_mesh_pts = utils.generate_hexagonal_grid(sec1.bbox, self._mesh_spacing)  # mesh_spacing 为 正六边形的纵向高度，六边形为pointy topped类型
        sec2_mesh_pts = utils.generate_hexagonal_grid(sec2.bbox, self._mesh_spacing)

        sec1_tiles_centers = [[(t.bbox[0] + t.bbox[1])/2, (t.bbox[2] + t.bbox[3])/2] for t in sec1.tiles()]
        sec1_tiles_centers_kdtree = KDTree(sec1_tiles_centers)
        sec1_tiles_mfov_tile_idxs = np.array([[t.mfov_index, t.tile_index] for t in sec1.tiles()])
        sec2_tiles_centers = [[(t.bbox[0] + t.bbox[1])/2, (t.bbox[2] + t.bbox[3])/2] for t in sec2.tiles()]
        sec2_tiles_centers_kdtree = KDTree(sec2_tiles_centers)
        sec2_tiles_mfov_tile_idxs = np.array([[t.mfov_index, t.tile_index] for t in sec2.tiles()])

        # TODO - split the work in a smart way between the processes
        # Group the mesh points of sec1 by its mfovs_tiles and make sure the points are in tiles
        sec1_mesh_pts_mfov_tile_idxs = sec1_tiles_mfov_tile_idxs[sec1_tiles_centers_kdtree.query(sec1_mesh_pts)[1]]
        sec1_per_region_mesh_pts = defaultdict(list)
        for sec1_pt, sec1_pt_mfov_tile_idx in zip(sec1_mesh_pts, sec1_mesh_pts_mfov_tile_idxs):
            sec1_tile = sec1.get_mfov(sec1_pt_mfov_tile_idx[0]).get_tile(sec1_pt_mfov_tile_idx[1])
            if FeaturesBlockMatcherDispatcher._is_point_in_img(sec1_tile.bbox, sec1_pt):
                sec1_per_region_mesh_pts[tuple(sec1_pt_mfov_tile_idx)].append(sec1_pt)

        # Group the mesh pts of sec2 by the mfov on sec1 which they should end up on (mfov1 that after applying its transformation is closest to that point)
        # Transform sec1 tiles centers to their estimated location on sec2
        sec1_tiles_centers_per_mfov = defaultdict(list)
        for sec1_tile_center, sec1_tiles_mfov_tile_idx in zip(sec1_tiles_centers, sec1_tiles_mfov_tile_idxs):
            sec1_tiles_centers_per_mfov[sec1_tiles_mfov_tile_idx[0]].append(sec1_tile_center)
        sec1_tiles_centers_on_sec2 = [sec1_to_sec2_mfovs_transforms[mfov_index].apply(np.atleast_2d(mfov1_tiles_centers)) \
                                      for mfov_index, mfov1_tiles_centers in sec1_tiles_centers_per_mfov.items()]
        sec1_tiles_centers_on_sec2 = np.vstack(tuple(sec1_tiles_centers_on_sec2))

        sec1_tiles_centers_on_sec2_kdtree = KDTree(sec1_tiles_centers_on_sec2)
        sec2_mesh_pts_sec1_closest_tile_idxs = sec1_tiles_centers_on_sec2_kdtree.query(sec2_mesh_pts)[1]
        sec2_mesh_pts_mfov_tile_idxs = sec2_tiles_mfov_tile_idxs[sec2_tiles_centers_kdtree.query(sec2_mesh_pts)[1]]
        sec2_per_region1_mesh_pts = defaultdict(list)
        for sec2_pt, (sec2_pt_mfov_idx, sec2_pt_tile_idx), sec1_tile_center_idx in zip(sec2_mesh_pts, sec2_mesh_pts_mfov_tile_idxs, \
            sec2_mesh_pts_sec1_closest_tile_idxs):
            sec2_tile = sec2.get_mfov(sec2_pt_mfov_idx).get_tile(sec2_pt_tile_idx)
            if FeaturesBlockMatcherDispatcher._is_point_in_img(sec2_tile.bbox, sec2_pt):
                sec2_per_region1_mesh_pts[tuple(sec1_tiles_mfov_tile_idxs[sec1_tile_center_idx])].append(sec2_pt)
    
        # logger.report_event("<=========== Computing per tile block features ===========>", log_level=logging.INFO)
        # tile_res = []
        # for region1_key, sec1_region_mesh_pts in sec1_per_region_mesh_pts.items():
        #     sec2_mesh_pts_cur_sec1_region = sec2_per_region1_mesh_pts[region1_key]
        #     fine_matcher_key = "block_matcher_{},{},{}".format(sec1.canonical_section_name, sec2.canonical_section_name, region1_key[0])
        #     thread_local_store = ThreadLocalStorageLRU()
        #     if fine_matcher_key in thread_local_store.keys():
        #         fine_matcher = thread_local_store[fine_matcher_key]
        #     else:
        #         fine_matcher = FeaturesBlockMatcherDispatcher.FeaturesBlockMatcher(sec1, sec2, sec1_to_sec2_mfovs_transforms[region1_key[0]], \
        #         sec1_cache[FeaturesBlockMatcherDispatcher.BLOCK_FEATURES_KEY], sec2_cache[FeaturesBlockMatcherDispatcher.BLOCK_FEATURES_KEY], **self._kwargs)
        #         if self._debug_dir is not None:
        #             fine_matcher.set_debug_dir(self._debug_dir)
        #         thread_local_store[fine_matcher_key] = fine_matcher
           
        #     sec1_to_sec2_transform_1,template_side,search_window_side,scaling,sec1_tiles_rtree,sec1 = fine_matcher.get_params_1()
        #     inverse_model,template_side,scaling,sec2_tiles_rtree,sec2 = fine_matcher.get_params_2()

        #     if not len(sec1_region_mesh_pts) == 0 :
        #         sec1_pts = np.atleast_2d(sec1_region_mesh_pts)
        #         sec1_mfov_pts_on_sec2 = sec1_to_sec2_transform_1.apply(sec1_pts)
        #         for sec1_pt, sec2_pt_estimated in zip(sec1_pts, sec1_mfov_pts_on_sec2):
        #             from_x1, from_y1 = sec1_pt - template_side
        #             to_x1, to_y1 = sec1_pt + template_side
        #             bbox1 = tuple(np.array([from_x1, to_x1, from_y1, to_y1]) * scaling)
        #             rect_res1 = sec1_tiles_rtree.search(bbox1)
        #             for t_idx, t in enumerate(sec1.tiles()):
        #                 k = "{}_t{}".format(sec1.canonical_section_name, t_idx)
        #                 if t_idx in rect_res1 and k not in sec1_cache[FeaturesBlockMatcherDispatcher.BLOCK_FEATURES_KEY]:
        #                     tile_pool = pool.apply_async(FeaturesBlockMatcherDispatcher.compute_features(t, sec1_cache[FeaturesBlockMatcherDispatcher.BLOCK_FEATURES_KEY], k, self._detector_type, self._detector_kwargs, self._scaling, self._use_clahe))
        #                     tile_res.append(tile_pool)
        #             from_x2, from_y2 = sec2_pt_estimated - search_window_side
        #             to_x2, to_y2 = sec2_pt_estimated + search_window_side
        #             bbox2 = tuple(np.array([from_x2, to_x2, from_y2, to_y2]) * scaling)
        #             rect_res2 = sec2_tiles_rtree.search(bbox2)
        #             for t_idx, t in enumerate(sec2.tiles()):
        #                 k = "{}_t{}".format(sec2.canonical_section_name, t_idx)
        #                 if t_idx in rect_res2 and k not in sec2_cache[FeaturesBlockMatcherDispatcher.BLOCK_FEATURES_KEY]:
        #                     tile_pool = pool.apply_async(FeaturesBlockMatcherDispatcher.compute_features(t, sec2_cache[FeaturesBlockMatcherDispatcher.BLOCK_FEATURES_KEY], k, self._detector_type, self._detector_kwargs, self._scaling, self._use_clahe))
        #                     tile_res.append(tile_pool)

        #     if not len(sec2_mesh_pts_cur_sec1_region) == 0:
        #         sec2_pts = np.atleast_2d(sec2_mesh_pts_cur_sec1_region)
        #         sec2_pts_on_sec1 = inverse_model.apply(sec2_pts)
        #         for sec2_pt, sec1_pt_estimated in zip(sec2_pts, sec2_pts_on_sec1):
        #             from_x2, from_y2 = sec2_pt - template_side
        #             to_x2, to_y2 = sec2_pt + template_side
        #             bbox3 = tuple(np.array([from_x2, to_x2, from_y2, to_y2]) * scaling)
        #             rect_res3 = sec2_tiles_rtree.search(bbox3)
        #             for t_idx, t in enumerate(sec2.tiles()):
        #                 k = "{}_t{}".format(sec2.canonical_section_name, t_idx)
        #                 if t_idx in rect_res2 and k not in sec2_cache[FeaturesBlockMatcherDispatcher.BLOCK_FEATURES_KEY]:
        #                     tile_pool = pool.apply_async(FeaturesBlockMatcherDispatcher.compute_features(t, sec2_cache[FeaturesBlockMatcherDispatcher.BLOCK_FEATURES_KEY], k, self._detector_type, self._detector_kwargs, self._scaling, self._use_clahe))
        #                     tile_res.append(tile_pool)
        #             from_x1, from_y1 = sec1_pt_estimated - search_window_side
        #             to_x1, to_y1 = sec1_pt_estimated + search_window_side
        #             bbox4 = tuple(np.array([from_x1, to_x1, from_y1, to_y1]) * scaling)
        #             rect_res4 = sec1_tiles_rtree.search(bbox4)
        #             for t_idx, t in enumerate(sec1.tiles()):
        #                 k = "{}_t{}".format(sec1.canonical_section_name, t_idx)
        #                 if t_idx in rect_res1 and k not in sec1_cache[FeaturesBlockMatcherDispatcher.BLOCK_FEATURES_KEY]:
        #                     tile_pool = pool.apply_async(FeaturesBlockMatcherDispatcher.compute_features(t, sec1_cache[FeaturesBlockMatcherDispatcher.BLOCK_FEATURES_KEY], k, self._detector_type, self._detector_kwargs, self._scaling, self._use_clahe))
        #                     tile_res.append(tile_pool)

        # for tile_pool in tile_res:
        #     tile_pool.get()
        # print('<=========== tile features detection all finished ===========>')

        logger.report_event("<=========== Computing tile indices ===========>", log_level=logging.INFO)
        tile_res1 = set()
        tile_res2 = set()
        for region1_key, sec1_region_mesh_pts in sec1_per_region_mesh_pts.items():
            sec2_mesh_pts_cur_sec1_region = sec2_per_region1_mesh_pts[region1_key]
            fine_matcher_key = "block_matcher_{},{},{}".format(sec1.canonical_section_name, sec2.canonical_section_name, region1_key[0])
            thread_local_store = ThreadLocalStorageLRU()
            if fine_matcher_key in thread_local_store.keys():
                fine_matcher = thread_local_store[fine_matcher_key]
            else:
                fine_matcher = FeaturesBlockMatcherDispatcher.FeaturesBlockMatcher(sec1, sec2, sec1_to_sec2_mfovs_transforms[region1_key[0]], \
                sec1_cache[FeaturesBlockMatcherDispatcher.BLOCK_FEATURES_KEY], sec2_cache[FeaturesBlockMatcherDispatcher.BLOCK_FEATURES_KEY], **self._kwargs)
                if self._debug_dir is not None:
                    fine_matcher.set_debug_dir(self._debug_dir)
                thread_local_store[fine_matcher_key] = fine_matcher
            sec1_to_sec2_transform_1, template_side, search_window_side, scaling, sec1_tiles_rtree, sec1 = fine_matcher.get_params_1()
            inverse_model, template_side, scaling, sec2_tiles_rtree, sec2 = fine_matcher.get_params_2() 

            if not len(sec1_region_mesh_pts) == 0:
                sec1_pts = np.atleast_2d(sec1_region_mesh_pts)
                sec1_mfov_pts_on_sec2 = sec1_to_sec2_transform_1.apply(sec1_pts)
                for sec1_pt, sec2_pt_estimated in zip(sec1_pts, sec1_mfov_pts_on_sec2):
                    from_x1, from_y1 = sec1_pt - template_side
                    to_x1, to_y1 = sec1_pt + template_side
                    bbox1 = tuple(np.array([from_x1, to_x1, from_y1, to_y1]) * scaling)
                    rect_res1 = sec1_tiles_rtree.search(bbox1)
                    for t_idx in rect_res1:
                        tile_res1.add(t_idx)
                    
                    from_x2, from_y2 = sec2_pt_estimated - search_window_side
                    to_x2, to_y2 = sec2_pt_estimated + search_window_side
                    bbox2 = tuple(np.array([from_x2, to_x2, from_y2, to_y2]) * scaling)
                    rect_res2 = sec2_tiles_rtree.search(bbox2)
                    for t_idx in rect_res2:
                        tile_res2.add(t_idx)

            if not len(sec2_mesh_pts_cur_sec1_region) == 0:
                sec2_pts = np.atleast_2d(sec2_mesh_pts_cur_sec1_region)
                sec2_pts_on_sec1 = inverse_model.apply(sec2_pts)
                for sec2_pt, sec1_pt_estimated in zip(sec2_pts, sec2_pts_on_sec1):
                    from_x2, from_y2 = sec2_pt - template_side
                    to_x2, to_y2 = sec2_pt + template_side
                    bbox3 = tuple(np.array([from_x2, to_x2, from_y2, to_y2]) * scaling)
                    rect_res3 = sec2_tiles_rtree.search(bbox3)
                    for t_idx in rect_res3:
                        tile_res2.add(t_idx)

                    from_x1, from_y1 = sec1_pt_estimated - search_window_side
                    to_x1, to_y1 = sec1_pt_estimated + search_window_side
                    bbox4 = tuple(np.array([from_x1, to_x1, from_y1, to_y1]) * scaling)
                    rect_res4 = sec1_tiles_rtree.search(bbox4)
                    for t_idx in rect_res4:
                        tile_res1.add(t_idx)
        print('<=========== tile indices detection finished ===========>')

        # For each section, detect the features for each tile, and transform them to their stitched location (store in cache for future comparisons)
        logger.report_event("Computing per tile block features", log_level=logging.INFO)
        pool_results = []
        print("detector_type:",self._detector_type)
        time.sleep(5)
        for sec1_t_idx, t in enumerate(sec1.tiles()):
            k = "{}_t{}".format(sec1.canonical_section_name, sec1_t_idx)
            if sec1_t_idx in tile_res1 and k not in sec1_cache[FeaturesBlockMatcherDispatcher.BLOCK_FEATURES_KEY]:
                res = pool.apply_async(FeaturesBlockMatcherDispatcher.compute_features, \
                    (t, sec1_cache[FeaturesBlockMatcherDispatcher.BLOCK_FEATURES_KEY], k, self._detector_type, self._detector_kwargs, \
                    self._scaling, self._use_clahe))
                pool_results.append(res)
        
        for sec2_t_idx, t in enumerate(sec2.tiles()):
            k = "{}_t{}".format(sec2.canonical_section_name, sec2_t_idx)
            if sec2_t_idx in tile_res2 and k not in sec2_cache[FeaturesBlockMatcherDispatcher.BLOCK_FEATURES_KEY]:
                res = pool.apply_async(FeaturesBlockMatcherDispatcher.compute_features, \
                    (t, sec2_cache[FeaturesBlockMatcherDispatcher.BLOCK_FEATURES_KEY], k, self._detector_type, self._detector_kwargs, \
                    self._scaling, self._use_clahe))
                pool_results.append(res)
        
        for res in pool_results:
            res.get()
        print('\n','<=========== section({},{}): tile features detection finished({} tiles computed)===========>\n'.format(
            sec1.layer, sec2.layer, len(pool_results)))
        time.sleep(5)
        # Activate the actual matching
        sec1_to_sec2_results = [[], []]
        sec2_to_sec1_results = [[], []]
        pool_results = []
        for region1_key, sec1_region_mesh_pts in sec1_per_region_mesh_pts.items():
            sec2_mesh_pts_cur_sec1_region = sec2_per_region1_mesh_pts[region1_key]
            #sec1_sec2_mfov_matches, sec2_sec1_mfov_matches = BlockMatcherPMCCDispatcher._perform_matching(sec1_mfov_index, sec1, sec2, sec1_to_sec2_mfovs_transforms[sec1_mfov_index], sec1_mfov_mesh_pts, sec2_mesh_pts_cur_sec1_mfov, self._debug_dir, **self._matcher_kwargs)
            res_pool = pool.apply_async(FeaturesBlockMatcherDispatcher._perform_matching, (region1_key, sec1, sec2, \
                sec1_to_sec2_mfovs_transforms[region1_key[0]], sec1_cache[FeaturesBlockMatcherDispatcher.BLOCK_FEATURES_KEY], \
                sec2_cache[FeaturesBlockMatcherDispatcher.BLOCK_FEATURES_KEY], sec1_region_mesh_pts, sec2_mesh_pts_cur_sec1_region, \
                self._debug_dir, self._kwargs, self._detector_type, self._detector_kwargs, self._scaling, self._use_clahe))
            pool_results.append(res_pool)
        print("perform_match finished")
        time.sleep(5)

        for res_pool in pool_results:
            sec1_region_index, sec1_sec2_region_matches, sec2_sec1_region_matches = res_pool.get()
            if len(sec1_sec2_region_matches[0]) > 0:
                sec1_to_sec2_results[0].append(sec1_sec2_region_matches[0])
                sec1_to_sec2_results[1].append(sec1_sec2_region_matches[1])
            if len(sec2_sec1_region_matches[0]) > 0:
                sec2_to_sec1_results[0].append(sec2_sec1_region_matches[0])
                sec2_to_sec1_results[1].append(sec2_sec1_region_matches[1])
        
        print("perform_matching finished")
        time.sleep(5)
        if len(sec1_to_sec2_results[0]) == 0 or len(sec2_to_sec1_results[0]) == 0:
            return [], []
        return np.array([np.vstack(sec1_to_sec2_results[0]), np.vstack(sec1_to_sec2_results[1])]), \
               np.array([np.vstack(sec2_to_sec1_results[0]), np.vstack(sec2_to_sec1_results[1])])
