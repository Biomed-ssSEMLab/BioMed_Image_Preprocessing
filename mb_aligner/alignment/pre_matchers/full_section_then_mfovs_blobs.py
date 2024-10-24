# coding = UTF-8

import os
import cv2
import numpy as np
from mb_aligner.common.detectors.blob_detector_2d import BlobDetector2D
from mb_aligner.common.matcher import FeaturesMatcher
from mb_aligner.common.thread_local_storage_lru import ThreadLocalStorageLRU
import mb_aligner.common.ransac
import rh_logger
from rh_logger.api import logger
import logging
import tinyr
import time
from PIL import Image

class PreMatch3DFullSectionThenMfovsBlobs(object):
    """
    Performs a section to section pre-matching by detecting blobs in each section,
    then performing a global section matching, and then a per-mfov (of sec1) local refinement of the matches.
    """
    OVERLAP_DELTAS = np.array([[0, 0], [0, -10000], [-5000, 5000], [5000, 5000]]) # The location of the points relative to the center of an mfov

    def __init__(self, **kwargs):
        self._kwargs = kwargs
        self._blob_detector = BlobDetector2D.create_detector(**self._kwargs.get("blob_detector", {}))
        self._matcher = FeaturesMatcher(BlobDetector2D.create_matcher, **self._kwargs.get("matcher_params", {}))

    @staticmethod
    def detect_mfov_blobs(blob_detector_args, mfov, load):
        """
        Receives a tilespec of an mfov (all the tiles in that mfov),
        detects the blobs on each of the thumbnails of the mfov tiles,
        and returns the locations of the blobs (in stitched global coordinates), and their
        descriptors.
        """
        
        #wafer1+wafer2
        #print("mfov.class", type(mfov))
        print("start detect mfov blobs")
        time.sleep(5)
        for tile in mfov.tiles():
            num_wafer_add = 0
            if 'wafer2' in tile.img_fname:
                num_wafer_add = 557
            elif 'wafer3' in tile.img_fname:
                num_wafer_add = 1022
            elif 'wafer4' in tile.img_fname:
                num_wafer_add = 1552
            elif 'wafer5' in tile.img_fname:
                num_wafer_add = 2092
            elif 'wafer6' in tile.img_fname:
                num_wafer_add = 2648
            elif 'wafer7' in tile.img_fname:
                num_wafer_add = 3187
            mfov_layer = mfov.layer + num_wafer_add

        thread_local_store = ThreadLocalStorageLRU()
        if 'blob_detector' not in thread_local_store.keys():
            # Initialize the blob_detector, and store it in the local thread storage
            blob_detector = BlobDetector2D.create_detector(**blob_detector_args)
            thread_local_store['blob_detector'] = blob_detector
        else:
            blob_detector = thread_local_store['blob_detector']
        #         blob_detector = getattr(threadLocal, 'blob_detector', None)
        #         if blob_detector is None:
        #             # Initialize the blob_detector, and store it in the local thread storage
        #             blob_detector = BlobDetector2D.create_detector(**blob_detector_args)
        #             threadLocal.blob_detector = blob_detector

        ds_rate = blob_detector_args.get("ds_rate", None)

        all_kps_descs = [[], []]
        #feature_mfov_exists, mfov_result = load.load_prev_results('pre_matches', 'features_S{}M{}'.format(mfov.layer, mfov.mfov_index))
        # wafer1+ wafer2
        feature_mfov_exists, mfov_result = load.load_prev_results('pre_matches', 'features_S{}M{}'.format(mfov_layer, mfov.mfov_index))

        if feature_mfov_exists:
            all_kps_descs[0] = mfov_result['all_kps_descs0']
            all_kps_descs[1] = mfov_result['all_kps_descs1']
        else:
            for tile in mfov.tiles():
                
                
                img = tile.image
                '''
                if mfov_layer == 1679:
                    print("旋转90度")
                    time.sleep(5)
                    h,w = img.shape[:2]
                    rotate_center = (w/2, h/2)
                    M = cv2.getRotationMatrix2D(rotate_center, 90, 1.0)
                    new_w = int(h * np.abs(M[0, 1]) + w * np.abs(M[0, 0]))
                    new_h = int(h * np.abs(M[0, 0]) + w * np.abs(M[0, 1]))
                    M[0, 2] += (new_w - w) / 2
                    M[1, 2] += (new_h - h) / 2
                    img = cv2.warpAffine(img, M, (new_w, new_h))
                '''
                '''
                #rotate img before key point detect
                img = np.dot([[np.cos(-np.pi/2), -np.sin(-np.pi/2)],
                        [np.sin(-np.pi/2), np.cos(-np.pi/2)]],
                        img.T).T + np.asarray(np.array([0,0])).reshape((1, 2))
                print("rotate 90 before detect key point")
                time.sleep(5)
                '''
                if ds_rate is not None:
                    img = cv2.resize(img, None, fx=ds_rate, fy=ds_rate, interpolation=cv2.INTER_AREA)
                kps, descs = blob_detector.detectAndCompute(img)

                if len(kps) == 0:
                    continue

                kps_pts = np.empty((len(kps), 2), dtype=np.float64)
                for kp_i, kp in enumerate(kps):
                    kps_pts[kp_i][:] = kp.pt

                # upsample the thumbnail coordinates to original tile coordinates
                if ds_rate is not None:
                    kps_pts[:, 0] /= ds_rate
                    kps_pts[:, 1] /= ds_rate

                # Apply the transformation to the points
                assert(len(tile.transforms) == 1)
                model = tile.transforms[0]
                kps_pts = model.apply(kps_pts)

                all_kps_descs[0].extend(kps_pts)
                all_kps_descs[1].extend(descs)

            mfov_results = {'all_kps_descs0': all_kps_descs[0],
                            'all_kps_descs1': all_kps_descs[1]}
            #load.store_result('pre_matches', 'features_S{}M{}'.format(mfov.layer, mfov.mfov_index), mfov_results)
            #print("保存mfov特征: section{} mfov{}".format(mfov.layer, mfov.mfov_index))
            
            #wafer1+wafer2
            load.store_result('pre_matches', 'features_S{}M{}'.format(mfov_layer, mfov.mfov_index), mfov_results)
            print("保存mfov特征: section{} mfov{}".format(mfov_layer, mfov.mfov_index))



        return mfov.mfov_index, all_kps_descs

    def compute_section_blobs(self, sec, sec_cache, pool, load):
        # Create nested caches is needed
#        if "pre_match_blobs" not in sec_cache:
#            #sec_cache.create_dict("pre_match_blobs")
#            sec_cache["pre_match_blobs"] = {}

        total_features_num = 0
        # create the mfovs blob computation jobs
        async_results = []
        for mfov in sec.mfovs():
            if "pre_match_blobs/" + str(mfov.mfov_index) in sec_cache:
                continue
            res = pool.apply_async(PreMatch3DFullSectionThenMfovsBlobs.detect_mfov_blobs, (self._kwargs.get("blob_detector", {}), mfov, load))
            async_results.append(res)

        for res in async_results:
            mfov_index, mfov_kps_descs = res.get()
            #sec_cache["pre_match_blobs"].create_dict(mfov_index)
            sec_cache["pre_match_blobs/" + str(mfov_index)] = mfov_kps_descs
            print("Adding {} kps for mfov {}".format(len(mfov_kps_descs[0]), mfov_index))
            total_features_num += len(mfov_kps_descs[0])
        return total_features_num

    @staticmethod
    def collect_all_features(sec_cache):
        # TODO - need to see if pre-allocation can improve speed
        all_kps_arrays = [kps_descs[0] for key, kps_descs in sec_cache.items() if key.startswith("pre_match_blobs/") and len(kps_descs[0]) > 0]
        all_descs_arrays = [kps_descs[1] for key, kps_descs in sec_cache.items() if key.startswith("pre_match_blobs/") and len(kps_descs[1]) > 0]
        return np.vstack(all_kps_arrays), np.vstack(all_descs_arrays)

    @staticmethod
    def get_overlapping_mfovs(mfov1, sec2, sec1_to_sec2_model, sec2_rtree):
        # TODO - for single beam data, it might be better to take the boundaries of all tiles in mfov1,
        #        and return their overlapping mfovs on sec2
        # Take mfov1's center
        mfov1_center = np.array([(mfov1.bbox[0] + mfov1.bbox[1]) / 2, 
                                 (mfov1.bbox[2] + mfov1.bbox[3]) / 2])

        # Add the triangle points
        sec1_points = PreMatch3DFullSectionThenMfovsBlobs.OVERLAP_DELTAS + mfov1_center
        sec1_on_sec2_points = sec1_to_sec2_model.apply(sec1_points)
        overlapping_mfovs = set()
        for sec1_on_sec2_point in sec1_on_sec2_points:
            rect_res = sec2_rtree.search([sec1_on_sec2_point[0], sec1_on_sec2_point[0] + 1, sec1_on_sec2_point[1], sec1_on_sec2_point[1] + 1])
            for other_t in rect_res:
                overlapping_mfovs.add(other_t.mfov_index)

        # TODO:将mfov1的bbox的四条边的点和sec2_rtree做交集判断，然后得到mfov_index是否更合理？
        return overlapping_mfovs
        
    @staticmethod
    def match_mfovs_features(matcher_params, sec1_cache, sec2_cache, mfovs1, mfovs2):
        """
        Matches the features in mfovs1 (of sec1) to the features in mfovs2 (of sec2).
        This method is run by a process that loads the matcher from its local thread storage.
        """        
        thread_local_store = ThreadLocalStorageLRU()
        if 'matcher' in thread_local_store.keys():
            matcher = thread_local_store['matcher']
        else:
            # Initialize the matcher, and store it in the local thread storage
            matcher = FeaturesMatcher(BlobDetector2D.create_matcher, **matcher_params)
            thread_local_store['matcher'] = matcher
            
#         matcher = getattr(threadLocal, 'matcher', None)
#         if matcher is None:
#             # Initialize the matcher, and store it in the local thread storage
#             matcher = FeaturesMatcher(BlobDetector2D.create_matcher, **matcher_params)
#             threadLocal.matcher = matcher

        def get_kps_descs(mfovs, sec_cache):
            mfovs = list(mfovs)
            if len(mfovs) == 1:
                mfovs_kps = np.array(sec_cache["pre_match_blobs/" + str(mfovs[0])][0])
                mfovs_descs = np.array(sec_cache["pre_match_blobs/" + str(mfovs[0])][1])
            else:
                mfovs_kps_arrays = []
                mfovs_descs_arrays = []
                for mfov in mfovs:
                    kps_descs = sec_cache["pre_match_blobs/" + str(mfov)]
                    if len(kps_descs[0]) > 0:
                        mfovs_kps_arrays.append(kps_descs[0])
                        mfovs_descs_arrays.append(kps_descs[1])
                if len(mfovs_kps_arrays) == 0:
                    mfovs_kps = np.array([])
                    mfovs_descs = np.array([])
                elif len(mfovs_kps_arrays) == 1:
                    mfovs_kps = mfovs_kps_arrays[0]
                    mfovs_descs = mfovs_descs_arrays[0]
                else:
                    mfovs_kps = np.vstack(mfovs_kps_arrays)
                    mfovs_descs = np.vstack(mfovs_descs_arrays)
            return np.array(mfovs_kps), np.array(mfovs_descs)

        mfovs1_kps, mfovs1_descs = get_kps_descs(mfovs1, sec1_cache)
        mfovs2_kps, mfovs2_descs = get_kps_descs(mfovs2, sec2_cache)

        model, filtered_matches = matcher.match_and_filter(mfovs1_kps, mfovs1_descs, mfovs2_kps, mfovs2_descs)
        return mfovs1, model, filtered_matches

    def pre_match_sections(self, sec1, sec2, sec1_cache, sec2_cache, pool, load):
        """
        Performs a section to section pre-matching by detecting blobs in each section,
        then performing a global section matching, and then a per-mfov (of sec1) refinement of the matches.
        Returns a map between an mfov of sec1, and a tuple that holds its transformation model to sec2, and the filtered_matches
        """
        pre_match_res = {}

        # dispatch blob computation
        self.compute_section_blobs(sec1, sec1_cache, pool, load)
        self.compute_section_blobs(sec2, sec2_cache, pool, load)

        # compute a section to section global affine transform
        sec1_kps, sec1_descs = PreMatch3DFullSectionThenMfovsBlobs.collect_all_features(sec1_cache)
        sec2_kps, sec2_descs = PreMatch3DFullSectionThenMfovsBlobs.collect_all_features(sec2_cache)

        # global_model, global_filtered_matches = self._matcher.match_and_filter(sec1_kps, sec1_descs, sec2_kps, sec2_descs)
        global_model, global_filtered_matches = self._matcher.match_and_filter(sec1_kps, sec1_descs, sec2_kps, sec2_descs, sec1.layer, sec2.layer, load)
        if global_model is None:
            logger.report_event("No global model found between section {} (all mfovs) and section {} (all mfovs)".format(
                sec1.canonical_section_name, sec2.canonical_section_name), log_level=logging.WARNING)
            return None

        logger.report_event("Global model found between section {} (all mfovs) and section {} (all mfovs):\n{}".format(
            sec1.canonical_section_name, sec2.canonical_section_name, global_model.get_matrix()), log_level=logging.INFO)
        # print("DECOMPOSED MATRIX: ", mb_aligner.common.ransac.decompose_affine_matrix(global_model.get_matrix()))

        if sec1.mfovs_num == 1:
            logger.report_event("Section {} has a single mfov, using the global model between section {} and section {}:\n{}".format(
                sec1.canonical_section_name, sec1.canonical_section_name, sec2.canonical_section_name, global_model.get_matrix()), log_level=logging.INFO)            
            mfov_index = next(sec1.mfovs()).mfov_index
            pre_match_res[mfov_index] = (global_model, global_filtered_matches)
            return pre_match_res

        # Create section2 tile's bounding box rtree, so it would be faster to search it
        # TODO - maybe store it in cache, because it might be used by other comparisons of this section
        sec2_rtree = tinyr.RTree(interleaved=False, max_cap=5, min_cap=2)
        for t in sec2.tiles():
            sec2_rtree.insert(t, t.bbox)

        # refine the global transform to a local one
        async_results = []
        for mfov1 in sec1.mfovs():
            # find overlapping mfovs in sec2
            mfovs2 = PreMatch3DFullSectionThenMfovsBlobs.get_overlapping_mfovs(mfov1, sec2, global_model, sec2_rtree)
            logger.report_event("Finding local model between section {} (mfov {}) and section {} (mfovs {})".format(
                sec1.canonical_section_name, mfov1.mfov_index, sec2.canonical_section_name, mfovs2), log_level=logging.INFO)
            # Note - the match_mfovs_features only reads from secX_cache, so we can send secX_cache._dict (the manager's part of that)
            res = pool.apply_async(PreMatch3DFullSectionThenMfovsBlobs.match_mfovs_features, (self._kwargs.get("matcher_params", {}), 
            sec1_cache._dict, sec2_cache._dict, [mfov1.mfov_index], mfovs2))
            async_results.append(res)

        for res in async_results:
            mfovs1, mfovs1_model, mfovs1_filtered_matches = res.get()
            assert(len(mfovs1) == 1)
            mfov_index = mfovs1[0]

            if mfovs1_model is None:
                logger.report_event("No local model found between section {} (mfov {}) and section {}".format(
                    sec1.canonical_section_name, mfov_index, sec2.canonical_section_name), log_level=logging.INFO)
            else:
                logger.report_event("Found local model between section {} (mfov {}) and section {}:\n{}".format(
                    sec1.canonical_section_name, mfov_index, sec2.canonical_section_name, mfovs1_model.get_matrix()), log_level=logging.INFO)
                # print("DECOMPOSED MATRIX: ", mb_aligner.common.ransac.decompose_affine_matrix(mfovs1_model.get_matrix()))
            pre_match_res[mfov_index] = (mfovs1_model, mfovs1_filtered_matches)

        return pre_match_res

#     @staticmethod
#     def create(**params):
#         """
#         Returns the object of the pre-matcher, the classname, and a list of method names that are supported.
#         """
#         new_obj = PreMatch3DFullSectionThenMfovsBlobs(**params)
#         return [new_obj, new_obj.__classname__, ["_pre_match_compute_mfov_blobs", "_pre_match_sections_local"]]

TRIANGLE_DELTAS = np.array([[0, -10000], [-5000, 5000], [5000, 5000]]) # The location of the points relative to the center of an mfov

def find_triangle_angles(p1, p2, p3):
    A = p2 - p1
    B = p3 - p2
    C = p1 - p3

    angles = []
    for e1, e2 in ((A, -B), (B, -C), (C, -A)):
        num = np.dot(e1, e2)
        denom = np.linalg.norm(e1) * np.linalg.norm(e2)
        angles.append(np.arccos(num/denom) * 180 / np.pi)
    return angles

def triangles_similarity_compartor(tri_angles1, dist_thresh, center2, model2):
    #logger.report_event("RANSAC threshold met: checking triangles similarity", log_level=logging.INFO)
    tri_angles2 = find_triangle_angles(*model2.apply(center2 + TRIANGLE_DELTAS))
    dists = np.abs([a - b for a,b in zip(tri_angles1, tri_angles2)])
    if np.any(dists > dist_thresh):
        #logger.report_event("Similarity threshold wasn't met - initial transform angles: {}, current transform angles: {}".format(tri_angles1, tri_angles2), log_level=logging.INFO)
        return False
    return True
