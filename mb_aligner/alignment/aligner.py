import sys
# sys.path.append('.')
import os
import glob
import yaml
import argparse
import cv2
import ujson
import numpy as np
import time
import importlib
import gc
import tinyr
import functools
import xlwt
import fs
import fs.path
import multiprocessing as mp
from collections import defaultdict
import scripts.common
from scripts import file_utils
import logging
import rh_logger
from rh_logger.api import logger
from mb_aligner.dal.section import Section
from multiprocessing.pool import ThreadPool # for debug ?
from mb_aligner.common.section_cache import SectionCacheProcesses as SectionCache
from mb_aligner.alignment.mesh_pts_model_exporter import MeshPointsModelExporter
from mb_aligner.alignment.normalize_coordinates import normalize_coordinates
from mb_aligner.common.intermediate_results_dal_pickle import IntermediateResultsDALPickle
from mb_aligner.common.thread_local_storage_lru import ThreadLocalStorageLRU

def set_style(name,height,bold=False):
    style = xlwt.XFStyle()
    font = xlwt.Font()
    font.name = name
    font.bold = bold
    font.color_index = 4
    font.height = height
    style.font = font
    return style

class StackAligner(object):
    def __init__(self, conf, out_dir):
        self._conf = conf
        self._out_dir = out_dir
        self._create_directories()

        # self._processes_factory = ProcessesFactory(self._conf)
        if 'process_lru_size' in conf.keys():
            ThreadLocalStorageLRU.LRU_SIZE = conf.get('process_lru_size')
        self._processes_num = conf.get('processes_num', 1)
        assert(self._processes_num > 0)
        self._processes_pool = None
        self._restart_pool()

        self._continue_on_error = conf.get('continue_on_error', False)
        self._compare_distance = conf.get('compare_distance', 1)

        # Initialize the pre_matcher, block_matcher and optimization algorithms
        pre_match_type = conf.get('pre_match_type')
        pre_match_params = conf.get('pre_match_params', {})
        self._pre_matcher = StackAligner.load_plugin(pre_match_type)(**pre_match_params)

        fine_match_type = conf.get('fine_match_type', None)
        self._fine_matcher = None
        self._fine_matcher_filter = None
        if fine_match_type is not None:
            fine_match_params = conf.get('fine_match_params', {})
            self._fine_matcher = StackAligner.load_plugin(fine_match_type)(**fine_match_params)

            fine_match_filter_type = conf.get('fine_match_filter_type', None)
            if fine_match_filter_type is not None:
                fine_match_filter_params = conf.get('fine_match_filter_params', {})
                self._fine_matcher_filter = StackAligner.load_plugin(fine_match_filter_type)(**fine_match_filter_params)

        optimizer_type = conf.get('optimizer_type')
        optimizer_params = conf.get('optimizer_params', {})
        optimizer_params["checkpoints_dir"] = self._out_dir
        self._optimizer = StackAligner.load_plugin(optimizer_type)(**optimizer_params)

        self._inter_results_dal = IntermediateResultsDALPickle(self._out_dir)

    def __del__(self):
        self._processes_pool.close()
        self._processes_pool.join()

    def _restart_pool(self):
        if self._processes_pool is not None:
            self._processes_pool.close()
            self._processes_pool.join()
            self._processes_pool = None
        # self._processes_pool = ProcessPool(processes=self._processes_num)
        self._processes_pool = mp.Pool(processes=self._processes_num)

    def _create_directories(self):
        def create_dir(dir_name):
            if not os.path.exists(dir_name):
                os.makedirs(dir_name)

        self._pre_matches_dir = os.path.join(self._out_dir, 'pre_matches')
        self._fine_matches_dir = os.path.join(self._out_dir, 'fine_matches')
        self._fine_matches_flt_dir = os.path.join(self._out_dir, 'fine_matches_filtered')
        self._post_opt_dir = os.path.join(self._out_dir, 'post_align')  # 存放optimizer后的json
        self._output_dir = os.path.join(self._out_dir, 'json')  # 存放最终json
        create_dir(self._out_dir)
        create_dir(self._pre_matches_dir)  # TODO: 后面把成员变量传入给相应的模块输出目录变量中
        create_dir(self._fine_matches_dir)
        create_dir(self._fine_matches_flt_dir)
        create_dir(self._post_opt_dir)
        create_dir(self._output_dir)

    @staticmethod
    def _read_directory(in_dir):
        fnames_set = set(glob.glob(os.path.join(in_dir, '*')))
        return fnames_set

    @staticmethod
    def load_plugin(class_full_name):
        package, class_name = class_full_name.rsplit('.', 1)
        plugin_module = importlib.import_module(package)
        plugin_class = getattr(plugin_module, class_name)
        return plugin_class

    @staticmethod
    def read_imgs(folder):
        img_fnames = sorted(glob.glob(os.path.join(folder, '*')))[:10]
        print("Loading {} images from {}.".format(len(img_fnames), folder))
        imgs = [cv2.imread(img_fname, 0) for img_fname in img_fnames]
        return img_fnames, imgs

    @staticmethod
    def load_conf_from_file(conf_fname):
        '''
        Loads a given configuration file from a yaml file
        '''
        print("Using config file: {}.".format(conf_fname))
        if conf_fname is None:
            return {}
        with open(conf_fname, 'r') as stream:
            conf = yaml.load(stream, Loader=yaml.FullLoader)
            conf = conf['alignment']
        
        logger.report_event("loaded configuration: {}".format(conf), log_level=logging.INFO)
        return conf

    @staticmethod
    def _compute_l2_distance(pts1, pts2):
        delta = pts1 - pts2
        s = np.sum(delta**2, axis=1)
        return np.sqrt(s)

    @staticmethod
    def _create_section_rtree(section):
        '''
        Receives a section, and returns an rtree of all the section's tiles bounding boxes
        '''
        tiles_rtree = tinyr.RTree(interleaved=False, max_cap=5, min_cap=2)
        # Insert all tiles bounding boxes to an rtree
        for t in section.tiles():
            bbox = t.bbox  # x_min, x_max, y_min, y_max
            tiles_rtree.insert(t, bbox)
        return tiles_rtree

    @staticmethod
    def _find_overlapping_tiles_gen(section):
        '''
        Receives a section, and yields triplets of (tile1, tile2, overlap_bbox ())
        '''
        tiles_rtree = Stitcher._create_section_rtree(section)
        # Iterate over the section tiles, and for each tile find all of its overlapping tiles
        for t in section.tiles():
            bbox = t.bbox
            rect_res = tiles_rtree.search(bbox)
            for overlap_t in rect_res:
                # We want to create a directed comparison (each tile with tiles that come after it in a lexicographical order)
                if overlap_t.mfov_index > t.mfov_index or (overlap_t.mfov_index == t.mfov_index and overlap_t.tile_index > t.tile_index):
                    yield t, overlap_t

    def align_sections(self, sections):
        '''
        Receives a list of sections that were already stitched and need to be registered into a single volume.
        '''
        logger.report_event("align_sections starting.", log_level=logging.INFO)
        layout = {}
        layout['sections'] = sections
        layout['neighbors'] = defaultdict(set)

        # TODO - read the intermediate results directories (so we won't recompute steps)

        # write a excel file
        f = xlwt.Workbook()
        sheet1 = f.add_sheet('align', cell_overwrite_ok=True)

        if conf.get('pre_match_params')["matcher_params"]['model_index'] == 1:
            a = "Rigid"
        elif conf.get('pre_match_params')["matcher_params"]['model_index'] == 0:
            a = "Translation"
        else:
            a = "Affine"
        pre_model = ["pre_trans_model", a]
        for i in range(0, len(pre_model)):
            sheet1.write(0, i , pre_model[i], set_style('Times New Roman', 220, True))

        fine_type = ["fine_detector_type", conf.get('fine_match_params')["detector_type"]]
        for i in range(0, len(fine_type)):
            sheet1.write(1, i, fine_type[i], set_style('Times New Roman', 220, True))

        if conf.get('fine_match_params')["matcher_params"]["model_index"] == 1:
            b = "Rigid"
        elif conf.get('fine_match_params')["matcher_params"]["model_index"] == 0:
            b = "Translation"
        else:
            b = "Affine"
        fine_model = ["fine_trans_model", b]
        for i in range(0, len(fine_model)):
            sheet1.write(2, i, fine_model[i], set_style('Times New Roman', 220, True))

        row0 = ["pre-match","fine-match","fine-match filter","sum"]
        for i in range(0, len(row0)):
            sheet1.write(4, i + 1, row0[i], set_style('Times New Roman', 220, True))

        error_found = False
        # for each pair of neighboring sections (up to compare_distance distance)
        pre_match_results, fine_match_results = {}, {}
        sec_caches = defaultdict(SectionCache)
        prev_sec = None

        sum_pre, sum_fine, sum_filter = 0, 0, 0
        row = 5
        total_time0 = time.time()
        mystyle = set_style('Times New Roman', 220, True)

        for sec1_idx, sec1 in enumerate(sections):
            print("sec1_idx", sec1_idx)
            # No need to keep the caches of sec1, can free up some memory
            if prev_sec is not None and prev_sec.layer in sec_caches:
                del sec_caches[prev_sec.layer]
                gc.collect()
            prev_sec = sec1

            # Compare to neighboring sections
            for j in range(1, self._compare_distance + 1):
                sec2_idx = sec1_idx + j
                if sec2_idx >= len(sections):
                    break

                sec2 = sections[sec2_idx]
                sheet1.write(row, 0, "sec{} - sec{}".format(sec1_idx + 1, sec2_idx + 1), mystyle)

                # TODO - check if the pre-match was already computed
                logger.report_event("Performing pre-matching between sections {} and {}".format(sec1.layer, sec2.layer), log_level=logging.INFO)
                pre_time0 = time.time()
                prev_result_exists, prev_result = self._inter_results_dal.load_prev_results('pre_matches', '{}_{}'.format(
                    sec1.canonical_section_name_no_layer, sec2.canonical_section_name_no_layer))

                if prev_result_exists:
                    pre_match_results[sec1_idx, sec2_idx] = prev_result['contents']
                    print("prev_result_exists")
                else:
                    # Result will be a map between mfov index in sec1, and (the model and filtered matches to section 2)
                    print("prev_result_not_exists")
                    pre_match_results[sec1_idx, sec2_idx] = self._pre_matcher.pre_match_sections(sec1, sec2, \
                        sec_caches[sec1.layer], sec_caches[sec2.layer], self._processes_pool, self._inter_results_dal)
                    
                    # Make sure that there are pre-matches between the two sections
                    if pre_match_results[sec1_idx, sec2_idx] is None or len(pre_match_results[sec1_idx, sec2_idx]) == 0 or \
                        np.all([model is None for (model, _) in pre_match_results[sec1_idx, sec2_idx].values()]):
                
                        if self._continue_on_error:
                            error_found = True
                            print("error_found", 1)
                            logger.report_event("No pre-match between the sections {} and {} were found".format(
                                sec1.canonical_section_name_no_layer, sec2.canonical_section_name_no_layer), log_level=logging.ERROR)
                            print("sec1:", sec1.canonical_section_name_no_layer, sec1_idx)
                            print("sec2:", sec2.canonical_section_name_no_layer, sec2_idx)

                            continue
                        else:
                            raise Exception("No pre-match between the sections {} and {} were found".format(
                                sec1.canonical_section_name_no_layer, sec2.canonical_section_name_no_layer))
                    # Save the intermediate results
                    intermed_results = {'metadata':{'sec1':sec1.canonical_section_name_no_layer,
                                                    'sec2':sec2.canonical_section_name_no_layer},
                                        'contents':pre_match_results[sec1_idx, sec2_idx]}
                    print("save pre match results")
                    self._inter_results_dal.store_result('pre_matches', '{}_{}'.format(sec1.canonical_section_name_no_layer, \
                        sec2.canonical_section_name_no_layer), intermed_results)

                pre_time = time.time() - pre_time0
                sum_pre += pre_time
                sheet1.write(row, 1, round(pre_time, 3), mystyle)
        
                layout['neighbors'][sec1_idx].add(sec2_idx)
                layout['neighbors'][sec2_idx].add(sec1_idx)
                if self._fine_matcher is None:
                    # No block matching, use the pre-match results as bi-directional fine-matches
                    cur_matches = [filtered_matches for model, filtered_matches in \
                        pre_match_results[sec1_idx, sec2_idx].values() if filtered_matches is not None]
                    if len(cur_matches) == 1:
                        fine_match_results[sec1_idx, sec2_idx] = cur_matches[0]
                        fine_match_results[sec2_idx, sec1_idx] = [fine_match_results[sec1_idx, sec2_idx][1], fine_match_results[sec1_idx, sec2_idx][0]]
                    else:
                        fine_match_results[sec1_idx, sec2_idx] = np.concatenate(cur_matches, axis=1)
                        fine_match_results[sec2_idx, sec1_idx] = [fine_match_results[sec1_idx, sec2_idx][1], fine_match_results[sec1_idx, sec2_idx][0]]
                else:
                    # Perform block matching
                    # TODO - check if the fine-match was already computed
                    logger.report_event("Performing fine-matching between sections {} and {}".format(sec1.layer, sec2.layer), log_level=logging.INFO)

                    fine_time0 = time.time()
                    prev_result_exists, prev_result = self._inter_results_dal.load_prev_results('fine_matches', '{}_{}'.format(
                        sec1.canonical_section_name_no_layer, sec2.canonical_section_name_no_layer))
                    if prev_result_exists:
                        sec1_sec2_matches, sec2_sec1_matches = prev_result['contents']
                    else:
                        sec1_sec2_matches, sec2_sec1_matches = self._fine_matcher.match_layers_fine_matching(sec1, sec2, \
                            sec_caches[sec1.layer], sec_caches[sec2.layer], pre_match_results[sec1_idx, sec2_idx], self._processes_pool)

                        # make sure enough fine matches were found
                        if len(sec1_sec2_matches[0]) == 0 or len(sec2_sec1_matches[0]) == 0:
                            if self._continue_on_error:
                                error_found = True
                                print("error_found", 2 )
                                logger.report_event("No fine matches found between the sections {} and {} ({} in 1->2 and {} in 1<-2)".format(
                                    sec1.canonical_section_name_no_layer, sec2.canonical_section_name_no_layer, \
                                    len(sec1_sec2_matches[0]), len(sec2_sec1_matches[0])), log_level=logging.ERROR)
                                continue
                            else:
                                raise Exception("No fine matches found between the sections {} and {} ({} in 1->2 and {} in 1<-2)".format(
                                    sec1.canonical_section_name_no_layer, sec2.canonical_section_name_no_layer, \
                                    len(sec1_sec2_matches[0]), len(sec2_sec1_matches[0])))

                        intermed_results = {'metadata':{'sec1':sec1.canonical_section_name_no_layer,
                                                        'sec2':sec2.canonical_section_name_no_layer},
                                            'contents':[sec1_sec2_matches, sec2_sec1_matches]}
 
                        self._inter_results_dal.store_result('fine_matches', '{}_{}'.format(
                            sec1.canonical_section_name_no_layer, sec2.canonical_section_name_no_layer), intermed_results)
                    logger.report_event("fine-matching between sections {0} and {1} results: {0}->{1} {2} matches, {0}<-{1} {3} matches ".format(
                        sec1.layer, sec2.layer, len(sec1_sec2_matches[0]), len(sec2_sec1_matches[0])), log_level=logging.INFO)
                    fine_match_results[sec1_idx, sec2_idx] = sec1_sec2_matches
                    fine_match_results[sec2_idx, sec1_idx] = sec2_sec1_matches

                    fine_time = time.time() - fine_time0
                    sum_fine += fine_time
                    sheet1.write(row, 2, round(fine_time, 3), mystyle)

                    if self._fine_matcher_filter is not None:
                        logger.report_event("Performing fine-matching filter between sections {} and {}".format(sec1.layer, sec2.layer), log_level=logging.INFO)
                        
                        filter_time0 = time.time()
                        prev_result_exists, prev_result = self._inter_results_dal.load_prev_results('fine_matches_filtered', '{}_{}'.format(
                            sec1.canonical_section_name_no_layer, sec2.canonical_section_name_no_layer))
                        if prev_result_exists:
                            sec1_sec2_matches_filtered, sec2_sec1_matches_filtered = prev_result['contents']
                        else:
                            sec1_sec2_matches_filtered = self._fine_matcher_filter.filter_matches(fine_match_results[sec1_idx, sec2_idx], self._processes_pool)
                            sec2_sec1_matches_filtered = self._fine_matcher_filter.filter_matches(fine_match_results[sec2_idx, sec1_idx], self._processes_pool)

                            # make sure enough fine matches were found after filter
                            if len(sec1_sec2_matches_filtered[0]) == 0 or len(sec2_sec1_matches_filtered[0]) == 0:
                                if self._continue_on_error:
                                    error_found = True
                                    print("error_found", 3)
                                    logger.report_event("No post-filter fine matches found between the sections {} and {} ({} in 1->2 and {} in 1<-2)".format(
                                        sec1.canonical_section_name_no_layer, sec2.canonical_section_name_no_layer, \
                                        len(sec1_sec2_matches_filtered[0]), len(sec2_sec1_matches_filtered[0])), log_level=logging.ERROR)
                                    continue
                                else:
                                    raise Exception("No post-filter fine matches found between the sections {} and {} ({} in 1->2 and {} in 1<-2)".format(
                                        sec1.canonical_section_name_no_layer, sec2.canonical_section_name_no_layer, \
                                        len(sec1_sec2_matches_filtered[0]), len(sec2_sec1_matches_filtered[0])))

                            intermed_results = {'metadata':{'sec1':sec1.canonical_section_name_no_layer,
                                                            'sec2':sec2.canonical_section_name_no_layer},
                                                'contents':[sec1_sec2_matches_filtered, sec2_sec1_matches_filtered]}
     
                            self._inter_results_dal.store_result('fine_matches_filtered', '{}_{}'.format(
                                sec1.canonical_section_name_no_layer, sec2.canonical_section_name_no_layer), intermed_results)

                        # 分别为 match_layers_fine_matching 的前两个参数输出
                        # fine match results； section1 to section2 ; section2 to section1
                        fine_match_results[sec1_idx, sec2_idx] = sec1_sec2_matches_filtered
                        fine_match_results[sec2_idx, sec1_idx] = sec2_sec1_matches_filtered
                        logger.report_event("fine-matching-filter between sections {0} and {1} results: {0}->{1} {2} matches, {0}<-{1} {3} matches ".format(
                            sec1.layer, sec2.layer, len(sec1_sec2_matches_filtered[0]), len(sec2_sec1_matches_filtered[0])), log_level=logging.INFO)

                        filter_time = time.time() - filter_time0
                        sum_filter += filter_time
                        # the excel is too big
                        #mystyle = set_style('Times New Roman', 220, True)
                        sheet1.write(row, 3, round(filter_time, 3), mystyle)

                        #sheet1.write(row, 3, round(filter_time, 3), set_style('Times New Roman', 220, True))
                        sum_time = fine_time + filter_time + pre_time
                        sheet1.write(row, 4, round(sum_time, 3), mystyle)
                        #sheet1.write(row, 4, round(sum_time, 3), set_style('Times New Roman', 220, True))
                row += 1
        
        total_time = time.time() - total_time0
        sheet1.write(row, 0, 'sum', mystyle)
        sheet1.write(row, 1, round(sum_pre, 3), mystyle)
        sheet1.write(row, 2, round(sum_fine, 3), mystyle)
        sheet1.write(row, 3, round(sum_filter, 3), mystyle)
        sheet1.write(row, 4, round(total_time, 3), mystyle)
        #sheet1.write(row, 0, 'sum', set_style('Times New Roman', 220, True))
        #sheet1.write(row, 1, round(sum_pre, 3), set_style('Times New Roman', 220, True))
        #sheet1.write(row, 2, round(sum_fine, 3), set_style('Times New Roman', 220, True))
        #sheet1.write(row, 3, round(sum_filter, 3), set_style('Times New Roman', 220, True))
        #sheet1.write(row, 4, round(total_time, 3), set_style('Times New Roman', 220, True))
        num = row - 5

        if error_found:
            raise Exception("Cannot run optimization, because an error occured previously but was skipped")

        # optimize the matches
        logger.report_event("Optimizing the matches...", log_level=logging.INFO)

        row += 2
        sheet1.write(row, 0, 'time-optimizing', set_style('Times New Roman', 220, True))
        opt0_time = time.time()

        # 定义偏函数 functial
        update_section_post_optimization_and_save_partial = functools.partial(update_section_post_optimization_and_save, out_dir=self._post_opt_dir)
        # self._optimizer.optimize(layout, fine_match_results, lambda section, orig_pts, new_pts, mesh_spacing: update_section_post_optimization_and_save(section, orig_pts, new_pts, mesh_spacing, self._post_opt_dir), self._processes_pool)
        print("optimize method:", self._optimizer.optimize)
        time.sleep(5)
        self._optimizer.optimize(layout, fine_match_results, update_section_post_optimization_and_save_partial, self._processes_pool)
        
        print("optimize finished")
        time.sleep(5)

        # TODO - normalize all the sections (shift everything so we'll have a (0, 0) coordinate system for the stack)
        normalize_coordinates([self._post_opt_dir], self._output_dir, self._processes_pool)
       
        opt_time  = time.time() - opt0_time
        sheet1.write(row, 1, round(opt_time,3), set_style('Times New Roman', 220, True))
        sheet1.write(row+2, 0, 'total_time', set_style('Times New Roman', 220, True))
        sheet1.write(row+2, 1, round(total_time+opt_time,3), set_style('Times New Roman', 220, True))
        sheet1.write(row+2, 1, round(total_time+opt_time,3), set_style('Times New Roman', 220, True))
        sheet1.write(row+3, 0, 'align_num', set_style('Times New Roman', 220, True))
        sheet1.write(row+3, 1, num, set_style('Times New Roman', 220, True))
        sheet1.write(row+4, 0, 'average_time', set_style('Times New Roman', 220, True))
        sheet1.write(row+4, 1, round((total_time+opt_time)/num,3), set_style('Times New Roman', 220, True))
        f.save(str(self._out_dir)+'/align.xls')

def update_section_post_optimization_and_save(section, orig_pts, new_pts, mesh_spacing, out_dir):
    logger.report_event("Exporting section {}".format(section.canonical_section_name), log_level=logging.INFO)
    print("start pointmodel export:")
    time.sleep(5)
    exporter = MeshPointsModelExporter()
    exporter.update_section_points_model_transform(section, orig_pts, new_pts, mesh_spacing)

    # TODO - should also save the mesh as h5s

    # save the output section
    out_fname = os.path.join(out_dir, '{}.json'.format(section.canonical_section_name))
    print('Writing output to: {}'.format(out_fname))
    section.save_as_json(out_fname)

def check_args(args):
    args.in_dir = file_utils.get_abs_dir(args.in_dir)
    args.out_dir = file_utils.create_dir(args.out_dir)
    all_json_files = scripts.common.get_ts_files(args.in_dir)
    if len(all_json_files) == 0:
        print("No json files to align, quitting...")
        sys.exit(1)

    all_layer_idx = scripts.common.get_all_layer_idx(all_json_files)
    all_layer_idx = sorted(all_layer_idx)

    args.start_section_idx = max(args.start_section_idx, all_layer_idx[0])
    args.end_section_idx = all_layer_idx[-1] if args.end_section_idx == -1 else min(args.end_section_idx, all_layer_idx[-1])

    if args.start_section_idx > all_layer_idx[-1] or args.start_section_idx > args.end_section_idx:
        raise Exception("Error! args.start_section_idx {} greater than max_sec_idx {} or greater than args.end_section_idx {}.".format(
            args.start_section_idx, all_layer_idx[-1], args.end_section_idx))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='This is a large scale electron microscope data alignment scripy.')
    parser.add_argument("-i", "--in_dir", type=str, default="/media/hqjin/Elements/em_data/Experiment_20220508_18-33-18/out/stitch/json", help="stitch json dir")
    parser.add_argument("-o", "--out_dir", type=str, default="/media/hqjin/Elements/em_data/Experiment_20220508_18-33-18/out/align", help="output dir")
    parser.add_argument("-s", "--start_section_idx", type=int, default=1, help="section start index to align")
    parser.add_argument("-e", "--end_section_idx", type=int, default=-1, help="section end index to align, -1 means all section")
    args = parser.parse_args()

    # set the directory where the current script is located as the running directory
    os.chdir(os.path.dirname(os.path.realpath(__file__)))

    check_args(args)
    start_time = time.time()

    all_files = scripts.common.get_ts_files(args.in_dir)
    all_files = [f.replace("file://", "") for f in all_files]
    filtered_layers, filtered_layers_idx = scripts.common.filter_files_layers(all_files, args.start_section_idx, args.end_section_idx)

    # config file
    conf_fname = '../../conf/conf_example.yaml'

    logger.start_process('main', 'aligner.py', [filtered_layers, conf_fname])
    conf = StackAligner.load_conf_from_file(conf_fname)
    logger.report_event("Initializing aligner", log_level=logging.INFO)
    aligner = StackAligner(conf, args.out_dir)

    logger.report_event("Loading sections", log_level=logging.INFO)
    sections = []
    # TODO - Should be done in a parallel fashion
    for sec_fname in filtered_layers:
        with open(sec_fname, 'rt') as in_f:
            tilespec = ujson.load(in_f)
        wafer_num = 1
        sec_num = int(os.path.basename(sec_fname).split('S')[1].split('M')[0])
        if sec_num > 557 and sec_num <= 1022:
            wafer_num = 2
        elif sec_num > 1022 and sec_num <= 1552:
            wafer_num = 3
        elif sec_num > 1552 and sec_num <= 2092:
            wafer_num = 4
        elif sec_num > 2092 and sec_num <= 2648:
            wafer_num = 5
        elif sec_num > 2648 and sec_num <= 3187:
            wafer_num = 6
        elif sec_num > 3187:
            wafer_num = 7
        print("sec_num", sec_num)
        sections.append(Section.create_from_tilespec(tilespec, wafer_section=(wafer_num, sec_num)))
    

    logger.report_event("Aligning sections", log_level=logging.INFO)
    sections.sort(key=lambda c: c.layer, reverse=False)
    aligner.align_sections(sections) # will align and update the section tiles' transformations

    del aligner
    logger.end_process('main ending', rh_logger.ExitCode(0))
    print(f'Success! Total time: {(time.time() - start_time) / 3600:.4f}h')
