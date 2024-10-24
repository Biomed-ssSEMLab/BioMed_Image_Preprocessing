import argparse
import sys
from rh_logger.api import logger
import logging
import rh_logger
import os
import re
import glob
import common
import pickle
from mb_aligner.dal.section import Section
import fs
import multiprocessing as mp
from mb_aligner.stitching.stitcher import Stitcher

import time
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


def sec_dir_to_wafer_section(sec_dir, args_wafer_num=None):

    # wafer_folder = sec_dir.split(os.sep)[-4]
    # section_folder = sec_dir.split(os.sep)[-2]

    if args_wafer_num is None:
        m = re.match('.*[W|w]([0-9]+).*', wafer_folder)
        if m is None:
            raise Exception("Couldn't find wafer number from section directory {} (wafer dir is: {})".format(sec_dir, wafer_folder))
        wafer_num = int(m.group(1))
    else:
        wafer_num = args_wafer_num

    # m = re.match('.*_S([0-9]+)R1+.*', section_folder)
    # if m is None:
    #     raise Exception("Couldn't find section number from section directory {} (section dir is: {})".format(sec_dir, section_folder))
    # sec_num = int(m.group(1))


    section_folder = sec_dir.split(os.sep)[-2]
    sec_num = int(section_folder.split('_')[0])
    wafer_num = args_wafer_num

    return wafer_num, sec_num


def get_layer_num(sec_num, initial_layer_num, reverse, max_sec_num):
    if reverse:
        layer_num = max_sec_num - sec_num + initial_layer_num
    else:
        layer_num = sec_num + initial_layer_num - 1
    return layer_num


def parse_args(args=None):
    parser = argparse.ArgumentParser(
        description="Given a wafer's folder, searches for the recent sections, and creates a per-section tilespec file.")
    parser.add_argument("--wafer_folder", metavar="wafer_folder",
                        help="a folder of a single wafer containing workflow folders",
                        default='/media/liuxz/3EA0B4CEA0B48E41/mb_aligner/Experiment_0326_coatedTape_SS6_20210326_16-11-56')
                        # default='/media/liuxz/3EA0B4CEA0B48E41/mb_aligner/test')
    parser.add_argument("-o", "--output_dir", metavar="output_dir",
                        help="The output directory name, where each section folder will have a json tilespec there",
                        default="/media/liuxz/3EA0B4CEA0B48E41/mb_aligner/my_mb_aligner/output1/stitch1")
    parser.add_argument("-i", "--initial_layer_num", metavar="initial_layer_num", type=int,
                        help="The layer# of the first section in the list. (default: 1)",
                        default=1)
    parser.add_argument("-f", "--filtered_mfovs_pkl", metavar="filtered_mfovs_pkl", type=str,
                        help="The name of the pkl file that has a per section mfovs list (default: None)",
                        default=None)
    parser.add_argument("-w", "--wafer_num", metavar="wafer_num", type=int,
                        help="Manually set the wafer number for the output files (default: parse from wafer folder)",
                        default=1)
    parser.add_argument('--reverse', action='store_true',
                        help='reverse the section numbering (reversed filename lexicographical order)')
    parser.add_argument("-p", "--processes_num", metavar="processes_num", type=int,
                        help="The unmber of processes to use (default: 1)",
                        default=1)

    return parser.parse_args(args)

def create_and_save_single_section(sec_relevant_mfovs, sections_map_sec_num, layer_num, wafer_folder, out_ts_fname):

    cur_fs = fs.open_fs(wafer_folder)
    if isinstance(sections_map_sec_num, list):
        # TODO - not implemented yet
        section = Section.create_from_mfovs_image_coordinates(sections_map_sec_num, layer_num, cur_fs=cur_fs, relevant_mfovs=sec_relevant_mfovs)
    else:
        section = Section.create_from_full_image_coordinates(sections_map_sec_num, layer_num, cur_fs=cur_fs, relevant_mfovs=sec_relevant_mfovs)
    section.save_as_json(out_ts_fname)


def parse_filtered_mfovs(filtered_mfovs_pkl):
    with open(filtered_mfovs_pkl, 'rb') as in_f:
        data = pickle.load(in_f)
    filtered_mfovs_map = {}
    # map the filtered_mfovs_map and the sorted_sec_keys
    for k, v in data.items():
        wafer_num = int(k.split('_')[0][1:])
        section_num = int(k.split('_')[1][1:4])
#         v[0] = v[0].replace('\\', '/')
#         v[1] = set(int(mfov_num) for mfov_num in v[1])
#         filtered_mfovs_map[wafer_num, section_num] = v
        filtered_mfovs_map[wafer_num, section_num] = set(int(mfov_num) for mfov_num in v[1])



    return filtered_mfovs_map

def create_tilespecs(args):

    cur_fs = fs.open_fs(args.wafer_folder)

    # parse the workflows directory
    sections_map = common.parse_workflow_folder(cur_fs, args.wafer_folder)

    logger.report_event("Finished parsing sections", log_level=logging.INFO)

    sorted_sec_keys = sorted(list(sections_map.keys()))
    print(list(sections_map.keys()))
    if min(sorted_sec_keys) != 1:
        logger.report_event("Minimal section # found: {}".format(min(sorted_sec_keys)), log_level=logging.WARN)

    logger.report_event("Found {} sections in {}".format(len(sections_map), args.wafer_folder), log_level=logging.INFO)
    max_sec_num = max(sorted_sec_keys)
    if len(sorted_sec_keys) != max_sec_num:
        logger.report_event("There are {} sections, but maximal section # found: {}".format(len(sections_map), max(sorted_sec_keys)), log_level=logging.WARN)
        missing_sections = [i for i in range(1, max(sorted_sec_keys)) if i not in sections_map]
        logger.report_event("Missing sections: {}".format(missing_sections), log_level=logging.WARN)

    # if there's a filtered mfovs file, parse it
    filtered_mfovs_map = None
    if args.filtered_mfovs_pkl is not None:
        logger.report_event("Filtering sections mfovs", log_level=logging.INFO)
        filtered_mfovs_map = parse_filtered_mfovs(args.filtered_mfovs_pkl)

    logger.report_event("Outputing sections to tilespecs directory: {}".format(args.output_dir), log_level=logging.INFO)

    common.fs_create_dir(args.output_dir)

    pool = mp.Pool(processes=args.processes_num)

    pool_results = []



    # write a excel file
    f = xlwt.Workbook()
    sheet1 = f.add_sheet('stitch', cell_overwrite_ok=True)
    sheet1.write(0, 0, 'detector_type', set_style('Times New Roman', 220, True))
    sheet1.write(0, 1, conf.get('detector_type'), set_style('Times New Roman', 220, True))

    if conf.get('matcher_params')['model_index'] == 1:
        a = "Rigid"
    elif conf.get('matcher_params')['model_index'] == 0:
        a = "Translation"
    else:
        a = "Affine"
    pre_model = ["transformation_model", a]
    for i in range(0, 2):
        sheet1.write(1, i, pre_model[i], set_style('Times New Roman', 220, True))

    row0 = ["section" + str(sec_num) for sec_num in sorted_sec_keys]
    for i in range(0, len(row0)):
        sheet1.write(3, i+1, row0[i], set_style('Times New Roman', 220, True))
    sheet1.write(4, 0, 'time', set_style('Times New Roman', 220, True))
    sheet1.write(5, 0, 'sum_time', set_style('Times New Roman', 220, True))
    sum_time = 0.0
    num = 1

    for sec_num in sorted_sec_keys:
        # extract wafer and section# from directory name
        starttime = time.time()
        if isinstance(sections_map[sec_num], list):
            wafer_num, sec_num = sec_dir_to_wafer_section(os.path.dirname(sections_map[sec_num][0]), args.wafer_num)
        else:
            wafer_num, sec_num = sec_dir_to_wafer_section(sections_map[sec_num], args.wafer_num)
            # wafer_num, sec_num = 1, 1
        out_ts_fname = os.path.join(args.output_dir, 'W{}_Sec{}_montaged.json'.format(str(wafer_num).zfill(2), str(sec_num).zfill(3)))
        if os.path.exists(out_ts_fname):
            logger.report_event("Already found tilespec: {}, skipping".format(os.path.basename(out_ts_fname)), log_level=logging.INFO)
            continue
        layer_num = get_layer_num(sec_num, args.initial_layer_num, args.reverse, max_sec_num)

        sec_relevant_mfovs = None
        if filtered_mfovs_map is not None:
            if (wafer_num, sec_num) not in filtered_mfovs_map:
                logger.report_event("WARNING: cannot find filtered data for (wafer, sec): {}, skipping".format((wafer_num, sec_num)), log_level=logging.INFO)
                continue
            sec_relevant_mfovs = filtered_mfovs_map[wafer_num, sec_num]
        sub_folders = cur_fs.glob("*/")

        res = pool.apply_async(create_and_save_single_section, (sec_relevant_mfovs, sections_map[sec_num], layer_num, args.wafer_folder, out_ts_fname))
        pool_results.append(res)

        # 写出时间到Excel
        times =  time.time() - starttime
        sum_time += times
        sheet1.write(4, num, round((time.time() - starttime),3), set_style('Times New Roman', 220, True))
        num += 1

    sheet1.write(5, 1, round((sum_time),3), set_style('Times New Roman', 220, True))
    num = len(sorted_sec_keys)
    sheet1.write(6, 0, 'sections_num', set_style('Times New Roman', 220, True))
    sheet1.write(6, 1, num, set_style('Times New Roman', 220, True))
    sheet1.write(7, 0, 'average_time', set_style('Times New Roman', 220, True))
    sheet1.write(7, 1, round((sum_time/num),3), set_style('Times New Roman', 220, True))

    f.save('/media/liuxz/3EA0B4CEA0B48E41/mb_aligner/my_mb_aligner/output1/stitch1/stitch.xls')

    for res in pool_results:
        res.get()

if __name__ == '__main__':
    args = parse_args()

    logger.start_process('main', 'create_tilespecs.py', [args])
    conf_fname = '../conf/conf_example.yaml'
    conf = Stitcher.load_conf_from_file(conf_fname)
    create_tilespecs(args)
    logger.end_process('main ending', rh_logger.ExitCode(0))


