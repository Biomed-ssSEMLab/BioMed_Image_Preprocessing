import sys
import os.path
import os
import datetime
import time
from collections import defaultdict
import argparse
import glob
import ujson as json
import math
import tinyr
from rh_renderer.multiple_tiles_renderer import BlendType
import multiprocessing as mp
import acc_pmcc
import my_render_core
import common
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
import xlwt
from mb_aligner.alignment.aligner import StackAligner
import fs
import fs.path
from mb_aligner.dal.section import Section
from scipy.spatial import cKDTree as KDTree
from statistics import mean

from PIL import Image
from skimage import io
Image.MAX_IMAGE_PIXELS = None

def generate_hexagonal_grid(boundingbox, spacing, compare_radius):
    """Generates an hexagonal grid inside a given bounding-box with a given spacing between the vertices"""
    hexheight = spacing
    hexwidth = math.sqrt(3) * spacing / 2

    # for debug
    assert ( compare_radius < int(hexwidth/2) )

    vertspacing = 0.75 * hexheight
    horizspacing = hexwidth
    sizex = int((boundingbox[1] - boundingbox[0]) / horizspacing) 
    sizey = int(((boundingbox[3] - boundingbox[2]) - hexheight) / vertspacing) + 1
    # if sizey % 2 == 0:
    #     sizey += 1
    pointsret = []
    for i in range(0, sizex):
        for j in range(0, sizey):
            xpos = int(i * horizspacing + horizspacing/2)
            ypos = int(j * vertspacing + hexheight/2)
            if j % 2 == 1:
                xpos += int(horizspacing * 0.5)
            if (xpos>boundingbox[1]) or ((xpos+compare_radius) > boundingbox[1] ) or ((ypos+compare_radius) > boundingbox[3]):
                continue
            assert int(xpos + boundingbox[0]) < boundingbox[1]
            pointsret.append([int(xpos + boundingbox[0]), int(ypos + boundingbox[2])])
    print('\n\nboundingbox for mesh is:{}'.format(boundingbox))
    print('the last element of meshbox is:{}\n'.format(pointsret[-1]))
    return pointsret

def get_ts_files(ts_folder):

    with fs.open_fs(ts_folder) as cur_fs:
        all_ts_fnames = []
        all_ts_fnames_glob = cur_fs.glob("*.json")
        for ts_fname_glob in all_ts_fnames_glob:
            ts_fname = ts_fname_glob.path
            stitch_fname = cur_fs.geturl(ts_fname).split('//')[1]
            all_ts_fnames.append(stitch_fname)
        return all_ts_fnames


def read_layer_from_filename(fname):
    num = int(os.path.basename(fname)[:4])
    return num


def binary_search_layer(files, ts_to_layer, target_layer_num):
    first = 0
    last = len(files) - 1
    found = False

    while first <= last and not found:
        midpoint = (first + last) // 2
        mid_file = files[midpoint]
        if mid_file not in ts_to_layer:
            ts_to_layer[mid_file] = read_layer_from_filename(mid_file)
        mid_layer = ts_to_layer[mid_file]
        if mid_layer == target_layer_num:
            found = True
            return midpoint
        else:
            if target_layer_num < mid_layer:
                last = midpoint - 1
            else:
                first = midpoint + 1

    # The target layer was not found, returning the one before
    return first


def filter_files_layers(orig_files, from_layer, to_layer):
    filtered_files = sorted(orig_files)
    ts_to_layer = {}
    if from_layer != -1:
        # binary search for the initial layer
        ret_val = binary_search_layer(filtered_files, ts_to_layer, from_layer)
        # if ret_val is -1, need to use the first layer from the filtered_files
        if ret_val >= 0:

            if ts_to_layer[filtered_files[ret_val]] >= from_layer:
                filtered_files = filtered_files[ret_val:]
            else:
                filtered_files = filtered_files[ret_val + 1:]

    if to_layer != -1:
        # binary search for the last layer
        ret_val = binary_search_layer(filtered_files, ts_to_layer, to_layer)
        # if ret_val is bigger than length of the files, need to use all the filtered files
        if ret_val < len(filtered_files):

            if ts_to_layer[filtered_files[ret_val]] <= to_layer:
                filtered_files = filtered_files[:ret_val]
            else:
                filtered_files = filtered_files[:ret_val + 1]

    # print("filtered_files: {}".format(filtered_files))
    return filtered_files


def find_relevant_tiles(in_fname, tile_size, from_x, from_y, to_x, to_y):
    relevant_tiles = set()

    # load the tilespec, and each of the tiles bboxes into an rtree
    with open(in_fname, 'r') as f:
        tilespecs = json.load(f)

    tiles_rtree = RTree()
    for ts in tilespecs:
        bbox = ts["bbox"]
        # pyrtree uses the (x_min, y_min, x_max, y_max) notation
        tiles_rtree.insert(ts, Rect(int(bbox[0]), int(bbox[2]), int(bbox[1]), int(bbox[3])))

    # iterate through all the possible output tiles and include the ones that are overlapping with tiles from the tilespec
    for y_idx, cur_y in enumerate(range(from_y, to_y, tile_size)):
        for x_idx, cur_x in enumerate(range(from_x, to_x, tile_size)):
            rect_res = tiles_rtree.query_rect(Rect(cur_x, cur_y, cur_x + tile_size, cur_y + tile_size))
            for rtree_node in rect_res:
                if not rtree_node.is_leaf():
                    continue
                # found a tilespec that is in the output tile
                relevant_tiles.add((y_idx + 1, x_idx + 1))

    return relevant_tiles

def map_hist_adjuster_files(filtered_files, hist_adjuster_dir):
    if hist_adjuster_dir is None:
        return None

    ret = []

    hist_adjuster_files = glob.glob(os.path.join(hist_adjuster_dir, '*.pkl'))
    hist_adjuster_files_map = {os.path.basename(fname): fname for fname in hist_adjuster_files}

    for ts_fname in filtered_files:
        matching_base_fname = os.path.basename(ts_fname).replace('.json', '.pkl')  # change suffix to .pkl
        if matching_base_fname in hist_adjuster_files_map.keys():
            ret.append(hist_adjuster_files_map[matching_base_fname])
        else:
            matching_base_fname = '_'.join(matching_base_fname.split('_')[1:])  # remove the layer number
            if matching_base_fname in hist_adjuster_files_map.keys():
                ret.append(hist_adjuster_files_map[matching_base_fname])
            else:
                ret.append(None)
        print("Histogram adjuster file {} (for {})".format(ret[-1], os.path.basename(ts_fname)))

    return ret


def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)
    # common.fs_create_dir(path)

def set_style(name,height,bold=False):
    style = xlwt.XFStyle()
    font = xlwt.Font()
    font.name = name
    font.bold = bold
    font.color_index = 4
    font.height = height
    style.font = font
    return style

def in_bbox(inbox, mask, scale):
    mask_scaled = cv2.resize(mask, (int(mask.shape[1]/scale),int(mask.shape[0]/scale)))
    # area = mask[int(inbox[2]*scale):int(inbox[3]*scale),int(inbox[0]*scale):int(inbox[1]*scale)]
    area = mask_scaled[int(inbox[2]):int(inbox[3]),int(inbox[0]):int(inbox[1])]
    print(area)
    if (area == 255).all():
        return True
    return False


if __name__ == '__main__':
    # Command line parser
    parser = argparse.ArgumentParser(description='Renders a given set of images using the SLURM cluster commands.')
    parser.add_argument('--scale', type=int,
                        help='set the scale of the rendered images (default: full image)',
                        default=0.25)
    parser.add_argument('--from_layer', type=int,
                        help='the layer to start from (inclusive, default: the first layer in the data)',
                        default=-1)
    parser.add_argument('--to_layer', type=int,
                        help='the last layer to render (inclusive, default: the last layer in the data)',
                        default=-1)
    parser.add_argument('--hop', type=int,
                        help='the number of sections to skip between the range [from_layer, to_layer) (default: 1 - no skipped sections)',
                        default=1)
    parser.add_argument('--from_x', type=int,
                        help='the left coordinate (default: 0)',
                        default=0)
    parser.add_argument('--from_y', type=int,
                        help='the top coordinate (default: 0)',
                        default=0)
    parser.add_argument('--to_x', type=int,
                        help='the right coordinate (default: full image)',
                        default=-1)
    parser.add_argument('--to_y', type=int,
                        help='the bottom coordinate (default: full image)',
                        default=-1)
    parser.add_argument('--tile_size', type=int,
                        help='the size (square side) of each tile (default: 0 - whole image)',
                        default=1280000000)
    parser.add_argument('--output_type', type=str,
                        help='The output type format',
                        default='png')
    parser.add_argument('-t', '--threads_num', type=int,
                        help='the number of threads to use (default: 1) - not used at the moment',
                        default=12)
    parser.add_argument('-i', '--invert_image', action='store_true', default=True,
                        help='store an inverted image')
    parser.add_argument('--relevant_tiles_only', action='store_true', default=False,
                        help='if set, and a tile_size is specified, only stores tiles that are overlapping with the tilespec of each section')
    parser.add_argument('--bbox', type=str,
                        help='A pre-computed bbox if the entire dataset (optional, if not supplied will be calculated. Format: x_min,x_max,y_min,y_max)',
                        default=None)
    parser.add_argument('-e', '--empty_placeholder', action='store_true',
                        help='store an empty file name (suffix will be "_empty"), when the tile/image has no data')
    parser.add_argument('--hist_adjuster_alg_type', type=str,
                        help='the type of algorithm to use for a general per-tile histogram normalization. Supported types: CLAHE, GB11CLAHE (gaussian blur (11,11) and then clahe) (default: None)',
                        default='CLAHE')
    parser.add_argument('--per_job_cols_rows', type=str,
                        help='Only used when tile_size is set. The maximal number of columns and rows for a single job in the cluster, of the from "cols_num,rows_num".\
                        If not set, each job will render a single tile (default: None)',
                        default=None)
    parser.add_argument('--from_to_cols_rows', type=str,
                        help='Only to be used with tiled output (the tile_size argument is set). The input includes 4 numbers separated by commas, \
                        in the form "from_col,from_row,to_col,to_row" and only the output tiles in the given range (including from, excluding to) will be saved. (default: None)',
                        default=None)
    parser.add_argument('--blend_type', type=str,
                        help='The type of blending to use. Values = {} (default: MULTI_BAND_SEAM,NO_BLENDING,AVERAGING,LINEAR)'.format(
                            BlendType.__members__.keys()),
                        default='MULTI_BAND_SEAM')
    parser.add_argument('-p', '--processes_num', type=int,
                        help='the number of processes to use (default: 1)',
                        default=24)
    parser.add_argument('--index', type=str,
	                    default='1-1000')
    parser.add_argument('--mesh_spacing',type=int,help='mesh-spacing should be at least 2.5 times of compare_distance',
                        default=8000)
    parser.add_argument('--compare_radius',type=int,default=250)
    parser.add_argument('--variance',type=int,default=1500)
    parser.add_argument('--out_image',action='store_true',default=True)
    args = parser.parse_args()

    working_dir = '/braindat/lab/xzliu/output/output/acc_test/2/acc_100'

    img_out_dir = working_dir + '/render'
    if not os.path.exists(img_out_dir):
        os.makedirs(img_out_dir)
    tiles_dir = working_dir
    
    fname_from = int(args.index.split('-')[0])
    fname_to = int(args.index.split('-')[1])
    secs_ts_fnames = sorted(get_ts_files(tiles_dir))
    for ind, sec in enumerate(secs_ts_fnames):
        index_sec = int(os.path.basename(sec).split('Sec')[1].split('.')[0])
        if index_sec == fname_from:
            fname_from = ind+1   
        if index_sec == fname_to:
            fname_to = ind+1
            break
    secs_ts_fnames = [secs_ts_fnames[i] for i in range(fname_from-1,fname_to)]
    if len(secs_ts_fnames) == 0:
        print("No json files to render, quitting")
        sys.exit(1)
    print('\nprocessing sections: {}\n'.format(secs_ts_fnames))

    sections = []
    fnames = []
    for ts_fname in secs_ts_fnames:
        with open(ts_fname, 'rt') as in_f:
            tilespec = ujson.load(in_f)
        wafer_num = 1
        sec_num = int(os.path.basename(sec).split('Sec')[1].split('.')[0])
        sections.append(Section.create_from_tilespec(tilespec, wafer_section=(wafer_num, sec_num)))
        fnames.append(ts_fname)

    from_to_cols_rows = None
    args.blend_type = BlendType[args.blend_type]
    pool = mp.Pool(processes=args.processes_num)

    print("Computing entire 3d volume bounding box (including filtered due to layer# restrictions)")
    if args.bbox is None:
        entire_image_bbox = common.read_bboxes_grep_pool(secs_ts_fnames, pool)
    else:
        entire_image_bbox = [float(f) for f in args.bbox.split(',')]
    print("Final bbox for the 3d image: {}".format(entire_image_bbox))

    # Set the boundaries according to the entire_image_bbox
    if args.from_x == 0:
        args.from_x = int(math.floor(entire_image_bbox[0]))
    if args.from_y == 0:
        args.from_y = int(math.floor(entire_image_bbox[2]))
    if args.to_x == -1:
        args.to_x = int(math.ceil(entire_image_bbox[1]))
    if args.to_y == -1:
        args.to_y = int(math.ceil(entire_image_bbox[3]))

    inbbox = (args.from_x, args.to_x, args.from_y, args.to_y)
    if inbbox[1] == -1 or inbbox[3] == -1:
        image_bbox = common.read_bboxes_grep(args.tile_size)
        image_bbox[0] = max(image_bbox[0], inbbox[0])
        image_bbox[2] = max(image_bbox[2], inbbox[2])
        if inbbox[1] > 0:
            image_bbox[1] = inbbox[1]
        if inbbox[3] > 0:
            image_bbox[3] = inbbox[3]
    else:
        image_bbox = inbbox
    scaled_bbox = [
        int(math.floor(image_bbox[0] * args.scale)),
        int(math.ceil(image_bbox[1] * args.scale)),
        int(math.floor(image_bbox[2] * args.scale)),
        int(math.ceil(image_bbox[3] * args.scale))
    ]

    # Single image (no tiles)
    print("**********************\nRendering full section\n**********************")
    mask_dir = working_dir + '/mask'
    if not os.path.exists(mask_dir):
        os.makedirs(mask_dir)
    mask_path = mask_dir + '/mask_logical_and.png'
    if os.path.exists(mask_path):
        mask = io.imread(mask_path)
    else:
        imgs_job = []
        for in_fname in fnames:
            image_path = mask_dir+'/mask_'+ os.path.basename(in_fname).split('.')[0] +'.png'
            if os.path.exists(image_path):
                job_render = io.imread(image_path)
            else:
                # job_render = pool.apply_async(my_render_core.render_tilespec, (in_fname, working_dir, args.scale, [args.from_x, args.to_x, args.from_y, args.to_y],
                #                     args.threads_num, args.hist_adjuster_alg_type, from_to_cols_rows, args.blend_type))
            
                job_render = my_render_core.render_tilespec(in_fname, working_dir, args.scale, [args.from_x, args.to_x, args.from_y, args.to_y],
                                    args.threads_num, args.hist_adjuster_alg_type, from_to_cols_rows, args.blend_type)
            imgs_job.append(job_render)
    
        # pool.close()
        # pool.join()

        mask1 = imgs_job[0]
        mask2 = imgs_job[1]
        mask = np.logical_and(mask1,mask2).astype(np.uint8)
        del imgs_job
        mask[mask>0] = 255
        out = mask_dir+'/mask_logical_and.png'
        cv2.imwrite(out,mask)    

    # Render each section to fit the out_shape
    for fname_idx, f in  enumerate(fnames):
        if fname_idx+1 == len(fnames):
            break
        in_fname_compare = fnames[fname_idx + 1]
        in_fname  = [f,in_fname_compare]
        sec1 = sections[fname_idx]
        sec2 = sections[fname_idx + 1]
       
        # find the nearest transformations for mfovs1 that are missing in sec1_to_sec2_mfovs_transforms and for sec2 to sec1
        sec1_idxs_centers = [[], []] 
        for mfov1 in sec1.mfovs():
            mfov1_center = np.array([(mfov1.bbox[0] + mfov1.bbox[1])/2, (mfov1.bbox[2] + mfov1.bbox[3])/2])
            sec1_idxs_centers[0].append(mfov1.mfov_index)
            sec1_idxs_centers[1].append(mfov1_center)
        sec2_idxs_centers = [[], []] 
        for mfov2 in sec2.mfovs():
            mfov2_center = np.array([(mfov1.bbox[0] + mfov1.bbox[1])/2, (mfov1.bbox[2] + mfov1.bbox[3])/2])
            sec2_idxs_centers[0].append(mfov2.mfov_index)
            sec2_idxs_centers[1].append(mfov2_center)
        
        sec1_kdtree = KDTree(sec1_idxs_centers[1])
        overlap_mfovs_idxs = sec1_kdtree.query(sec2_idxs_centers[1])[1]
        overlap_mfovs = [sec1_idxs_centers[0][i] for i in overlap_mfovs_idxs]
        print('\noverlaped mfovs:{}\n'.format(overlap_mfovs))

        array = [[],[],[],[]]
        for mfov in sec1.mfovs():
            if mfov.mfov_index in overlap_mfovs:
                array[0].append(mfov.bbox[0])
                array[1].append(mfov.bbox[1])
                array[2].append(mfov.bbox[2])
                array[3].append(mfov.bbox[3])
        overlap_bbox = [min(array[0]), max(array[1]),min(array[2]), max(array[3])]
        print('\noverlapped-mfovs bounding box:{}\n'.format(overlap_bbox))
        print("Computing grid points and distributing work")
        # Lay a grid on top of each section
        sec1_mesh_pts = generate_hexagonal_grid(entire_image_bbox, args.mesh_spacing, args.compare_radius)

        mesh_bbox = []
        for sec1_pt in sec1_mesh_pts:
            sec1_bbox = [max(sec1_pt[0]-args.compare_radius,entire_image_bbox[0]),min(sec1_pt[0]+args.compare_radius,entire_image_bbox[1]),
                        max(sec1_pt[1]-args.compare_radius,entire_image_bbox[2]),min(sec1_pt[1]+args.compare_radius,entire_image_bbox[3])]
            mesh_bbox.append(sec1_bbox)


        results = {}
        results['dhash'] = []
        results['ncc'] = []
        pool_result = []
        for scaled_bbox in mesh_bbox:
            if in_bbox(scaled_bbox, mask, args.scale): 
                job_render = pool.apply_async(acc_pmcc.render_tilespec,(
                scaled_bbox, entire_image_bbox, sec1, sec2, in_fname, img_out_dir, args.scale, args.variance, 
                args.out_image, args.invert_image, dict(hist_adjuster_alg_type=args.hist_adjuster_alg_type,
                                                            blend_type=args.blend_type)))
                
                # job_render = acc_pmcc.render_tilespec(
                # scaled_bbox, entire_image_bbox, sec1, sec2, in_fname, img_out_dir, args.scale, args.variance,
                # args.out_image, args.invert_image, dict(hist_adjuster_alg_type=args.hist_adjuster_alg_type,
                #                                             blend_type=args.blend_type))
                
                pool_result.append(job_render)
                

        pool.close()
        pool.join()
    
    for job_render in pool_result:
        job_acc = job_render.get()
        # job_acc = job_render
        if job_acc[0] == True:
            results['dhash'].append(job_acc[1])
            results['ncc'].append(job_acc[2])    
    results['avg_dhash'] = mean(results['dhash'])
    results['avg_ncc'] = mean(results['ncc'])

    out_path = working_dir+'/acc.json'
    with open(out_path, 'wt') as out_f:
        json.dump(results, out_f, sort_keys=True, indent=4)

    del results
    print("All jobs finished, shutting down...")

