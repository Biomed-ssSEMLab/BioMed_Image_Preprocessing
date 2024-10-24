#
# Executes the rendering process on the 3d output transformations
# It takes a collection of tilespec files, each describing a montage of a single section,
# where all sections are aligned to a single coordinate system,
# and outputs a per-section image (or images/tiles).
# The input is a directory with tilespec files in json format (each file for a single layer).
# For tiles output, splits the image into tiles, and renders each in a separate process
#
import sys
import os
import os.path
import numpy as np
import time
import argparse
import math
import ujson
import multiprocessing as mp

import render_core
import common
import file_utils
from rh_renderer.multiple_tiles_renderer import BlendType
from rh_renderer.tilespec_renderer import TilespecRenderer
from rh_renderer import models
from rh_renderer.normalization.histogram_clahe import HistogramCLAHE, HistogramGB11CLAHE

def check_args(args):
    args.in_dir = file_utils.get_abs_dir(args.in_dir)
    args.out_dir = file_utils.create_dir(args.out_dir)

    if args.scale > 1 or args.scale <= 0:
        raise Exception("Error! args.scale {} should be in (0, 1].".format(args.scale))

    args.tile_size = np.iinfo(np.int32).max if args.tile_size == -1 else args.tile_size
    if args.tile_size <= 0:
        raise Exception("Error! args.tile_size {} should be greater than 0.".format(args.tile_size))
    elif args.tile_size != np.iinfo(np.int32).max and args.tile_size % 128 != 0:
        raise Exception("Error! args.tile_size {} should be {} or a multiple of 128.".format(args.tile_size, np.iinfo(np.int32).max))

    all_json_files = common.get_ts_files(args.in_dir)
    if len(all_json_files) == 0:
        print("No json files to render, quitting...")
        sys.exit(1)

    all_layer_idx = common.get_all_layer_idx(all_json_files)
    all_layer_idx = sorted(all_layer_idx)

    args.from_layer = max(args.from_layer, all_layer_idx[0])
    args.to_layer = all_layer_idx[-1] if args.to_layer == -1 else args.to_layer
    args.to_layer = min(args.to_layer, all_layer_idx[-1])

    if args.from_layer > all_layer_idx[-1] or args.from_layer > args.to_layer:
        raise Exception("Error! args.from_layer {} greater than all_layer_idx[-1] {} or args.to_layer {}.".format(args.from_layer, all_layer_idx[-1], args.to_layer))

    if args.keep_same_dimension != 0 and args.keep_same_dimension != 1:
        raise Exception("Error! args.keep_same_dimension {} should be 0 or 1.".format(args.keep_same_dimension))

    if args.hist_adjuster_alg_type not in ["CLAHE", "GB11CLAHE"]:
        raise Exception("Error! args.hist_adjuster_alg_type {} should be \"CLAHE\" or \"GB11CLAHE\".".format(args.hist_adjuster_alg_type))

    if args.blend_type not in ["NO_BLENDING", "AVERAGING", "LINEAR", "MULTI_BAND_SEAM"]:
        raise Exception("Error! args.blend_type {} should be \"NO_BLENDING\", \"AVERAGING\", \"LINEAR\" or \"MULTI_BAND_SEAM\".".format(args.blend_type))

    if args.output_type.lower() not in ["bmp", "jpg", "jpeg", "png"]:
        raise Exception("Error! args.output_type {} should be \"bmp\", \"jpg\", \"jpeg\" or \"png\".".format(args.output_type))

    args.process_num = os.cpu_count() if args.process_num == -1 else max(args.process_num, 1)

def render(tilespec, hist_adjuster, scale, blend_type, from_x, to_x, from_y, to_y, max_row, max_col, tile_size, actual_tile_size, \
    in_fname, out_fname_prefix, output_type, invert_image):
    renderer = TilespecRenderer(tilespec, hist_adjuster=hist_adjuster, dynamic=(scale != 1.0), blend_type=blend_type)
    print("hist_adjuster_type:", hist_adjuster)
    if scale != 1.0:
        downsample = models.AffineModel(np.array([[scale, 0., 0.],
                                                  [0., scale, 0.],
                                                  [0., 0., 1.]]))
        renderer.add_transformation(downsample)

    # Iterate over each row and column and save the tile
    for cur_row in range(max_row):
        tmp_from_y = from_y + cur_row * actual_tile_size
        tmp_to_y = min(from_y + (cur_row + 1) * actual_tile_size, to_y)

        for cur_col in range(max_col):
            out_fname = "{}_tr{}-tc{}.{}".format(out_fname_prefix, cur_row + 1, cur_col + 1, output_type)
            
            print("out_fname:", out_fname)

            if not os.path.exists(out_fname):
                print("out_fname not exist")
                tmp_from_x = from_x + cur_col * actual_tile_size
                tmp_to_x = min(from_x + (cur_col + 1) * actual_tile_size, to_x)

                render_core.render_tilespec(max_row, max_col, in_fname, renderer, out_fname, scale, output_type, \
                    [tmp_from_x, tmp_to_x, tmp_from_y, tmp_to_y], tile_size, invert_image)
    
            print("out_fname exist")
    del renderer

def async_render(args):
    start_time = time.time()
    check_args(args)

    args.blend_type = BlendType[args.blend_type]
    if args.hist_adjuster_alg_type == "CLAHE":
        hist_adjuster = HistogramCLAHE()
    elif args.hist_adjuster_alg_type == "GB11CLAHE":
        hist_adjuster = HistogramGB11CLAHE()

    all_files = common.get_ts_files(args.in_dir)
    all_files = ['/' + file.split('///')[1] for file in all_files]

    print("Computing entire 3d volume bounding box (including filtered due to layer# restrictions)")
    # Compute the width and height of the entire 3d volume
    # so all images will have the same dimensions (needed for many image viewing applications, e.g., Fiji)
    entire_image_bbox = common.read_bboxes(all_files)
    
    print("Final bbox for the 3d image: {}".format(entire_image_bbox))
    time.sleep(5)
    ''' 
    entire_image_bbox[0] = 60000
    entire_image_bbox[2] = 60000
    entire_image_bbox[1] = 64096
    entire_image_bbox[3] = 64096
    print("the region select is done: {}".format(entire_image_bbox))
    '''
    '''
    entire_image_bbox[0] = 60000
    entire_image_bbox[2] = 60000
    entire_image_bbox[1] = 70000
    entire_image_bbox[3] = 68192
    print("the region select is done: {}".format(entire_image_bbox))
    '''
    '''
    entire_image_bbox[0] = 60000
    entire_image_bbox[2] = 60000
    entire_image_bbox[1] = 72000
    entire_image_bbox[3] = 72000
    print("the region select is done: {}".format(entire_image_bbox))
    '''
    '''
    #N92 stitching render part
    entire_image_bbox[0] = 229376
    entire_image_bbox[2] = 237568
    entire_image_bbox[1] = 245760
    entire_image_bbox[3] = 253952
    '''

    '''    
    #N92 stitching render block part
    entire_image_bbox[0] = 204800
    entire_image_bbox[2] = 578600
    entire_image_bbox[1] = 268000
    entire_image_bbox[3] = 642600
    '''
    '''
    #S1585 stitching rendder block part
    entire_image_bbox[0] = 49000
    entire_image_bbox[2] = 57000
    entire_image_bbox[1] = 53000
    entire_image_bbox[3] = 61000
    print("the region select is done: {}".format(entire_image_bbox))
    '''
    '''
    #RB706 stitching render part
    entire_image_bbox[0] = 256000
    entire_image_bbox[2] = 960000
    entire_image_bbox[1] = 272384
    entire_image_bbox[3] = 976384
    '''

    # Set the boundaries according to the entire_image_bbox
    from_x = int(math.floor(entire_image_bbox[0]))
    from_y = int(math.floor(entire_image_bbox[2]))
    to_x = int(math.ceil(entire_image_bbox[1]))
    to_y = int(math.ceil(entire_image_bbox[3]))

    # Set the max_col and max_row (in case of rendering tiles)
    actual_tile_size = int(math.ceil(args.tile_size / args.scale))
    max_col = int(math.ceil((to_x - from_x) / float(actual_tile_size)))
    max_row = int(math.ceil((to_y - from_y) / float(actual_tile_size)))

    filtered_files, filtered_layers_idx = common.filter_files_layers(all_files, args.from_layer, args.to_layer)

    print("SELECT REGION")
    time.sleep(5)

    pool = mp.Pool(processes=args.process_num)  # Create the pool of processes
    res_l = []
    for in_fname, layer_idx in zip(filtered_files, filtered_layers_idx):
        if args.keep_same_dimension == 0:
            entire_image_bbox = common.read_bboxes([in_fname])
            from_x = int(math.floor(entire_image_bbox[0]))
            from_y = int(math.floor(entire_image_bbox[2]))
            to_x = int(math.ceil(entire_image_bbox[1]))
            to_y = int(math.ceil(entire_image_bbox[3]))
            max_col = int(math.ceil((to_x - from_x) / float(actual_tile_size)))
            max_row = int(math.ceil((to_y - from_y) / float(actual_tile_size)))


        cur_out_dir = os.path.join(args.out_dir, str(layer_idx).zfill(3))
        if not os.path.exists(cur_out_dir):
            os.makedirs(cur_out_dir)
        out_fname_prefix = os.path.join(cur_out_dir, os.path.basename(in_fname).rsplit(".")[0])

        
        with open(in_fname, 'r') as data:
            tilespec = ujson.load(data)
        
        print("start res")
        res = pool.apply_async(render, (tilespec, hist_adjuster, args.scale, args.blend_type, \
            from_x, to_x, from_y, to_y, max_row, max_col, args.tile_size, actual_tile_size, \
            in_fname, out_fname_prefix, args.output_type, args.invert_image))
        res_l.append(res)

    pool.close()
    pool.join()

    for i in res_l:
        res = i.get()
    print(f'Success! Total time: {time.time() - start_time:.2f}s')

def main():
    parser = argparse.ArgumentParser(description='Renders a given set of images using the SLURM cluster commands.')
    parser.add_argument("--in_dir", type=str, default="/media/hqjin/Elements/em_data/Experiment_20220508_18-33-18/out/stitch/json", help="json dir")
    parser.add_argument("--out_dir", type=str, default="/media/hqjin/Elements/em_data/Experiment_20220508_18-33-18/out/stitch/img_8nm", help="output dir")
    parser.add_argument("--scale", type=float, default=0.5, help="set the scale of the rendered images (default: full image)")
    parser.add_argument("--tile_size", type=int, default=-1, help="the size (square side) of each tile, -1 means the whole image. (Set the value to a multiple of 128(for neuroglancer display) or -1)")
    parser.add_argument("--from_layer", type=int, default=1, help="the layer to start from")
    parser.add_argument("--to_layer", type=int, default=-1, help="the last layer to render, -1 means the last of json list")
    parser.add_argument("--keep_same_dimension", type=int, default=1, choices=[0, 1], help="all save images whether have the same dimensions, 0 means not, 1(default) means yes")
    parser.add_argument("--hist_adjuster_alg_type", type=str, default="GB11CLAHE", choices=["CLAHE", "GB11CLAHE"], 
                        help="the type of algorithm to use for a general per-tile histogram normalization. Supported types: CLAHE, GB11CLAHE (gaussian blur (11,11) and then clahe) (default: None)")
    parser.add_argument("--blend_type", type=str, default="LINEAR", choices=["NO_BLENDING", "AVERAGING", "LINEAR", "MULTI_BAND_SEAM"], help="The type of blending to use")
    parser.add_argument("--output_type", type=str, default="png", help="The output type format")
    parser.add_argument("--invert_image", action="store_true", default=False, help="store an inverted image")
    parser.add_argument("--process_num", type=int, default=10, help="the number of processes to use (default: 1)")
    args = parser.parse_args()
    
    async_render(args)

if __name__ == '__main__':
    main()
