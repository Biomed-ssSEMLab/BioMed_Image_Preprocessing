import sys
import cv2
import rh_logger
from mb_aligner.stitching.stitcher import Stitcher
from rh_renderer.tilespec_renderer import TilespecRenderer
from rh_renderer.multiple_tiles_renderer import BlendType
from rh_renderer import models
import argparse
import numpy as np
import ujson
import os
import math
from rh_renderer.normalization.histogram_clahe import HistogramCLAHE, HistogramGB11CLAHE
from scripts import common
import rh_img_access_layer
from rh_logger.api import logger
import time
from mb_aligner.dal.section import Section

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

def render_tilespec(tile_fname, output, scale, output_type, in_bbox, tile_size, invert_image, threads_num=1,
                    empty_placeholder=False, hist_adjuster_alg_type=None, from_to_cols_rows=None,
                    blend_type=BlendType.MULTI_BAND_SEAM):
    """Renders a given tilespec.
       If the in_bbox to_x/to_y values are -1, uses the tilespecs to determine the output size.
       If tile_size is 0, the output will be a single image, otherwise multiple tiles will be created.
       output is either a single filename to save the output in (using the output_type),
       or a prefix for the tiles output, which will be of the form: {prefix}_tr%d-tc%d.{output_type}
       and the row (tr) and column (tc) values will be one-based."""

    start_time = time.time()

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

    scaled_bbox1 = [
        int(math.floor(image_bbox[0])),
        int(math.ceil(image_bbox[1])),
        int(math.floor(image_bbox[2])),
        int(math.ceil(image_bbox[3]))
    ]
    # Set the post-scale out shape of the image
    out_shape = (scaled_bbox[1] - scaled_bbox[0], scaled_bbox[3] - scaled_bbox[2])
    print("Final out_shape for the image: {}".format(out_shape))

    hist_adjuster = None
    if hist_adjuster_alg_type is not None:
        if hist_adjuster_alg_type.upper() == 'CLAHE':
            hist_adjuster = HistogramCLAHE()
        if hist_adjuster_alg_type.upper() == 'GB11CLAHE':
            hist_adjuster = HistogramGB11CLAHE()

    # with rh_img_access_layer.FSAccess(tile_fname,False) as data:
    with open(tile_fname, 'r') as data:
        tilespec = ujson.load(data)
    renderer = TilespecRenderer(tilespec, hist_adjuster=hist_adjuster, dynamic=(scale != 1.0), blend_type=blend_type)

    if scale != 1.0:
        downsample = models.AffineModel(np.array([
            [scale, 0., 0.],
            [0., scale, 0.],
            [0., 0., 1.]
        ]))
        # renderer.add_transformation(downsample,scaled_bbox1[0], scaled_bbox1[2], scaled_bbox1[1] - 1, scaled_bbox1[3] - 1,scale)
        renderer.add_transformation(downsample)

    if tile_size == 0:
        # no tiles, just render a single file
        out_fname = "{}.{}".format(os.path.splitext(output)[0], output_type)
        out_fname_empty = "{}_empty".format(out_fname)

        # Render the image
        img, start_point = renderer.crop(scaled_bbox[0], scaled_bbox[2], scaled_bbox[1] - 1, scaled_bbox[3] - 1)
        print("Rendered cropped and downsampled version")
        # cv2.imwrite('/media/liuxz/3EA0B4CEA0B48E41/output/test1/shiyanshiyan.png',img)

        if empty_placeholder:
            if img is None or np.all(img == 0):
                # create the empty file, and return
                print("saving empty image {}".format(out_fname_empty))
                # open(out_fname_empty, 'a').close()
                rh_img_access_layer.FSAccess(out_fname_empty, True, read=False).close()
                print("Rendering and saving empty file {} took {} seconds.".format(out_fname_empty,
                                                                                   time.time() - start_time))
                return

        if img is None:
            # No actual image, set a blank image of the wanted size
            img = np.zeros((out_shape[1], out_shape[0]), dtype=np.uint8)
            start_point = (0, 0)

        print("Padding image")
        img = pad_image(img, scaled_bbox[0], scaled_bbox[2], start_point)
        # cv2.imwrite('/media/liuxz/3EA0B4CEA0B48E41/output/test1/pading.png', img)

        if invert_image:
            img = 255 - img

        print("saving image {}".format(out_fname))
        rh_img_access_layer.write_image_file(out_fname, img)

    print("Rendering and saving {} took {} seconds.".format(tile_fname, time.time() - start_time))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Renders a given tilespec to a file (or multiple files/tiles).\
                                                      Note that the images output sizes will be the (to_x - from_x, to_y - from_y) if given, or the entire image size.')
    parser.add_argument('-t', '--threads_num', type=int,
                        help='the number of threads to use (default: 1) - not used at the moment',
                        default=1)
    parser.add_argument('-s', '--scale', type=float,
                        help='the scale of the output images (default: 0.1)',
                        default=0.5)
    parser.add_argument('--output_type', type=str,
                        help='the images output type (default: png)',
                        default='png')
    parser.add_argument('--from_x', type=int,
                        help='the left coordinate, full res (default: 0)',
                        default=0)
    parser.add_argument('--from_y', type=int,
                        help='the top coordinate, full res (default: 0)',
                        default=0)
    parser.add_argument('--to_x', type=int,
                        help='the right coordinate, full res (default: full image)',
                        default=-1)
    parser.add_argument('--to_y', type=int,
                        help='the bottom coordinate, full res (default: full image)',
                        default=-1)
    parser.add_argument('--tile_size', type=int,
                        help='the size (square side) of each tile, post-scale (default: 0 - no tiles)',
                        default=0)
    parser.add_argument('-i', '--invert_image', action='store_true',
                        help='store an inverted image', default=True)
    parser.add_argument('-e', '--empty_placeholder', action='store_true',
                        help='store an empty file name (suffix will be "_empty"), when the tile/image has no data')
    parser.add_argument('--hist_adjuster_alg_type', type=str,
                        help='the type of algorithm to use for a general per-tile histogram normalization. Supported typed: CLAHE (default: None)',
                        default='CLAHE')
    parser.add_argument('--from_to_cols_rows', type=str,
                        help='Only to be used with tiled output (the tile_size argument is set). The input includes 4 numbers separated by commas, \
    in the form "from_col,from_row,to_col,to_row" and only the output tiles in the given range (including from, excluding to) will be saved. (default: None)',
                        default=None)
    parser.add_argument('--blend_type', type=str,
                        help='The type of blending to use. Values = {} (default: MULTI_BAND_SEAM)'.format(
                            BlendType.__members__.keys()),
                        default='MULTI_BAND_SEAM')
    args = parser.parse_args()
    print(args)

    blend_type = BlendType[args.blend_type]
    from_to_cols_rows = None
    if args.from_to_cols_rows is not None:
        assert (args.tile_size > 0)
        from_to_cols_rows = [int(i) for i in args.from_to_cols_rows.split(',')]
        assert (len(from_to_cols_rows) == 4)
    index = 1
    section_dir = '/MultiSEM_data/00'+str(index)+'_S'+str(index)+'R1/full_image_coordinates_corrected_1.txt'
    # section_dir = '/media/liuxz/3EA0B4CEA0B48E41/mb_aligner/Experiment_0326_coatedTape_SS6_20210326_16-11-56/001_S1R1/full_image_coordinates_2.txt'
    section_num = 1
    conf_fname = '../../conf/conf_example.yaml'
    processes_num = 8
    out_fname = '/home/liuxz/aligner/results_1/s3m1/image/sift.png'

    logger.start_process('main', 'stitcher.py', [section_dir, conf_fname])
    section = Section.create_from_full_image_coordinates(section_dir, section_num)
    conf = Stitcher.load_conf_from_file(conf_fname)
    stitcher = Stitcher(conf)
    stitcher.stitch_section(section)  # will stitch and update the section tiles' transformations

    import json
    with open(out_fname+'.json', 'wt') as out_f:
        json.dump(section.tilespec, out_f, sort_keys=True, indent=4)
    tilespec = out_fname+'.json'

    render_tilespec(tilespec, out_fname, args.scale, args.output_type,
                    (args.from_x, args.to_x, args.from_y, args.to_y), args.tile_size, args.invert_image,
                    args.threads_num, args.empty_placeholder, args.hist_adjuster_alg_type, from_to_cols_rows,
                    blend_type)
    logger.end_process('main ending', rh_logger.ExitCode(0))
    sys.exit