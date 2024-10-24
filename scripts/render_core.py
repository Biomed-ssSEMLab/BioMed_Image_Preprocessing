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
'''
def pad_image(img, from_x, from_y, start_point):
    """Pads the image (zeros) that starts from start_point (returned from the renderer), to (from_x, from_y)"""
    # Note that start_point is (y, x)
    print("start pad image")

    if start_point[0] == from_y and start_point[1] == from_x:
        # Nothing to pad, return the image as is
        return img

    full_height_width = (img.shape + np.array([start_point[1] - from_y, start_point[0] - from_x])).astype(int)
    full_img = np.zeros(full_height_width, dtype=img.dtype)
    full_img[int(start_point[1] - from_y):int(start_point[1] - from_y + img.shape[0]),
    int(start_point[0] - from_x):int(start_point[0] - from_x + img.shape[1])] = img
    return full_img
'''
# pad_image from EMSA
def pad_image(img, from_x, from_y, start_point):
    # Forked from rh-renderer(https://github.com/Rhoana/rh_renderer)
    """Pads the image (zeros) that starts from start_point (returned from the renderer), to (from_x, from_y)"""
    # Note that start_point is (y, x)
    print("start pad image")

    if start_point[0] == from_y and start_point[1] == from_x:
        # Nothing to pad, return the image as is
        return img
    from_x = int(from_x)
    from_y = int(from_y)
    start_point = list(start_point)
    start_point[0] = int(start_point[0])
    start_point[1] = int(start_point[1])
    full_height_width = img.shape + np.array([start_point[1] - from_y, start_point[0] - from_x])
    full_img = np.zeros(full_height_width, dtype=img.dtype)
    full_img[start_point[1] - from_y:
             start_point[1] - from_y + img.shape[0], start_point[0] - from_x:
                                                     start_point[0] - from_x + img.shape[1]] = img
    return full_img

# def render_tilespec(tile_fname, output, scale, output_type, in_bbox, tile_size, invert_image, threads_num=1, empty_placeholder=False, hist_adjuster_fname=None, hist_adjuster_alg_type=None, from_to_cols_rows=None):
# def render_tilespec(max_row, max_col, tile_fname, output, scale, output_type, in_bbox, tile_size, invert_image, threads_num=1,
#                     empty_placeholder=False, hist_adjuster_alg_type=None, from_to_cols_rows=None,
#                     blend_type=BlendType.MULTI_BAND_SEAM):
def render_tilespec(max_row, max_col, tile_fname, renderer, output, scale, output_type, in_bbox, tile_size, invert_image, threads_num=1,
                    empty_placeholder=False, hist_adjuster_alg_type=None, from_to_cols_rows=None,
                    blend_type=BlendType.MULTI_BAND_SEAM):
    """Renders a given tilespec.
       If the in_bbox to_x/to_y values are -1, uses the tilespecs to determine the output size.
       If tile_size is 0, the output will be a single image, otherwise multiple tiles will be created.
       output is either a single filename to save the output in (using the output_type),
       or a prefix for the tiles output, which will be of the form: {prefix}_tr%d-tc%d.{output_type}
       and the row (tr) and column (tc) values will be one-based."""

    start_time = time.time()
    print("start render tilespec")
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

    #============================== hqjin ==============================#
    scaled_bbox[1] = min(scaled_bbox[0] + tile_size, scaled_bbox[1])
    scaled_bbox[3] = min(scaled_bbox[2] + tile_size, scaled_bbox[3])
    #============================== hqjin ==============================#


    # row = os.path.basename(output).split('.')[0].split('_')[-1].split('-')[0].split('r')[1]
    # col = os.path.basename(output).split('.')[0].split('_')[-1].split('-')[1].split('c')[1]
    # # Set the post-scale out shape of the image
    # if not row == max_row: 
    #     if (scaled_bbox[3] - scaled_bbox[2])%64 == 1:
    #         scaled_bbox[3] = scaled_bbox[3] - 1
    #     if (scaled_bbox[3] - scaled_bbox[2])%64 == 63:
    #         scaled_bbox[3] = scaled_bbox[3] + 1
    # if not col == max_col: 
    #     if (scaled_bbox[1] - scaled_bbox[0])%64 == 1:
    #         scaled_bbox[1] = scaled_bbox[1] - 1
    #     if (scaled_bbox[1] - scaled_bbox[0])%64 == 63:
    #         scaled_bbox[1] = scaled_bbox[1] + 1

    out_shape = (scaled_bbox[1] - scaled_bbox[0], scaled_bbox[3] - scaled_bbox[2])
   
    print("Final out_shape for the image: {}".format(out_shape))

    # #============================== hqjin ==============================#
    # hist_adjuster = HistogramCLAHE()
    # # hist_adjuster = None
    # # if hist_adjuster_alg_type is not None:
    # #     if hist_adjuster_alg_type.upper() == 'CLAHE':
    # #         hist_adjuster = HistogramCLAHE()
    # #     if hist_adjuster_alg_type.upper() == 'GB11CLAHE':
    # #         hist_adjuster = HistogramGB11CLAHE()

    # # with rh_img_access_layer.FSAccess(tile_fname, False) as data:
    # with open(tile_fname, 'r') as data:
    #     tilespec = ujson.load(data)

    # renderer = TilespecRenderer(tilespec, hist_adjuster=hist_adjuster, dynamic=(scale != 1.0), blend_type=blend_type)

    # # FOR THE IARPA latest dataset
    # # trans_model = models.AffineModel(np.array([
    # #                                          [1., 0., 13679.],
    # #                                          [0., 1., 2108.],
    # #                                          [0., 0., 1.]
    # #                                         ]))
    # # renderer.add_transformation(trans_model)

    # # Add the downsampling transformation
    # if scale != 1.0:
    #     downsample = models.AffineModel(np.array([
    #         [scale, 0., 0.],
    #         [0., scale, 0.],
    #         [0., 0., 1.]
    #     ]))
    #     renderer.add_transformation(downsample)
    # #============================== hqjin ==============================#

    # FOR THE IARPA R2B1 first dataset
    # rot = 0.29441193975
    # rot_model = models.AffineModel(np.array([
    #                                          [math.cos(rot), -math.sin(rot), 0.],
    #                                          [math.sin(rot), math.cos(rot), 0.],
    #                                          [0., 0., 1.]
    #                                         ]))
    # renderer.add_transformation(rot_model)

    # FOR THE IARPA R2B1 Layers_1_2 dataset
    # rot = 0.3490658504
    # rot_model = models.AffineModel(np.array([
    #                                          [math.cos(rot), -math.sin(rot), 0.],
    #                                          [math.sin(rot), math.cos(rot), 0.],
    #                                          [0., 0., 1.]
    #                                         ]))
    # renderer.add_transformation(rot_model)

    tile_size = 0
    if tile_size == 0:
        #  no tiles, just render a single file
        out_fname = "{}.{}".format(os.path.splitext(output)[0], output_type)
        out_fname_empty = "{}_empty".format(out_fname)
        
        print("start render crop")
        # Render the image
        img, start_point = renderer.crop(scaled_bbox[0], scaled_bbox[2], scaled_bbox[1] - 1, scaled_bbox[3] - 1)

        if empty_placeholder:
            if img is None or np.all(img == 0):
                # create the empty file, and return
                print("saving empty image {}".format(out_fname_empty))
                # open(out_fname_empty, 'a').close()
                rh_img_access_layer.FSAccess(out_fname_empty, True, read=False).close()
                # print("Rendering and saving empty file {} took {} seconds.".format(out_fname_empty, time.time() - start_time))
                return

        if img is None:
            # No actual image, set a blank image of the wanted size
            img = np.zeros((out_shape[1], out_shape[0]), dtype=np.uint8)
            start_point = (0, 0)
           
        img = pad_image(img, scaled_bbox[0], scaled_bbox[2], start_point)

        if invert_image:
            img = 255 - img

        cv2.imwrite(out_fname, img)
        # rh_img_access_layer.write_image_file(out_fname, img)

    else:
        # Tile the image
        rows = int(math.ceil(out_shape[1] / float(tile_size)))
        cols = int(math.ceil(out_shape[0] / float(tile_size)))

        from_row = 0
        from_col = 0
        to_row = rows
        to_col = cols
        if from_to_cols_rows is not None:
            from_col, from_row, to_col, to_row = from_to_cols_rows

        # Iterate over each row and column and save the tile
        for cur_row in range(from_row, to_row):
            from_y = scaled_bbox[2] + cur_row * tile_size
            to_y = min(scaled_bbox[2] + (cur_row + 1) * tile_size, scaled_bbox[3])
            for cur_col in range(from_col, to_col):
                # tile_start_time = time.time()
                # out_fname = "{}_tr{}-tc{}.{}".format(output, str(cur_row + 1), str(cur_col + 1), output_type)
                out_fname = output
                out_fname_empty = "{}_empty".format(out_fname)
                from_x = scaled_bbox[0] + cur_col * tile_size
                to_x = min(scaled_bbox[0] + (cur_col + 1) * tile_size, scaled_bbox[1])

                # Render the tile
                img, start_point = renderer.crop(from_x, from_y, to_x - 1, to_y - 1)
                print("Rendered cropped and downsampled version")

                if empty_placeholder:
                    if img is None or np.all(img == 0):
                        # create the empty file, and return
                        print("saving empty image {}".format(out_fname_empty))
                        # open(out_fname_empty, 'a').close()
                        rh_img_access_layer.FSAccess(out_fname_empty, True, read=False).close()
                        continue

                if img is None:
                    # No actual image, set a blank image of the wanted size
                    img = np.zeros((to_y - from_y, to_x - from_x), dtype=np.uint8)
                    start_point = (from_y, from_x)

                print("Padding image")
                img = pad_image(img, from_x, from_y, start_point)

                if invert_image:
                    print("inverting image")
                    img = 255 - img

                print("saving image {}".format(out_fname))
                cv2.imwrite(out_fname, img)

                # # a part for aligning testing, delete this module finally
                # x, y = img.shape[0:2]
                # img = cv2.resize(img, (int(y /4), int(x /4)))

                # rh_img_access_layer.write_image_file(out_fname, img)

                # print("single tile rendering and saving to {} took {} seconds.".format(out_fname,
                #                                                                        time.time() - tile_start_time))

    print("Rendering and saving {} took {} seconds.".format(tile_fname, time.time() - start_time))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Renders a given tilespec to a file (or multiple files/tiles).\
                                                  Note that the images output sizes will be the (to_x - from_x, to_y - from_y) if given, or the entire image size.')
    # parser.add_argument('tilespec', metavar='tilespec', type=str,
    #                     help='the tilespec to render')
    # parser.add_argument('output', type=str,
    #                     help='the output filename (in case of a single image output), or a filename prefix (in case of tiled output)')
    parser.add_argument('-t', '--threads_num', type=int,
                        help='the number of threads to use (default: 1) - not used at the moment',
                        default=1)
    parser.add_argument('-s', '--scale', type=float,
                        help='the scale of the output images (default: 0.1)',
                        default=0.25)
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
    #     parser.add_argument('--hist_adjuster', type=str,
    #                         help='A location of a pkl file that containg the histogram adjuster object file to adjust the images with (default: None)',
    #                         default=None)
    parser.add_argument('--hist_adjuster_alg_type', type=str,
                        help='the type of algorithm to use for a general per-tile histogram normalization. Supported typed: CLAHE (default: None)',
                        default='CLAHE')
    # parser.add_argument('--entire_image_bbox', type=str,
    #                    help='the 2D pre-scale image bbox for the entire 3D image (default: use current section bbox)',
    #                    default=None)
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

    output1 = '/userhome/output/test_retina//W01_Sec309_16nm.png'
    tilespec = '/userhome/output/test_retina/ECS_test9_cropped_010_S309R1-1.json'

    blend_type = BlendType[args.blend_type]

    from_to_cols_rows = None
    if args.from_to_cols_rows is not None:
        assert (args.tile_size > 0)
        from_to_cols_rows = [int(i) for i in args.from_to_cols_rows.split(',')]
        assert (len(from_to_cols_rows) == 4)

    #     render_tilespec(args.tilespec, args.output, args.scale, args.output_type,
    #                     (args.from_x, args.to_x, args.from_y, args.to_y), args.tile_size, args.invert_image,
    # #                    args.threads_num, args.empty_placeholder, args.hist_adjuster, args.hist_adjuster_alg_type, from_to_cols_rows)
    #                     args.threads_num, args.empty_placeholder, args.hist_adjuster_alg_type, from_to_cols_rows,
    #                     blend_type)

    render_tilespec(tilespec, output1, args.scale, args.output_type,
                    (args.from_x, args.to_x, args.from_y, args.to_y), args.tile_size, args.invert_image,
                    #                    args.threads_num, args.empty_placeholder, args.hist_adjuster, args.hist_adjuster_alg_type, from_to_cols_rows)
                    args.threads_num, args.empty_placeholder, args.hist_adjuster_alg_type, from_to_cols_rows,
                    blend_type)
