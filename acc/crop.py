import cv2
import numpy as np
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
import math
import sys



def in_bbox(inbox, mask, scale):
    mask_scaled = cv2.resize(mask, (int(mask.shape[1]/scale),int(mask.shape[0]/scale)))
    # area = mask[int(inbox[2]*scale):int(inbox[3]*scale),int(inbox[0]*scale):int(inbox[1]*scale)]
    area = mask_scaled[int(inbox[2]):int(inbox[3]),int(inbox[0]):int(inbox[1])]
    if (area == 255).all():
        print('***** add mesh_point *****\n')
        return True
    return False

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
    return pointsret

def crop1(img1, img2, scaled_bbox, moving,out_path, i):
    crop1 = img1[scaled_bbox[2]:scaled_bbox[3], scaled_bbox[0]:scaled_bbox[1]]
    # if np.var(crop1) > 1500:
    crop2 = img2[scaled_bbox[2]+moving[1]:scaled_bbox[3]+moving[1], scaled_bbox[0]+moving[0]:scaled_bbox[1]+moving[0]]
    cv2.imwrite(out_path+'/crop_image_'+str(i)+'_1.png', 255-crop1)
    cv2.imwrite(out_path+'/crop_image_'+str(i)+'_2.png', 255-crop2)
    print('croped {} image pairs.\n'.format(i))


def crop(img1, img2, angle, moving,out_path, mesh_bbox):
    from multiprocessing import Pool
    pool = Pool(8)
    if not angle == 0:  
        img2 = img2.rotate(angle)
    img1 = np.array(img1)
    img2 = np.array(img2)
    mask1 = (img1 > 0).astype(np.uint8)
    mask2 = (img2 > 0).astype(np.uint8)
    mask = np.logical_and(mask1,mask2).astype(np.uint8)
    mask[mask>0] = 255
    print('***** mask created... *****')
    i = 1
    for scaled_bbox in mesh_bbox:
        if in_bbox(scaled_bbox, mask, 1):
            crop1 = img1[scaled_bbox[2]:scaled_bbox[3], scaled_bbox[0]:scaled_bbox[1]]
            if np.var(crop1) > 1000:
                crop2 = img2[scaled_bbox[2]+moving[1]:scaled_bbox[3]+moving[1], scaled_bbox[0]+moving[0]:scaled_bbox[1]+moving[0]]
                cv2.imwrite(out_path+'/crop_image_'+str(i)+'_1.png', 255-crop1)
                cv2.imwrite(out_path+'/crop_image_'+str(i)+'_2.png', 255-crop2)
                print('croped {} image pairs.\n'.format(i))
                i += 1
    # pool.close()
    # pool.join()


img1 = '/braindat/lab/xzliu/output/output/render/test/0310_W01_Sec310.png'
img2 = '/braindat/lab/xzliu/output/output/render/test/0311_W01_Sec311.png'
img1 = Image.open(img1)
img2 = Image.open(img2)
print('\nimages read...\n')

out_path1 = '/braindat/lab/xzliu/output/output/render/test/test1'
out_path2 = '/braindat/lab/xzliu/output/output/render/test/test2'
out_path3 = '/braindat/lab/xzliu/output/output/render/test/test3'

shape = np.array(img1).shape
print('shape computed...')
bbox = [0, shape[1], 0, shape[0]]
compare_radius = 250
mesh_pts = generate_hexagonal_grid(bbox, 2000, compare_radius)
mesh_bbox = []
for sec1_pt in mesh_pts:
    sec_bbox = [sec1_pt[0]-compare_radius,sec1_pt[0]+compare_radius,
                sec1_pt[1]-compare_radius,sec1_pt[1]+compare_radius]
    mesh_bbox.append(sec_bbox)
  

crop(img1, img2, 2, [0, 0],out_path1, mesh_bbox)
print("crop1 finished\n")
crop(img1, img2, 0, [40, 40],out_path2, mesh_bbox)
print("crop2 finished\n")
crop(img1, img2, 2, [20, 30],out_path3, mesh_bbox)
print("crop3 finished\n")

sys.exit(0)