import os
import sys
import glob
import skimage
import skimage.io
import skimage.measure
import numpy as np


def generate_outline(im, mask):
    mask_contours = skimage.measure.find_contours(mask, 0.5)
  
    for n, contour in enumerate(mask_contours):
        contour = contour.astype(np.int32)

        # set line width to 3 px
        contour_p1 = contour + 1
        contour_m1 = contour - 1
        contour_p1[contour_p1[:, 0] >= im.shape[0], 0] = im.shape[0] - 1
        contour_p1[contour_p1[:, 1] >= im.shape[1], 1] = im.shape[1] - 1
        contour_m1[contour_m1[:, 0] < 0, 0] = 0
        contour_m1[contour_m1[:, 1] < 0, 1] = 0
        line_rgb = [227, 0, 72]
        for ch in range(len(im.shape)):
            im[contour[:, 0], contour[:, 1], ch] = line_rgb[ch]
            im[contour_p1[:, 0], contour_p1[:, 1], ch] = line_rgb[ch]
            im[contour_m1[:, 0], contour_m1[:, 1], ch] = line_rgb[ch]
    
    return im


def main(dpath):
    
    for fpath in glob.glob(os.path.join(dpath, '*_mask.tif')):
        print(fpath)
        im = skimage.io.imread(fpath[:-9] + '.tif')
        mask = skimage.io.imread(fpath)
        
        img_outline = generate_outline(im, mask)
        skimage.io.imsave(fpath[:-9] + '_outline.png', img_outline)
        
        


if __name__ == '__main__':
    
    main(sys.argv[1])

