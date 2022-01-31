import os
import sys
import glob
import shutil
import joblib
import math
import numpy as np
import skimage
import skimage.io
import skimage.measure
import skimage.morphology
import agrolens
import agrolens.models
import datetime

def write_exe_time(msg=''):
    t = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    with open('exetime_log.txt', 'a') as outfh:
        outfh.write('{}\t{}\n'.format(t, msg))


def process_image(fpath):
    
    im = skimage.io.imread(fpath)
    im_org = im.copy()
    im[im > 0] = 255
    if len(im.shape) >= 3:
        im = im[:, :, 0]
    leaf_area_ratio = np.sum(im > 0) / im.size
    n_contours = 0
    
    # remove the small noises and then check the image is valid for training or not
    valid = False
    if 0.1 < leaf_area_ratio and leaf_area_ratio < 0.8:
        im = skimage.morphology.remove_small_objects(im, min_size=1000)
        im = skimage.morphology.remove_small_holes(skimage.img_as_bool(im), area_threshold=100)
        im = skimage.img_as_ubyte(im)
        n_contours = len(skimage.measure.find_contours(im, 0.5))
        if n_contours < 10:
            valid = True

    if valid:
        try:
            # save the mask (after removing the small noises)
            nim = np.zeros((im.shape[0], im.shape[1], 3)).astype(np.uint8)
            nim[:, :, 0] = im
            nim[:, :, 1] = im
            nim[:, :, 2] = im
            skimage.io.imsave(fpath, nim, check_contrast=False)
        except:
            valid = False
    
    return valid

 
    

def draw_fig(imgann, image_dpath):
    mask_fpath = os.path.join(image_dpath, os.path.basename(imgann.image_path) + '.mask.png')
    imgann.draw('mask', mask_fpath)

    
def main(ws):
    
    for i in range(6):
        print('--- iteration: {} ---'.format(i))
        write_exe_time('START\t{}'.format(i))
        images_0_dpath = os.path.join(ws, 'data', 'images')
        images_i_dpath = os.path.join(ws, 'data', 'masks_{}'.format(i))
        os.makedirs(images_i_dpath, exist_ok=True)
        for _ in glob.glob(os.path.join(images_i_dpath, '*')):
            os.remove(_)
        
        print('>>> summmarize images ...')
        
        images_0_fpath = []
        for image_0_fpath in glob.glob(os.path.join(images_0_dpath, '*')):
            if os.path.splitext(image_0_fpath)[1].lower() in ['.jpg', '.jpeg', '.png']:
                images_0_fpath.append(image_0_fpath)
        
        print('>>> detecting objects ...')
        n_batch = 1000
        for j in range(math.ceil(len(images_0_fpath) / n_batch)):
            batch_from = j * n_batch
            batch_to = (j + 1) * n_batch if (j + 1) * n_batch < len(images_0_fpath) - 1 else len(images_0_fpath) - 1
            images_0_fpath_batch = images_0_fpath[batch_from:batch_to]
        
            weight = os.path.join(ws, 'weights', 'u2net.{}.pth'.format(i))
            u2net = agrolens.models.U2Net(weight)
            write_exe_time('START_DETECTION\t{}\tMINIBATCH {}'.format(i, j))
            outputs = u2net.inference(images_0_fpath_batch, batch_size=16, strategy='resize', gpu=1, cpu=32)
            write_exe_time('FINISH_DETECTION\t{}\tMINIBATCH {}'.format(i, j))
        
            print('    writing mask images ...')
            write_exe_time('START_WRITEMASK\t{}\tMINIBATCH {}'.format(i, j))
            _ = joblib.Parallel(n_jobs=32, backend='multiprocessing')(
                    joblib.delayed(draw_fig)(outputs[k], images_i_dpath) for k in range(len(outputs)))
            write_exe_time('FINISH_WRITEMASK\t{}\tMINIBATCH {}'.format(i, j))
        
        print('>>> validating images ...')
        write_exe_time('START_VALIDMASK\t{}'.format(i))
        for mask_fpath in sorted(glob.glob(os.path.join(images_i_dpath, '*'))):
            valid = process_image(mask_fpath)
            if not valid:
                shutil.move(mask_fpath, mask_fpath[:-4] + '._____.png')
        write_exe_time('FINISH_VALIDMASK\t{}'.format(i))
        
        print('>>> training model ...')
        u2net = agrolens.models.U2Net(weight)
        train_image_list = 'train_images_iterate{}'.format(i + 1)
        with open(train_image_list, 'w') as outfh:
            for image_fpath in glob.glob(os.path.join(images_0_dpath, '*')):
                mask_fpath = os.path.join(images_i_dpath, os.path.basename(image_fpath) + '.mask.png')
                if os.path.exists(mask_fpath):
                    outfh.write('{}\t{}\n'.format(image_fpath, mask_fpath))
        write_exe_time('START_TRAINMODEL\t{}'.format(i))
        u2net.train(train_image_list, batch_size=16, epoch=10, strategy='resize', gpu=1, cpu=32)
        write_exe_time('FINISH_TRAINMODEL\t{}'.format(i))
        updated_weight_fpath = os.path.join(ws, 'weights', 'u2net.{}.pth'.format(i + 1))
        u2net.save(updated_weight_fpath)


        write_exe_time('FINISH\t{}'.format(i))


if __name__ == '__main__':
    main(ws='.')




