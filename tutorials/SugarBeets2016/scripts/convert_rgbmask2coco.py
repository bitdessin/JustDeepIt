import os
import sys
import json
import random
import shutil
import glob
import joblib
import tqdm
from justdeepit.utils import ImageAnnotation, ImageAnnotations

random.seed(2022)

def get_images(img_dpath, ann_dpath):
    #' get image list with the corresponding annotations
    #'
    img_dict = {}
    for img_fpath in glob.glob(os.path.join(img_dpath, '*.png'), recursive=True):
        if os.path.basename(img_fpath) not in img_dict:
            img_dict[os.path.basename(img_fpath)] = 0
        img_dict[os.path.basename(img_fpath)] += 1
    for ann_fpath in glob.glob(os.path.join(ann_dpath, '*.png'), recursive=True):
        if os.path.basename(ann_fpath) not in img_dict:
            img_dict[os.path.basename(ann_fpath)] = 0
        img_dict[os.path.basename(ann_fpath)] += 1
        
    img_list = []
    ann_list = []
    for img_fpath, val in img_dict.items():
        if val == 2:
            img_list.append(os.path.join(img_dpath, img_fpath))
            ann_list.append(os.path.join(ann_dpath, img_fpath))
    
    return img_list, ann_list
    
    

def format2coco(img_dpath, ann_dpath, output_dpath):
    #' convert annotations (RGB mask) to COOC format 
    #'
    shutil.copy(img_dpath, os.path.join(output_dpath, os.path.basename(img_dpath)))
    shutil.copy(ann_dpath, os.path.join(output_dpath + '_mask', os.path.basename(ann_dpath)))
    return ImageAnnotation(img_dpath, ann_dpath,
                           'rgbmask',
                           rgb2class={'255,0,0': 'weeds', '0,255,0': 'sugarbeets'})



def main(data_dpath, output_dpath):
    train_dpath = os.path.join(output_dpath, 'train')
    test_dpath = os.path.join(output_dpath, 'test')
    os.makedirs(train_dpath, exist_ok=True)
    os.makedirs(test_dpath, exist_ok=True)
    os.makedirs(train_dpath + '_mask', exist_ok=True)
    os.makedirs(test_dpath + '_mask', exist_ok=True)
    
    images = []
    anns = []
    
    for dpath in glob.glob(os.path.join(data_dpath, 'CKA_*')):
        img_dpath = os.path.join(dpath, 'images', 'rgb')
        ann_dpath = os.path.join(dpath, 'annotations', 'dlp', 'colorCleaned')
        img_list, ann_list = get_images(img_dpath, ann_dpath)
        images.extend(img_list)
        anns.extend(ann_list)
    
    rand_idx = random.sample(range(len(images)), k=5000 + 1000)
    train_idx = rand_idx[:5000]
    test_idx = rand_idx[5000:]
    
    train_anns = joblib.Parallel(n_jobs = 64)(
        joblib.delayed(format2coco)(images[i], anns[i], train_dpath) for i in tqdm.tqdm(train_idx)
    )
    train_anns = ImageAnnotations(train_anns)
    train_anns.format('coco', os.path.join(output_dpath, 'train.json'))
    
    test_anns = joblib.Parallel(n_jobs = 64)(
        joblib.delayed(format2coco)(images[i], anns[i], test_dpath) for i in tqdm.tqdm(test_idx)
    )
    test_anns = ImageAnnotations(test_anns)
    test_anns.format('coco', os.path.join(output_dpath, 'test.json'))
    

    
    

if __name__ == '__main__':
    data_dpath = sys.argv[1]
    output_dpath = sys.argv[2]
    main(data_dpath, output_dpath)



