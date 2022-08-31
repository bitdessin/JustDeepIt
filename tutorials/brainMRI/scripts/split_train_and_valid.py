import sys
import os
import glob
import random
import shutil
random.seed(12345)

def main(img_from, img_to):
    
    train_samples = []
    test_samples = []
    
    samples = []
    for patient_dpath in glob.glob(os.path.join(img_from, '*')):
        if 'TCGA_' in patient_dpath:
            samples.append('_'.join(os.path.basename(patient_dpath).split('_')[1:3]))
    
    samples = list(set(samples))
    
    for sample in samples:
        if random.random() > 0.1:
            train_samples.append(sample)
        else:
            test_samples.append(sample)
    
    
    for patient_dpath in sorted(glob.glob(os.path.join(img_from, '*'))):
        if 'TCGA_' in patient_dpath:
            sample = '_'.join(os.path.basename(patient_dpath).split('_')[1:3])
            if sample in train_samples:
                for img in sorted(glob.glob(os.path.join(patient_dpath, '*.tif'))):
                    dest = ''
                    if '_mask.tif' not in img:
                        dest = os.path.join(img_to, 'train', os.path.basename(img)[:-4] + '_image.tif')
                    else:
                        dest = os.path.join(img_to, 'train', os.path.basename(img))
                    shutil.copy(img, dest)
            else:
                for img in sorted(glob.glob(os.path.join(patient_dpath, '*.tif'))):
                    if '_mask.tif' not in img:
                        shutil.copy(img, os.path.join(img_to, 'test', os.path.basename(img)))



if __name__ == '__main__':
    
    main(sys.argv[1], sys.argv[2])
    


