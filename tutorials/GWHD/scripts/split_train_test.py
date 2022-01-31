import os
import sys
import glob
import random
import shutil

def main(input_dpaht, train_dpath, query_dpath):
    random.seed(12345)
    
    os.makedirs(train_dpath, exist_ok=True)
    os.makedirs(query_dpath, exist_ok=True)
    
    for fpath in glob.glob(os.path.join(input_dpaht, '*.png')):
        img_fpath = fpath
        
        if random.random() < 0.8:
            shutil.copy2(img_fpath, os.path.join(train_dpath, os.path.basename(img_fpath)))
        else:
            shutil.copy2(img_fpath, os.path.join(query_dpath, os.path.basename(img_fpath)))
            

if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2], sys.argv[3])
    
    
    

