import os
import platform
import shutil
import glob
import unittest
import torch
from agrolens import SOD



class TestSODAPPMask(unittest.TestCase):
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.dataset = os.path.join('sample_datasets', 'sod')
        self.ws = os.path.join('outputs', 'sodappmask')
        os.makedirs(self.ws, exist_ok=True)    
        
        self.image_suffix = '_image.jpg'
        self.mask_suffix = '_mask.png'
        
        self.batchsize = 4
        self.epoch = 110
        self.gpu = 0
        self.cpu = 4
        if platform.system() == 'Darwin':
            self.cpu = 0
        if torch.cuda.is_available():
            self.gpu = 1
            self.cpu = 8
        
        
    
    def __test_app(self, weight, strategy, is_series):
        inference_strategy = 'resize' if strategy == 'resize' else 'slide'
        
        app = SOD(self.ws)
        
        for fpath in glob.glob(os.path.join(self.dataset, '*.jpg')):
            shutil.copy(fpath, os.path.join(self.ws, 'train_dataset'))
            shutil.copy(fpath, os.path.join(self.ws, 'query_dataset'))
        for fpath in glob.glob(os.path.join(self.dataset, '*.png')):
            shutil.copy(fpath, os.path.join(self.ws, 'train_dataset'))
        
        # training
        app.sort_train_images(os.path.join(self.ws, 'train_dataset'),
                              image_suffix=self.image_suffix, mask_suffix=self.mask_suffix)
        app.train_model(weight, self.batchsize, self.epoch, self.cpu, self.gpu,
                        strategy, 340)
        
        # detection
        app.sort_query_images(os.path.join(self.ws, 'query_dataset'))
        app.detect_objects(weight, self.batchsize, inference_strategy,
                           0.8, 0, 0, 340,
                           cpu=self.cpu, gpu=self.gpu)
        
        # summarization
        if platform.system() == 'Darwin':
            self.cpu = 1
        app.generate_movie(10.0, 1.0, 'mp4v', '.mp4')
        
    
    def test_app_1(self):
        weight = os.path.join(self.ws, 'sodapp.t1.pth')
        self.__test_app(weight, 'resize', False)
    
    def test_app_2(self):
        weight = os.path.join(self.ws, 'sodapp.t2.pth')
        self.__test_app(weight, 'randomcrop', True)
    


if __name__ == '__main__':
    unittest.main()


