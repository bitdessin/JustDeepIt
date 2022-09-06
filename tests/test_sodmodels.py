import os
import platform
import glob
import unittest
import logging
import torch
from justdeepit.models import SOD
logging.basicConfig(level=logging.WARNING)
logging.getLogger('justdeepit.models.utils.u2net').setLevel(level=logging.WARNING)


class TestModels(unittest.TestCase):
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.dataset = os.path.join('sample_datasets', 'sod')
        self.ws = os.path.join('outputs', 'sodmodel')
        os.makedirs(self.ws, exist_ok=True)    
        
        self.train_images = os.path.join(self.dataset, 'images')
        self.train_masks = os.path.join(self.dataset, 'masks')
        self.query_images = self.train_images
        
        self.batchsize = 4
        self.epoch = 110
        self.gpu = 0
        self.cpu = 4
        if torch.cuda.is_available():
            self.gpu = 1
            self.cpu = 8
        
    
    
    
    def __test_model(self, weight, train_strategy, detect_strategy):
    
        trained_weight = os.path.join(self.ws, weight)
        
        # training
        u2net = SOD(workspace=self.ws)
        u2net.train(self.train_images, self.train_masks,
                    batch_size=self.batchsize, epoch=self.epoch, cpu=self.cpu, gpu=self.gpu,
                    strategy=train_strategy)
        u2net.save(trained_weight)
        
        # detection
        u2net = SOD(model_weight=trained_weight, workspace=self.ws)
        outputs = u2net.inference(self.query_images, strategy=detect_strategy, u_cutoff=0.5,
                                 batch_size=self.batchsize, cpu=self.cpu, gpu=self.gpu)
        for output in outputs:
            output.draw('bbox+contour', os.path.join(self.ws,
                        os.path.splitext(os.path.basename(output.image_path))[0] + '.' + detect_strategy + '.contour.png'), label=True)
            output.draw('mask', os.path.join(self.ws,
                        os.path.splitext(os.path.basename(output.image_path))[0] + '.' + detect_strategy + '.mask.png'))
            output.draw('rgbmask', os.path.join(self.ws,
                        os.path.splitext(os.path.basename(output.image_path))[0] + '.' + detect_strategy + '.rgbmask.png'))
 
 
    
    def test_model_1(self):
        self.__test_model('trained_weight.t1.pth', 'resizing', 'resizing')
        
    
    
    def test_model_2(self):
        self.__test_model('trained_weight.t2.pth', 'randomcrop', 'sliding')
        



if __name__ == '__main__':
    unittest.main()


