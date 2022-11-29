import os
import logging
import justdeepit
from justdeepit.webapp import OD

logger = logging.getLogger(__name__)       


class IS(OD):
    
    def __init__(self, workspace):
        super().__init__(workspace)
        self.app = 'IS'
        self.images = []
    
    
    def train_model(self, class_label, image_dpath, annotation_path, annotation_format,
                    model_arch='maskrcnn', model_config=None, model_weight=None, 
                    optimizer=None, scheduler=None,
                    batchsize=8, epoch=100, score_cutoff=0.5, cpu=4, gpu=1, backend='mmdetection'):
        return super().train_model(class_label, image_dpath, annotation_path, annotation_format,
                                   model_arch, model_config, model_weight, 
                                   optimizer, scheduler,
                                   batchsize, epoch, score_cutoff, cpu, gpu, backend)
    
    
    def detect_objects(self, class_label, image_dpath,
                       model_arch='maskrcnn', model_config=None, model_weight=None,
                       score_cutoff=0.5, batchsize=8, cpu=4, gpu=1, backend='mmdetection'):
        return super().detect_objects(class_label, image_dpath,
                                      model_arch, model_config, model_weight,
                                      score_cutoff, batchsize, cpu, gpu, backend)



