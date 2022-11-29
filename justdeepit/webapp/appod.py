import os
import datetime
import glob
import json
import tqdm
import logging
import threading
import joblib
import traceback
import numpy as np
import skimage.io
import skimage.color
import skimage.exposure
import torch
import justdeepit
from justdeepit.webapp import AppBase

logger = logging.getLogger(__name__)       


class OD(AppBase):
    
    def __init__(self, workspace):
        super().__init__(workspace)
        self.app = 'OD'
    
    
     
    def __build_model(self, class_label, model_arch, model_config, model_weight, backend):
        init_model_weight = model_weight if os.path.exists(model_weight) else None
        if self.app == 'OD':
            return justdeepit.models.OD(class_label, model_arch, model_config, init_model_weight,
                                        os.path.join(self.workspace_, 'tmp'), backend)
        elif self.app == 'IS':
            return justdeepit.models.IS(class_label, model_arch, model_config, init_model_weight,
                                        os.path.join(self.workspace_, 'tmp'), backend)
    
    
    
    def train_model(self, class_label, image_dpath, annotation_path, annotation_format,
                    model_arch='fasterrcnn', model_config=None, model_weight=None, 
                    optimizer=None, scheduler=None,
                    batchsize=8, epoch=100, score_cutoff=0.5, cpu=4, gpu=1, backend='mmdetection'):
        job_status = self.set_jobstatus(self.code.TRAINING, self.code.JOB__TRAIN_MODEL, self.code.STARTED, '')
        try:
            self.check_training_images(image_dpath, annotation_path, annotation_format, class_label)
            model = self.__build_model(class_label, model_arch, model_config, model_weight, backend)
            model.train(image_dpath, annotation_path, annotation_format,
                        optimizer, scheduler,
                        batchsize, epoch, score_cutoff, cpu, gpu)
            model.save(model_weight)
            
            job_status = self.set_jobstatus(self.code.TRAINING, self.code.JOB__TRAIN_MODEL, self.code.FINISHED, '')
        except KeyboardInterrupt:
            job_status = self.set_jobstatus(self.code.TRAINING, self.code.JOB__TRAIN_MODEL, self.code.INTERRUPT, '')
        
        except BaseException as e:
            traceback.print_exc()
            job_status = self.set_jobstatus(self.code.TRAINING, self.code.JOB__TRAIN_MODEL, self.code.ERROR, str(e))
        else:
            job_status = self.set_jobstatus(self.code.TRAINING, self.code.JOB__TRAIN_MODEL, self.code.COMPLETED,
                                            'Params: batchsize {}; epoch {}; optimizer: {}; scheduler: {}.'.format(batchsize, epoch, optimizer, scheduler))
        
        return job_status
    
    
    
    def detect_objects(self, class_label, image_dpath,
                       model_arch='fasterrcnn', model_config=None, model_weight=None,
                       score_cutoff=0.8, batchsize=8, cpu=4, gpu=1, backend='mmdetection'):
        
        def __save_outputs(ws, image_fpath, output, app_id):
            image_name = os.path.splitext(os.path.basename(image_fpath))[0]
            if app_id == 'OD':
                output.draw('bbox', os.path.join(ws, 'outputs', image_name + '.bbox.png'), label=True, score=True)
            elif app_id == 'IS':
                output.draw('contour+bbox', os.path.join(ws, 'outputs', image_name + '.bbox.png'), label=True, score=True)
        
        job_status = self.set_jobstatus(self.code.INFERENCE, self.code.JOB__INFER, self.code.STARTED, '')
        try:
            self.check_query_images(image_dpath)
            self.seek_query_images()
        
            valid_ws = os.path.join(self.workspace_, 'tmp')
            # if config is not given, use the config saved during training process
            if model_config is None or model_config == '':
                model_config = os.path.join(os.path.splitext(model_weight)[0] + '.yaml')
            model = self.__build_model(class_label, model_arch, model_config, model_weight, backend)
            outputs = model.inference(self.images, score_cutoff, batchsize, cpu, gpu)
                
            joblib.Parallel(n_jobs=cpu)(
                joblib.delayed(__save_outputs)(self.workspace_, self.images[i], outputs[i], self.app) for i in range(len(self.images)))
            outputs.format('coco', os.path.join(self.workspace_, 'outputs', 'annotation.json'))
            
            job_status = self.set_jobstatus(self.code.INFERENCE, self.code.JOB__INFER, self.code.FINISHED, '')
        except KeyboardInterrupt:
            job_status = self.set_jobstatus(self.code.INFERENCE, self.code.JOB__INFER, self.code.INTERRUPT, '')
                   
        except BaseException as e:
            traceback.print_exc()
            job_status = self.set_jobstatus(self.code.INFERENCE, self.code.JOB__INFER, self.code.ERROR, str(e))
        else:
            job_status = self.set_jobstatus(self.code.INFERENCE, self.code.JOB__INFER, self.code.COMPLETED, '')

        return job_status
    


  
    def summarize_objects(self, cpu):

        def RGB(regionmask, intensity): 
            r = np.median(intensity[:, :, 0][regionmask[:, :, 0]])
            g = np.median(intensity[:, :, 1][regionmask[:, :, 1]])
            b = np.median(intensity[:, :, 2][regionmask[:, :, 2]])
            return (r, g, b)
    
        def HSV(regionmask, intensity): 
            intensity = skimage.color.rgb2hsv(intensity)
            h = np.median(intensity[:, :, 0][regionmask[:, :, 0]])
            s = np.median(intensity[:, :, 1][regionmask[:, :, 1]])
            v = np.median(intensity[:, :, 2][regionmask[:, :, 2]])
            return (h, s, v)
        
        def Lab(regionmask, intensity): 
            intensity = skimage.color.rgb2lab(intensity)
            l = np.median(intensity[:, :, 0][regionmask[:, :, 0]])
            a = np.median(intensity[:, :, 1][regionmask[:, :, 1]])
            b = np.median(intensity[:, :, 2][regionmask[:, :, 2]])
            return (l, a, b)
        
        
        def __summarize_objects(image_meta):
            image_fpath, ann_fpath, output_fpath = image_meta
            image = skimage.io.imread(image_fpath)
            ann = justdeepit.utils.ImageAnnotation(image_fpath, ann_fpath, 'coco')
            with open(output_fpath, 'w') as outfh:
                outfh.write('{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\n'.format(
                        'image_path', 'object_id', 'class', 'score', 'xmin', 'ymin', 'xmax', 'ymax'
                    ))
                for region in ann.regions:
                    outfh.write('{}\t{}\t{}\t{}\t{}\t{}\t{}\n'.format(
                        ann.image_path, region['id'], region['class'], region['score'], *region['bbox']
                    ))
            
        job_status = self.set_jobstatus(self.code.INFERENCE, self.code.JOB__SUMMARIZE, self.code.STARTED, '')
        try:
            self.seek_query_images()
            logger.info('Finding objects and calculate the summary data using {} CPUs.'.format(cpu))
            
            images_meta = []
            for image in self.images:
                images_meta.append([image,
                                    os.path.join(self.workspace_, 'outputs', 'annotation.json'),
                                    os.path.join(self.workspace_, 'outputs', os.path.splitext(os.path.basename(image))[0] + '.object.txt')])
            
            images_meta = joblib.Parallel(n_jobs=cpu)(
                joblib.delayed(__summarize_objects)(images_meta[i]) for i in range(len(images_meta)))
 

            job_status = self.set_jobstatus(self.code.INFERENCE, self.code.JOB__SUMMARIZE, self.code.FINISHED, '')
        except KeyboardInterrupt:
            job_status = self.set_jobstatus(self.code.INFERENCE, self.code.JOB__SUMMARIZE, self.code.INTERRUPT, '')
        
        except BaseException as e:
            traceback.print_exc()
            job_status = self.set_jobstatus(self.code.INFERENCE, self.code.JOB__SUMMARIZE, self.code.ERROR, str(e))
        else:
            job_status = self.set_jobstatus(self.code.INFERENCE, self.code.JOB__SUMMARIZE, self.code.COMPLETED, '')

        return job_status
               
    

    def summarise_objects(self, erosion_size, dilation_size, aligned_images, cpu):
        job_status = self.summarize_objects(erosion_size, dilation_size, aligned_images, cpu)
        return job_status

    




