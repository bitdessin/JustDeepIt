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
from justdeepit.app import AppBase

logger = logging.getLogger(__name__)       


class OD(AppBase):
    
    def __init__(self, workspace):
        super().__init__(workspace)
        self.app = 'OD'
    
    
    def build_model(self,
                      class_label,
                      model_arch,
                      model_config,
                      model_weight):
        init_model_weight = model_weight if os.path.exists(model_weight) else None
        if self.app == 'OD':
            return justdeepit.models.OD(class_label,
                                        model_arch,
                                        model_config,
                                        init_model_weight,
                                        self.tmp_dpath)
        elif self.app == 'IS':
            return justdeepit.models.IS(class_label,
                                        model_arch,
                                        model_config,
                                        init_model_weight,
                                        self.tmp_dpath)
    
    
    def train_model(self,
                    class_label,
                    train_dataset,
                    valid_dataset,
                    test_dataset,
                    model_arch='fasterrcnn',
                    model_config=None,
                    model_weight=None, 
                    optimizer=None,
                    scheduler=None,
                    batchsize=8,
                    epoch=100,
                    score_cutoff=0.5,
                    cpu=4,
                    gpu=1):
        job_status = self.set_jobstatus(self.app_code.TRAINING, self.app_code.JOB__TRAIN_MODEL, self.app_code.STARTED, '')
        try:
            train_images = self.check_images(train_dataset['images'],
                                             train_dataset['annotations'],
                                             train_dataset['annotation_format'])
            if valid_dataset is not None:
                valid_images = self.check_images(valid_dataset['images'],
                                                 valid_dataset['annotations'],
                                                 valid_dataset['annotation_format'])
            if test_dataset is not None:
                test_images = self.check_images(test_dataset['images'],
                                                test_dataset['annotations'],
                                                test_dataset['annotation_format'])

            if len(train_images) > 0:
                model = self.build_model(class_label, model_arch, model_config, model_weight)
                model.train(train_dataset, valid_dataset, test_dataset,
                            optimizer=optimizer, scheduler=scheduler,
                            score_cutoff=score_cutoff,
                            batchsize=batchsize, epoch=epoch, cpu=cpu, gpu=gpu)
                model.save(model_weight)
                job_status = self.set_jobstatus(self.app_code.TRAINING, self.app_code.JOB__TRAIN_MODEL, self.app_code.FINISHED, '')
            else:
                logger.info('No images for model training. Finished the process.')
                job_status = self.set_jobstatus(self.app_code.TRAINING, self.app_code.JOB__TRAIN_MODEL, self.app_code.FINISHED,
                                                'No images for training.')

        except KeyboardInterrupt:
            job_status = self.set_jobstatus(self.app_code.TRAINING, self.app_code.JOB__TRAIN_MODEL, self.app_code.INTERRUPT, '')
        
        except BaseException as e:
            traceback.print_exc()
            job_status = self.set_jobstatus(self.app_code.TRAINING, self.app_code.JOB__TRAIN_MODEL, self.app_code.ERROR, str(e))

        else:
            job_status = self.set_jobstatus(self.app_code.TRAINING, self.app_code.JOB__TRAIN_MODEL, self.app_code.COMPLETED,
                                            'Params: batchsize {}; epoch {}; optimizer: {}; scheduler: {}.'.format(batchsize, epoch, optimizer, scheduler))
        
        return job_status
    
    
    
    def detect_objects(self,
                       class_label,
                       image_dpath,
                       model_arch='fasterrcnn',
                       model_config=None,
                       model_weight=None,
                       score_cutoff=0.5,
                       batchsize=8,
                       cpu=4,
                       gpu=1):
        
        def __save_outputs(output_dpath, image_fpath, output, app_id):
            output_fmt = 'contour+bbox' if app_id == 'IS' else 'bbox'
            output_pfx = os.path.join(output_dpath, os.path.splitext(os.path.basename(image_fpath))[0])
            output.draw(output_fmt, output_pfx + '.bbox.png', label=True, score=True)
            output.format('json', output_pfx + '.json')

        job_status = self.set_jobstatus(self.app_code.INFERENCE, self.app_code.JOB__INFER, self.app_code.STARTED, '')
        try:
            images = self.check_images(image_dpath, None, None)
        
            # if config is not given, use the config saved during training process
            if model_config is None or model_config == '':
                model_config = os.path.join(os.path.splitext(model_weight)[0] + '.py')
            
            model = self.build_model(class_label, model_arch, model_config, model_weight)
            outputs = model.inference(images, score_cutoff, batchsize, cpu, gpu)
            
            joblib.Parallel(n_jobs=cpu)(
                joblib.delayed(__save_outputs)(
                    os.path.join(self.workspace, 'outputs'),
                    images[i],
                    outputs[i],
                    self.app) for i in range(len(images)))
            outputs.format('coco', os.path.join(self.workspace, 'outputs', 'annotations.json'))
            
            job_status = self.set_jobstatus(self.app_code.INFERENCE, self.app_code.JOB__INFER, self.app_code.FINISHED, '')
        except KeyboardInterrupt:
            job_status = self.set_jobstatus(self.app_code.INFERENCE, self.app_code.JOB__INFER, self.app_code.INTERRUPT, '')
                   
        except BaseException as e:
            traceback.print_exc()
            job_status = self.set_jobstatus(self.app_code.INFERENCE, self.app_code.JOB__INFER, self.app_code.ERROR, str(e))
        else:
            job_status = self.set_jobstatus(self.app_code.INFERENCE, self.app_code.JOB__INFER, self.app_code.COMPLETED, '')

        return job_status
    
  
    def summarize_objects(self, image_dpath, cpu):

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
            ann = justdeepit.utils.ImageAnnotation(image_fpath, ann_fpath, 'json')
            with open(output_fpath, 'w') as outfh:
                outfh.write('image_path\tobject_id\tclass\tscore\txmin\tymin\txmax\tymax\n')
                for region in ann.regions:
                    outfh.write(
                        f"{ann.image_path}\t" + \
                        f"{region['id']}\t{region['class']}\t{region['score']}\t" + \
                        '{}\n'.format('\t'.join([str(x) for x in region['bbox']]))
                    )
            
        job_status = self.set_jobstatus(self.app_code.INFERENCE, self.app_code.JOB__SUMMARIZE, self.app_code.STARTED, '')
        try:
            images = self.check_images(image_dpath)
            logger.info('Finding objects and calculate the summary data using {} CPUs.'.format(cpu))
            
            images_meta = []
            for image in images:
                images_meta.append([
                    image,
                    os.path.join(self.workspace, 'outputs', os.path.splitext(os.path.basename(image))[0] + '.json'),
                    os.path.join(self.workspace, 'outputs', os.path.splitext(os.path.basename(image))[0] + '.objects.txt')
                ])
            
            images_meta = joblib.Parallel(n_jobs=cpu)(
                joblib.delayed(__summarize_objects)(images_meta[i]) for i in range(len(images_meta)))
 

            job_status = self.set_jobstatus(self.app_code.INFERENCE, self.app_code.JOB__SUMMARIZE, self.app_code.FINISHED, '')
        except KeyboardInterrupt:
            job_status = self.set_jobstatus(self.app_code.INFERENCE, self.app_code.JOB__SUMMARIZE, self.app_code.INTERRUPT, '')
        
        except BaseException as e:
            traceback.print_exc()
            job_status = self.set_jobstatus(self.app_code.INFERENCE, self.app_code.JOB__SUMMARIZE, self.app_code.ERROR, str(e))
        else:
            job_status = self.set_jobstatus(self.app_code.INFERENCE, self.app_code.JOB__SUMMARIZE, self.app_code.COMPLETED, '')

        return job_status
