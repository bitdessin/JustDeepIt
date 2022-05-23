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
        self.images = []
    
    
    def save_initial_model(self, class_label, model_arch, model_config, model_weight, backend):
        
        job_status = self.set_jobstatus(self.code.CONFIG, self.code.JOB__SAVE_INIT_MODEL, self.code.STARTED, '')
        try:
            config_dpath = os.path.join(self.workspace_, 'config')
        
            model = self.__build_model(class_label, model_arch, model_config, model_weight, backend)
            model.save(os.path.join(config_dpath, 'default.pth'))
            
            job_status = self.set_jobstatus(self.code.CONFIG, self.code.JOB__SAVE_INIT_MODEL, self.code.FINISHED, '')
        except BaseException as e:
            traceback.print_exc()
            job_status = self.set_jobstatus(self.code.CONFIG, self.code.JOB__SAVE_INIT_MODEL, self.code.ERROR, str(e))
        else:
            job_status = self.set_jobstatus(self.code.CONFIG, self.code.JOB__SAVE_INIT_MODEL, self.code.COMPLETED, '')

        return job_status
    
     
    def __build_model(self, class_label, model_arch, model_config, model_weight, backend):
        model_arch = model_arch.replace('-', '').replace(' ', '').lower()
        if model_config is None or model_config == '':
            logger.info('Config file is not specified, use the preset config to build model.')
            model_config = None
        else:
            if not os.path.exists(model_config):
                FileNotFoundError('The specified config file [{}] is not found.'.format(model_config))
        if model_weight is not None:
            if not os.path.exists(model_weight):
                FileNotFoundError('The specified weight [{}] is not found.'.format(model_weight))
        
        if self.app == 'OD':
            return justdeepit.models.OD(class_label, model_arch, model_config, model_weight,
                                    os.path.join(self.workspace_, 'tmp'), backend)
        elif self.app == 'IS':
            return justdeepit.models.IS(class_label, model_arch, model_config, model_weight,
                                    os.path.join(self.workspace_, 'tmp'), backend)
    
    
    def sort_train_images(self, class_label=None, image_dpath=None, annotation_fpath=None, annotation_format='coco'):
        
        job_status = self.set_jobstatus(self.code.TRAINING, self.code.JOB__SORT_IMAGES, self.code.STARTED, '')
        
        try:
            images = []
            with open(annotation_fpath, 'r') as infh:
                image_records = json.load(infh)
                for image_record in image_records['images']:
                    images.append(image_record['file_name'])
                
            with open(os.path.join(self.workspace_, 'data', 'train', 'train_images.txt'), 'w') as outfh:
                outfh.write('CLASS_LABEL\t{}\n'.format(class_label))
                outfh.write('IMAGES_DPATH\t{}\n'.format(image_dpath))
                outfh.write('ANNOTATION_FPATH\t{}\n'.format(annotation_fpath))
                outfh.write('ANNOTATION_FORMAT\t{}\n'.format(annotation_format))
                outfh.write('N_IMAGES\t{}\n'.format(len(images)))
            logger.info('There are {} images are valid for model training.'.format(len(images)))
            
            job_status = self.set_jobstatus(self.code.TRAINING, self.code.JOB__SORT_IMAGES, self.code.FINISHED, '')
        except BaseException as e:
            traceback.print_exc()
            job_status = self.set_jobstatus(self.code.TRAINING, self.code.JOB__SORT_IMAGES, self.code.ERROR, str(e))
        else:
            job_status = self.set_jobstatus(self.code.TRAINING, self.code.JOB__SORT_IMAGES, self.code.COMPLETED, '')
    
        return job_status
   
    
    def train_model(self, class_label=None, model_arch='fasterrcnn', model_config=None, model_weight=None, 
                    batchsize=32, epoch=1000, lr=0.0001, score_cutoff=0.7, cpu=8, gpu=1, backend='mmdetection'):
        
        job_status = self.set_jobstatus(self.code.TRAINING, self.code.JOB__TRAIN_MODEL, self.code.STARTED, '')
        try:
            train_data_info = {}
            with open(os.path.join(self.workspace_, 'data', 'train', 'train_images.txt'), 'r') as infh:
                for kv in infh:
                    k, v = kv.replace('\n', '').split('\t')
                    train_data_info[k] = v
                
            init_model_weight = model_weight if os.path.exists(model_weight) else None
            
            model = self.__build_model(class_label, model_arch, model_config, init_model_weight, backend)
            model.train(train_data_info['ANNOTATION_FPATH'], train_data_info['IMAGES_DPATH'],
                        batchsize, epoch, lr, score_cutoff, cpu, gpu)
            model.save(model_weight)  # save .pth and .yaml with same name
            
            job_status = self.set_jobstatus(self.code.TRAINING, self.code.JOB__TRAIN_MODEL, self.code.FINISHED, '')
            
        except BaseException as e:
            traceback.print_exc()
            job_status = self.set_jobstatus(self.code.TRAINING, self.code.JOB__TRAIN_MODEL, self.code.ERROR, str(e))
        else:
            job_status = self.set_jobstatus(self.code.TRAINING, self.code.JOB__TRAIN_MODEL, self.code.COMPLETED,
                                            'Params: batchsize {}; epoch {}; lr: {}.'.format(batchsize, epoch, lr))
        
        return job_status
    
    
    
    def sort_query_images(self, query_image_dpath=None):
        
        job_status = self.set_jobstatus(self.code.INFERENCE, self.code.JOB__SORT_IMAGES, self.code.STARTED, '')
        try:
            image_files = []
            for f in sorted(glob.glob(os.path.join(query_image_dpath, '**'), recursive=True)):
                if os.path.splitext(f)[1].lower() in self.image_ext:
                    image_files.append(f)
                
            # write image files to text file
            with open(os.path.join(self.workspace_, 'data', 'query', 'query_images.txt'), 'w') as outfh:
                for image_file in image_files:
                    outfh.write('{}\n'.format(image_file))
        
            job_status = self.set_jobstatus(self.code.INFERENCE, self.code.JOB__SORT_IMAGES, self.code.FINISHED, '')
        except BaseException as e:
            traceback.print_exc()
            job_status = self.set_jobstatus(self.code.INFERENCE, self.code.JOB__SORT_IMAGES, self.code.ERROR, str(e))
        else:
            job_status = self.set_jobstatus(self.code.INFERENCE, self.code.JOB__SORT_IMAGES, self.code.COMPLETED, '')

        return job_status
    
    
    
    def __seek_images(self):
        self.images = []
        with open(os.path.join(self.workspace_, 'data', 'query', 'query_images.txt'), 'r') as infh:
            for _image in infh:
                _image_info = _image.replace('\n', '').split('\t')
                self.images.append(_image_info[0])
    
    
    
    def detect_objects(self, class_label=None, model_arch='fasterrcnn', model_config=None, model_weight=None,
                       score_cutoff=0.7, batchsize=32, cpu=8, gpu=1, backend='mmdetection'):
        
        def __save_outputs(ws, image_fpath, output, app_id):
            image_name = os.path.splitext(os.path.basename(image_fpath))[0]
            if app_id == 'OD':
                output.draw('bbox', os.path.join(ws, 'outputs', image_name + '.bbox.png'), label=True, score=True)
            elif app_id == 'IS':
                output.draw('contour+bbox', os.path.join(ws, 'outputs', image_name + '.bbox.png'), label=True, score=True)
        
        job_status = self.set_jobstatus(self.code.INFERENCE, self.code.JOB__INFER, self.code.STARTED, '')
        try:
            self.__seek_images()
        
            valid_ws = os.path.join(self.workspace_, 'tmp')
            # if config is not given, use the config saved during training process
            if model_config is None or model_config == '':
                model_config = os.path.join(os.path.splitext(model_weight)[0] + '.yaml')
            model = self.__build_model(class_label, model_arch, model_config, model_weight, backend)
            outputs = model.inference(self.images, score_cutoff, batchsize, cpu, gpu)
                
            joblib.Parallel(n_jobs=cpu)(
                joblib.delayed(__save_outputs)(self.workspace_, self.images[i], outputs[i], self.app) for i in range(len(self.images)))
            #for image_fpath, output in zip(self.images, outputs):
            #    image_name = os.path.splitext(os.path.basename(image_fpath))[0]
            #    output.draw('bbox', os.path.join(self.workspace, 'detection_results', image_name + '.outline.png'), label=True, score=True)
            
            outputs = justdeepit.utils.ImageAnnotations(outputs)
            outputs.format('coco', os.path.join(self.workspace_, 'outputs', 'annotation.json'))
            
            job_status = self.set_jobstatus(self.code.INFERENCE, self.code.JOB__INFER, self.code.FINISHED, '')
                
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
            self.__seek_images()
            logger.info('Finding objects and calculate the summary data using {} CPUs.'.format(cpu))
            
            images_meta = []
            for image in self.images:
                images_meta.append([image,
                                    os.path.join(self.workspace_, 'outputs', 'annotation.json'),
                                    os.path.join(self.workspace_, 'outputs', os.path.splitext(os.path.basename(image))[0] + '.object.txt')])
            
            images_meta = joblib.Parallel(n_jobs=cpu)(
                joblib.delayed(__summarize_objects)(images_meta[i]) for i in range(len(images_meta)))
 

            job_status = self.set_jobstatus(self.code.INFERENCE, self.code.JOB__SUMMARIZE, self.code.FINISHED, '')
        except BaseException as e:
            traceback.print_exc()
            job_status = self.set_jobstatus(self.code.INFERENCE, self.code.JOB__SUMMARIZE, self.code.ERROR, str(e))
        else:
            job_status = self.set_jobstatus(self.code.INFERENCE, self.code.JOB__SUMMARIZE, self.code.COMPLETED, '')

        return job_status
               
    

    def summarise_objects(self, erosion_size, dilation_size, aligned_images, cpu):
        job_status = self.summarize_objects(erosion_size, dilation_size, aligned_images, cpu)
        return job_status

    




