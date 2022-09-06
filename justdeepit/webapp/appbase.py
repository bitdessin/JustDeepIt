import os
import glob
import datetime
import pathlib
import json
import logging
import traceback


logger = logging.getLogger(__name__)


class AppCode:
    
    def __init__(self):
        self.CONFIG = 'CONFIG'
        self.TRAINING = 'TRAINING'
        self.INFERENCE = 'INFERENCE'
        self.STARTED = 'STARTED'
        self.RUNNING = 'RUNNING'
        self.FINISHED = 'FINISHED'
        self.ERROR = 'ERROR'
        self.INTERRUPT = 'INTERRUPT'
        self.COMPLETED = 'COMPLETED'
        
        self.JOB__INIT_WORKSPACE = 'INIT_WORKSPACE'
        self.JOB__SAVE_INIT_MODEL = 'SAVE_INIT_MODEL'
        self.JOB__SORT_IMAGES = 'SORT_IMAGES'
        self.JOB__TRAIN_MODEL = 'TRIAN_MODEL'
        self.JOB__INFER = 'INFER'
        self.JOB__SUMMARIZE = 'SUMMARIZE_RESULTS'
        

class AppBase:
    
    def __init__(self, workspace):
        
        self.app = '(base)'
        self.code = AppCode()
        self.workspace = workspace
        self.workspace_ = os.path.join(workspace, 'justdeepitws')
        self.image_ext = ('.jpg', '.jpeg', '.png', '.tiff', '.tif')
        self.job_status_fpath = os.path.join(self.workspace_, 'config', 'job_status.txt')
        self.init_workspace()
        self.images = []
        
    
    def init_workspace(self):
        
        try:
            workspace_subdirs = ['',
                                 'tmp',
                                 'config',
                                 'data', 'data/train', 'data/query',
                                 'log',
                                 'outputs']
            for workspace_subdir in workspace_subdirs:
                workspace_subdir_abspath = os.path.join(self.workspace_, workspace_subdir)
                if not os.path.exists(workspace_subdir_abspath):
                    os.mkdir(workspace_subdir_abspath)
            
            job_status = self.set_jobstatus(self.code.CONFIG, self.code.JOB__INIT_WORKSPACE, self.code.STARTED, '')
            job_status = self.set_jobstatus(self.code.CONFIG, self.code.JOB__INIT_WORKSPACE, self.code.FINISHED, '')
            
        except KeyboardInterrupt:
            job_status = self.set_jobstatus(self.code.CONFIG, self.code.JOB__INIT_WORKSPACE, self.code.INTERRUPT, '')
        
        except BaseException as e:
            traceback.print_exc()
            job_status = self.set_jobstatus(self.code.CONFIG, self.code.JOB__INIT_WORKSPACE, self.code.ERROR, str(e))
        else:
            job_status = self.set_jobstatus(self.code.CONFIG, self.code.JOB__INIT_WORKSPACE, self.code.COMPLETED, '')
        
        return job_status
    
    
    def set_jobstatus(self, module_name, job_code, job_status_code, msg=''):
        
        with open(self.job_status_fpath, 'a') as outfh:
            outfh.write('{}\t{}\t{}\t{}\t{}\t{}\n'.format(
                self.app,
                datetime.datetime.now().isoformat(),
                module_name,
                job_code,
                job_status_code,
                msg
            ))
        
        return {'status': job_status_code, 'msg': msg}
    
    
    def sort_train_images(self, *args, **kwargs):
        raise NotImplementedError()
    
    
    def train_model(self, *args, **kwargs):
        raise NotImplementedError()
    

    def sort_query_images(self, *args, **kwargs):
        raise NotImplementedError()
    
    
    def detect_objects(self, *args, **kwargs):
        raise NotImplementedError()
    
    
    def summarize_objects(self, *args, **kwargs):
        raise NotImplementedError()
    
    
    
    def check_training_images(self, image_dpath, annotation_path, annotation_format, class_label='NA'):
        images = []
        
        if self.__norm_str(annotation_format) == 'coco':
            with open(annotation_path, 'r') as infh:
                image_records = json.load(infh)
                for f in image_records['images']:
                    if os.path.exists(os.path.join(image_dpath, os.path.basename(f['file_name']))):
                        images.append(f)
        
        elif (self.__norm_str(annotation_format) == 'mask') or ('voc' in self.__norm_str(annotation_format)):
            fdict = {}
            for f in glob.glob(os.path.join(image_dpath, '*')):
                fname = os.path.splitext(os.path.basename(f))[0]
                if os.path.splitext(f)[1].lower() in self.image_ext:
                    if fname not in fdict:
                        fdict[fname] = 0
                    fdict[fname] += 1
            for f in glob.glob(os.path.join(annotation_path, '*')):
                fname = os.path.splitext(os.path.basename(f))[0]
                if fname not in fdict:
                    fdict[fname] = 0
                fdict[fname] += 1
            for fname, fval in fdict.items():
                if fval == 2:
                    images.append(fname)
        
        else:
            raise NotImplementedError('JustDeepIt does not support {} format.'.format(annotation_format))
    
        logger.info('There are {} images for model training.'.format(len(images)))
        
        with open(os.path.join(self.workspace_, 'data', 'train', 'train_images.txt'), 'w') as outfh:
            outfh.write('CLASS_LABEL\t{}\n'.format(class_label))
            outfh.write('IMAGES_DPATH\t{}\n'.format(image_dpath))
            outfh.write('ANNOTATION_FPATH\t{}\n'.format(annotation_path))
            outfh.write('ANNOTATION_FORMAT\t{}\n'.format(annotation_format))
            outfh.write('N_IMAGES\t{}\n'.format(len(images)))
    
    
    
    def check_query_images(self, image_dpath):
        images = []
        for f in sorted(glob.glob(os.path.join(image_dpath, '**'), recursive=True)):
            if os.path.splitext(f)[1].lower() in self.image_ext:
                images.append(f)
        with open(os.path.join(self.workspace_, 'data', 'query', 'query_images.txt'), 'w') as outfh:
            for image in images:
                outfh.write('{}\n'.format(image))

        logger.info('There are {} images for inference.'.format(len(images)))
    
    
    
    def seek_query_images(self):
        self.images = []
        with open(os.path.join(self.workspace_, 'data', 'query', 'query_images.txt'), 'r') as infh:
            for _image in infh:
                _image_info = _image.replace('\n', '').split('\t')
                self.images.append(_image_info[0])

    
    
    def __norm_str(self, x):
        return x.replace('-', '').replace(' ', '').lower()


