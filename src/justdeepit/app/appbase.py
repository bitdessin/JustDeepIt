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
        # mode
        self.CONFIG = 'BASE'
        self.TRAINING = 'TRAIN'
        self.INFERENCE = 'INFERENCE'
        # status
        self.STARTED = 'STARTED'
        self.RUNNING = 'RUNNING'
        self.FINISHED = 'FINISHED'
        self.ERROR = 'ERROR'
        self.INTERRUPT = 'INTERRUPT'
        self.COMPLETED = 'COMPLETED'
        # job name
        self.JOB__INIT_WORKSPACE = 'INIT_WORKSPACE'
        self.JOB__SAVE_INIT_MODEL = 'SAVE_INIT_MODEL'
        self.JOB__SORT_IMAGES = 'SORT_IMAGES'
        self.JOB__TRAIN_MODEL = 'TRIAN_MODEL'
        self.JOB__INFER = 'INFER'
        self.JOB__SUMMARIZE = 'SUMMARIZE_RESULTS'
        
    def __contains__(self, key):
        return key in self.__dict__.values()
        

class AppBase:
    
    def __init__(self, workspace):
        self.app = 'unset'
        self.app_code = AppCode()
        self.workspace = workspace
        self.image_ext = ('.jpg', '.jpeg', '.png', '.tiff', '.tif')
        self.job_status_fpath = os.path.join(self.workspace, 'job_status.txt')
        self.tmp_dpath = os.path.join(workspace, 'justdeepit.tmp')
        
        #self.init_workspace()
        
    
    def init_workspace(self):
        job_status = self.set_jobstatus(self.app_code.CONFIG, self.app_code.JOB__INIT_WORKSPACE, self.app_code.STARTED, '')
        try:
            ws_subdirs = ['',
                          'justdeepit.tmp',
                          'inputs',
                          'outputs']
            for ws_subdir in ws_subdirs:
                ws_subdir_abspath = os.path.join(self.workspace, ws_subdir)
                if not os.path.exists(ws_subdir_abspath):
                    os.mkdir(ws_subdir_abspath)
            job_status = self.set_jobstatus(self.app_code.CONFIG, self.app_code.JOB__INIT_WORKSPACE, self.app_code.FINISHED, '')
        except KeyboardInterrupt:
            job_status = self.set_jobstatus(self.app_code.CONFIG, self.app_code.JOB__INIT_WORKSPACE, self.app_code.INTERRUPT, '')
        except BaseException as e:
            traceback.print_exc()
            job_status = self.set_jobstatus(self.app_code.CONFIG, self.app_code.JOB__INIT_WORKSPACE, self.app_code.ERROR, str(e))
        else:
            job_status = self.set_jobstatus(self.app_code.CONFIG, self.app_code.JOB__INIT_WORKSPACE, self.app_code.COMPLETED, '')
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
        
    
    def train_model(self, *args, **kwargs):
        raise NotImplementedError()
        
    
    def detect_objects(self, *args, **kwargs):
        raise NotImplementedError()
    
    
    def summarize_objects(self, *args, **kwargs):
        raise NotImplementedError()
    
    
    def check_images(self, image_dpath=None, annotation_path=None, annotation_format=None):
        if image_dpath is None:
            return []

        images = []
        if (annotation_path is not None) and (annotation_format is not None):
            if self.__norm_str(annotation_format) == 'coco':
                with open(annotation_path, 'r') as infh:
                    image_records = json.load(infh)
                for f in image_records['images']:
                    if os.path.exists(os.path.join(image_dpath, os.path.basename(f['file_name']))):
                        images.append(f)            
            elif self.__norm_str(annotation_format) == 'vott':
                with open(annotation_path, 'r') as infh:
                    image_records = json.load(infh)
                for fid, f in image_records['assets'].items():
                    if os.path.exists(os.path.join(image_dpath, os.path.basename(f['asset']['name']))):
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
        else:
            for f in sorted(glob.glob(os.path.join(image_dpath, '**'), recursive=True)):
                if os.path.splitext(f)[1].lower() in self.image_ext:
                    images.append(f)

        logger.info('{} images are found.'.format(len(images)))
        return images
    
    

    
    def __norm_str(self, x):
        return x.replace('-', '').replace(' ', '').lower()


