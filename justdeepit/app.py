import os
import datetime
import pathlib
import logging
import traceback
import justdeepit


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
        self.workspace_ = os.path.join(workspace, 'justdeepit')
        self.image_ext = ('.jpg', '.jpeg', '.png', '.tiff')
        self.job_status_fpath = os.path.join(workspace, 'justdeepit/config', 'job_status.txt')
        self.init_workspace()
        
        
    
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



