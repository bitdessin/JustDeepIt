import os
import datetime
import pathlib
import logging
import multiprocessing
#multiprocessing.set_start_method('spawn')
#multiprocessing.set_start_method('fork')
import tkinter
import tkinter.ttk
import tkinter.filedialog
import ttkbootstrap
import agrolens


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)       


class AppBase:
    """Base class of an application
    
    Attributes:
        workspace (str): A path to workspace. All inputs and outputs should be performed
                         in this directory.
        job_status (dict): Job status to record which analysis is uncomplete or completed.
        images (list): A path to images for analysis.
    """
    
    def __init__(self, app, workspace=None):
        self.app       = app
        self.workspace = None
        self._job_status_fpath = None
        self.job_status = None
        if workspace is not None:
            self.init_workspace(workspace)
        self.image_ext = ('.jpg', '.jpeg', '.png', '.tiff')
        self.images = []
    
    
    def init_workspace(self, workspace=None):
        """Initialize workspace

        When a path is given, this function will generate some subdirectories
        in the path to prepare model training or image analysis.

        Args:
            workspace (str): A path to workspace.

        """
        if workspace is not None:
            self.workspace = workspace
            self._job_status_fpath = os.path.join(workspace, 'JOB_STATUS')
            self.job_status = None
        
        ws = ['train_dataset', 'query_dataset', 
              'detection_results',
              'init_params',
              'tmp', 'tmp/train', 'tmp/detection']
        
        for ws_ in ws:
            ws_dpath = os.path.join(self.workspace, ws_)
            if not os.path.exists(ws_dpath):
                os.mkdir(ws_dpath)
        
        if not os.path.exists(self._job_status_fpath):
            self.set_jobstatus('EVAL', 'INIT', 'COMPLETED')
            self.set_jobstatus('TRAIN', 'INIT', 'COMPLETED')
        
        self.init_jobstatus()
        self.refresh_jobstatus()
    
    
    
    
    def set_jobstatus(self, module_name, job_code, status, msg=''):
        """Write job status into log file

        Write job status and additional information into log file.

        Args:
            module_name (str): Module name. Module name of model training and
                               image analysis (including object segmentaion) should be
                               TRAIN and ANALYSIS, respectively.
            msg (str): Additional messages to be recorded in the log file.
            status (str): Job status. It can be set UNLOADED, UNCOMPLETE, COMPLETED, ERROR.

        """
        with open(os.path.join(self._job_status_fpath), 'a') as outfh:
            outfh.write('{}\t{}\t{}\t{}\t{}\t{}\n'.format(
                datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                self.app,
                module_name,
                job_code,
                status,
                msg
            ))



    def init_jobstatus(self):
        """Get the latest job status

        Load the log file in the workshop and refresh the job status.

        Args:
            update (bool): If True, load log file and refresh the job status according to log file.
                           If False, just initiate all jobs with UNLOADED status.

        """
        
        if self.app == 'SOD':
            job_status = {
                'TRAIN': {
                    'INIT':  {'id': 0, 'title': 'Preparation',    'datetime': '', 'status': 'UNLOADED'},
                    'SORT':  {'id': 1, 'title': 'Image Sorting' , 'datetime': '', 'status': 'UNLOADED'},
                    'TRAIN': {'id': 2, 'title': 'Model Training', 'datetime': '', 'status': 'UNLOADED'},
                },
                'EVAL': {
                    'INIT':      {'id': 0, 'title': 'Preparation',      'datetime': '', 'status': 'UNLOADED'},
                  'SORT':      {'id': 1, 'title': 'Image Sorting' ,   'datetime': '', 'status': 'UNLOADED'},
                    'DETECT':    {'id': 2, 'title': 'Object Detection', 'datetime': '', 'status': 'UNLOADED'},
                    'SUMMARIZE': {'id': 3, 'title': 'Summarization',    'datetime': '', 'status': 'UNLOADED'},
                    'MOVIE':     {'id': 4, 'title': 'Movie Generation', 'datetime': '', 'status': 'UNLOADED'},
                },
            }
        elif self.app == 'OD' or self.app == 'IS':
            job_status =  {
                'TRAIN': {
                    'INIT':  {'id': 0, 'title': 'Preparation',    'datetime': '', 'status': 'UNLOADED'},
                    'SORT':  {'id': 1, 'title': 'Image Sorting' , 'datetime': '', 'status': 'UNLOADED'},
                    'TRAIN': {'id': 2, 'title': 'Model Training', 'datetime': '', 'status': 'UNLOADED'},
                },
                'EVAL': {
                    'INIT':      {'id': 0, 'title': 'Preparation',      'datetime': '', 'status': 'UNLOADED'},
                    'SORT':      {'id': 1, 'title': 'Image Sorting' ,   'datetime': '', 'status': 'UNLOADED'},
                    'DETECT':    {'id': 2, 'title': 'Object Detection', 'datetime': '', 'status': 'UNLOADED'},
                },
            }
        
        else:
            raise ValueError('The current version only supports OD for object detection or SOD for salient object detection.')
        
        
        self.job_status = job_status
    
    
    
    
    def refresh_jobstatus(self):
        # replace UNLOADED with UNCOMPLETE
        for module_name in self.job_status.keys():
            for job_code in self.job_status[module_name].keys():
                self.job_status[module_name][job_code]['status'] = 'UNCOMPLETE'

        # update status (replace UNCOMPLETE with the actually status)
        with open(self._job_status_fpath, 'r') as infh:
            for buf in infh:
                bufs = buf.replace('\n', '').split('\t')
                if len(bufs) >= 6 and bufs[2] != 'INIT':
                    if self.app == bufs[1]:
                        if bufs[3] in self.job_status[bufs[2]]:
                            self.job_status[bufs[2]][bufs[3]]['datetime'] = bufs[0]
                            self.job_status[bufs[2]][bufs[3]]['status']   = bufs[4]
    
    
    
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
    
    
    
    
