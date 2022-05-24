import os
import sys
import time
import datetime
import argparse
import json
import pkg_resources
import logging
import ctypes
import inspect
import torch
import justdeepit
import threading
import uvicorn
import traceback
from fastapi import FastAPI, Request, Form, BackgroundTasks
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, BaseSettings


APP_LOG_FPATH = '.justdeepit.log'
logging.basicConfig(level=logging.INFO,
                    handlers=[logging.FileHandler(filename=APP_LOG_FPATH, mode='w'),
                              logging.StreamHandler()],
                    force=True)
logger = logging.getLogger(__name__)



class ModuleFrame():
 
    def __init__(self, workspace=None):
        
        # basic parameters
        self.workspace = workspace
        self.active_module = None
        self.module = None
        self.module_desc = None
        self.__params_file = os.path.join(workspace, 'config.json')
        self.params = self.__init_params()
        
        # threading parameters
        self.thread = None
        self.stop_threads = threading.Event()
        self.status = {'module': None, 'mode': None, 'status': None, 'timestamp': None}
    
    
    def activate_module(self, module_id):
        
        if self.module is None:
            self.params = self.__init_params(module_id)
            self.active_module = module_id
        
            if module_id == 'OD':
                self.module = justdeepit.webapp.OD(self.workspace)
                self.module_desc = 'Object Detection'
            elif module_id == 'IS':
                self.module = justdeepit.webapp.IS(self.workspace)
                self.module_desc = 'Instance Segmentation'
            elif module_id == 'SOD':
                self.module = justdeepit.webapp.SOD(self.workspace)
                self.module_desc = 'Salient Object Detection'
        
            logger.info('JustDeepIt module for {} has been activated.'.format(self.module_desc))
            
            
    
    def __init_params(self, module_id=None):
        params = {}
        params.update({'module': module_id})
        
        if module_id == 'OD' or module_id == 'IS':
            params.update({
                'config': {
                    'status': 'WAITING',
                    'backend': 'MMDetection',
                    'architecture': 'Faster R-CNN' if module_id == 'OD' else 'Mask R-CNN',
                    'config': '',
                    'class_label': '',
                    'cpu': os.cpu_count(),
                    'gpu': 1 if torch.cuda.is_available() else 0,
                    'workspace': self.workspace,
                },
                'training': {
                    'status': 'WAITING',
                    'model_weight': '',
                    'image_folder': '',
                    'annotation_format': 'COCO',
                    'annotation_path': '',
                    'batchsize': 8,
                    'epoch': 1000,
                    'lr': 0.001,
                    'cutoff': 0.5,
                },
                'inference': {
                    'status': 'WAITING',
                    'model_weight': '',
                    'image_folder': '',
                    'batchsize': 8,
                    'cutoff': 0.5,
                }
            })
        elif module_id == 'SOD':    
             params.update({
                'config': {
                    'status': 'WAITING',
                    'architecture': 'U2Net',
                    'config': '',
                    'class_label': '',
                    'cpu': os.cpu_count(),
                    'gpu': 1 if torch.cuda.is_available() else 0,
                    'workspace': self.workspace,
                },
                'training': {
                    'status': 'WAITING',
                    'model_weight': '',
                    'image_folder': '',
                    'image_suffix': '',
                    'mask_suffix': '',
                    'strategy': 'resizing',
                    'windowsize': 320,
                    'batchsize': 8,
                    'epoch': 1000,
                    'lr': 0.001,
                    'cutoff': 0.5,
                },
                'inference': {
                    'status': 'WAITING',
                    'model_weight': '',
                    'image_folder': '',
                    'batchsize': 8,
                    'cutoff': 0.5,
                    'strategy': 'resizing',
                    'windowsize': 320,
                    'openingks': 0,
                    'closingks': 0,
                    'align_images': False,
                    'timeseries': False,
                }
            })
        
        
        if os.path.exists(self.__params_file):
            with open(self.__params_file, 'r') as paramsfh:
                params_ = json.load(paramsfh)
            if params['module'] == params_['module']:
                params = params_
        
        return params
    
    
    
    def update_params(self, params, mode=None):
        if mode is None:
            # update parameters for all modes
            pass
        else:
            params = dict(params)
            params = self.__update_checkboxes(params, mode)
            for param_id in params:
                v = params.get(param_id)
                v = v.strip(' ')
                if param_id in ['lr', 'cutoff']:
                    v = float(v)
                elif param_id in ['batchsize', 'epoch', 'cpu', 'gpu', 'openingks', 'closingks', 'windowsize']:
                    v = int(v)
                self.params[mode].update({param_id: v})
         
        # save the latest parameters       
        with open(self.__params_file, 'w') as outfh:
            json.dump(self.params, outfh)


    def __update_checkboxes(self, params, mode):
        if self.active_module is not None:
            chkboxes = [['SOD', 'inference', 'align_images'],
                        ['SOD', 'inference', 'timeseries']]
            for _1, _2, _3 in chkboxes:
                if self.active_module == _1 and mode == _2:
                    params[_3] = True if _3 in params and  params[_3] == 'on' else False
        return params
    
    
 
    def update_status(self, mode, status):
        self.status.update({
            'module': self.active_module,
            'mode': mode,
            'status': status,
            'timestamp': datetime.datetime.now().isoformat()})
        self.params[mode]['status'] = status





moduleframe = ModuleFrame(os.getcwd())
app = FastAPI()
app.mount('/static',
          StaticFiles(directory=pkg_resources.resource_filename('justdeepit', 'webapp/static')),
          name='static')
templates = Jinja2Templates(directory=pkg_resources.resource_filename('justdeepit', 'webapp/templates'))








@app.get('/')
def index(request: Request):
    return templates.TemplateResponse('index.html', {'request': request,
        'justdeepit_version': justdeepit.__version__})



@app.get('/module/{module_id}')
def od(request: Request, module_id):
    
    moduleframe.activate_module(module_id)
    
    # initialize module and parameters
    req_dict = {
        'request': request,
        'justdeepit_version': justdeepit.__version__,
        'module': module_id,
        'module_name': moduleframe.module_desc,
    }
    if module_id == 'OD' or module_id == 'IS':
        req_dict.update({
            'backends': ['MMDetection', 'Detectron2'],
            'architectures': get_architectures(module_id, 'MMDetection', 'list'),
        })
    elif module_id == 'SOD':
        req_dict.update({
             'architectures': ['U2Net'], 
        })
    
    return templates.TemplateResponse('module.html', req_dict)




@app.post('/module/{module_id}/{mode}')
async def module_action(request: Request, module_id, mode):
    
    # update latest params from HTML form
    form = await request.form()
    moduleframe.update_params(form, mode)
    
    # set threading to run threading job
    thread_job = None
    moduleframe.stop_threads.clear()
    if module_id == 'OD' or module_id == 'IS':
        if mode == 'config':
            moduleframe.update_status('config', 'STARTED')
            thread_job = threading.Thread(target=od_config)
        elif mode == 'training':
            moduleframe.update_status('training', 'STARTED')
            thread_job = threading.Thread(target=od_training)
        elif mode == 'inference':
            moduleframe.update_status('inference', 'STARTED')
            thread_job = threading.Thread(target=od_inference)
    elif module_id == 'SOD':
        if mode == 'config':
            moduleframe.update_status('config', 'STARTED')
            thread_job = threading.Thread(target=sod_config)
        elif mode == 'training':
            moduleframe.update_status('training', 'STARTED')
            thread_job = threading.Thread(target=sod_training)
        elif mode == 'inference':
            moduleframe.update_status('inference', 'STARTED')
            thread_job = threading.Thread(target=sod_inference)
    
    if thread_job is not None:
        moduleframe.thread_job = thread_job
        moduleframe.thread_job.start()
    
    return JSONResponse(content={})




def od_config():
    logger.info('Task Started: [Load Workspace] ...')
    statuses = []
    while not moduleframe.stop_threads.is_set():
    
        try:
            moduleframe.update_status('config', 'RUNNING')
        
            status_ = moduleframe.module.init_workspace()
            statuses.append(status_)
            status_ = moduleframe.module.save_initial_model(moduleframe.params['config']['class_label'],
                                                 moduleframe.params['config']['architecture'],
                                                 moduleframe.params['config']['config'],
                                                 None,
                                                 moduleframe.params['config']['backend'])
            statuses.append(status_)
        
            if all(_['status'] == 'COMPLETED' for _ in statuses):
                model_config_ext = '.py' if moduleframe.params['config']['backend'].replace(' ', '').replace('-', '').lower()[0:5] == 'mmdet' else '.yaml'
                moduleframe.update_params({'config': os.path.join(moduleframe.params['config']['workspace'],
                                                             'justdeepitws/config/default') + model_config_ext},
                                     'config')
                moduleframe.update_status('config', 'COMPLETED')
                logger.info('[[!SUCCEEDED]] Task succeeded.')
            else:
                moduleframe.update_status('config', 'ERROR')
                logger.error('Workspace loading and initialization are failed probably due to JustDeepIt bugs.')
        except BaseException as e:
            traceback.print_exc()
            moduleframe.update_status('config', 'ERROR')
            logger.error('Confirmation process is failed.')
        
        finally:
            moduleframe.stop_threads.set()
            logger.info('Task Finished: [Load Workspace] ...')
            logger.info('[[!NEWPAGE]]')



def od_training():
    logger.info('Task Started: [Train Model] ...')
    statuses = []
    while not moduleframe.stop_threads.is_set():
        try:
            moduleframe.update_status('training', 'RUNNING')
            
            status_ = moduleframe.module.save_initial_model(moduleframe.params['config']['class_label'],
                                                 moduleframe.params['config']['architecture'],
                                                 moduleframe.params['config']['config'],
                                                 None,
                                                 moduleframe.params['config']['backend'])
            statuses.append(status_)
            status_ = moduleframe.module.sort_train_images(moduleframe.params['config']['class_label'],
                                                moduleframe.params['training']['image_folder'],
                                                moduleframe.params['training']['annotation_path'],
                                                moduleframe.params['training']['annotation_format'])
            statuses.append(status_)
            status_ = moduleframe.module.train_model(moduleframe.params['config']['class_label'],
                                          moduleframe.params['config']['architecture'],
                                          moduleframe.params['config']['config'],
                                          moduleframe.params['training']['model_weight'],
                                          moduleframe.params['training']['batchsize'],
                                          moduleframe.params['training']['epoch'],
                                          moduleframe.params['training']['lr'],
                                          moduleframe.params['training']['cutoff'],
                                          moduleframe.params['config']['cpu'],
                                          moduleframe.params['config']['gpu'],
                                          moduleframe.params['config']['backend'])
            statuses.append(status_)
        
            if all(_['status'] == 'COMPLETED' for _ in statuses):
                moduleframe.update_status('training', 'COMPLETED')
                logger.info('[[!SUCCEEDED]] Task succeeded.')
            else:
                moduleframe.update_status('training', 'ERROR')
                logger.error('Workspace loading and initialization are failed probably due to JustDeepIt bugs.')

    
        except BaseException as e:
            traceback.print_exc()
            moduleframe.update_status('training', 'ERROR')
            logger.error('Confirmation process is failed.')
        
        finally:
            moduleframe.stop_threads.set()
            logger.info('Task Finished: [Train Model] ...')
            logger.info('[[!NEWPAGE]]')



def od_inference():
    logger.info('Task Started: [Inference] ...')
    statuses = []
    while not moduleframe.stop_threads.is_set():
        try:
            moduleframe.update_status('inference', 'RUNNING')
 

            status_ = moduleframe.module.save_initial_model(moduleframe.params['config']['class_label'],
                                                 moduleframe.params['config']['architecture'],
                                                 moduleframe.params['config']['config'],
                                                 None,
                                                 moduleframe.params['config']['backend'])
            statuses.append(status_)
            status_ = moduleframe.module.sort_query_images(moduleframe.params['inference']['image_folder'])
            statuses.append(status_)
            status_ = moduleframe.module.detect_objects(moduleframe.params['config']['class_label'],
                                             moduleframe.params['config']['architecture'],
                                             moduleframe.params['config']['config'],
                                             moduleframe.params['inference']['model_weight'],
                                             moduleframe.params['inference']['cutoff'],
                                             moduleframe.params['inference']['batchsize'],
                                             moduleframe.params['config']['cpu'],
                                             moduleframe.params['config']['gpu'],
                                             moduleframe.params['config']['backend'])
            statuses.append(status_)
            status_ = moduleframe.module.summarize_objects(moduleframe.params['config']['cpu'])
            statuses.append(status_)
        
            if all(_['status'] == 'COMPLETED' for _ in statuses):
                moduleframe.update_status('inference', 'COMPLETED')
                logger.info('[[!SUCCEEDED]] Task succeeded.')
            else:
                moduleframe.update_status('inference', 'ERROR')
                logger.error('Workspace loading and initialization are failed probably due to JustDeepIt bugs.')
    
        except BaseException as e:
            traceback.print_exc()
            moduleframe.update_status('inference', 'ERROR')
            logger.error('Confirmation process is failed.')
        
        finally:
            moduleframe.stop_threads.set()
            logger.info('Task Finished: [inference] ...')
            logger.info('[[!NEWPAGE]]')




def sod_config():
    logger.info('Task Started: [Load Workspace] ...')
    statuses = []
    while not moduleframe.stop_threads.is_set():
    
        try:
            moduleframe.update_status('config', 'RUNNING')
        
            status_ = moduleframe.module.init_workspace()
            statuses.append(status_)
        
            if all(_['status'] == 'COMPLETED' for _ in statuses):
                moduleframe.update_status('config', 'COMPLETED')
                logger.info('[[!SUCCEEDED]] Task succeeded.')
            else:
                moduleframe.update_status('config', 'ERROR')
                logger.error('Workspace loading and initialization are failed probably due to JustDeepIt bugs.')
        except BaseException as e:
            traceback.print_exc()
            moduleframe.update_status('config', 'ERROR')
            logger.error('Confirmation process is failed.')
        
        finally:
            moduleframe.stop_threads.set()
            logger.info('Task Finished: [Load Workspace] ...')
            logger.info('[[!NEWPAGE]]')



def sod_training():
    logger.info('Task Started: [Train Model] ...')
    statuses = []
    while not moduleframe.stop_threads.is_set():
        try:
            moduleframe.update_status('training', 'RUNNING')
            
            status_ = moduleframe.module.sort_train_images(moduleframe.params['training']['image_folder'],
                                                           None,
                                                           'mask',
                                                           moduleframe.params['training']['image_suffix'],
                                                           moduleframe.params['training']['mask_suffix'])
            statuses.append(status_)
            status_ = moduleframe.module.train_model(moduleframe.params['config']['architecture'],
                                                     moduleframe.params['training']['model_weight'],
                                                     moduleframe.params['training']['batchsize'],
                                                     moduleframe.params['training']['epoch'],
                                                     moduleframe.params['training']['lr'],
                                                     moduleframe.params['config']['cpu'],
                                                     moduleframe.params['config']['gpu'],
                                                     moduleframe.params['training']['strategy'],
                                                     moduleframe.params['training']['windowsize'])
            statuses.append(status_)
        
            if all(_['status'] == 'COMPLETED' for _ in statuses):
                moduleframe.update_status('training', 'COMPLETED')
                logger.info('[[!SUCCEEDED]] Task succeeded.')
            else:
                moduleframe.update_status('training', 'ERROR')
                logger.error('Workspace loading and initialization are failed probably due to JustDeepIt bugs.')

    
        except BaseException as e:
            traceback.print_exc()
            moduleframe.update_status('training', 'ERROR')
            logger.error('Confirmation process is failed.')
        
        finally:
            moduleframe.stop_threads.set()
            logger.info('Task Finished: [Train Model] ...')
            logger.info('[[!NEWPAGE]]')



def sod_inference():
    logger.info('Task Started: [Inference] ...')
    statuses = []
    while not moduleframe.stop_threads.is_set():
        try:
            moduleframe.update_status('inference', 'RUNNING')
            
            status_ = moduleframe.module.sort_query_images(moduleframe.params['inference']['image_folder'],
                                                           moduleframe.params['inference']['align_images'])
            statuses.append(status_)
            status_ = moduleframe.module.detect_objects(moduleframe.params['config']['architecture'],
                                             moduleframe.params['inference']['model_weight'],
                                             moduleframe.params['inference']['batchsize'],
                                             moduleframe.params['inference']['strategy'],
                                             moduleframe.params['inference']['cutoff'],
                                             moduleframe.params['inference']['openingks'],
                                             moduleframe.params['inference']['closingks'],
                                             moduleframe.params['inference']['windowsize'],
                                             moduleframe.params['config']['cpu'],
                                             moduleframe.params['config']['gpu'])
            statuses.append(status_)
            status_ = moduleframe.module.summarize_objects(moduleframe.params['config']['cpu'],
                                             moduleframe.params['inference']['timeseries'],
                                             moduleframe.params['inference']['openingks'],
                                             moduleframe.params['inference']['closingks'])
            
            if all(_['status'] == 'COMPLETED' for _ in statuses):
                moduleframe.update_status('inference', 'COMPLETED')
                logger.info('[[!SUCCEEDED]] Task succeeded.')
            else:
                moduleframe.update_status('inference', 'ERROR')
                logger.error('Workspace loading and initialization are failed probably due to JustDeepIt bugs.')
    
        except BaseException as e:
            traceback.print_exc()
            moduleframe.update_status('inference', 'ERROR')
            logger.error('Confirmation process is failed.')
        
        finally:
            moduleframe.stop_threads.set()
            logger.info('Task Finished: [inference] ...')
            logger.info('[[!NEWPAGE]]')











@app.get('/api/params')
def get_params():
    return JSONResponse(content=moduleframe.params)
    


@app.get('/api/architecture')
def get_architectures(module: str = '', backend: str = 'mmdetection', output_format: str = 'json'):
    archs = None
    m = None
    if module == 'OD':
        m = justdeepit.models.OD()
    elif module == 'IS':
        m = justdeepit.models.IS()
    elif module == 'SOD':
        m = justdeepit.models.SOD()
    
    if m is not None:
        archs = m.available_architectures[backend.lower()]
    
    if output_format == 'json':
        return JSONResponse(content=archs)
    else:
        return archs


@app.get('/api/status')
def get_status(request: Request):
    return JSONResponse(content=moduleframe.status)


@app.get('/api/log')
def get_log():
    log_records = '';
    with open(APP_LOG_FPATH, 'r') as infh:
        for log_record in infh:
            if ':uvicorn' not in log_record:
                log_record = log_record.replace('\n', '')
                if 'ERROR' in log_record:
                    log_record = '<p class="log-error">' + log_record + '</p>'
                elif '[[!SUCCEEDED]]' in log_record:
                    log_record = '<p class="log-succeeded">' + log_record.replace('[[!SUCCEEDED]]', '').replace('INFO:', 'SUCCEEDED:') + '</p>'
                elif '[[!NEWPAGE]]' in log_record:
                    log_record = '<p class="log-newpage"></p>'
                else:
                    log_record = '<p>' + log_record + '</p>'
                log_records += log_record
                
    return log_records



@app.get('/api/interrupt')
@app.post('/api/interrupt')
def interrupt_threads():
    
    if not moduleframe.stop_threads.is_set():
        moduleframe.stop_threads.set()

    # force stop threading
    tid = ctypes.c_long(moduleframe.thread_job.ident)
    exctype = KeyboardInterrupt
    if not inspect.isclass(exctype):
        exctype = type(exctype)
    res = ctypes.pythonapi.PyThreadState_SetAsyncExc(tid, ctypes.py_object(exctype))
    
    moduleframe.update_status('inference', 'ERROR')
    
    if res == 0:
        raise ValueError('Cannot terminate thread since the thread ID is invalid.')
    elif res != 1:
        ctypes.pythonapi.PyThhreadState_SetAsyncExc(tid, None)
        raise SystemError('PyThradState_SetAsyncExc failed.')
    
    time.sleep(10) # wailt for the process is exactly stopped by OS
    logger.error('The process is stopped by pressing STOP button.')



@app.get('/api/dirtree')
def get_dirtree(dirpath: str = None, include_file: int = 1):
    if dirpath is None:
        dirpath = moduleframe.workspace
    else:
        dirpath = os.path.join(moduleframe.workspace, dirpath)
    
    def __get_filetree(dirpath, idx_start, include_file, n_files_cutoff=20):
        filetree_dict = []
        n_files = [0, 0] # (directories, files)
        
        # check #files
        for fpath in sorted(os.listdir(dirpath)):
            if fpath.startswith('.'):
                continue
            fpath = os.path.join(dirpath, fpath)
            if  os.path.isdir(os.path.join(dirpath, fpath)):
                n_files[0] += 1
            if os.path.isfile(fpath):
                n_files[1] += 1
            
        # walk directories
        for fpath in sorted(os.listdir(dirpath)):
            if fpath.startswith('.'):
                continue
            fpath = os.path.join(dirpath, fpath)
            if os.path.isdir(fpath):
                idx_start[0] += 1
                filetree_dict.append({
                    'id': idx_start[0],
                    'name': os.path.basename(fpath),
                    'children': __get_filetree(fpath, idx_start, include_file),
                })
            elif os.path.isfile(fpath):
                if include_file == 1 and n_files[1] <= n_files_cutoff:
                    idx_start[0] += 1
                    filetree_dict.append({
                        'id': idx_start[0],
                        'name': os.path.basename(fpath),
                    })
                
        
        # if #files is larger than the cutoff, set '...' to represent files.
        if include_file == 1 and n_files[1] > n_files_cutoff:
            idx_start[0] += 1
            filetree_dict.append({
                'id': idx_start[0],
                'name': '...',
            })
        
        return filetree_dict
    

    if os.path.exists(dirpath):
        idx_start = [0]
        filetree_dict = [{
            'id': 0,
            'name': dirpath,
            'children': __get_filetree(dirpath, idx_start, include_file)
        }]
    else:
        filetree_dict = []
    
    return JSONResponse(content=filetree_dict)
        
    
    
   





def run_app():

    parser = argparse.ArgumentParser(description='JustDeepIt')
    parser.add_argument('--host', type=str, default='127.0.0.1', help='hostname')
    parser.add_argument('--port', type=int, default=8000, help='port')
    args = parser.parse_args() 

    uvicorn.run(app, host=args.host, port=args.port, log_config=None)


if __name__ == '__main__':
    run_app()



