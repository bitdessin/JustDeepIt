import os
import sys
import time
import datetime
import argparse
import json
import threading
import contextlib
import pkg_resources
import logging
import ctypes
import inspect
import torch
import justdeepit
import uvicorn
import traceback
from fastapi import FastAPI, Request, Form, BackgroundTasks
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, BaseSettings


parser = argparse.ArgumentParser(description='JustDeepIt')
parser.add_argument('--host', type=str, default='127.0.0.1', help='hostname')
parser.add_argument('--port', type=int, default=8000, help='port')
args = parser.parse_args()


APP_LOG_FPATH = '.justdeepit.log'
logging.basicConfig(level=logging.INFO,
                    handlers=[logging.FileHandler(filename=APP_LOG_FPATH, mode='w'),
                              logging.StreamHandler()],
                    force=True)
logger = logging.getLogger(__name__)




class Server(uvicorn.Server):
    def __init__(self, config):
        super().__init__(config)
        self.keep_running = True
    
    def install_signal_handlers(self):
        pass

    @contextlib.contextmanager
    def run_in_thread(self):
        thread = threading.Thread(target=self.run, name='Thread-Server')
        thread.start()
        try:
            while not self.started:
                time.sleep(1e-3)
            yield
        finally:
            self.should_exit = True
            thread.join()



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
        self.thread_job = None
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
                    'cpu': 4,
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
                    'epoch': 100,
                    'optimizer': '',
                    'scheduler': '',
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
                    'backend': 'PyTorch',
                    'architecture': 'U2Net',
                    'config': '',
                    'class_label': '',
                    'cpu': 4,
                    'gpu': 1 if torch.cuda.is_available() else 0,
                    'workspace': self.workspace,
                },
                'training': {
                    'status': 'WAITING',
                    'model_weight': '',
                    'image_folder': '',
                    'annotation_format': 'mask',
                    'annotation_path': '',
                    'strategy': 'resizing',
                    'windowsize': 320,
                    'batchsize': 8,
                    'epoch': 100,
                    'optimizer': 'Adam(params, lr=0.001)',
                    'scheduler': '',
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
            # update parameters for all modes, not implemented in this version
            pass
        else:
            params = dict(params)
            params = self.__update_checkboxes(params, mode)
            for param_id in params:
                v = params.get(param_id)
                
                if param_id in ['cutoff']:
                    v = float(v.strip(' '))
                elif param_id in ['batchsize', 'epoch', 'cpu', 'gpu', 'openingks', 'closingks', 'windowsize']:
                    v = int(v.strip(' '))
                elif param_id == 'config':
                    if v is not None and v == '':
                        v = None
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
server = Server(config=uvicorn.Config(app, host=args.host, port=args.port, log_config=None))




@app.get('/')
def index(request: Request):
    return templates.TemplateResponse('index.html', {'request': request,
        'justdeepit_version': justdeepit.__version__})



@app.get('/module/{module_id}')
def module(request: Request, module_id):
    moduleframe.activate_module(module_id)
    # initialize module and parameters
    req_dict = {
        'request': request,
        'justdeepit_version': justdeepit.__version__,
        'module': module_id,
        'module_name': moduleframe.module_desc,
        'backend': moduleframe.params['config']['backend'],
        'supported_formats': get_supported_formats(module_id),
    }
    if module_id == 'OD' or module_id == 'IS':
        req_dict.update({
            'backends': ['MMDetection', 'Detectron2'],
            'architectures': get_architectures(module_id, moduleframe.params['config']['backend'], 'list'),
        })
    elif module_id == 'SOD':
        req_dict.update({
            'architectures': get_architectures(module_id, output_format='list'),
        })
    else:
        raise NotImplementedError('JustDeepIt only supports OD, IS, and SOD.')
    
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
            thread_job = threading.Thread(target=od_config, name='Thread-Module')
        elif mode == 'training':
            moduleframe.update_status('training', 'STARTED')
            thread_job = threading.Thread(target=od_training, name='Thread-Module')
        elif mode == 'inference':
            moduleframe.update_status('inference', 'STARTED')
            thread_job = threading.Thread(target=od_inference, name='Thread-Module')
    elif module_id == 'SOD':
        if mode == 'config':
            moduleframe.update_status('config', 'STARTED')
            thread_job = threading.Thread(target=sod_config, name='Thread-Module')
        elif mode == 'training':
            moduleframe.update_status('training', 'STARTED')
            thread_job = threading.Thread(target=sod_training, name='Thread-Module')
        elif mode == 'inference':
            moduleframe.update_status('inference', 'STARTED')
            thread_job = threading.Thread(target=sod_inference, name='Thread-Module')
    
    if thread_job is not None:
        moduleframe.thread_job = thread_job
        moduleframe.thread_job.start()
    
    return JSONResponse(content={})



def validate_status(statuses):
    s = None
    
    if not isinstance(statuses, (list, tuple)):
        statuses = [statuses]
    
    if all(_['status'] == 'COMPLETED' for _ in statuses):
        s = 'COMPLETED'
        logger.info('[[!SUCCEEDED]] Task succeeded.')
    elif any(_['status'] == 'ERROR' for _ in statuses):
        s = 'ERROR'
        logger.error('The process is failed probably due to JustDeepIt bugs.')
    elif statuses[-1]['status'] == 'INTERRUPT':
        s = 'INTERRUPT'
        logger.error('The process has been interrupted by pressing the STOP button.')
    else:
        s= 'ERROR'
        logger.error('Unexpected error!')
    
    return s



def od_config():
    logger.info('Task Started: [Load Workspace] ...')
    while not moduleframe.stop_threads.is_set():
        try:
            moduleframe.update_status('config', 'RUNNING')
            status = moduleframe.module.init_workspace()
            
            status = validate_status(status)
            moduleframe.update_status('config', status)
        
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
    while not moduleframe.stop_threads.is_set():
        try:
            moduleframe.update_status('training', 'RUNNING')
            if os.path.splitext(moduleframe.params['training']['model_weight'])[1] != '.pth':
                moduleframe.update_params({'model_weight': moduleframe.params['training']['model_weight'] + '.pth'},
                                          'training')
            
            status = moduleframe.module.train_model(
                                moduleframe.params['config']['class_label'],
                                moduleframe.params['training']['image_folder'],
                                moduleframe.params['training']['annotation_path'],
                                moduleframe.params['training']['annotation_format'],
                                moduleframe.params['config']['architecture'],
                                moduleframe.params['config']['config'],
                                moduleframe.params['training']['model_weight'],
                                moduleframe.params['training']['optimizer'],
                                moduleframe.params['training']['scheduler'],
                                moduleframe.params['training']['batchsize'],
                                moduleframe.params['training']['epoch'],
                                moduleframe.params['training']['cutoff'],
                                moduleframe.params['config']['cpu'],
                                moduleframe.params['config']['gpu'],
                                moduleframe.params['config']['backend'])
            
            status = validate_status(status)
            if status == 'COMPLETED':
                model_config_ext = '.py' if moduleframe.params['config']['backend'][0].lower() == 'm' else '.yaml'
                moduleframe.update_params({'config': os.path.splitext(moduleframe.params['training']['model_weight'])[0] + model_config_ext},
                                          'config')
                moduleframe.update_params({'model_weight': moduleframe.params['training']['model_weight']},
                                          'inference')
            moduleframe.update_status('training', status)
    
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
    while not moduleframe.stop_threads.is_set():
        try:
            moduleframe.update_status('inference', 'RUNNING')
 
            status_1 = moduleframe.module.detect_objects(
                                moduleframe.params['config']['class_label'],
                                moduleframe.params['inference']['image_folder'],
                                moduleframe.params['config']['architecture'],
                                moduleframe.params['config']['config'],
                                moduleframe.params['inference']['model_weight'],
                                moduleframe.params['inference']['cutoff'],
                                moduleframe.params['inference']['batchsize'],
                                moduleframe.params['config']['cpu'],
                                moduleframe.params['config']['gpu'],
                                moduleframe.params['config']['backend'])
            status_2 = moduleframe.module.summarize_objects(moduleframe.params['config']['cpu'])
            
            status = validate_status([status_1, status_2])
            moduleframe.update_status('inference', status)
        
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
            
            status = validate_status(statuses)
            moduleframe.update_status('config', status)
            
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
    while not moduleframe.stop_threads.is_set():
        try:
            moduleframe.update_status('training', 'RUNNING')
            if os.path.splitext(moduleframe.params['training']['model_weight'])[1] != '.pth':
                moduleframe.update_params({'model_weight': moduleframe.params['training']['model_weight'] + '.pth'},
                                          'training')

            status = moduleframe.module.train_model(
                                moduleframe.params['training']['image_folder'],
                                moduleframe.params['training']['annotation_path'],
                                moduleframe.params['training']['annotation_format'],
                                moduleframe.params['config']['architecture'],
                                moduleframe.params['training']['model_weight'],
                                moduleframe.params['training']['optimizer'],
                                moduleframe.params['training']['scheduler'],
                                moduleframe.params['training']['batchsize'],
                                moduleframe.params['training']['epoch'],
                                moduleframe.params['config']['cpu'],
                                moduleframe.params['config']['gpu'],
                                moduleframe.params['training']['strategy'],
                                moduleframe.params['training']['windowsize'])
        
            status = validate_status(status)
            if status == 'COMPLETED':
                moduleframe.update_params({'model_weight': moduleframe.params['training']['model_weight']},
                                          'inference')
            moduleframe.update_status('training', status)

    
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
    while not moduleframe.stop_threads.is_set():
        try:
            moduleframe.update_status('inference', 'RUNNING')
            
            status_1 = moduleframe.module.sort_query_images(
                                moduleframe.params['inference']['image_folder'],
                                moduleframe.params['inference']['align_images'])
            status_2 = moduleframe.module.detect_objects(
                                moduleframe.params['config']['architecture'],
                                moduleframe.params['inference']['model_weight'],
                                moduleframe.params['inference']['batchsize'],
                                moduleframe.params['inference']['strategy'],
                                moduleframe.params['inference']['cutoff'],
                                moduleframe.params['inference']['openingks'],
                                moduleframe.params['inference']['closingks'],
                                moduleframe.params['inference']['windowsize'],
                                moduleframe.params['config']['cpu'],
                                moduleframe.params['config']['gpu'])
            status_3 = moduleframe.module.summarize_objects(
                                moduleframe.params['config']['cpu'],
                                moduleframe.params['inference']['timeseries'],
                                moduleframe.params['inference']['openingks'],
                                moduleframe.params['inference']['closingks'])
            status = validate_status([status_1, status_2, status_3])
            moduleframe.update_status('inference', status)

    
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
    backend = backend.lower()
    m = None
    if module == 'OD':
        if backend not in ['mmdetection', 'detectron2']:
            backend = 'mmdetection'
        m = justdeepit.models.OD(model_arch=None)
    elif module == 'IS':
        if backend not in ['mmdetection', 'detectron2']:
            backend = 'mmdetection'
        m = justdeepit.models.IS(model_arch=None)
    elif module == 'SOD':
        if backend not in ['pytorch']:
            backend = 'pytorch'
        m = justdeepit.models.SOD(model_arch=None)
    
    archs = None
    if m is not None:
        archs = m.available_architectures(backend)
    
    if output_format == 'json':
        return JSONResponse(content=archs)
    else:
        return archs



def get_supported_formats(module: str = ''):
    supported_formats = None
    m = None
    if module == 'OD':
        m = justdeepit.models.OD(model_arch=None)
    elif module == 'IS':
        m = justdeepit.models.IS(model_arch=None)
    elif module == 'SOD':
        m = justdeepit.models.SOD(model_arch=None)
    
    if m is not None:
        supported_formats = m.supported_formats()
    return supported_formats





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




@app.get('/api/modelconfig')
def get_modelconfig(config_fpath: str):
    response_data = {'data': '', 'status': ''}
    if os.path.exists(config_fpath):
        with open(config_fpath, 'r') as configfh:
            for line in configfh:
                response_data['data'] += line
        response_data['status'] = '[{}]     SUCCESS: The config file has been loaded.'.format(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    else:
        response_data['status'] = '[{}]     ERROR: FileNotFound, check the file path to the config file.'.format(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    return JSONResponse(content=response_data)


 
@app.post('/api/modelconfig')
async def update_modelconfig(request: Request):
    form = await request.form()
    config_fpath = form.get('file_path')
    config_data = form.get('data')
    response_data = {'data': config_data, 'status': ''}
    if os.path.exists(config_fpath):
        with open(config_fpath, 'w') as configfh:
            configfh.write(config_data)
        response_data['status'] = '[{}]     SUCCESS: The config file has been updated.'.format(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
        logger.info('The model config has been edited manually. The edits has been saved.')
        logger.info('[[!NEWPAGE]]')
    else:
        response_data['status'] = '[{}]     ERROR: FileNotFound, check the file path to the config file.'.format(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    
    return JSONResponse(content=response_data)

   



@app.post('/app/interrupt')
def interrupt_threads():
    if not moduleframe.stop_threads.is_set():
        moduleframe.stop_threads.set()

    # force stop threading
    if moduleframe.thread_job is not None:
        tid = ctypes.c_long(moduleframe.thread_job.ident)
        exctype = KeyboardInterrupt
        if not inspect.isclass(exctype):
            exctype = type(exctype)
        res = ctypes.pythonapi.PyThreadState_SetAsyncExc(tid, ctypes.py_object(exctype))
        if res == 0:
            raise ValueError('Cannot terminate thread since the thread ID is invalid.')
        elif res > 1:
            ctypes.pythonapi.PyThhreadState_SetAsyncExc(tid, None)
            raise SystemError('PyThradState_SetAsyncExc failed.')



def __shutdown_server():
    # interrupt_threads()
    global server
    server.keep_running = False


@app.post("/app/shutdown")
async def shutdown_server(request: Request, background_tasks: BackgroundTasks):
    background_tasks.add_task(__shutdown_server)
    return templates.TemplateResponse('shutdown.html', {'request': request,
        'justdeepit_version': justdeepit.__version__})





def run_app():
    with server.run_in_thread():
        while server.keep_running:
            time.sleep(1)


if __name__ == '__main__':
    run_app()



