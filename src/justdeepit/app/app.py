import os
import sys
import argparse

parser = argparse.ArgumentParser(description='JustDeepIt', add_help=False)
parser.add_argument('-h', '--host', type=str, default='0.0.0.0', help='hostname')
parser.add_argument('-p', '--port', type=int, default=8000, help='port')
parser.add_argument('--help', action='help', help='JustDeepIt Arguments')
args = parser.parse_args()
if hasattr(args, 'help'):
    parser.print_help()
    sys.exit(0)

import time
import datetime
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


# APP temporary directory
APP_ROOT_PATH = os.getcwd()
APP_CONFIG_PATH = os.path.join(APP_ROOT_PATH, '.justdeepit', 'config.json')
APP_LOG_PATH = os.path.join(APP_ROOT_PATH, '.justdeepit', 'log')
if not os.path.exists(os.path.dirname(APP_CONFIG_PATH)):
    os.makedirs(os.path.dirname(APP_CONFIG_PATH))
logging.basicConfig(level=logging.INFO,
                    handlers=[logging.FileHandler(filename=APP_LOG_PATH, mode='w'),
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


class AppCore():
    def __init__(self):
        # basic parameters
        self.id = None
        self.desc = None
        self.model = None
        self.config = None
        
        # threading parameters & status
        self.thread_job = None
        self.stop_threads = threading.Event()
        self.status = {
            'mode': None,
            'status': None,
            'timestamp': None
        }

        self.app_code = justdeepit.app.AppCode()

    
    def init(self, module_id):
        if self.id is None:
            self.id = module_id
            if self.id == 'OD':
                self.desc = 'Object Detection'
            elif self.id == 'SOD':
                self.desc = 'Salient Object Detection'
            elif self.id == 'IS':
                self.desc = 'Instance Segmentation'    
            self.config = self.__init_config()
            logger.info(f'JustDeepIt {self.id} ({self.desc}) has been activated.')

    
    def activate(self):
        ws = self.config[self.id]['BASE']['workspace']
        assert ws is not None, 'Workspace is not set.'

        if self.id == 'OD':
            self.model = justdeepit.app.OD(ws)
        elif self.id == 'IS':
            self.model = justdeepit.app.IS(ws)
        elif self.id == 'SOD':
            self.model = justdeepit.app.SOD(ws)
        else:
            raise ValueError(f'Invalid module id {self.id}.')


    def __get_default_backend(self, module_id):
        default_backends = {
            'OD': 'MMDetection',
            'IS': 'MMDetection',
            'SOD': 'PyTorch',
        }
        return default_backends.get(module_id)


    def __get_default_arch(self, module_id):
        default_archs = {
            'OD': 'Faster R-CNN',
            'IS': 'Mask R-CNN',
            'SOD': 'U2-Net',
        }
        return default_archs.get(module_id)


    def __get_default_cpus(self):
        return len(os.sched_getaffinity(0)) // 2
    

    def __get_default_gpus(self):
        return 1 if torch.cuda.is_available() else 0

    
    def __init_config(self):
        config = {self.id: {}}

        # default configs
        config[self.id].update({
            'BASE': {
                'backend': self.__get_default_backend(self.id),
                'architecture': self.__get_default_arch(self.id),
                'config': '',
                'class_label': '',
                'cpu': self.__get_default_cpus(),
                'gpu': self.__get_default_gpus(),
                'workspace': APP_ROOT_PATH,
            }
        })
        if self.id == 'OD' or self.id == 'IS':
            config[self.id].update({
                'TRAIN': {
                    'model_weight': '',
                    'train_image_dpath': '',
                    'train_annotation_fpath': '',
                    'train_annotation_format': 'COCO',
                    'valid_image_dpath': '',
                    'valid_annotation_fpath': '',
                    'valid_annotation_format': 'COCO',
                    'test_image_dpath': '',
                    'test_annotation_fpath': '',
                    'test_annotation_format': 'COCO',
                    'batchsize': 8,
                    'epoch': 100,
                    'optimizer': '',
                    'scheduler': '',
                    'cutoff': 0.5,
                },
                'INFERENCE': {
                    'model_weight': '',
                    'query_image_dpath': '',
                    'batchsize': 8,
                    'cutoff': 0.5,
                }
            })
        elif self.id == 'SOD':
            config[self.id].update({
                'TRAIN': {
                    'model_weight': '',
                    'image_folder': '',
                    'annotation_format': 'mask',
                    'annotation_path': '',
                    'strategy': 'resizing',
                    'windowsize': 320,
                    'batchsize': 8,
                    'epoch': 100,
                    'optimizer': '',
                    'scheduler': '',
                    'cutoff': 0.5,
                },
                'INFERENCE': {
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
        
        # update the latest configs
        if os.path.exists(APP_CONFIG_PATH):
            with open(APP_CONFIG_PATH, 'r') as fh:
                config_ = json.load(fh)
            if self.id in config_:
                config[self.id] = config_[self.id]
            logger.info(f'JustDeepIt {self.id} ({self.desc}) has loaded the previous configuration.')
        return config
    
    
    def update_config(self, module_id, mode, form_inputs):
        form_inputs = dict(form_inputs)

        for config_key in form_inputs:
            config_val = form_inputs.get(config_key)

            if config_key in ['cutoff']:
                config_val = float(config_val)
            elif config_key in ['batchsize', 'epoch', 'cpu', 'gpu', 'openingks', 'closingks', 'windowsize']:
                config_val = int(config_val)
            elif config_key in ['align_images', 'timeseries']:
                config_val = bool(config_val == 'on')
            elif config_val == '':
                config_val = None
            
            self.config[self.id][mode][config_key] = config_val
        
        with open(APP_CONFIG_PATH, 'w') as outfh:
            json.dump(self.config, outfh, indent=4)
        
        logger.info(f'JustDeepIt {self.id} ({self.desc}) configuration has been updated.')
    
    
    def update_status(self, mode_code, status_code):
        assert mode_code in self.app_code, f'Invalid status code {mode_code}.'
        assert status_code in self.app_code, f'Invalid status code {status_code}.'
        self.status.update({
            'mode': mode_code,
            'status': status_code,
            'timestamp': datetime.datetime.now().isoformat()})




appcore = AppCore()
app = FastAPI()
app.mount('/static',
          StaticFiles(directory=pkg_resources.resource_filename('justdeepit', 'app/static')),
          name='static')
templates = Jinja2Templates(directory=pkg_resources.resource_filename('justdeepit', 'app/templates'))
server = Server(config=uvicorn.Config(app, host=args.host, port=args.port, log_config=None))


@app.get('/')
def index(request: Request):
    return templates.TemplateResponse('index.html', {
        'request': request,
        'version': justdeepit.__version__
    })


@app.get('/module/{module_id}')
def module(request: Request, module_id):
    appcore.init(module_id)
    return templates.TemplateResponse('module.html', {
        'request': request,
        'version': justdeepit.__version__,
        'module': appcore.id,
        'module_desc': appcore.desc,
        'annotation_formats': get_annotation_formats(appcore.id),
        'architectures': get_architectures(None, appcore.id, 'list'),
    })


@app.post('/module/{module_id}/{mode}')
async def module_action(request: Request, module_id, mode):
    # update latest params from HTML form
    form = await request.form()
    appcore.update_config(module_id, mode, form)
    
    # set threading to run threading job
    thread_job = None
    appcore.stop_threads.clear()
    appcore.update_status(mode, 'STARTED')
    if mode == 'BASE':
        thread_job = threading.Thread(target=run_module_config,
                                      args=(module_id, ))
    elif mode == 'TRAIN':
        thread_job = threading.Thread(target=run_module_train,
                                      args=(module_id, ))
    elif mode == 'INFERENCE':
        thread_job = threading.Thread(target=run_module_inference,
                                      args=(module_id, ))

    # run the threading job
    if thread_job is not None:
        appcore.thread_job = thread_job
        appcore.thread_job.start()
    
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



def run_module_config(module_id):
    logger.info('Task Started: [Load Workspace] ...')
    while not appcore.stop_threads.is_set():
        try:
            appcore.activate()
            appcore.update_status('BASE', 'RUNNING')
            status_1 = appcore.model.init_workspace()
            status = validate_status(status_1)
            appcore.update_status('BASE', status)
        except BaseException as e:
            traceback.print_exc()
            appcore.update_status('BASE', 'ERROR')
            logger.error('Confirmation process is failed.')
        finally:
            appcore.stop_threads.set()
            logger.info('Task Finished: [Load Workspace] ...')
            logger.info('[[!NEWPAGE]]')



def run_module_train(module_id):
    logger.info('Task Started: [Train Model] ...')
    while not appcore.stop_threads.is_set():
        try:
            appcore.update_status('TRAIN', 'RUNNING')
            if module_id == 'OD' or module_id == 'IS':
                if os.path.splitext(appcore.config[module_id]['TRAIN']['model_weight'])[1] != '.pth':
                    appcore.update_config(module_id, 'TRAIN', {
                        'model_weight': appcore.config[module_id]['TRAIN']['model_weight'] + '.pth'
                    })
            status_1 = appcore.model.train_model(*__reshape_train_args(module_id))
            status = validate_status(status_1)
            if status == 'COMPLETED':
                if module_id == 'OD' or module_id == 'IS':
                    appcore.update_config(module_id, 'BASE', {
                        'config': os.path.splitext(appcore.config[module_id]['TRAIN']['model_weight'])[0] + '.py'
                    })
                appcore.update_config(module_id, 'INFERENCE', {
                    'model_weight': appcore.config[module_id]['TRAIN']['model_weight']
                })
            appcore.update_status('TRAIN', status)
    
        except BaseException as e:
            traceback.print_exc()
            appcore.update_status('TRAIN', 'ERROR')
            logger.error('Confirmation process is failed.')
        
        finally:
            appcore.stop_threads.set()
            logger.info('Task Finished: [Train Model] ...')
            logger.info('[[!NEWPAGE]]')


def __reshape_train_args_mmdetdataset(module_id, dataset_type):
    if appcore.config[module_id]['TRAIN'][dataset_type + 'images'] is None or \
            appcore.config[module_id]['TRAIN'][dataset_type + 'images'] == '':
            return None
    return {
        'images': appcore.config[module_id]['TRAIN'][dataset_type + 'images'],
        'annotations': appcore.config[module_id]['TRAIN'][dataset_type + 'ann'],
        'annotation_format': appcore.config[module_id]['TRAIN'][dataset_type + 'annfmt']
    }

def __reshape_train_args(module_id):
    if module_id == 'OD' or module_id == 'IS':
        return (appcore.config[module_id]['BASE']['class_label'],
                __reshape_train_args_mmdetdataset(module_id, 'train'),
                __reshape_train_args_mmdetdataset(module_id, 'valid'),
                __reshape_train_args_mmdetdataset(module_id, 'test'),
                appcore.config[module_id]['BASE']['architecture'],
                appcore.config[module_id]['BASE']['config'],
                appcore.config[module_id]['TRAIN']['model_weight'],
                appcore.config[module_id]['TRAIN']['optimizer'],
                appcore.config[module_id]['TRAIN']['scheduler'],
                appcore.config[module_id]['TRAIN']['batchsize'],
                appcore.config[module_id]['TRAIN']['epoch'],
                appcore.config[module_id]['TRAIN']['cutoff'],
                appcore.config[module_id]['BASE']['cpu'],
                appcore.config[module_id]['BASE']['gpu'])
    elif module_id == 'SOD':
        return (appcore.config[module_id]['TRAIN']['trainimages'],
                appcore.config[module_id]['TRAIN']['trainann'],
                appcore.config[module_id]['TRAIN']['trainannfmt'],
                appcore.config[module_id]['BASE']['architecture'],
                appcore.config[module_id]['TRAIN']['model_weight'],
                appcore.config[module_id]['TRAIN']['optimizer'],
                appcore.config[module_id]['TRAIN']['scheduler'],
                appcore.config[module_id]['TRAIN']['batchsize'],
                appcore.config[module_id]['TRAIN']['epoch'],
                appcore.config[module_id]['BASE']['cpu'],
                appcore.config[module_id]['BASE']['gpu'],
                appcore.config[module_id]['TRAIN']['strategy'],
                appcore.config[module_id]['TRAIN']['windowsize'])



def run_module_inference(module_id):
    logger.info('Task Started: [Inference] ...')
    while not appcore.stop_threads.is_set():
        try:
            appcore.update_status('INFERENCE', 'RUNNING')
            if module_id == 'OD' or module_id == 'IS':
                status_1 = appcore.model.detect_objects(*__reshape_inference_args(module_id))
                status_2 = appcore.model.summarize_objects(
                    appcore.config[module_id]['INFERENCE']['image_folder'],
                    appcore.config[module_id]['BASE']['cpu'])
                status = validate_status([status_1, status_2])
            elif module_id == 'SOD':
                status_1 = appcore.model.sort_query_images(
                                appcore.config[module_id]['INFERENCE']['image_folder'],
                                appcore.config[module_id]['INFERENCE']['align_images'])
                status_2 = appcore.model.detect_objects(*__reshape_inference_args(module_id))
                status_3 = appcore.model.summarize_objects(
                                appcore.config[module_id]['BASE']['cpu'],
                                appcore.config[module_id]['INFERENCE']['timeseries'],
                                appcore.config[module_id]['INFERENCE']['openingks'],
                                appcore.config[module_id]['INFERENCE']['closingks'])
                status = validate_status([status_1, status_2, status_3])
            appcore.update_status('INFERENCE', status)
        
        except BaseException as e:
            traceback.print_exc()
            appcore.update_status('INFERENCE', 'ERROR')
            logger.error('Confirmation process is failed.')
        
        finally:
            appcore.stop_threads.set()
            logger.info('Task Finished: [inference] ...')
            logger.info('[[!NEWPAGE]]')



def __reshape_inference_args(module_id):
    if module_id == 'OD' or module_id == 'IS':
        return (appcore.config[module_id]['BASE']['class_label'],
                appcore.config[module_id]['INFERENCE']['image_folder'],
                appcore.config[module_id]['BASE']['architecture'],
                appcore.config[module_id]['BASE']['config'],
                appcore.config[module_id]['INFERENCE']['model_weight'],
                appcore.config[module_id]['INFERENCE']['cutoff'],
                appcore.config[module_id]['INFERENCE']['batchsize'],
                appcore.config[module_id]['BASE']['cpu'],
                appcore.config[module_id]['BASE']['gpu'])
    elif module_id == 'SOD':
        return (appcore.config[module_id]['BASE']['architecture'],
                appcore.config[module_id]['INFERENCE']['model_weight'],
                appcore.config[module_id]['INFERENCE']['batchsize'],
                appcore.config[module_id]['INFERENCE']['strategy'],
                appcore.config[module_id]['INFERENCE']['cutoff'],
                appcore.config[module_id]['INFERENCE']['openingks'],
                appcore.config[module_id]['INFERENCE']['closingks'],
                appcore.config[module_id]['INFERENCE']['windowsize'],
                appcore.config[module_id]['BASE']['cpu'],
                appcore.config[module_id]['BASE']['gpu'])




@app.get('/api/config')
def get_config(request: Request, module = None):
    if appcore.config is not None:
        if module is None:
            return JSONResponse(content=appcore.config)
        elif module in appcore.config:
            return JSONResponse(content=appcore.config[module])
    return JSONResponse(content={})
    


@app.get('/api/architecture')
def get_architectures(request: Request, module='', output_format='json'):
    m = None
    if module == 'OD':
        m = justdeepit.models.OD(model_arch=None)
    elif module == 'IS':
        m = justdeepit.models.IS(model_arch=None)
    elif module == 'SOD':
        m = justdeepit.models.SOD(model_arch=None)
    
    archs = None
    if m is not None:
        archs = m.available_architectures()
    
    if output_format == 'json':
        return JSONResponse(content=archs)
    else:
        return archs


def get_annotation_formats(module=''):
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
    return JSONResponse(content=appcore.status)


@app.get('/api/log')
def get_log(request: Request):
    log_records = ''
    with open(APP_LOG_PATH, 'r') as infh:
        for log_record in infh:
            if ':uvicorn' not in log_record:
                log_record = log_record.replace('\n', '')
                if 'ERROR' in log_record:
                    log_record = f'<p class="log-error">{log_record}</p>'
                elif '[[!SUCCEEDED]]' in log_record:
                    log_record = '<p class="log-succeeded">' + log_record.replace('[[!SUCCEEDED]]', '').replace('INFO:', 'SUCCEEDED:') + '</p>'
                elif '[[!NEWPAGE]]' in log_record:
                    log_record = '<p class="log-newpage"></p>'
                else:
                    log_record = f'<p>{log_record}</p>'
                log_records += log_record
    return log_records


@app.post('/api/dirtree')
async def get_dirtree(request: Request):
    form = await request.form()    
    dirpath = form.get('dir')
    if dirpath is None or dirpath == 'null':
        dirpath = APP_ROOT_PATH
    else:
        dirpath = os.path.join(APP_ROOT_PATH, dirpath)
    
    ftree = ['<ul class="jqueryFileTree" style="display: none;">']
    try:
        ftree = ['<ul class="jqueryFileTree" style="display: none;">']
        for fname in os.listdir(dirpath):
            fpath =os.path.join(dirpath, fname)
            if os.path.isdir(fpath):
                ftree.append(f'<li class="directory collapsed"><a rel="{fpath}/">{fname}</a></li>')
            else:
                e = os.path.splitext(fname)[1][1:]
                ftree.append(f'<li class="file ext_{e}"><a rel="{fpath}">{fname}</a></li>')
        ftree.append('</ul>')
    except Exception(e):
        ftree.append(f'Could not load directory: {str(e)}')
    ftree.append('</ul>')
    return HTMLResponse(''.join(ftree))


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
def interrupt_threads(request: Request):
    if not appcore.stop_threads.is_set():
        appcore.stop_threads.set()

    # force stop threading
    if appcore.thread_job is not None:
        tid = ctypes.c_long(appcore.thread_job.ident)
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
    if os.path.exists(APP_LOG_PATH):
        os.remove(APP_LOG_PATH)
    global server
    server.keep_running = False
    

@app.post("/app/shutdown")
async def shutdown_server(request: Request, background_tasks: BackgroundTasks):
    background_tasks.add_task(__shutdown_server)
    return templates.TemplateResponse('shutdown.html', {
        'request': request,
        'version': justdeepit.__version__
    })


def run_app():
    with server.run_in_thread():
        while server.keep_running:
            time.sleep(1)


if __name__ == '__main__':
    run_app()
