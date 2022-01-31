import os
import datetime
import random
import pathlib
import glob
import json
import tqdm
import logging
import multiprocessing
import joblib
import traceback
import numpy as np
import pandas as pd
import PIL
import cv2
import skimage.io
import skimage.color
import skimage.exposure
import torch
import tkinter
import tkinter.filedialog
import ttkbootstrap as ttk
import agrolens


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)       
cv2.setNumThreads(1)


class IS(agrolens.app.AppBase):
    """Application for object detection
    
    """
    
    def __init__(self, workspace=None):
        super().__init__('IS', workspace)
        self.backend_available = ['MMDetection', 'Detectron2']
        self.models_available = {'MMDetection': ['Mask R-CNN'],
                                 'Detectron2': ['Mask R-CNN']}
        self.formats_available = ['COCO', 'Pascal VOC']
    
    
    
    def sort_train_images(self, class_label=None, image_dpath=None, annotation_fpath=None, annotation_format='coco'):
        """Sort training images
        
        This method check images and annotations in :file:`inputs/train_images`
        directory of workspace.
        All training images shoud directly be put in :file:`inputs/train_images` with
        the COCO format annotation.
        This method will check images are existed or not to prevent errors during model trainig.

        Args:
            image_dpath (str): A path to a directory which contains training images.
            annotation_fpath (str): A path to a file or directory which contains image annotations.
            annotation_format (str): A string to specify annotation format.
        Returs:
            (str): A status of running result.
        """
        
        run_status = True
        self.set_jobstatus('TRAIN', 'SORT', 'BEGIN')
        
        try:
            images = []
            with open(annotation_fpath, 'r') as infh:
                image_records = json.load(infh)
                for image_record in image_records['images']:
                    images.append(image_record['file_name'])
            
            with open(os.path.join(self.workspace, 'train_dataset', 'train_images.txt'), 'w') as outfh:
                outfh.write('CLASS_LABEL\t{}\n'.format(class_label))
                outfh.write('IMAGES_DPATH\t{}\n'.format(image_dpath))
                outfh.write('ANNOTATION_FPATH\t{}\n'.format(annotation_fpath))
                outfh.write('ANNOTATION_FORMAT\t{}\n'.format(annotation_format))
                outfh.write('N_IMAGES\t{}\n'.format(len(images)))
            logger.info('There are {} images are valid for model training.'.format(len(images)))
        
        except BaseException as e:
            traceback.print_exc()
            self.set_jobstatus('TRAIN', 'SORT', 'ERROR', str(e))
            run_status = False
        else:
            self.set_jobstatus('TRAIN', 'SORT', 'COMPLETED')
        
        return run_status
    
    
    
    def build_model(self, class_labels, model_arch, model_config, model_weight, ws, backend):
        
        # model architecture
        model_arch = model_arch.replace('-', '').replace(' ', '').lower()
        
        # config file
        if model_config is None or model_config == '':
            logger.info('Config file is not specified, use the preset config to build model.')
            model_config = None
        else:
            if not os.path.exists(model_config):
                FileNotFoundError('The specified config file [{}] is not found.'.format(model_config))
        
        # weight file
        if model_weight is not None:
            if not os.path.exists(model_weight):
                FileNotFoundError('The specified weight [{}] is not found.'.format(model_weight))
        
        return agrolens.models.IS(class_labels, model_arch, model_config, model_weight, ws, backend)
        
    
    
    def train_model(self, class_labels=None, model_arch='fasterrcnn', model_config=None, model_weight=None, 
                    batchsize=32, epoch=1000, lr=0.0001, score_cutoff=0.7, cpu=8, gpu=1, backend='mmdetection'):
        """Train model
        
        """
        
        run_status = True
        
        self.set_jobstatus('TRAIN', 'TRAIN', 'BEGIN')
        try:
            train_data_info = {}
            with open(os.path.join(self.workspace, 'train_dataset', 'train_images.txt'), 'r') as infh:
                for kv in infh:
                    k, v = kv.replace('\n', '').split('\t')
                    train_data_info[k] = v
            
            train_ws = os.path.join(self.workspace, 'tmp/train')
            init_model_weight = model_weight if os.path.exists(model_weight) else None
            
            model = self.build_model(class_labels, model_arch, model_config, init_model_weight, train_ws, backend)
            model.train(train_data_info['ANNOTATION_FPATH'], train_data_info['IMAGES_DPATH'],
                        batchsize, epoch, lr, score_cutoff, cpu, gpu)
            model.save(model_weight)  # save .pth and .yaml with same name
            
        except BaseException as e:
            traceback.print_exc()
            self.set_jobstatus('TRAIN', 'TRAIN', 'ERROR', str(e))
            run_status = False
        else:
            self.set_jobstatus('TRAIN', 'TRAIN', 'COMPLETED', 'Params: batchsize {}; epoch {}; lr: {}.'.format(batchsize, epoch, lr))
        
        return run_status
    
    
    
    
    
    
    def sort_query_images(self, query_image_dpath=None):
        """Sort images for analysis

        """

        run_status = True
        
        self.set_jobstatus('EVAL', 'SORT', 'BEGIN')
        
        
        try:
            image_files = []
            for f in sorted(glob.glob(os.path.join(query_image_dpath, '**'), recursive=True)):
                if os.path.splitext(f)[1].lower() in self.image_ext:
                    image_files.append(f)
            
            # write image files to text file
            with open(os.path.join(self.workspace, 'query_dataset', 'query_images.txt'), 'w') as outfh:
                for image_file in image_files:
                    outfh.write('{}\n'.format(image_file))
        
        
        except BaseException as e:
            traceback.print_exc()
            self.set_jobstatus('EVAL', 'SORT', 'ERROR', str(e))
            run_status = False
        else:
            self.set_jobstatus('EVAL', 'SORT', 'COMPLETED')
        

        return run_status



    
    
    def seek_images(self):
        """List up images for analysis

        This function loads query_images.txt file and list up all images with "pass" code.
        The valid images are stored as list into the attribute `image` of App class.

        """

        self.images = []
        with open(os.path.join(self.workspace, 'query_dataset', 'query_images.txt'), 'r') as infh:
            for _image in infh:
                _image_info = _image.replace('\n', '').split('\t')
                self.images.append(_image_info[0])
    
    
    
    
    def detect_objects(self, class_labels=None, model_arch='fasterrcnn', model_config=None, model_weight=None,
                       score_cutoff=0.7, batchsize=32, cpu=8, gpu=1, backend='mmdetection'):
        """Object segmentation

        Object segmentaion.


        Args:
            weight (str): A path to save the trained weight.

        Returs:
            (str): A status of running result.

        """

        run_status = True
        
        self.set_jobstatus('EVAL', 'DETECT', 'BEGIN')
        self.seek_images()
        
        def __save_outputs(ws, image_fpath, output):
            image_name = os.path.splitext(os.path.basename(image_fpath))[0]
            output.draw('bbox', os.path.join(ws, 'segmentation_results', image_name + '.outline.png'), label=True, score=True)
        
        try:
            valid_ws = os.path.join(self.workspace, 'tmp/detection')
            # if config is not given, use the config saved during training process
            if model_config is None or model_config == '':
                model_config = os.path.join(os.path.splitext(model_weight)[0] + '.yaml')
            model = self.build_model(class_labels, model_arch, model_config, model_weight, valid_ws, backend)
            outputs = model.inference(self.images, score_cutoff, batchsize, cpu, gpu)
            
            joblib.Parallel(n_jobs=cpu)(
                joblib.delayed(__save_outputs)(self.workspace, self.images[i], outputs[i]) for i in range(len(self.images)))
            #for image_fpath, output in zip(self.images, outputs):
            #    image_name = os.path.splitext(os.path.basename(image_fpath))[0]
            #    output.draw('bbox', os.path.join(self.workspace, 'segmentation_results', image_name + '.outline.png'), label=True, score=True)
            
            outputs = agrolens.utils.ImageAnnotations(outputs)
            outputs.format('coco', os.path.join(self.workspace, 'segmentation_results', 'annotation.json'))
                
                
        except BaseException as e:
            traceback.print_exc()
            self.set_jobstatus('EVAL', 'DETECT', 'ERROR', str(e))
            run_status = False
        else:
            self.set_jobstatus('EVAL', 'DETECT', 'COMPLETED')
    
        return run_status



       
  
    def summarize_objects(self, cpu):
        """Object summarizaiton

        Object summarizaiton.


        Args:
            erosion_size (int): Size
            dilation_size (int): Size
            aligned_images (bool): Use aligned or not
            cpu (int): Number of CPUs. 

        Returs:
            (str): A status of running result.
        """


        run_status = True
        self.set_jobstatus('EVAL', 'SUMMARIZE', 'BEGIN')
       
        self.seek_images()
     
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
            ann = agrolens.utils.ImageAnnotation(image_fpath, ann_fpath, 'coco')
            with open(output_fpath, 'w') as outfh:
                outfh.write('{}\t{}\t{}\t{}\t{}\t{}\t{}\n'.format(
                        'image_path', 'object_id', 'class', 'xmin', 'ymin', 'xmax', 'ymax'
                    ))
                for region in ann.regions:
                    outfh.write('{}\t{}\t{}\t{}\t{}\t{}\t{}\n'.format(
                        ann.image_path, region['id'], region['class'], *region['bbox']
                    ))
            
        try:
            logger.info('Finding objects and calculate the summary data using {} CPUs.'.format(cpu))
            
            images_meta = []
            for image in self.images:
                images_meta.append([image,
                                    #os.path.join(self.workspace, 'segmentation_results', os.path.splitext(os.path.basename(image))[0] + '.xml'),
                                    os.path.join(self.workspace, 'segmentation_results', 'annotation.json'),
                                    os.path.join(self.workspace, 'segmentation_results', os.path.splitext(os.path.basename(image))[0] + '.object.txt')])
            
            images_meta = joblib.Parallel(n_jobs=cpu)(
                joblib.delayed(__summarize_objects)(images_meta[i]) for i in range(len(images_meta)))
            

        except BaseException as e:
            traceback.print_exc()
            self.set_jobstatus('EVAL', 'SUMMARIZE', 'ERROR', str(e))
            run_status = False
        else:
            self.set_jobstatus('EVAL', 'SUMMARIZE', 'COMPLETED')

        return run_status

    

    def summarise_objects(self, erosion_size, dilation_size, aligned_images, cpu):
        run_status = self.summarize_objects(erosion_size, dilation_size, aligned_images, cpu)
        return run_status

    





class ISGUI(IS):
    """Base implementation of AgroLens GUI
    
    GUIBase class implements fundamental parts of GUI for AgroLens.
    
    
    Attributes:
       window (ThemedTk): A main window of GUI generated by tkinter.
       tasb (Notebook): A tab manager of GUI application window.
       config (dictionary): A dictionary to store inputs from GUI.
    """
    
    
    def __init__(self):
        super().__init__()
        
        
        # common configs
        self.__status_style = {
            'UNLOADED':   'secondary.TLabel',
            'UNCOMPLETE': 'TLabel',
            'COMPLETED':  'success.TLabel',
            'RUNNING':    'warning.TLabel',
            'ERROR':      'danger.TLabel',
        }
        
        
        # app window
        self.window = ttk.Window(themename='lumen')
        self.window.title(r'AgroLens - Object Detection')
        self.window_style = tkinter.ttk.Style()
        self.window_style.configure('.', font=('System', 10))
        
        # app tabs (3 tabs)
        tabs = [['config', 'Preferences',        tkinter.NORMAL],
                ['train',  'Model Training',   tkinter.DISABLED],
                ['eval',   'Image Analysis', tkinter.DISABLED]]
        self.tabs = ttk.Notebook(self.window, name='tab')
        self.tabs.grid(padx=10, pady=10)
        for name_, label_, state_ in tabs:
            self.tabs.add(ttk.Frame(self.tabs, name=name_),
                          text=label_, state=state_)
        self.__gui_flag = {'config': 0, 'train': 0, 'eval': 0}
        
        # modules
        self.config = {}
        self.setup_module('config')
        self.setup_module('train')
        self.setup_module('eval')
        
        # job runing status
        self.thread_job = None
        self.thread_module_name = None
        
        
        # startup modules
        self.locate_module('config')
        self.locate_module('train')
        self.locate_module('eval')

    
    
    def setup_module(self, module_name):
        if module_name == 'config':
            self.__setup_module_config()
        elif module_name == 'train':
            self.__setup_module_train()
        elif module_name == 'eval':
            self.__setup_module_eval()
        else:
            raise ValueError('Unsupported module name `{}`.'.format(module_name))
    
    
    
    def __setup_module_config(self):
        # main frame
        frame = ttk.Frame(self.tabs.nametowidget('config'), padding=10, name='module')
        
        # subframes
        subframe_desc = ttk.Frame(frame, name='desc')
        subframe_params = ttk.LabelFrame(frame, padding=5, name='params', text='  Settings  ')
        subframe_pref = ttk.Frame(subframe_params, name='preference')
        subframe_ws   = ttk.Frame(subframe_params, name='workspace')
        
        # subframe - app description
        app_title   = ttk.Label(subframe_desc, name='appTitle',
                                        width=40, text='AgroLens',
                                        style='TLabel', font=('System', 0, 'bold'))
        app_version = ttk.Label(subframe_desc, name='appVersion',
                                        width=10, text='',
                                        #width=10, text='v{}'.format(agrolens.__version__),
                                        style='TLabel')
        app_desc    = ttk.Label(subframe_desc, name='appDesc',
                                        width=86, text='Object Detection',
                                        style='TLabel')
        hr = ttk.Separator(subframe_desc, name='separator', orient='horizontal')
        
        # subframe - preferences
        cpu = tkinter.IntVar()
        cpu.set(multiprocessing.cpu_count())
        cpu_label = ttk.Label(subframe_pref, name='cpuLabel',
                                      text='CPU', width=5)
        cpu_input = ttk.Entry(subframe_pref, name='cpuInput',
                                      textvariable=cpu, width=5, justify=tkinter.RIGHT)
        gpu = tkinter.IntVar()
        gpu_input_state = tkinter.NORMAL
        gpu.set(1)
        if not torch.cuda.is_available():
            gpu.set(0)
            gpu_input_state = tkinter.DISABLED
        gpu_label = ttk.Label(subframe_pref, name='gpuLabel',
                                      text='GPU', width=5)
        gpu_input = ttk.Entry(subframe_pref, name='gpuInput',
                                      textvariable=gpu, width=5, justify=tkinter.RIGHT, state=gpu_input_state)
        
        pkgbackend = tkinter.StringVar()
        pkgbackend.set('MMDetection')
        pkgbackend_label = ttk.Label(subframe_pref, name='pkgbackendLabel',
                                            width=10, text='Backend')
        pkgbackend_input = ttk.Combobox(subframe_pref, name='pkgbackendInput',
                                            textvariable=pkgbackend,
                                            values=self.backend_available, width=20, justify=tkinter.CENTER)
                                            #state=tkinter.DISABLED)
        
        
        modelarch = tkinter.StringVar()
        modelarch_label = ttk.Label(subframe_pref, name='modelarchLabel',
                                            width=10, text='Architecture')
        modelarch_input = ttk.Combobox(subframe_pref, name='modelarchInput',
                                            textvariable=modelarch,
                                            postcommand=lambda: self.__modelarch_combobox(),
                                            width=20, justify=tkinter.CENTER)
        
        
        modelcfg = tkinter.StringVar()
        modelcfg.set('')
        modelcfg_label  = ttk.Label(subframe_pref, name='modelcfgLabel',
                                            width=10, text='Config')
        modelcfg_input  = ttk.Entry(subframe_pref, name='modelcfgInput',
                                            width=50, textvariable=modelcfg)
        modelcfg_button = ttk.Button(subframe_pref, name='modelcfgSelectButton',
                                             text='Select', width=18,
                                             command=lambda: self.__filedialog('config', modelcfg, required=False))
        
        cl = tkinter.StringVar()
        cl_label  = ttk.Label(subframe_pref, name='classlabelLabel',
                                          width=10, text='Class label')
        cl_input  = ttk.Entry(subframe_pref, name='classlabelInput',
                                          width=50, textvariable=cl)
        cl_button = ttk.Button(subframe_pref, name='classlabelSelectButton',
                                           text='Select', width=18,
                                           command=lambda: self.__filedialog('config', cl))
        
        
        
        # submodule - workspace
        workspace = tkinter.StringVar()
        workspace_label = ttk.Label(subframe_ws, name='wsLabel',
                                            width=10, text='Workspace')
        workspace_input = ttk.Entry(subframe_ws, name='wsInput',
                                            width=50, textvariable=workspace)
        workspace_button = ttk.Button(subframe_ws, name='wsSelectButton',
                                              width=18, text='Select',
                                              command=lambda: self.__filedialog('config', workspace, 'opendir'))
        loadworkspace_button = ttk.Button(subframe_ws, width=18, text='Load Workspace', 
                                                  name='loadButton',
                                                  state=tkinter.DISABLED,
                                                  command=lambda: self.load_workspace())
        
        
        self.config.update({
            'workspace':  workspace,
            'pkgbackend': pkgbackend,
            'modelarch':  modelarch,
            'modelcfg':   modelcfg,
            'classlabel': cl,
            'cpu':        cpu,
            'gpu':        gpu,
        })
    
    
    
    def __setup_module_train(self):
        # main frame
        frame = ttk.Frame(self.tabs.nametowidget('train'), padding=10, name='module')
        
        # sub frames
        subframe = ttk.LabelFrame(frame, padding=5, name='params', text='  Training Settings  ')
        subframe_sort = ttk.Frame(subframe, name='sort')
        subframe_weight = ttk.Frame(subframe, name='weight')
        subframe_params = ttk.Frame(subframe, name='trainparams')
        
        
        # subframe - model training / weight
        imagesdpath = tkinter.StringVar()
        imagesdpath_label  = ttk.Label(subframe_weight, name='imagesdpathLabel',
                                               width=16, text='Image folder')
        imagesdpath_input  = ttk.Entry(subframe_weight, name='imagesdpathInput',
                                               width=44, textvariable=imagesdpath)
        imagesdpath_button = ttk.Button(subframe_weight, name='imagesdpathSelectButton',
                                                text='Select', width=18,
                                                command=lambda: self.__filedialog('train', imagesdpath, 'opendir'))
        
        annfpath = tkinter.StringVar()
        annfpath_label  = ttk.Label(subframe_weight, name='annfpathLabel',
                                               width=16, text='Annotations')
        annfpath_input  = ttk.Entry(subframe_weight, name='annfpathInput',
                                               width=44, textvariable=annfpath)
        annfpath_button = ttk.Button(subframe_weight, name='annfpathSelectButton',
                                                text='Select', width=18,
                                                command=lambda: self.__filedialog('train', annfpath, 'openfd'))
       
        annformat = tkinter.StringVar()
        annformat_label = ttk.Label(subframe_weight, name='annformatLabel',
                                            width=16, text='Annotation format')
        annformat_input = ttk.Combobox(subframe_weight, name='annformatInput',
                                               textvariable=annformat,
                                               values=self.formats_available, width=6, justify=tkinter.CENTER)
        
        weight = tkinter.StringVar()
        weight_label  = ttk.Label(subframe_weight, name='weightLabel',
                                          width=16, text='Model weight')
        weight_input  = ttk.Entry(subframe_weight, name='weightInput',
                                          width=44, textvariable=weight)
        weight_button = ttk.Button(subframe_weight, name='weightSelectButton',
                                           text='Select', width=18,
                                           command=lambda: self.__filedialog('train', weight, 'savefile'))
        
        # subframe - model training / params
        batchsize = tkinter.IntVar()
        batchsize.set(2)
        batchsize_label = ttk.Label(subframe_params, name='batchsizeLabel',
                                            text='batch size', width=10)
        batchsize_input = ttk.Entry(subframe_params, name='batchsizeInput',
                                            textvariable=batchsize, width=6, justify=tkinter.RIGHT)
        epoch = tkinter.IntVar()
        epoch.set(10000)
        epoch_label = ttk.Label(subframe_params, name='epochLabel',
                                        text='epochs', width=10)
        epoch_input = ttk.Entry(subframe_params, name='epochInput',
                                        textvariable=epoch, width=6, justify=tkinter.RIGHT)
        lr = tkinter.DoubleVar()
        lr.set(0.001)
        lr_label = ttk.Label(subframe_params, name='lrLabel',
                                     text='learning rate', width=12)
        lr_input = ttk.Entry(subframe_params, name='lrInput',
                                     textvariable=lr, width=6, justify=tkinter.RIGHT)
        
        scorecutoff = tkinter.DoubleVar()
        scorecutoff.set(0.7)
        scorecutoff_label = ttk.Label(subframe_params, name='scorecutoffLabel',
                                     text='cutoff', width=10)
        scorecutoff_input = ttk.Entry(subframe_params, name='scorecutoffInput',
                                     textvariable=scorecutoff, width=6, justify=tkinter.RIGHT)
        
        # subframe job panel
        self.setup_panel_jobs('train')
        
        self.config.update({
            'train__weight'   : weight,
            'train__batchsize': batchsize,
            'train__epoch'    : epoch,
            'train__lr'       : lr,     
            'train__scorecutoff': scorecutoff,
            'train__imagedpath' : imagesdpath,
            'train__annfpath'   : annfpath,
            'train__annformat'  : annformat,
        })
          
    
       
    def __setup_module_eval(self):
        # main frame
        frame = ttk.Frame(self.tabs.nametowidget('eval'), padding=10, name='module')

        # sub frames
        subframe = ttk.LabelFrame(frame, padding=5, name='params', text='   Detection Settings   ')
        subframe_sort = ttk.Frame(subframe, name='sort')
        subframe_weight = ttk.Frame(subframe, name='weight')
        subframe_params = ttk.Frame(subframe, name='evalparams')
        subframe_objsum = ttk.LabelFrame(frame, name='objsum', text='   Object Summarization   ')
        
        # subframe - image sorting
        imagesdpath = tkinter.StringVar()
        imagesdpath_label  = ttk.Label(subframe_weight, name='imagesdpathLabel',
                                               width=16, text='Image folder')
        imagesdpath_input  = ttk.Entry(subframe_weight, name='imagesdpathInput',
                                               width=44, textvariable=imagesdpath)
        imagesdpath_button = ttk.Button(subframe_weight, name='imagesdpathSelectButton',
                                                text='Select', width=18,
                                                command=lambda: self.__filedialog('eval', imagesdpath, 'opendir'))
        
        # subframe - detection / weight
        weight = tkinter.StringVar()
        weight_label  = ttk.Label(subframe_weight, name='weightLabel',
                                          width=16, text='Model weight')
        weight_input  = ttk.Entry(subframe_weight, name='weightInput',
                                          width=44, textvariable=weight)
        weight_button = ttk.Button(subframe_weight, name='weightSelectButton',
                                           text='Select', width=18,
                                           command=lambda: self.__filedialog('eval', weight, 'openfile'))
        
        # subframe - detection / params
        batchsize = tkinter.IntVar()
        batchsize.set(2)
        batchsize_label = ttk.Label(subframe_params, name='batchsizeLabel',
                                            text='batch size', width=10)
        batchsize_input = ttk.Entry(subframe_params, name='batchsizeInput',
                                            textvariable=batchsize, width=6, justify=tkinter.RIGHT)
        scorecutoff = tkinter.DoubleVar()
        scorecutoff.set(0.7)
        scorecutoff_label = ttk.Label(subframe_params, name='scorecutoffLabel',
                                              text='cutoff', width=10)
        scorecutoff_input = ttk.Entry(subframe_params, name='scorecutoffInput',
                                              textvariable=scorecutoff, width=6, justify=tkinter.RIGHT)
        
        
        self.setup_panel_jobs('eval')
            
        self.config.update({
            'eval__weight'   : weight,
            'eval__batchsize': batchsize,
            'eval__imagedpath' : imagesdpath,
            'eval__scorecutoff': scorecutoff,
        })
    
        
    
    def setup_panel_jobs(self, module_name):
        if self._job_status_fpath is None:
            return False
        
        self.refresh_jobstatus()
        job_status = self.job_status[module_name.upper()]
        
        frame = self.tabs.nametowidget(module_name + '.module')
        
        jobpanel_subframe = ttk.LabelFrame(frame, padding=(5, 5), name='jobpanel', text='  Job Panel  ')
        jobpanel_subframe.grid(pady=(20, 5), sticky=tkinter.W + tkinter.E)
        
        job_headers = ['Job', 'Status', 'Datetime', 'Run']
        for i, job_header in enumerate(job_headers):
            job_header = ttk.Label(jobpanel_subframe, text=job_header, padding=0)
            job_header.grid(row=0, column=i)
        
        jobpanel_hr = ttk.Separator(jobpanel_subframe, name='separator', orient='horizontal')
        jobpanel_hr.grid(row=1, column=0, columnspan=len(job_headers), pady=(5, 10), sticky=tkinter.E + tkinter.W)
        
        # job items
        prev_job_status = 'COMPLETED'
        for _job_code, _job_status in sorted(job_status.items(), key=lambda kv: kv[1]['id']):
            if _job_code == 'INIT':
                continue
            if _job_status['status'] == 'BEGIN':
                _job_status['status'] = 'UNCOMPLETE'
            
            enable_run_button = tkinter.DISABLED
            if prev_job_status == 'COMPLETED':
                if module_name + '__weight' in self.config and len(self.config[module_name + '__weight'].get()) > 0:
                    enable_run_button = tkinter.NORMAL
            
            jobtitle_label  = ttk.Label(jobpanel_subframe, name=_job_code.lower() + 'Label',
                                                text=_job_status['title'], width=21, anchor='center')
            jobstatus_label = ttk.Label(jobpanel_subframe, name=_job_code.lower() + 'Status',
                                                text=_job_status['status'], width=17, anchor='center', padding=(2, 5),
                                                style=self.__status_style[_job_status['status']])
            jobendtime_label = ttk.Label(jobpanel_subframe, name=_job_code.lower() + 'Datetime',
                                                 text=_job_status['datetime'], width=22, anchor='center')
            jobrun_button = ttk.Button(jobpanel_subframe, text='RUN', name=_job_code.lower() + 'RunButton',
                                               width=14, state=enable_run_button,
                                               command=lambda jc = _job_code: self.run_job(module_name, jc))
            
            jobtitle_label.grid(row=_job_status['id'] + 1,   column=0, pady=1, sticky=tkinter.E + tkinter.W)
            jobstatus_label.grid(row=_job_status['id'] + 1,  column=1, pady=1, sticky=tkinter.E + tkinter.W)
            jobendtime_label.grid(row=_job_status['id'] + 1, column=2, pady=1, sticky=tkinter.E + tkinter.W)
            jobrun_button.grid(row=_job_status['id'] + 1,    column=3, pady=1, sticky=tkinter.E + tkinter.W)
            
            prev_job_status = _job_status['status']
 


    
    def load_workspace(self):
        self.init_workspace(self.config['workspace'].get())

        init_ws = os.path.join(self.config['workspace'].get(), 'init_params')
        model = self.build_model(self.config['classlabel'].get(), self.config['modelarch'].get(), self.config['modelcfg'].get(),
                                 None, init_ws, self.config['pkgbackend'].get())
        
        default_model_prefix = os.path.join(init_ws, 'default')
        model.save(default_model_prefix + '.pth', default_model_prefix + '.py')
        self.config['modelcfg'].set(default_model_prefix + '.py')
        
        self.setup_panel_jobs('train')
        self.tabs.tab(1, state=tkinter.NORMAL)
        self.setup_panel_jobs('eval')
        self.tabs.tab(2, state=tkinter.NORMAL)
        
    
    def startup(self):
        self.window.mainloop()
    
 
    
    def __filedialog(self, module_name, tkvar, mode='openfile', required=True):
        if mode == 'openfd':
            if self.config['train__annformat'].get() in ['COCO']:
                mode = 'openfile'
            else:
                mode = 'opendir'
        
        root_dpath = os.getcwd()
        if 'workspace' in self.config and self.config['workspace'].get():
            root_dpath = self.config['workspace'].get()
        
        if mode == 'openfile':
            fp = tkinter.filedialog.askopenfilename(filetypes=[('', '*')], initialdir=root_dpath)
        elif mode == 'opendir':
            fp = tkinter.filedialog.askdirectory(initialdir=root_dpath)
        elif mode == 'savefile':
            fp = tkinter.filedialog.asksaveasfilename(defaultextension='.pth', initialdir=root_dpath)
        else:
            raise ValueError('Unexpected mode.')

        if len(fp) > 0:
            tkvar.set(fp)
            if required:
                self.__gui_flag[module_name] += 1
                self.__enable_loadwsbutton()
                self.__enable_jobpanel()
    
    
    def __enable_loadwsbutton(self):
        if self.__gui_flag['config'] >= 2:
            self.tabs.nametowidget('config.module.params.workspace.loadButton').config(state=tkinter.NORMAL)
    
    
    def __enable_jobpanel(self):
        if self.__gui_flag['train'] >= 3:
            self.setup_panel_jobs('train')
        if self.__gui_flag['eval'] >= 2:
            self.setup_panel_jobs('eval')
    
    
    def __modelarch_combobox(self):
        self.tabs.nametowidget('config.module.params.preference.modelarchInput').config(
            value=self.models_available[self.config['pkgbackend'].get()]
        )
    
    
    def run_job(self, module_name, job_code):
        # disable tabs during job runing
        for i, tab_name in enumerate(self.tabs.tabs()):
            if tab_name != '.tab.' + module_name:
                self.tabs.tab(i, state=tkinter.DISABLED)
        
        for w in self.tabs.nametowidget(module_name + '.module.jobpanel').winfo_children():
            if 'RunButton' in str(w):
                w.config(state=tkinter.DISABLED)
        
        self.tabs.nametowidget(module_name + '.module.jobpanel.' + job_code.lower() + 'Status').config(text='RUNNING',
                               style=self.__status_style['RUNNING'])
        self.tabs.nametowidget(module_name + '.module.jobpanel.' + job_code.lower() + 'RunButton').config(text='RUNNING',
                                        state=tkinter.DISABLED)
        ##self.tabs.nametowidget(module_name + '.module.jobpanel.' + job_code.lower() + 'RunButton').config(text='STOP',
        ##                                state=tkinter.NORMAL, command=lambda: self.stop_job())
        self.tabs.nametowidget(module_name + '.module.jobpanel.' + job_code.lower() + 'Datetime').config(text='')
        self.tabs.nametowidget(module_name + '.module.jobpanel.' + job_code.lower() + 'RunButton').update()
        self.tabs.nametowidget(module_name + '.module.jobpanel.' + job_code.lower() + 'Status').update()
        self.tabs.nametowidget(module_name + '.module.jobpanel.' + job_code.lower() + 'Datetime').update()
        
        if module_name == 'train':
            if job_code == 'SORT':
                ##self.thread_job = multiprocessing.Process(target=self.sort_train_images,
                ##        args=(self.config['classlabel'].get(),
                ##              self.config['train__imagedpath'].get(),
                ##              self.config['train__annfpath'].get(),
                ##              self.config['train__annformat'].get(),))
                self.sort_train_images(
                        self.config['classlabel'].get(),
                        self.config['train__imagedpath'].get(),
                        self.config['train__annfpath'].get(),
                        self.config['train__annformat'].get()
                )
        
            elif job_code == 'TRAIN':
                ##self.thread_job = multiprocessing.Process(target=self.train_model,
                ##    args=(self.config['classlabel'].get(),
                ##          self.config['modelarch'].get(),
                ##          self.config['modelcfg'].get(),
                ##          self.config['train__weight'].get(),
                ##          self.config['train__batchsize'].get(),
                ##          self.config['train__epoch'].get(),
                ##          self.config['train__lr'].get(),
                ##          self.config['train__scorecutoff'].get(),
                ##          self.config['cpu'].get(),
                ##          self.config['gpu'].get(),
                ##          self.config['pkgbackend'].get(),))
                self.train_model(
                          self.config['classlabel'].get(),
                          self.config['modelarch'].get(),
                          self.config['modelcfg'].get(),
                          self.config['train__weight'].get(),
                          self.config['train__batchsize'].get(),
                          self.config['train__epoch'].get(),
                          self.config['train__lr'].get(),
                          self.config['train__scorecutoff'].get(),
                          self.config['cpu'].get(),
                          self.config['gpu'].get(),
                          self.config['pkgbackend'].get()
                )       


        elif module_name == 'eval':
            if job_code == 'SORT':
                ##self.thread_job = multiprocessing.Process(target=self.sort_query_images,
                ##        args=(self.config['eval__imagedpath'].get(),))
                self.sort_query_images(self.config['eval__imagedpath'].get())
                
            elif job_code == 'DETECT':
                ##self.thread_job = multiprocessing.Process(target=self.detect_objects,
                ##    args=(self.config['classlabel'].get(),
                ##          self.config['modelarch'].get(),
                ##          self.config['modelcfg'].get(),
                ##          self.config['eval__weight'].get(),
                ##          self.config['eval__scorecutoff'].get(),
                ##          self.config['eval__batchsize'].get(),
                ##          self.config['cpu'].get(),
                ##          self.config['gpu'].get(),
                ##          self.config['pkgbackend'].get(), ))
                self.detect_objects(
                          self.config['classlabel'].get(),
                          self.config['modelarch'].get(),
                          self.config['modelcfg'].get(),
                          self.config['eval__weight'].get(),
                          self.config['eval__scorecutoff'].get(),
                          self.config['eval__batchsize'].get(),
                          self.config['cpu'].get(),
                          self.config['gpu'].get(),
                          self.config['pkgbackend'].get()
                )

            elif job_code == 'SUMMARIZE':
                ##self.thread_job = multiprocessing.Process(target=self.summarize_objects,
                ##    args=(self.config['cpu'].get(),))
                self.summarize_objects(self.config['cpu'].get())

                    
        
        ##self.thread_job.start()
        self.thread_module_name = module_name
        self.check_thread_job()
        
 
    
    def stop_job(self):
        if self.thread_job is not None and self.thread_job.is_alive():
            self.thread_job.terminate()
    
    
    def check_thread_job(self):
        
        if self.thread_job is not None and self.thread_job.is_alive():
            self.window.after(1000, self.check_thread_job)
        
        else:
            if self.thread_job is not None:
                self.thread_job.join()
            self.setup_panel_jobs(self.thread_module_name)
            
            for i, tab_name in enumerate(self.tabs.tabs()):
                self.tabs.tab(i, state=tkinter.NORMAL)
            
            self.thread_job = None
            self.thread_module_name = None




    def locate_module(self, module_name):
        if module_name == 'config':
            self.__locate_module_config()
        elif module_name == 'train':
            self.__locate_module_train()
        elif module_name == 'eval':
            self.__locate_module_eval()
        else:
            raise ValueError('Unsupported module name.')
    

    def __locate_module_config(self):
        self.tabs.nametowidget('config.module').grid(sticky=tkinter.W + tkinter.E)
        self.tabs.nametowidget('config.module.desc').grid(pady=5, sticky=tkinter.W + tkinter.E)
        if True:
            self.tabs.nametowidget('config.module.desc.appTitle').grid(row=0, column=0, pady=5, sticky=tkinter.W)
            self.tabs.nametowidget('config.module.desc.appVersion').grid(row=0, column=1, pady=5, sticky=tkinter.E)
            self.tabs.nametowidget('config.module.desc.appDesc').grid(row=1, column=0, columnspan=2, pady=5, sticky=tkinter.E + tkinter.W)
            self.tabs.nametowidget('config.module.desc.separator').grid(row=2, column=0, columnspan=2, pady=5, sticky=tkinter.W + tkinter.E)
        self.tabs.nametowidget('config.module.params').grid(sticky=tkinter.W + tkinter.E)
        self.tabs.nametowidget('config.module.params.preference').grid(pady=5, sticky=tkinter.W + tkinter.E)
        if True:
            self.tabs.nametowidget('config.module.params.preference.pkgbackendLabel').grid(row=0, column=0, padx=(0, 5), pady=5, sticky=tkinter.W)
            self.tabs.nametowidget('config.module.params.preference.pkgbackendInput').grid(row=0, column=1, columnspan=2, pady=5, sticky=tkinter.W)
            self.tabs.nametowidget('config.module.params.preference.modelarchLabel').grid(row=1, column=0, padx=(0, 5), pady=5, sticky=tkinter.W)
            self.tabs.nametowidget('config.module.params.preference.modelarchInput').grid(row=1, column=1, columnspan=2, pady=5, sticky=tkinter.W)
            self.tabs.nametowidget('config.module.params.preference.modelcfgLabel').grid(row=2, column=0, padx=(0, 5), pady=5, sticky=tkinter.W + tkinter.E)
            self.tabs.nametowidget('config.module.params.preference.modelcfgInput').grid(row=2, column=1, columnspan=3, pady=5, sticky=tkinter.W + tkinter.E)
            self.tabs.nametowidget('config.module.params.preference.modelcfgSelectButton').grid(row=2, column=4, padx=(10, 0), pady=5, sticky=tkinter.W + tkinter.E)
            self.tabs.nametowidget('config.module.params.preference.classlabelLabel').grid(row=3, column=0, padx=(0, 5), pady=5, sticky=tkinter.W + tkinter.E)
            self.tabs.nametowidget('config.module.params.preference.classlabelInput').grid(row=3, column=1, columnspan=3, pady=5, sticky=tkinter.W + tkinter.E)
            self.tabs.nametowidget('config.module.params.preference.classlabelSelectButton').grid(row=3, column=4, padx=(10, 0), pady=5, sticky=tkinter.W + tkinter.E)
            self.tabs.nametowidget('config.module.params.preference.cpuLabel').grid(row=4, column=0, sticky=tkinter.W)
            self.tabs.nametowidget('config.module.params.preference.cpuInput').grid(row=4, column=1, sticky=tkinter.W)
            self.tabs.nametowidget('config.module.params.preference.gpuLabel').grid(row=4, column=2, padx=(20, 0), sticky=tkinter.W)
            self.tabs.nametowidget('config.module.params.preference.gpuInput').grid(row=4, column=3, sticky=tkinter.W)

        self.tabs.nametowidget('config.module.params.workspace').grid(pady=5, sticky=tkinter.W + tkinter.E)
        if True:
            self.tabs.nametowidget('config.module.params.workspace.wsLabel').grid(row=0, column=0, padx=(0, 5), pady=5, sticky=tkinter.W + tkinter.E)
            self.tabs.nametowidget('config.module.params.workspace.wsInput').grid(row=0, column=1, pady=5, sticky=tkinter.W + tkinter.E)
            self.tabs.nametowidget('config.module.params.workspace.wsSelectButton').grid(row=0, column=2, padx=(10, 0), pady=5, sticky=tkinter.W + tkinter.E)
            self.tabs.nametowidget('config.module.params.workspace.loadButton').grid(row=1, column=2, padx=(10, 0), pady=5, sticky=tkinter.W + tkinter.E)
    
    
    
    def __locate_module_train(self):
        self.tabs.nametowidget('train.module').grid(sticky=tkinter.W + tkinter.E)
        self.tabs.nametowidget('train.module.params').grid(sticky=tkinter.W + tkinter.E)
        self.tabs.nametowidget('train.module.params.weight').grid(pady=5, sticky=tkinter.W + tkinter.E)
        if True:
            self.tabs.nametowidget('train.module.params.weight.weightLabel').grid(row=0, column=0, pady=5, sticky=tkinter.W + tkinter.E)
            self.tabs.nametowidget('train.module.params.weight.weightInput').grid(row=0, column=1, columnspan=2, pady=5, sticky=tkinter.W + tkinter.E)
            self.tabs.nametowidget('train.module.params.weight.weightSelectButton').grid(row=0, column=3, padx=(10, 0), pady=5, sticky=tkinter.W + tkinter.E)
            self.tabs.nametowidget('train.module.params.weight.imagesdpathLabel').grid(row=1, column=0, pady=5, sticky=tkinter.W + tkinter.E)
            self.tabs.nametowidget('train.module.params.weight.imagesdpathInput').grid(row=1, column=1, columnspan=2, pady=5, sticky=tkinter.W + tkinter.E)
            self.tabs.nametowidget('train.module.params.weight.imagesdpathSelectButton').grid(row=1, column=3, padx=(10, 0), pady=5, sticky=tkinter.W + tkinter.E)
            self.tabs.nametowidget('train.module.params.weight.annformatLabel').grid(row=2, column=0, pady=5, sticky=tkinter.W + tkinter.E)
            self.tabs.nametowidget('train.module.params.weight.annformatInput').grid(row=2, column=1, pady=5, sticky=tkinter.W + tkinter.E)
            self.tabs.nametowidget('train.module.params.weight.annfpathLabel').grid(row=3, column=0, pady=5, sticky=tkinter.W + tkinter.E)
            self.tabs.nametowidget('train.module.params.weight.annfpathInput').grid(row=3, column=1, columnspan=2, pady=5, sticky=tkinter.W + tkinter.E)
            self.tabs.nametowidget('train.module.params.weight.annfpathSelectButton').grid(row=3, column=3, padx=(10, 0), pady=5, sticky=tkinter.W + tkinter.E)

        self.tabs.nametowidget('train.module.params.trainparams').grid(pady=5, sticky=tkinter.W + tkinter.E)
        if True:
            self.tabs.nametowidget('train.module.params.trainparams.batchsizeLabel').grid(row=0, column=0, pady=5, sticky=tkinter.W)
            self.tabs.nametowidget('train.module.params.trainparams.batchsizeInput').grid(row=0, column=1, pady=5, sticky=tkinter.W)
            self.tabs.nametowidget('train.module.params.trainparams.epochLabel').grid(row=0, column=3, padx=(50, 0), pady=5, sticky=tkinter.W)
            self.tabs.nametowidget('train.module.params.trainparams.epochInput').grid(row=0, column=4, pady=5, sticky=tkinter.W)
            self.tabs.nametowidget('train.module.params.trainparams.lrLabel').grid(row=1, column=0, pady=5, sticky=tkinter.W)
            self.tabs.nametowidget('train.module.params.trainparams.lrInput').grid(row=1, column=1, pady=5, sticky=tkinter.W)
            self.tabs.nametowidget('train.module.params.trainparams.scorecutoffLabel').grid(row=1, column=3, padx=(50, 0), pady=5, sticky=tkinter.W)
            self.tabs.nametowidget('train.module.params.trainparams.scorecutoffInput').grid(row=1, column=4, pady=5, sticky=tkinter.W)
            
    
       
    def __locate_module_eval(self):
        self.tabs.nametowidget('eval.module').grid(sticky=tkinter.W + tkinter.E)
        self.tabs.nametowidget('eval.module.params').grid(sticky=tkinter.W + tkinter.E)
        self.tabs.nametowidget('eval.module.params.weight').grid(pady=5, sticky=tkinter.W + tkinter.E)
        if True:
            self.tabs.nametowidget('eval.module.params.weight.weightLabel').grid(row=0, column=0, pady=5, sticky=tkinter.W + tkinter.E)
            self.tabs.nametowidget('eval.module.params.weight.weightInput').grid(row=0, column=1, pady=5, sticky=tkinter.W + tkinter.E)
            self.tabs.nametowidget('eval.module.params.weight.weightSelectButton').grid(row=0, column=2, padx=(10, 0), pady=5, sticky=tkinter.W + tkinter.E)
            self.tabs.nametowidget('eval.module.params.weight.imagesdpathLabel').grid(row=1, column=0, pady=5, sticky=tkinter.W + tkinter.E)
            self.tabs.nametowidget('eval.module.params.weight.imagesdpathInput').grid(row=1, column=1, pady=5, sticky=tkinter.W + tkinter.E)
            self.tabs.nametowidget('eval.module.params.weight.imagesdpathSelectButton').grid(row=1, column=2, padx=(10, 0), pady=5, sticky=tkinter.W + tkinter.E)
        self.tabs.nametowidget('eval.module.params.evalparams').grid(pady=5, sticky=tkinter.W + tkinter.E)
        if True:
            self.tabs.nametowidget('eval.module.params.evalparams.batchsizeLabel').grid(row=0, column=0, pady=5, sticky=tkinter.W)
            self.tabs.nametowidget('eval.module.params.evalparams.batchsizeInput').grid(row=0, column=1, pady=5, sticky=tkinter.W)
            self.tabs.nametowidget('eval.module.params.evalparams.scorecutoffLabel').grid(row=0, column=2, padx=(15, 0), pady=5, sticky=tkinter.W)
            self.tabs.nametowidget('eval.module.params.evalparams.scorecutoffInput').grid(row=0, column=3, pady=5, sticky=tkinter.W)
        


if __name__ == '__main__':
    print('God,,,, no more bugs please!')


