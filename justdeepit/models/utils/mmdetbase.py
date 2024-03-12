import os
import shutil
import json
import glob
import math
import tqdm
import tempfile
import logging
import datetime
import numpy as np
import skimage
import skimage.measure
import PIL
import torch
import torch.multiprocessing
from justdeepit.models.abstract import ModuleTemplate, JDIError
from justdeepit.utils import ImageAnnotation, ImageAnnotations, load_images
from justdeepit.models.utils.data import DataClass, DataPipeline, DataLoader

logger = logging.getLogger(__name__)

try:
    import mim
    import mmcv
    import mmdet
    import mmcv.ops
    import mmdet.utils
    import mmdet.apis
    import mmdet.models
    import mmdet.datasets
    import mmengine.config
    import mmengine.registry
    import mmengine.runner
except ImportError:
    msg = ('JustDeepIt requires the installation of '
           'the following Python packages:\n'
           '     - mmcv>=2.0.0\n'
           '     - mmdet>=3.0.0\n'
           '     - mmengine>=0.7.3\n'
           'to build detection/segmentation models. '
           'Please ensure that these packages are '
           'already installed with the correct versions.\n')
    logger.error(msg)
    raise JDIError(msg)





class MMDetBase(ModuleTemplate):
    
    
    def __init__(self,
                 class_labels=None,
                 model_arch=None,
                 model_config=None,
                 model_weight=None,
                 model_class=None,
                 workspace=None,
                 seed=None):
        
        # model settings
        self.model_arch = model_arch
        self.class_labels = DataClass(class_labels)
        self.model_class = model_class
        self.cfg_fpath = model_config
        self.cfg = self.__set_config(model_config, model_weight, self.class_labels.class_labels)
        
        # workspace
        self.tempd = None
        if workspace is None:
            self.tempd = tempfile.TemporaryDirectory()
            self.workspace = self.tempd.name
        else:
            self.workspace = workspace
        self.cfg.work_dir = os.path.abspath(self.workspace)
        logger.info(f'The workspace is set to `{self.workspace}`. '
                    f'Please locate the intermediate and final results '
                    f'within this workspace.')
        
        # random seed
        if seed is None:
            seed = int(datetime.datetime.utcnow().timestamp())
        self.cfg.seed = seed
    
    

    def __del__(self):
        try:
            if self.tempd is not None:
                self.tempd.cleanup()
        except:
            logger.info(f'The temporary directory (`{self.workspace}`) created by JustDeepIt '
                        f'cannot be removed automatically. Please remove it manually.')
    
    

    def __set_config(self, model_config, model_weight, class_labels):
        cfg = None
        if model_config is None or model_config == '':
            raise JDIError(f'JustDeepIt requires a configuration file to build models. '
                           f'Set up a path to a configuration file or '
                           f'run `mim search mmcls --valid-config` to set the pre-defined configuration.')
        try:
            # if config does not exist, download the config from MMLab.
            if not os.path.exists(model_config):
                cache_dpath = os.path.join(os.path.expanduser('~'), '.cache', 'mim')
                if os.path.splitext(model_config)[1] in ['.py', '.yaml']:
                    model_config = os.path.splitext(model_config)[0]
                model_chkpoint = mim.commands.download(package='mmdet',
                                                       configs=[model_config])[0]
                model_config = os.path.join(cache_dpath, model_config + '.py')
                model_chkpoint =  os.path.join(cache_dpath, model_chkpoint)
            cfg = mmengine.config.Config.fromfile(model_config)
            
            # if weights not given, download the weights from MMLab
            if (model_weight is None) or (not os.path.exists(model_weight)):
                model_weight = model_chkpoint
            cfg.load_from = model_weight
            cfg.launcher = 'none'
            cfg.resume = False

            # update class labels
            cfg = self.__set_class_labels(cfg, class_labels)
        except:
            raise JDIError('JustDeepIt cannot find or download the configuration file.'
                           'Please check the file path or the internet connection and try agian.')
        
        return  cfg
        
    

    def __set_class_labels(self, cfg, class_labels):
        def __set_cl(cfg, class_labels):
            for cfg_key in cfg:
                if isinstance(cfg[cfg_key], dict):
                    __set_cl(cfg[cfg_key], class_labels)
                elif isinstance(cfg[cfg_key], (list, tuple)):
                    if isinstance(cfg[cfg_key][0], dict):
                        for elem in cfg[cfg_key]: 
                            __set_cl(elem, class_labels)
                else:
                    if cfg_key == 'classes':
                        cfg[cfg_key] = class_labels
                    elif cfg_key == 'num_classes' or cfg_key == 'num_things_classes':
                        cfg[cfg_key] = len(class_labels)
            return cfg
        
        cfg.data_root = ''
        cfg.merge_from_dict(dict(metainfo = dict(classes=class_labels)))
        cfg.model = __set_cl(cfg.model, class_labels)
        # for RetinaNet: ResNet: init_cfg and pretrained cannot be specified at the same time
        if 'pretrained' in cfg.model:
            del cfg.model['pretrained']
        return cfg
    


    def train(self,
              train_dataset,
              valid_dataset=None,
              test_dataset=None,
              optimizer=None,
              scheduler=None,
              score_cutoff=0.5,
              batchsize=8,
              epoch=100,
              cpu=8,
              gpu=1):
        
        # CPU/GPUs
        self.__set_device(gpu)

        # training params
        self.__set_optimizer(optimizer)
        self.__set_scheduler(scheduler)
       
        # datasets
        if self.model_class == 'od':
            dataloader = DataLoader(self.cfg,
                                    train_dataset, valid_dataset, test_dataset,
                                    batchsize, epoch, cpu,
                                    with_bbox=True, with_mask=False)
        elif self.model_class == 'is':
            dataloader = DataLoader(self.cfg,
                                    train_dataset, valid_dataset, test_dataset,
                                    batchsize, epoch, cpu,
                                    with_bbox=True, with_mask=True)
        else:
            raise ValueError('The model class is not supported.')
        self.cfg.merge_from_dict(dataloader.cfg)
        self.cfg.default_hooks.checkpoint.interval = 20

        # training
        runner = mmengine.runner.Runner.from_cfg(self.cfg)
        runner.train()
    
    
    
    def __set_device(self, gpu=0):
        if gpu != 0 and gpu != 1:
            raise JDIError('The current JustDeepIt does not support multiple GPUs for trianing. '
                           'Set `gpu` to 0 or 1.')
        gpu = gpu if torch.cuda.is_available() else 0
        gpu = torch.cuda.device_count() if torch.cuda.device_count() < gpu else gpu
        self.cfg.device = 'cuda:0' if gpu > 0 else 'cpu'
    

    
    def __set_optimizer(self, optimizer):
        if optimizer is not None and optimizer.replace(' ', '') != '':
            if optimizer[0] != '{' or optimizer[0:4] != 'dict':
                optimizer = 'dict(' + optimizer + ')'
            self.cfg.optimizer = eval(optimizer)
    
    

    def __set_scheduler(self, scheduler):
        if scheduler is not None and scheduler.replace(' ', '') != '':
            if scheduler[0] != '{' or scheduler[0:4] != 'dict':
                scheduler = 'dict(' + scheduler + ')'
            self.cfg.scheduler = eval(scheduler)
    


    def save(self, weight_fpath, config_fpath=None):
        # weight
        if not weight_fpath.endswith('.pth'):
            weight_fpath + '.pth'
        with open(os.path.join(self.cfg.work_dir, 'last_checkpoint')) as chkf:
            last_chk = chkf.readline().strip()
            shutil.copy2(last_chk, weight_fpath)
        # config
        if config_fpath is None:
            config_fpath = os.path.splitext(weight_fpath)[0] + '.py'
        self.cfg.dump(config_fpath)



    def inference(self,
                  images,
                  score_cutoff=0.5,
                  batchsize=8,
                  cpu=4,
                  gpu=1):
        
        self.__set_device(gpu)
        
        # load images for inference
        target_images = load_images(images)
        assert len(target_images) > 0, 'No images found in {}'.format(images)
        
        # set params
        pipeline = DataPipeline()
        self.cfg.merge_from_dict(
            dict(test_dataloader=dict(
                _delete_=True,
                batch_size=1,
                num_workers=cpu,
                persistent_workers=True,
                drop_last=False,
                sampler=dict(type='DefaultSampler', shuffle=False),
                dataset=dict(
                    type='CocoDataset',
                    pipeline = pipeline.inference,
                    metainfo=self.cfg.metainfo))))


        # load model
        model = mmdet.apis.init_detector(self.cfg,
                                         self.cfg.load_from,
                                         device=self.cfg.device)
        
        # inference
        outputs = mmdet.apis.inference_detector(model, target_images)

        # format
        outputs_fmt = []
        for target_image, output in zip(target_images, outputs):
            outputs_fmt.append(
                ImageAnnotation(target_image,
                                self.__format_annotation(output,
                                                         score_cutoff)))
        
        return ImageAnnotations(outputs_fmt)
    
    
    
    def __format_annotation(self, output, score_cutoff):
        if 'bboxes' in output.pred_instances:
            pred_bboxes = output.pred_instances.bboxes.detach().cpu().numpy().tolist()
            pred_labels = output.pred_instances.labels.detach().cpu().numpy().tolist()
            pred_scores = output.pred_instances.scores.detach().cpu().numpy().tolist()
        else:
            pred_bboxes = []
            pred_labels = []
            pred_scores = []
        
        if 'masks' in output.pred_instances:
            pred_masks = output.pred_instances.masks.detach().cpu().numpy()
        else:
            pred_masks = [None] * len(pred_bboxes)
        
        regions = []
        for pred_bbox, pred_label, pred_score, pred_mask in zip(
                pred_bboxes, pred_labels, pred_scores, pred_masks):
            if pred_score > score_cutoff:
                region = {
                    'id': len(regions) + 1,
                    'class': self.class_labels[pred_label],
                    'score': pred_score,
                    'bbox': pred_bbox #.astype(np.int32)
                }
                if pred_mask is not None:
                    sg = skimage.measure.find_contours(pred_mask, 0.5)
                    region['contour'] = sg[0][:, [1, 0]]
                regions.append(region)
        
        return regions
        
    




