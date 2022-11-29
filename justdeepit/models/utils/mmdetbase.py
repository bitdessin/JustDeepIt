import os
import datetime
import shutil
import json
import glob
import math
import tqdm
import tempfile
import logging
import time
import numpy as np
import skimage
import skimage.measure
import PIL
import torch
import torch.multiprocessing
from justdeepit.models.abstract import ModuleTemplate
from justdeepit.utils import ImageAnnotation, ImageAnnotations


logger = logging.getLogger(__name__)

try:
    import mim
    import mmcv
    #import mmcv.parallel
    import mmdet
    import mmcv.ops
    import mmdet.utils
    import mmcv.runner
    import mmdet.apis
    import mmdet.models
    import mmdet.datasets
    import mmdet.datasets.pipelines
except ImportError:
    msg = 'JustDeepIt requires mmdetection library to build models. Make sure MMDetection and the related packages (mmcv-full, mmdetection, openmim, mmengine) have been installed already.'
    logger.error(msg)
    raise ImportError(msg)




class MMDetBase(ModuleTemplate):

    def __init__(self, class_labels=None, model_arch=None, model_config=None, model_weight=None, workspace=None, seed=None):
        
        # model
        self.model_arch = model_arch
        self.trainer  = None
        self.detector = None
        
        # config
        self.cfg = self.__get_config(model_config, model_weight)
        
        # workspace
        self.tempd = None
        if workspace is None:
            self.tempd = tempfile.TemporaryDirectory()
            self.workspace = self.tempd.name
        else:
            self.workspace = workspace
        self.cfg.work_dir = os.path.abspath(self.workspace)
        logger.info('Workspace for MMDetection-based model is set at `{}`.'.format(self.workspace))
        
        # class labels
        self.class_labels = self.__parse_class_labels(class_labels)
        self.cfg = self.__set_class_labels(self.cfg, self.class_labels)

        # setup time stmp
        self.timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
        self.log_file = os.path.join(self.cfg.work_dir, '{}.log'.format(self.timestamp))
        self.logger = mmdet.utils.get_root_logger(log_file=self.log_file, log_level=self.cfg.log_level)

        # random seed
        if seed is None:
            seed = int(datetime.datetime.utcnow().timestamp())
        self.cfg.seed = seed
        
        # create metadata
        meta = dict()
        meta['env_info'] = '\n'.join([(f'{k}: {v}') for k, v in mmdet.utils.collect_env().items()])
        meta['config'] = self.cfg.pretty_text
        meta['exp_name'] = os.path.basename(model_config)
        meta['seed'] = seed
        self.meta = meta



    def __del__(self):
        try:
            if self.tempd is not None:
                self.tempd.cleanup()
        except:
            pass

    
    def __get_config(self, model_config, model_weight):
        if model_config is None or model_config == '':
            raise ValueError('Configuration file for MMDetection cannot be empty.')
        
        cfg = None
        try:
            if not os.path.exists(model_config):
                # if the given path does not exist, set the chkpoint from mmdet Lab as a initial params
                if os.path.splitext(model_config)[1] in ['.py', '.yaml']:
                    model_config = os.path.splitext(model_config)[0]
                model_chkpoint = mim.commands.download(package='mmdet', configs=[model_config])[0]
                model_config = os.path.join(os.path.expanduser('~'), '.cache', 'mim', model_config + '.py')
                model_chkpoint =  os.path.join(os.path.expanduser('~'), '.cache', 'mim', model_chkpoint)
            cfg = mmcv.utils.Config.fromfile(model_config)
        
            if (model_weight is None) or (not os.path.exists(model_weight)):
                model_weight = model_chkpoint
            cfg.load_from = model_weight
            
        except:
            raise FileNotFoundError('The path or name of the configuration file `{}` is incorrect. JustDeepIt cannot find or download.'.format(model_config))
            
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
        cfg.merge_from_dict(dict(classes=class_labels,
                                 num_classes=len(class_labels),
                                 num_things_classes=len(class_labels)))
        cfg.data = __set_cl(cfg.data, class_labels)
        cfg.model = __set_cl(cfg.model, class_labels)
        # for RetinaNet: ResNet: init_cfg and pretrained cannot be specified at the same time
        if 'pretrained' in cfg.model:
            del cfg.model['pretrained']
        return cfg
    

    def __parse_class_labels(self, class_labels):
        cl = None
        if class_labels is None:
            raise ValueError('`class_labels` is required to build model.')
        else:
            if isinstance(class_labels, list):
                cl = class_labels
            elif isinstance(class_labels, str):
                cl = []
                with open(class_labels, 'r') as infh:
                    for cl_ in infh:
                        cl.append(cl_.replace('\n', ''))
            else:
                raise ValueError('Unsupported data type of `class_labels`. Set a path to a file which contains class labels or set a list of class labels.')
        return cl



    def __get_device(self, gpu=1):
        device = 'cpu'
        if gpu > 0:
            if torch.cuda.is_available():
                device = 'cuda'
        return device 
    
    
    
    def __find_free_port(self):
        import socket
        from contextlib import closing
        with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
            s.bind(('', 0))
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            return str(s.getsockname()[1])



    def __setup_process_group(self, local_rank,  world_size, backend='nccl', master_addr='127.0.0.1', master_port='29500'):
        os.environ['MASTER_ADDR'] = master_addr
        os.environ['MASTER_PORT'] = master_port
        torch.cuda.set_device(local_rank)
        torch.distributed.init_process_group(backend, init_method='env://',
                                             world_size=world_size, rank=local_rank)


    
    def __generate_config(self, input_fpath, output_fpath=None):
        cfg = mmcv.Config.fromfile(input_fpath)
        if output_fpath is None:
            return cfg
        else:
            cfg.dump(output_fpath)
    
    
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
       
    
    def train(self, image_dpath, annotation,
              optimizer=None, scheduler=None,
              batchsize=8, epoch=100, score_cutoff=0.5, cpu=4, gpu=1):
        
        if not torch.cuda.is_available():
            gpu = 0
        if torch.cuda.device_count() < gpu:
            gpu = torch.cuda.device_count()
        
        if gpu == 0:
            self.cfg.device = 'cpu'
        else:
            self.cfg.device = 'cuda'
        
        if gpu == 0:
            raise EnvironmentError('CPU training is not supported by MMDetection. Make sure the system can recognize GPU and set gpu more than zero.')
        elif gpu == 1:
            self.__train(None, image_dpath, annotation,
                         optimizer, scheduler,
                         batchsize, epoch, score_cutoff, cpu, gpu)
        elif gpu > 1:
            raise EnvironmentError('The current JustDeepIt does not support multiple GPUs for trianing.')
            
            dist_avail = torch.distributed.is_available()
            nccl_avail = torch.distributed.is_nccl_available()
            if not dist_avail:
                raise ValueError('Torch version does not support distributed computing.')
            if not nccl_avail:
                raise ValueError('Backend NCCL for multiGPUs computing is not available.')
        
            if torch.distributed.is_initialized():
                torch.distributed.destroy_process_group()
            s_per_gpu = 2
            w_per_gpu = 1
            if int(batchsize / gpu) > s_per_gpu:
                s_per_gpu = int(batchsize / gpu)
            if int(cpu / gpu) > w_per_gpu:
                w_per_gpu = int(cpu / gpu)

            master_addr = '127.0.0.1'
            master_port = self.__find_free_port()
            verbose = False
            _stop_event = torch.multiprocessing.Event()

            try:
                torch.multiprocessing.spawn(self.__train,
                         args=(image_dpath, annotation,
                               s_per_gpu, epoch, lr, score_cutoff, w_per_gpu, gpu,
                               True, master_addr, master_port),
                         nprocs=gpu)
            except KeyboardInterrupt:
                try:
                    torch.distributed.destroy_process_group()
                except KeyboardInterrupt:
                    _stop_event.set()
                    os.system("kill $(ps aux | grep multiprocessing.spawn | grep -v grep | awk '{print $2}')")

        else:
            raise EnvironmentError('Set the number of GPU correctly.')
    


    def __train(self, rank, image_dpath, annotation,
                optimizer, scheduler,
                batchsize, epoch, score_cutoff, cpu, gpu,
                distributed=False, master_addr='127.0.0.1', master_port='29555'):
        self.cfg.gpu_ids = range(gpu)
        if gpu > 1:
            self.__setup_process_group(rank, gpu, backend='nccl', master_addr=master_addr, master_port=master_port)
        
        train_cfg = dict(
            dataset_type = 'CocoDataset',
            classes = self.class_labels,
            data = dict(
                samples_per_gpu = batchsize,
                workers_per_gpu = cpu,
                train = dict(
                    type = 'RepeatDataset',
                    times = 1,
                    dataset = dict(
                        type = 'CocoDataset',
                        classes = self.class_labels,
                        img_prefix = image_dpath,
                        ann_file = annotation,
                        pipeline = self.cfg.train_pipeline))),
            checkpoint_config = dict(interval = 100))
        
        
        self.cfg.merge_from_dict(train_cfg)
        self.cfg.runner = dict(type='EpochBasedRunner', max_epochs=epoch)
        self.cfg.total_epochs = epoch
        self.__set_optimizer(optimizer)
        self.__set_scheduler(scheduler)
        
        datasets = [mmdet.datasets.build_dataset(self.cfg.data.train)]
        model = mmdet.models.build_detector(self.cfg.model)
        
        if self.cfg.load_from is not None:
            checkpoint = mmcv.runner.load_checkpoint(model, self.cfg.load_from, map_location='cpu')
        else:
            model.init_weights()
        model.CLASSES = self.class_labels
        model.train()
        mmdet.apis.train_detector(model, datasets, self.cfg, distributed=distributed, validate=False,
                                  timestamp=self.timestamp, meta=self.meta)


    
    
    def inference(self, image_path, score_cutoff=0.5, batchsize=8, cpu=4, gpu=1):
        
        if not torch.cuda.is_available():
            gpu = 0
        if torch.cuda.device_count() < gpu:
            gpu = torch.cuda.device_count()
        
        if gpu == 0:
            self.cfg.device = 'cpu'
        else:
            self.cfg.device = 'cuda'
          
        images_fpath = []
        if isinstance(image_path, list):
            images_fpath = image_path
        elif os.path.isfile(image_path):
            images_fpath = [image_path]
        else:
            for f in glob.glob(os.path.join(image_path, '*')):
                if os.path.splitext(f)[1].lower() in ['.jpg', '.png', '.tiff']:
                    images_fpath.append(f)
        
        #test_cfg = dict(
        #    classes = self.class_labels,
        #    data = dict(
        #        samples_per_gpu = batchsize,
        #        workers_per_gpu = cpu,
        #))
        #self.cfg.merge_from_dict(test_cfg)
        
        # build model for inference
        model = None
        if self.detector is not None:
            model = self.detector
        else:
            self.cfg.model.pretrained = None
            self.cfg.model.train_cfg = None
            model = mmdet.models.build_detector(self.cfg.model)
            checkpoint = mmcv.runner.load_checkpoint(model, self.cfg.load_from, map_location='cpu')
            model.CLASSES = self.class_labels
            model.cfg = self.cfg
            model.to(self.__get_device(gpu))
            model.eval()
            self.detector = model
        
        self.cfg.data.test.pipeline[0].type = 'LoadImageFromFile'
        self.cfg.data.test.pipeline = mmdet.datasets.replace_ImageToTensor(self.cfg.data.test.pipeline)
        test_pipeline = mmdet.datasets.pipelines.Compose(self.cfg.data.test.pipeline)
        
        # mini-batch inferences
        outputs = []
        for batch_id in tqdm.tqdm(range(math.ceil(len(images_fpath) / batchsize)), desc='Processed batches: ', leave=True):
            batch_id_from = batch_id * batchsize
            batch_id_to = min((batch_id + 1) * batchsize, len(images_fpath))
            batch_images_fpath = images_fpath[batch_id_from:batch_id_to]
            
            inputs = []
            for image_fpath in batch_images_fpath:
                #original_image = PIL.Image.open(image_fpath)
                #if original_image.mode == 'RGBA':
                #    original_image = original_image.convert('RGB')
                #original_image = np.array(PIL.ImageOps.exif_transpose(original_image))
                #inputs.append(test_pipeline(dict(img=original_image)))
                inputs.append(test_pipeline(dict(img_prefix=None, img_info=dict(filename=image_fpath))))
            
            inputs = mmcv.parallel.collate(inputs, samples_per_gpu=len(inputs))
            inputs['img_metas'] = [img_metas.data[0] for img_metas in inputs['img_metas']]
            inputs['img'] = [img.data[0] for img in inputs['img']]
        
            if next(model.parameters()).is_cuda:
                inputs = mmcv.parallel.scatter(inputs, [self.__get_device(gpu)])[0]
            else:
                for m in model.modules():
                    assert not isinstance(m, mmcv.ops.RoIPool), 'CPU inference with RoIPool is not supported currently.'
            
            # inference
            with torch.no_grad():
                batch_outputs = model(return_loss=False, rescale=True, **inputs)
        
            # format outputs
            for image_fpath, output in zip(batch_images_fpath, batch_outputs):
                output_fmt = self.__format_annotation(output, score_cutoff)
                outputs.append(ImageAnnotation(image_fpath, output_fmt))
        
        return ImageAnnotations(outputs)
    
    
    
    def __format_annotation(self, output, score_cutoff):
        
        bbox_result = None
        segm_result = None
        if isinstance(output, tuple):
            bbox_result, segm_result = output 
        else:
            bbox_result = output
        
        if segm_result is None:
            segm_result = [None] * len(bbox_result)
        
        regions = []
        for i, (bboxes, segms) in enumerate(zip(bbox_result, segm_result)):
            if segms is None:
                segms = [None] * len(bboxes)
            
            cl = self.class_labels[i]
            for bbox, segm in zip(bboxes, segms):
                if bbox[4] > score_cutoff:
                    bb = bbox[0:4].astype(np.int32)

                    sg = None
                    if (segm is not None) and (np.sum(segm) > 0):
                        sg = skimage.measure.find_contours(segm.astype(np.uint8), 0.5)
                    
                    region = {
                        'id': len(regions) + 1,
                        'class': cl,
                        'score': bbox[4],
                        'bbox': bb,
                    }
                    if sg is not None:
                        region['contour'] = sg[0][:, [1, 0]]
                    regions.append(region)
        
        return regions
        


            
        
    def save(self, weight_fpath, config_fpath=None):
        if not weight_fpath.endswith('.pth'):
            weight_fpath + '.pth'

        # pth file
        if os.path.exists(os.path.join(self.cfg.work_dir, 'latest.pth')):
            checkpoint = torch.load(os.path.join(self.cfg.work_dir, 'latest.pth'),
                                    map_location='cpu')
            if 'optimizer' in checkpoint:
                del checkpoint['optimizer']
            torch.save(checkpoint, weight_fpath)
        
        # config file
        if config_fpath is None:
            config_fpath = os.path.splitext(weight_fpath)[0] + '.py'
        self.cfg.dump(config_fpath)



