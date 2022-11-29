import os
import sys
import json
import glob
import tqdm
import logging
import tempfile
import math
import numpy as np
import torch
import PIL.Image
import skimage
import skimage.measure
import cv2
from justdeepit.models.abstract import ModuleTemplate
from justdeepit.utils import ImageAnnotation, ImageAnnotations

logger = logging.getLogger(__name__)

try:
    import detectron2
    import detectron2.config
    import detectron2.model_zoo
    import detectron2.structures
    import detectron2.engine
    import detectron2.modeling
    import detectron2.checkpoint
    import detectron2.data.transforms
except ImportError:
    msg = 'JustDeepIt requires detectron2 library to build models. Make sure detectron2 has been installed already.'
    logger.error(msg)
    raise ImportError(msg)




class DetectronBase(ModuleTemplate):

    def __init__(self, class_labels, model_arch=None, model_config=None, model_weight=None, workspace=None):
        # model
        self.model_arch = model_arch
        self.model = None
        self.detector = None
        
        # config and weight
        self.cfg = self.__get_config(model_config, model_weight)
        
        # workspace
        if workspace is None:
            self.tempd = tempfile.TemporaryDirectory()
            self.workspace = self.tempd.name
        else:
            self.tempd = None
            self.workspace = workspace
        self.cfg.OUTPUT_DIR = os.path.abspath(self.workspace)
        logger.info('Workspace for Detectron2-based model is set at `{}`.'.format(self.workspace))
        
        
        # class labels
        self.class_labels = self.__parse_class_labels(class_labels)
        self.cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(self.class_labels)
        self.cfg.MODEL.RETINANET.NUM_CLASSES = len(self.class_labels)
        
        # default params
        self.cfg.MODEL.DEVICE = 'cpu'
    
    
    def __del__(self):
        try:
            if self.tempd is not None:
                self.tempd.cleanup()
        except:
            pass
    
    
    def __get_config(self, model_config, model_weight):
        if model_config is None or model_config == '':
            raise ValueError('Configuration file for Detectron2 cannot be empty.')
        
        cfg = None
        try:
            model_config_fpath = model_config
            if not os.path.exists(model_config_fpath):
                # if the given path does not exist, set the chkpoint from github as a initial params
                model_config_fpath = detectron2.model_zoo.get_config_file(model_config_fpath)
            cfg = detectron2.config.get_cfg()
            cfg.merge_from_file(model_config_fpath)
        
            if (model_weight is None) or (not os.path.exists(model_weight)):
                model_weight = detectron2.model_zoo.get_checkpoint_url(model_config)
            cfg.MODEL.WEIGHTS = model_weight
        
        except:
            raise FileNotFoundError('The path or name of the configuration file `{}` is incorrect. JustDeepIt cannot find or download.'.format(model_config))
           
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
        # device = torch.device(device)
        return device
   
    
        

    def __dataloader(self, train_data_fpath):
        data_dict = []

        with open(train_data_fpath, mode='r', encoding='utf-8') as datafh:
            for imdata in datafh:
                imdata = imdata.replace('\n', '').split('\t')
                imann = ImageAnnotation(train_data_fpath[0], train_data_fpath[1], train_data_fpath[2])
                data_dict.append(imann.format('detectron'))

        return data_dict


    
    def __n_train_images(self, image_dpath):
        n = 0
        for fpath in glob.glob(os.path.join(image_dpath, '*')):
            if os.path.splitext(fpath)[1].lower() in ['.jpg', '.jpeg', '.png', '.tif', '.tiff']:
                n += 1
        return n
    
    
    def train(self, image_dpath, annotation,
              optimizer=None, scheduler=None,
              batchsize=8, epoch=100, score_cutoff=0.5, cpu=4, gpu=1):
        lr = 0.001
        
        if not torch.cuda.is_available():
            gpu = 0
        if torch.cuda.device_count() < gpu:
            gpu = torch.cuda.device_count()
        
        # train settings
        train_dataset_id = 'ds:{}_train'.format(image_dpath)
        self.cfg.MODEL.DEVICE =self.__get_device(gpu)
        self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = score_cutoff
        self.cfg.DATALOADER.NUM_WORKERS = cpu
        self.cfg.SOLVER.IMS_PER_BATCH   = batchsize
        self.cfg.SOLVER.BASE_LR  = lr
        self.cfg.SOLVER.MAX_ITER = int(self.__n_train_images(image_dpath) / batchsize * epoch)
        self.cfg.DATASETS.TRAIN = (train_dataset_id, )
        self.cfg.DATASETS.TEST = ()

        # datasest settings
        detectron2.data.DatasetCatalog.clear()
        detectron2.data.datasets.register_coco_instances(train_dataset_id, {}, annotation, image_dpath)
        detectron2.data.MetadataCatalog.get(train_dataset_id).set(thing_classes=self.class_labels)
        
        # train model
        trainer = detectron2.engine.DefaultTrainer(self.cfg)
        trainer.resume_or_load(resume=False)
        trainer.train()
        
        self.model = trainer.model

    
    def inference(self, image_path, score_cutoff=0.5, batchsize=8, cpu=4, gpu=1):
        images_fpath = []
        if isinstance(image_path, list):
            images_fpath = image_path
        elif os.path.isfile(image_path):
            images_fpath = [image_path]
        else:
            for f in glob.glob(os.path.join(image_path, '*')):
                if os.path.splitext(f)[1].lower() in ['.jpg', '.png', '.tiff']:
                    images_fpath.append(f)
            
        # build model for inference
        if self.detector is not None:
            model = self.detector
        else:
            self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = score_cutoff
            self.cfg.MODEL.DEVICE = self.__get_device(gpu)
            self.cfg.DATALOADER.NUM_WORKERS = cpu
            self.cfg.SOLVER.IMS_PER_BATCH = batchsize
            model = detectron2.modeling.build_model(self.cfg)
            model.eval()
            checkpointer = detectron2.checkpoint.DetectionCheckpointer(model)
            checkpointer.load(self.cfg.MODEL.WEIGHTS)
            self.detector = model
        
        aug = detectron2.data.transforms.ResizeShortestEdge(
            [self.cfg.INPUT.MIN_SIZE_TEST, self.cfg.INPUT.MIN_SIZE_TEST],
            self.cfg.INPUT.MAX_SIZE_TEST
        )
        # mini-batch inferences
        outputs = []
        for batch_id in tqdm.tqdm(range(math.ceil(len(images_fpath) / batchsize)), desc='Processed batches:', leave=True):
            batch_id_from = batch_id * batchsize
            batch_id_to = min((batch_id + 1) * batchsize, len(images_fpath))
            batch_images_fpath = images_fpath[batch_id_from:batch_id_to]
            inputs = []
            for image_fpath in batch_images_fpath:
                pil_im = PIL.Image.open(image_fpath).convert('RGB')
                original_image = np.array(PIL.ImageOps.exif_transpose(pil_im))
                original_image = original_image[:, :, ::-1]
                height, width = original_image.shape[:2]
                image = aug.get_transform(original_image).apply_image(original_image)
                image = torch.as_tensor(image.astype('float32').transpose(2, 0, 1))
                inputs.append({'image': image, 'height': height, 'width': width})
                pil_im.close()
        
            # inference
            with torch.no_grad():
                batch_outputs = model(inputs)
        
            for image_fpath, output in zip(batch_images_fpath, batch_outputs):
                output_fmt = None
                if 'instances' in output:
                    output['instances'] =  output['instances'].to('cpu')
                    pred_classes = output['instances'].pred_classes.numpy()
                    scores = output['instances'].scores.numpy()
                    pred_boxes = np.array([k.numpy() for k in output['instances'].pred_boxes]).astype(np.int32)
                    if hasattr(output['instances'], 'pred_masks'):
                        pred_masks = np.array([k.numpy() for k in output['instances'].pred_masks]).astype(np.uint8)
                    else:
                        pred_masks = [None] * len(pred_boxes)
                
                    output_fmt = self.__format_annotation(
                        pred_classes, scores, pred_boxes, pred_masks, score_cutoff, image_fpath
                    )
                outputs.append(ImageAnnotation(image_fpath, output_fmt))
        
        return ImageAnnotations(outputs)
    
    
    def __format_annotation(self, pred_classes, scores, pred_boxes, pred_masks, score_cutoff, tmp):
        regions = []
        for cl, score, bbox, segm  in zip(pred_classes, scores, pred_boxes, pred_masks):
            if score > score_cutoff:
                cl = self.class_labels[cl]
                bb = bbox.tolist()
                sg = None
                if (segm is not None) and (np.sum(segm) > 0):
                    sg = skimage.measure.find_contours(segm.astype(np.uint8), 0.5)
                
                region = {
                    'id'     : len(regions) + 1,
                    'class'  : cl,
                    'score'  : score,
                    'bbox'   : bb,
                }
                if sg is not None:
                    region['contour'] = sg[0][:, [1, 0]]
                regions.append(region)

        return regions
        
    
    def save(self, weight_fpath, config_fpath=None):
        if not weight_fpath.endswith('.pth'):
            weight_fpath + '.pth'
        
        # pth file
        if self.model is not None:
            torch.save(self.model.to('cpu').state_dict(), weight_fpath)
        
        # config file
        if config_fpath is None:
            config_fpath = os.path.splitext(weight_fpath)[0] + '.yaml'
        with open(config_fpath, 'w') as outfh:
            outfh.write(self.cfg.dump())


