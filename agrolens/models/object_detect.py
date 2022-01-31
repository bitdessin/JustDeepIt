import os
import pkg_resources

class OD:
    """Base module to generate object detection model
    
    Class :class:`OD <agrolens.models.OD>` generates object detection models
    by internally calling the MMDetection or Detectron2 library.
    This class generates a model from the configuration file
    (``model_config``) considering the backend.

    If ``backend`` is specified to ``detectron2``,
    then calls Detectron2 package to generate a model,
    otherwise if ``mmdetection`` is specified then calls MMDetection to generate a model. 


    Args:
        class_label (str): A path to a file which contains class labels.
                       The file should be multiple rows with one column,
                       and string in each row represents a class label.
        model_arch (str): A string to specify model architecture.
                       If ``model_config`` is given, this option will be ignored.
        model_config (str): A path to configure file for building models.
                        If the configure file is not given or does not exist at the specified
                        path, then load the default configure file according ``model_arch``.
        model_weight (str): A path to a model weights. If ``None``, then use the initial
                        value that randomly generated by the packages.
        workspace (str): A path to workspace directory. Log information and checkpoints of model
                     training will be stored in this directory.
                     If ``None``, then create temporary directory in system temporary directory
                     such as :file:`/tmp` and will be removed after finishing the program.
        backend (str): Specify the backend to build object detection mode.
                   ``detectron2`` or ``mmdetection`` can be speficied.
    
    Examples:
        >>> from agrolens.models import OD
        >>> 
        >>> # initialize Faster RCNN with random weights using MMDetection backend
        >>> model = OD('./class_label.txt', model_arch='fasterrcnn', backend='mmdetection')
        >>> 
        >>> # initialize Faster RCNN with randomm weights using Detectron2 backend
        >>> model = OD('./class_label.txt', model_arch='fasterrcnn', backend='detectron2')
        >>> 
        >>> # initialize RetinaNet with trained weights using MMDetection backend
        >>> model = OD('./class_label.txt', model_arch='retinanet', mdoel_weight='trained_weight.pth',
        >>>            backend='mmdetection')
        >>> 
        >>> # initialize RetinaNet with trained weights using Detectron2 backend
        >>> model = OD('./class_label.txt', model_arch='retinanet', model_weight='trained_weight.pth',
        >>>            backend='detectron2')
        >>> 
        >>> # check the avaliable architectures
        >>> model = OD()
        >>> print(model.available_architectures)
    """  
    
    
    
    def __init__(self, class_label=None, model_arch=None, model_config=None, model_weight=None, workspace=None, backend='mmdetection'):
        
        self.module = None
        self.available_architectures = {
            'mmdetection': ['fasterrcnn', 'retinanet', 'yolo3', 'yolof',
                            'ssd', 'fcos', 'userdefined'],
            'detectron2': ['fasterrcnn', 'retinanet']
        } 
        self.backend = backend
        self.model_arch = model_arch
        self.workspace = workspace
        self.module = self.__init_module(class_label, model_arch, model_config, model_weight, workspace, backend)
    
    
    
    
    def __init_module(self, class_label, model_arch, model_config, model_weight, workspace, backend):
        if model_arch is None and model_config is None:
            return None
        
        module = None
        model_arch = model_arch.replace('-', '').replace(' ', '').lower()
        backend = backend.lower()
        
        if backend in ['mm', 'mmdet', 'mmdetection']:
            backend = 'mmdet'
        elif backend in ['d2', 'detectron', 'detectron2']:
            backend = 'd2'
        
        if backend == 'mmdet':
            if  model_arch == 'fasterrcnn':
                model_config = pkg_resources.resource_filename('agrolens', 'models/configs/mmdet/faster_rcnn_r101_fpn_mstrain_3x_coco.py')
            elif model_arch == 'retinanet':
                model_config = pkg_resources.resource_filename('agrolens', 'models/configs/mmdet/retinanet_r101_fpn_mstrain_640-800_3x_coco.py')
            elif model_arch == 'ssd':
                model_config = pkg_resources.resource_filename('agrolens', 'models/configs/mmdet/ssd512_coco.py')
            elif model_arch == 'yolo3':
                model_config = pkg_resources.resource_filename('agrolens', 'models/configs/mmdet/yolov3_d53_mstrain-608_273e_coco.py')
            elif model_arch == 'yolof':
                model_config = pkg_resources.resource_filename('agrolens', 'models/configs/mmdet/yolof_r50_c5_8x8_1x_coco.py')
            elif model_arch == 'fcos':
                model_config = pkg_resources.resource_filename('agrolens', 'models/configs/mmdet/fcos_r101_caffe_fpn_gn-head_mstrain_640-800_2x_coco.py')
            else:
                NotImplementedError('AgroLens does not support {} archtecture when MMDetection is specified as a backend.'.format(backend))
            from agrolens.models.utils.mmdetbase import MMDetBase
            module = MMDetBase(class_label, model_arch, model_config, model_weight, workspace)
        
        elif backend == 'd2':
            if  model_arch == 'fasterrcnn':
                model_config = 'COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml'
            elif model_arch == 'retinanet':
                model_config = 'COCO-Detection/retinanet_R_101_FPN_3x.yaml'
            else:
                NotImplementedError('AgroLens does not support {} archtecture when Detectron2 is specified as a backend. Try to set backend as MMDetection.'.format(backend))
                
            from agrolens.models.utils.detectron2base import DetectronBase
            module = DetectronBase(class_label, model_arch, model_config, model_weight, workspace)
        
        return module
    
    
    def train(self, annotation, image_dpath,
              batchsize=32, epoch=1000, lr=0.0001, score_cutoff=0.7, cpu=8, gpu=1):
        """Train model
        
        Method :func:`train <agrolens.models.OD.train>` is used for
        training models based on MMDetection or Detectron2.
        This method requires a COCO format annotation file and
        a path to the directory containing the training images.
        
        Args:
            annotation (str): A file path to COCO format annotation file.
            image_dpath (str): A path to directory which contains all training images.
            batch_size (int): Batch size for each GPU.
            epoch (int): Epoch.
            lr (float): Learning rate.
            score_cutoff (float): Cutoff of score for object detection.
            gpu (int): Number of GPUs for model training.
            cpu (int): Number of workers for pre-prociessing images for each GPU.
        
        Examples:
            >>> from agrolens.models import OD
            >>> 
            >>> model = OD('./class_label.txt', model_arch='fasterrcnn')
            >>> model.train('./train_images', './train_images/annotations.coco.json')
        """
        self.module.train(annotation, image_dpath,
                          batchsize=batchsize, epoch=epoch, lr=lr, score_cutoff=score_cutoff,
                          cpu=cpu, gpu=gpu)
    
    
    
    def save(self, weight_fpath, config_fpath=None):
        '''Save the trained model
        
        Method :func:`save <agrolens.models.OD.save>`
        stores the trained model weights and model configuration.
        
        Args:
            weight_fpath (str): A path to save the weights.
            config_fpath (str): A path to save the model configure. If ``None``,
                                then save the configure to file with same name
                                of ``weight_fpath`` but different extension.
        
        Examples:
            >>> from agrolens.models import OD
            >>> 
            >>> model = OD('./class_label.txt', model_arch='fasterrcnn')
            >>> model.train('./train_images', './train_images/annotations.coco.json')
            >>> odnet.save('./trained_weight.pth')
        ''' 
        self.module.save(weight_fpath, config_fpath)
    
    
    
    def inference(self, images, score_cutoff=0.7, batchsize=32, cpu=8, gpu=1):
        '''Detect objects from images
        
        Method :func:`inference <agrolens.models.OD.inference>` is used to
        detect objects from an image or images with a given model (weights).
        
        Args:
            images (str): A path to a image file or a path to a directory which contains
                          multiple images.
            score_cutoff (float): Cutoff for object detection.
            batchsize (int): Number of batches.
            gpu (int): Number of GPUs.
            cpu (int): Number of CPUs.
        
        Returns:
            :class:`ImageAnnotation <agrolens.utils.ImageAnnotation>` class object
            or a list of :class:`ImageAnnotation <agrolens.utils.ImageAnnotation>` class object.
        
        Examples:
            >>> import os
            >>> from agrolens.models import OD
            >>> 
            >>> model = OD('./class_label.txt', model_arch='fasterrcnn')
            >>> 
            >>> # inference single image
            >>> output = model.inference('sample.jpg')
            >>> output.draw('contour', 'output/sample.png')
            >>> 
            >>> # inference multiple images
            >>> test_images = ['sample1.jpg', 'sample2.jpg', 'sample3.jpg']
            >>> outputs = model.inference(sample_images)
            >>> for test_image, output in zip(test_images, outputs):
            >>>     bbox_img_fpath = os.path.splitext(test_image)[0] + '.bbox.png'
            >>>     output.draw('bbox', bbox_img_fpath)
            >>> 
        '''
        
        return self.module.inference(images, score_cutoff=score_cutoff, batchsize=batchsize, cpu=cpu, gpu=gpu)
    
    





