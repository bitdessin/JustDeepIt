import sys
import os
from justdeepit.models import OD
from justdeepit.utils import ImageAnnotation, ImageAnnotations


def train(dataset_dpath, model_backend):
    
    # traininng images
    train_images = os.path.join(dataset_dpath, 'train')
    train_images_annotation = os.path.join(dataset_dpath, 'train.json')
    class_label = os.path.join(dataset_dpath, 'class_label.txt')
    # temporary folder
    ws_dpath = os.path.join('outputs', model_backend)
    os.makedirs(ws_dpath, exist_ok=True)

    if model_backend == 'mmdetection':
        # download faster_rcnn_r101_fpn_mstrain_3x_coco_20210524_110822-4d4d2ca8.pth from
        # https://github.com/open-mmlab/mmdetection/tree/master/configs/faster_rcnn
        init_weight = os.path.join(os.path.dirname(__file__),
                    'faster_rcnn_r101_fpn_mstrain_3x_coco_20210524_110822-4d4d2ca8.pth')
    else:
        # the pre-trained weights will be automatically downloaded in detectron2 function
        init_weight = None
    
    net = OD(class_label, model_arch='fasterrcnn', model_weight=init_weight,
             workspace=ws_dpath, backend=model_backend)
    net.train(train_images, train_images_annotation,
              batchsize=8, epoch=100, lr=0.001,
              gpu=1, cpu=16)
    net.save(os.path.join(ws_dpath, 'gwhd2021.fasterrcnn.' + model_backend + '.pth'))


def test(dataset_dpath, model_backend):

    # test images
    test_images = os.path.join(dataset_dpath, 'test')
    class_label = os.path.join(dataset_dpath, 'class_label.txt')
    # temporary folder
    ws_dpath = os.path.join('outputs', model_backend)
    trained_weight = os.path.join(ws_dpath, 'gwhd2021.fasterrcnn.' + model_backend + '.pth')
    
    net = OD(class_label, model_arch='fasterrcnn',
             model_weight=trained_weight, workspace=ws_dpath, backend=model_backend)
    detect_outputs = net.inference(test_images, score_cutoff=0.7, batchsize=8, gpu=1, cpu=16)
    
    for detect_output in detect_outputs:
        detect_output.draw('bbox', os.path.join(ws_dpath, os.path.basename(detect_output.image_path)))
    
    img_anns = ImageAnnotations(detect_outputs)
    img_anns.format('coco', os.path.join(ws_dpath, 'annotation.json'))




if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Usage: python scripts/run_justdeepit.py train')
        print('       python scripts/run_justdeepit.py test')
        sys.exit(0)

    run_mode = sys.argv[1]
    model_backend = sys.argv[2] if len(sys.argv) >= 3 else 'detectron2'

    dataset_dpath = '.'

    if run_mode == 'train':
        train(dataset_dpath, model_backend)
    elif run_mode == 'test':
        test(dataset_dpath, model_backend)
    else:
        print('unsupported running mode, set `train` or `test`.')


