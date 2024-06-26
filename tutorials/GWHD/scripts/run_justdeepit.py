import sys
import os
from justdeepit.models import OD
from justdeepit.utils import ImageAnnotation, ImageAnnotations


def train(dataset_dpath, model_backend):
    
    # traininng images
    train_dataset = {
        'images': os.path.join(dataset_dpath, 'train'),
        'annotations': os.path.join(dataset_dpath, 'train.json'),
        'annotation_format': 'coco'
    }
    class_label = os.path.join(dataset_dpath, 'class_label.txt')
    # temporary folder
    ws_dpath = 'outputs'
    os.makedirs(ws_dpath, exist_ok=True)

    
    net = OD(class_label, model_arch='fasterrcnn', workspace=ws_dpath)
    net.train(train_dataset, batchsize=8, epoch=100, gpu=1, cpu=16)
    net.save(os.path.join(ws_dpath, 'gwhd2021.fasterrcnn.pth'))


def test(dataset_dpath, model_backend):

    # test images
    test_images = os.path.join(dataset_dpath, 'test')
    class_label = os.path.join(dataset_dpath, 'class_label.txt')
    # temporary folder
    ws_dpath = 'outputs'
    trained_weight = os.path.join(ws_dpath, 'gwhd2021.fasterrcnn..pth')
    
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


