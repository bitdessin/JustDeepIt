import os
import glob
import logging
from justdeepit.models import OD
logging.basicConfig(level=logging.WARNING)
logging.getLogger('detectron2.utils.events').setLevel(level=logging.WARNING)
logging.getLogger('fvcore.common.checkpoint').setLevel(level=logging.WARNING)
logging.getLogger('mmdet').setLevel(level=logging.WARNING)



def run_fasterrcnn(backend, train_images, train_annotation, class_label, query_images, ws_dpath):

    weight = os.path.join(ws_dpath, 'fasterrcnn.pth')
    
    # training
    net = OD(class_label, model_arch='fasterrcnn', workspace=ws_dpath, backend=backend)
    net.train(train_annotation, train_images, batchsize=8, epoch=1000, lr=0.001, gpu=1, cpu=32)
    net.save(weight)
    
    # detection
    net = OD(class_label, model_arch='fasterrcnn', model_weight=weight, workspace=ws_dpath, backend=backend)
    outputs = net.inference(query_images, batchsize=8, gpu=1, cpu=32)
    for output in outputs:
        output.draw('bbox+contour', os.path.join(ws_dpath,
                    os.path.splitext(os.path.basename(output.image_path))[0] + '.png'),
                    label=True, score=True, alpha=0.3)
        outputs.format('coco', os.path.join(ws_dpath, 'result_coco.json'))
        
    

if __name__ == '__main__':

    train_images = './data/images'
    train_annotation = './data/annotations/COCO/instances_default.json'
    class_label = './data/class_label.txt'
    query_images = './data/images'

    ws_dpath = './workspace_mmdet'
    os.makedirs(ws_dpath, exist_ok=True)
    run_fasterrcnn('mmdet', train_images, train_annotation, class_label, query_images, ws_dpath)

    ws_dpath = './workspace_d2'
    os.makedirs(ws_dpath, exist_ok=True)
    run_fasterrcnn('detectron2', train_images, train_annotation, class_label, query_images, ws_dpath)



