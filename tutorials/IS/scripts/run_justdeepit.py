import os
import glob
import logging
from justdeepit.models import IS
logging.basicConfig(level=logging.WARNING)
logging.getLogger('fvcore.common.checkpoint').setLevel(level=logging.WARNING)
logging.getLogger('mmdet').setLevel(level=logging.WARNING)



def run_maskrcnn(train_dataset, class_label, query_images, ws_dpath):

    weight = os.path.join(ws_dpath, 'maskrcnn.pth')
    
    # training
    net = IS(class_label, model_arch='maskrcnn', workspace=ws_dpath)
    net.train(train_dataset,
              batchsize=8, epoch=500, gpu=1, cpu=32)
    net.save(weight)
    
    # detection
    net = IS(class_label, model_arch='maskrcnn', model_weight=weight, workspace=ws_dpath)
    outputs = net.inference(query_images, batchsize=8, gpu=1, cpu=32)
    for output in outputs:
        output.draw('bbox+contour', os.path.join(ws_dpath,
                    os.path.splitext(os.path.basename(output.image_path))[0] + '.png'),
                    label=True, score=True, alpha=0.3)
        outputs.format('coco', os.path.join(ws_dpath, 'result_coco.json'))
        
    

if __name__ == '__main__':
    
    train_dataset = {
        'images': './data/images',
        'annotations': './data/annotations/COCO/instances_default.json',
        'annotation_format': 'coco'
    }
    class_label = './data/class_label.txt'
    query_images = './data/images'

    ws_dpath = './workspace_tmp'
    os.makedirs(ws_dpath, exist_ok=True)
    run_maskrcnn(train_dataset, class_label, query_images, ws_dpath)




