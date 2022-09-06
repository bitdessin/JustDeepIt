import os
import glob
from justdeepit.models import SOD


def run_u2net(dataset_dpath, ws_dpath, weight_fpath, train_strategy, detect_strategy):
    
    train_images = os.path.join(dataset_dpath, 'images')
    mask_images = os.path.join(dataset_dpath, 'masks')
    query_images = os.path.join(dataset_dpath, 'images')
    
    # training
    model = SOD('u2net', workspace=ws_dpath)
    model.train(train_images, mask_images,
                batch_size=8, epoch=100, cpu=8, gpu=1, strategy=train_strategy)
    model.save(weight_fpath)
    
    # detection
    model = SOD(model_weight=weight_fpath, workspace=ws_dpath)
    outputs = model.inference(query_images, strategy=detect_strategy, u_cutoff=0.5, batch_size=8, cpu=8, gpu=1)
    for output in outputs:
        output.draw('bbox+contour', os.path.join(ws_dpath,
                    os.path.splitext(os.path.basename(output.image_path))[0] + '.contour.png'), label=True)
        output.draw('mask', os.path.join(ws_dpath,
                    os.path.splitext(os.path.basename(output.image_path))[0] + '.mask.png'))
        output.draw('rgbmask', os.path.join(ws_dpath,
                    os.path.splitext(os.path.basename(output.image_path))[0] + '.rgbmask.png'))
 


if __name__ == '__main__':
    
    dataset_dpath = './data'
    
    
    # strategy: resize, resize
    ws_dpath = './workspace_resize'
    os.makedirs(ws_dpath, exist_ok=True)
    run_u2net(dataset_dpath, ws_dpath, os.path.join(ws_dpath, 'u2net.pth'), 'resizing', 'resizing')
    
    
    # strategy: randomcrop, slide
    ws_dpath = './workspace_crop'
    os.makedirs(ws_dpath, exist_ok=True)
    run_u2net(dataset_dpath, ws_dpath, os.path.join(ws_dpath, 'u2net.pth'), 'randomcrop', 'sliding')




