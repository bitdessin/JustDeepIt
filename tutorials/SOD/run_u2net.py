import os
import glob
from justdeepit.models import SOD


def run_u2net(dataset_dpath, ws_dpath, weight_fpath, train_strategy, detect_strategy):
    
    image_suffix = '_image.jpg'
    label_suffix = '_mask.png'
    
    
    # check and summarize images and labels (mask images)
    train_images = os.path.join(ws_dpath, 'train_images.txt')
    query_images = []
    with open(train_images, 'w') as outfh:
        for train_image_fpath in glob.glob(os.path.join(dataset_dpath, '*' + image_suffix), recursive=True):
            query_images.append(train_image_fpath)
            label_image_fpath = train_image_fpath.replace(image_suffix, label_suffix)
            outfh.write('{}\t{}\n'.format(train_image_fpath, label_image_fpath))
    
    
    # training
    u2net = SOD('u2net', workspace=ws_dpath)
    u2net.train(train_images, batch_size=32, epoch=100, cpu=32, gpu=1, strategy=train_strategy)
    u2net.save(weight_fpath)
    
    
    # detection
    u2net = SOD(model_weight=weight_fpath, workspace=ws_dpath)
    outputs = u2net.inference(query_images, strategy=detect_strategy, u_cutoff=0.5, batch_size=16, cpu=4, gpu=1)
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
    run_u2net(dataset_dpath, ws_dpath, os.path.join(ws_dpath, 'u2net.pth'), 'resize', 'resize')
    
    
    # strategy: randomcrop, slide
    ws_dpath = './workspace_crop'
    os.makedirs(ws_dpath, exist_ok=True)
    run_u2net(dataset_dpath, ws_dpath, os.path.join(ws_dpath, 'u2net.pth'), 'randomcrop', 'slide')




