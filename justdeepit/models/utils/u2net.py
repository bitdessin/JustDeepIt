import os
import logging
import datetime
import gc
import joblib
import glob
import tqdm
import tempfile
import math
import numpy as np
import torch
import torchvision
import skimage
import skimage.io
import skimage.transform
import skimage.measure
import skimage.morphology
import justdeepit.utils
from justdeepit.models.abstract import ModuleTemplate
logger = logging.getLogger(__name__)


class REBNCONV(torch.nn.Module):
    def __init__(self,in_ch=3, out_ch=3, dirate=1):
        super().__init__()
        self.conv_s1 = torch.nn.Conv2d(in_ch, out_ch, 3, padding=1 * dirate, dilation=1 * dirate)
        self.bn_s1 = torch.nn.BatchNorm2d(out_ch)
        self.relu_s1 = torch.nn.ReLU(inplace=True)

    def forward(self,x):
        hx = x
        xout = self.relu_s1(self.bn_s1(self.conv_s1(hx)))
        return xout


def _upsample_like(src,tar):
    src = torch.nn.functional.interpolate(src,size=tar.shape[2:], mode='bilinear', align_corners=True)
    return src


class RSU7(torch.nn.Module):
    def __init__(self, in_ch=3, mid_ch=12, out_ch=3):
        super().__init__()

        self.rebnconvin = REBNCONV(in_ch, out_ch, dirate=1)
        self.rebnconv1 = REBNCONV(out_ch, mid_ch, dirate=1)
        self.pool1 = torch.nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.rebnconv2 = REBNCONV(mid_ch, mid_ch, dirate=1)
        self.pool2 = torch.nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.rebnconv3 = REBNCONV(mid_ch, mid_ch, dirate=1)
        self.pool3 = torch.nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.rebnconv4 = REBNCONV(mid_ch, mid_ch, dirate=1)
        self.pool4 = torch.nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.rebnconv5 = REBNCONV(mid_ch, mid_ch, dirate=1)
        self.pool5 = torch.nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.rebnconv6 = REBNCONV(mid_ch, mid_ch, dirate=1)
        self.rebnconv7 = REBNCONV(mid_ch, mid_ch, dirate=2)

        self.rebnconv6d = REBNCONV(mid_ch*2,mid_ch, dirate=1)
        self.rebnconv5d = REBNCONV(mid_ch*2,mid_ch, dirate=1)
        self.rebnconv4d = REBNCONV(mid_ch*2,mid_ch, dirate=1)
        self.rebnconv3d = REBNCONV(mid_ch*2,mid_ch, dirate=1)
        self.rebnconv2d = REBNCONV(mid_ch*2,mid_ch, dirate=1)
        self.rebnconv1d = REBNCONV(mid_ch*2,out_ch, dirate=1)

    def forward(self,x):
        hx = x
        hxin = self.rebnconvin(hx)
        hx1 = self.rebnconv1(hxin)
        hx = self.pool1(hx1)
        hx2 = self.rebnconv2(hx)
        hx = self.pool2(hx2)
        hx3 = self.rebnconv3(hx)
        hx = self.pool3(hx3)
        hx4 = self.rebnconv4(hx)
        hx = self.pool4(hx4)
        hx5 = self.rebnconv5(hx)
        hx = self.pool5(hx5)
        hx6 = self.rebnconv6(hx)
        hx7 = self.rebnconv7(hx6)
        hx6d =  self.rebnconv6d(torch.cat((hx7, hx6), 1))
        hx6dup = _upsample_like(hx6d,hx5)
        hx5d =  self.rebnconv5d(torch.cat((hx6dup, hx5), 1))
        hx5dup = _upsample_like(hx5d,hx4)
        hx4d = self.rebnconv4d(torch.cat((hx5dup, hx4), 1))
        hx4dup = _upsample_like(hx4d,hx3)
        hx3d = self.rebnconv3d(torch.cat((hx4dup, hx3), 1))
        hx3dup = _upsample_like(hx3d,hx2)
        hx2d = self.rebnconv2d(torch.cat((hx3dup, hx2), 1))
        hx2dup = _upsample_like(hx2d,hx1)
        hx1d = self.rebnconv1d(torch.cat((hx2dup, hx1), 1))

        return hx1d + hxin


class RSU6(torch.nn.Module):
    def __init__(self, in_ch=3, mid_ch=12, out_ch=3):
        super().__init__()

        self.rebnconvin = REBNCONV(in_ch, out_ch, dirate=1)
        self.rebnconv1 = REBNCONV(out_ch, mid_ch, dirate=1)
        self.pool1 = torch.nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.rebnconv2 = REBNCONV(mid_ch, mid_ch, dirate=1)
        self.pool2 = torch.nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.rebnconv3 = REBNCONV(mid_ch, mid_ch, dirate=1)
        self.pool3 = torch.nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.rebnconv4 = REBNCONV(mid_ch, mid_ch, dirate=1)
        self.pool4 = torch.nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.rebnconv5 = REBNCONV(mid_ch, mid_ch, dirate=1)
        self.rebnconv6 = REBNCONV(mid_ch, mid_ch, dirate=2)

        self.rebnconv5d = REBNCONV(mid_ch * 2, mid_ch, dirate=1)
        self.rebnconv4d = REBNCONV(mid_ch * 2, mid_ch, dirate=1)
        self.rebnconv3d = REBNCONV(mid_ch * 2, mid_ch, dirate=1)
        self.rebnconv2d = REBNCONV(mid_ch * 2, mid_ch, dirate=1)
        self.rebnconv1d = REBNCONV(mid_ch * 2, out_ch, dirate=1)

    def forward(self,x):

        hx = x
        hxin = self.rebnconvin(hx)
        hx1 = self.rebnconv1(hxin)
        hx = self.pool1(hx1)
        hx2 = self.rebnconv2(hx)
        hx = self.pool2(hx2)
        hx3 = self.rebnconv3(hx)
        hx = self.pool3(hx3)
        hx4 = self.rebnconv4(hx)
        hx = self.pool4(hx4)
        hx5 = self.rebnconv5(hx)
        hx6 = self.rebnconv6(hx5)

        hx5d =  self.rebnconv5d(torch.cat((hx6, hx5), 1))
        hx5dup = _upsample_like(hx5d, hx4)
        hx4d = self.rebnconv4d(torch.cat((hx5dup, hx4), 1))
        hx4dup = _upsample_like(hx4d, hx3)
        hx3d = self.rebnconv3d(torch.cat((hx4dup, hx3), 1))
        hx3dup = _upsample_like(hx3d, hx2)
        hx2d = self.rebnconv2d(torch.cat((hx3dup, hx2), 1))
        hx2dup = _upsample_like(hx2d, hx1)
        hx1d = self.rebnconv1d(torch.cat((hx2dup, hx1), 1))

        return hx1d + hxin


class RSU5(torch.nn.Module):
    def __init__(self, in_ch=3, mid_ch=12, out_ch=3):
        super().__init__()

        self.rebnconvin = REBNCONV(in_ch, out_ch, dirate=1)
        self.rebnconv1 = REBNCONV(out_ch, mid_ch, dirate=1)
        self.pool1 = torch.nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.rebnconv2 = REBNCONV(mid_ch, mid_ch, dirate=1)
        self.pool2 = torch.nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.rebnconv3 = REBNCONV(mid_ch, mid_ch, dirate=1)
        self.pool3 = torch.nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.rebnconv4 = REBNCONV(mid_ch, mid_ch, dirate=1)
        self.rebnconv5 = REBNCONV(mid_ch, mid_ch, dirate=2)

        self.rebnconv4d = REBNCONV(mid_ch * 2, mid_ch, dirate=1)
        self.rebnconv3d = REBNCONV(mid_ch * 2, mid_ch, dirate=1)
        self.rebnconv2d = REBNCONV(mid_ch * 2, mid_ch, dirate=1)
        self.rebnconv1d = REBNCONV(mid_ch * 2, out_ch, dirate=1)

    def forward(self,x):

        hx = x
        hxin = self.rebnconvin(hx)
        hx1 = self.rebnconv1(hxin)
        hx = self.pool1(hx1)
        hx2 = self.rebnconv2(hx)
        hx = self.pool2(hx2)
        hx3 = self.rebnconv3(hx)
        hx = self.pool3(hx3)
        hx4 = self.rebnconv4(hx)
        hx5 = self.rebnconv5(hx4)
        hx4d = self.rebnconv4d(torch.cat((hx5, hx4), 1))
        hx4dup = _upsample_like(hx4d, hx3)
        hx3d = self.rebnconv3d(torch.cat((hx4dup, hx3), 1))
        hx3dup = _upsample_like(hx3d, hx2)
        hx2d = self.rebnconv2d(torch.cat((hx3dup, hx2), 1))
        hx2dup = _upsample_like(hx2d, hx1)
        hx1d = self.rebnconv1d(torch.cat((hx2dup, hx1), 1))

        return hx1d + hxin


class RSU4(torch.nn.Module):
    def __init__(self, in_ch=3, mid_ch=12, out_ch=3):
        super().__init__()

        self.rebnconvin = REBNCONV(in_ch, out_ch, dirate=1)
        self.rebnconv1 = REBNCONV(out_ch, mid_ch, dirate=1)
        self.pool1 = torch.nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.rebnconv2 = REBNCONV(mid_ch, mid_ch, dirate=1)
        self.pool2 = torch.nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.rebnconv3 = REBNCONV(mid_ch, mid_ch, dirate=1)
        self.rebnconv4 = REBNCONV(mid_ch, mid_ch, dirate=2)

        self.rebnconv3d = REBNCONV(mid_ch * 2, mid_ch, dirate=1)
        self.rebnconv2d = REBNCONV(mid_ch * 2, mid_ch, dirate=1)
        self.rebnconv1d = REBNCONV(mid_ch * 2, out_ch, dirate=1)

    def forward(self,x):

        hx = x
        hxin = self.rebnconvin(hx)
        hx1 = self.rebnconv1(hxin)
        hx = self.pool1(hx1)
        hx2 = self.rebnconv2(hx)
        hx = self.pool2(hx2)
        hx3 = self.rebnconv3(hx)
        hx4 = self.rebnconv4(hx3)
        hx3d = self.rebnconv3d(torch.cat((hx4, hx3), 1))
        hx3dup = _upsample_like(hx3d, hx2)
        hx2d = self.rebnconv2d(torch.cat((hx3dup, hx2), 1))
        hx2dup = _upsample_like(hx2d, hx1)
        hx1d = self.rebnconv1d(torch.cat((hx2dup, hx1), 1))

        return hx1d + hxin


class RSU4F(torch.nn.Module):
    def __init__(self, in_ch=3, mid_ch=12, out_ch=3):
        super().__init__()

        self.rebnconvin = REBNCONV(in_ch, out_ch, dirate=1)
        self.rebnconv1 = REBNCONV(out_ch, mid_ch, dirate=1)
        self.rebnconv2 = REBNCONV(mid_ch, mid_ch, dirate=2)
        self.rebnconv3 = REBNCONV(mid_ch, mid_ch, dirate=4)
        self.rebnconv4 = REBNCONV(mid_ch, mid_ch, dirate=8)
        
        self.rebnconv3d = REBNCONV(mid_ch * 2, mid_ch, dirate=4)
        self.rebnconv2d = REBNCONV(mid_ch * 2, mid_ch, dirate=2)
        self.rebnconv1d = REBNCONV(mid_ch * 2, out_ch, dirate=1)

    def forward(self,x):

        hx = x
        hxin = self.rebnconvin(hx)
        hx1 = self.rebnconv1(hxin)
        hx2 = self.rebnconv2(hx1)
        hx3 = self.rebnconv3(hx2)
        hx4 = self.rebnconv4(hx3)

        hx3d = self.rebnconv3d(torch.cat((hx4, hx3), 1))
        hx2d = self.rebnconv2d(torch.cat((hx3d, hx2), 1))
        hx1d = self.rebnconv1d(torch.cat((hx2d, hx1), 1))

        return hx1d + hxin


class U2NetArch(torch.nn.Module):

    def __init__(self, in_ch=3, out_ch=1):
        super().__init__()
        
        self.stage1 = RSU7(in_ch, 32, 64)
        self.pool12 = torch.nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.stage2 = RSU6(64, 32, 128)
        self.pool23 = torch.nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.stage3 = RSU5(128, 64, 256)
        self.pool34 = torch.nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.stage4 = RSU4(256, 128, 512)
        self.pool45 = torch.nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.stage5 = RSU4F(512, 256, 512)
        self.pool56 = torch.nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.stage6 = RSU4F(512, 256, 512)

        # decoder
        self.stage5d = RSU4F(1024, 256, 512)
        self.stage4d = RSU4(1024, 128, 256)
        self.stage3d = RSU5(512, 64, 128)
        self.stage2d = RSU6(256, 32, 64)
        self.stage1d = RSU7(128, 16, 64)

        self.side1 = torch.nn.Conv2d(64, out_ch, 3, padding=1)
        self.side2 = torch.nn.Conv2d(64, out_ch, 3, padding=1)
        self.side3 = torch.nn.Conv2d(128, out_ch, 3, padding=1)
        self.side4 = torch.nn.Conv2d(256, out_ch, 3, padding=1)
        self.side5 = torch.nn.Conv2d(512, out_ch, 3, padding=1)
        self.side6 = torch.nn.Conv2d(512, out_ch, 3, padding=1)

        self.outconv = torch.nn.Conv2d(6 * out_ch, out_ch, 1)
        
        # use for loss function
        self.bce_loss_ = torch.nn.BCELoss(reduction='mean')

    def forward(self,x):

        hx = x
        hx1 = self.stage1(hx)
        hx = self.pool12(hx1)
        hx2 = self.stage2(hx)
        hx = self.pool23(hx2)
        hx3 = self.stage3(hx)
        hx = self.pool34(hx3)
        hx4 = self.stage4(hx)
        hx = self.pool45(hx4)
        hx5 = self.stage5(hx)
        hx = self.pool56(hx5)
        hx6 = self.stage6(hx)
        hx6up = _upsample_like(hx6, hx5)

        hx5d = self.stage5d(torch.cat((hx6up, hx5), 1))
        hx5dup = _upsample_like(hx5d, hx4)
        hx4d = self.stage4d(torch.cat((hx5dup, hx4), 1))
        hx4dup = _upsample_like(hx4d, hx3)
        hx3d = self.stage3d(torch.cat((hx4dup, hx3), 1))
        hx3dup = _upsample_like(hx3d, hx2)
        hx2d = self.stage2d(torch.cat((hx3dup, hx2), 1))
        hx2dup = _upsample_like(hx2d, hx1)
        hx1d = self.stage1d(torch.cat((hx2dup, hx1), 1))

        d1 = self.side1(hx1d)
        d2 = self.side2(hx2d)
        d2 = _upsample_like(d2, d1)
        d3 = self.side3(hx3d)
        d3 = _upsample_like(d3, d1)
        d4 = self.side4(hx4d)
        d4 = _upsample_like(d4, d1)
        d5 = self.side5(hx5d)
        d5 = _upsample_like(d5, d1)
        d6 = self.side6(hx6)
        d6 = _upsample_like(d6, d1)

        d0 = self.outconv(torch.cat((d1, d2, d3, d4, d5, d6), 1))
        
        return torch.sigmoid(d0), torch.sigmoid(d1), torch.sigmoid(d2), torch.sigmoid(d3), torch.sigmoid(d4), torch.sigmoid(d5), torch.sigmoid(d6)
    
    
    def bce_loss(self, d0, d1, d2, d3, d4, d5, d6, labels_v):
        # labels_v has 3 channels, but only the first one contain data
        loss0 = self.bce_loss_(d0, labels_v[:, [0], :, :])
        loss1 = self.bce_loss_(d1, labels_v[:, [0], :, :])
        loss2 = self.bce_loss_(d2, labels_v[:, [0], :, :])
        loss3 = self.bce_loss_(d3, labels_v[:, [0], :, :])
        loss4 = self.bce_loss_(d4, labels_v[:, [0], :, :])
        loss5 = self.bce_loss_(d5, labels_v[:, [0], :, :])
        loss6 = self.bce_loss_(d6, labels_v[:, [0], :, :])
        loss = loss0 + loss1 + loss2 + loss3 + loss4 + loss5 + loss6
        return loss0, loss






class TrainDatasetLoader(torch.utils.data.Dataset):
    def __init__(self, dataset, transform=None):
        self.images, self.labels = self.__parse_dataset(dataset)
        self.transform = transform

    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, i):
        x = {'image': self.__get_image(i), 'label': self.__get_label(i)}
        if self.transform:
            x = self.transform(x)
        return x
    
    def __parse_dataset(self, dataset):
        images = []
        labels = []
        with open(dataset, 'r') as infh:
            for line in infh:
                d = line.replace('\n', '').split('\t')
                images.append(d[0])
                if (len(d) > 1):
                    if d[1] != '':
                        labels.append(d[1])
        
        # valid dataset
        if (len(labels) > 0) and (len(labels) != len(images)):
            raise ValueError('The number of labels do not equal to the number of images!')
        
        return images, labels
    
    
    def __get_image(self, i):
        image = skimage.io.imread(self.images[i])
        if (len(image.shape) == 2):
            image = image[:, :, np.newaxis]
        elif (len(image.shape) == 3):
            pass
        else:
            raise ValueError('Irregular images!')
        
        return image
       
        
    def __get_label(self, i):
        if (len(self.labels) == 0):
            label = np.zeros(image.shape)
        else:
            label = skimage.io.imread(self.labels[i])
            if (len(label.shape) == 2):
                label = label[:, :, np.newaxis]
       
        return label



class InferenceDatasetLoader(torch.utils.data.Dataset):
    
    def __init__(self, image_fpath, transform=None):
        self.images = image_fpath
        self.transform = transform
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, i):
        x = {
            'image': skimage.io.imread(self.images[i]) if isinstance(self.images[i], str) else self.images[i],
            'label': None
        }
        if self.transform is not None:
            x = self.transform(x)
        return x




class ToTensor():
    def __init__(self):
        pass
   
    def __call__(self, sample):
        return {'image': torch.from_numpy(sample['image'].transpose((2, 0, 1))).type(torch.FloatTensor),
                'label': torch.from_numpy(sample['label'].transpose((2, 0, 1))).type(torch.FloatTensor) if sample['label'] is not None else torch.from_numpy(np.array([np.nan]))}


class Normalize():
    def __init__(self):
        pass
    
    def __call__(self, sample):
        image = sample['image'] / np.max(sample['image'])
        image_std = np.zeros((image.shape[0], image.shape[1], 3))
        if image.shape[2] == 1:
            image_std[:, :, 0] = (image[: ,:, 0] - 0.485) / 0.229
            image_std[:, :, 1] = (image[: ,:, 0] - 0.485) / 0.229
            image_std[:, :, 2] = (image[: ,:, 0] - 0.485) / 0.229
        else:
            image_std[:, :, 0] = (image[: ,:, 0] - 0.485) / 0.229
            image_std[:, :, 1] = (image[: ,:, 1] - 0.456) / 0.224
            image_std[:, :, 2] = (image[: ,:, 2] - 0.406) / 0.225
        
        # label has 3 channels, but only use the first channel
        label_std = None
        if sample['label'] is not None:
            label_std = np.zeros(sample['label'].shape)
            label_std[:, :, 0] = sample['label'][:, :, 0] + 0.0
            if (np.max(label_std) > 1e-6):
                label_std = label_std / np.max(label_std)
        
        return {'image': image_std, 'label': label_std}




class Resize():
    def __init__(self, crop_size):
        self.crop_size = crop_size
    
    def __call__(self, sample):
        image = skimage.transform.resize(sample['image'], (self.crop_size, self.crop_size), mode='constant', order=0, preserve_range=True)
        label = skimage.transform.resize(sample['label'], (self.crop_size, self.crop_size), mode='constant', order=0, preserve_range=True) if sample['label'] is not None else None
        return {'image': image, 'label': label}


class RandomScaledCrop():
    def __init__(self, crop_size):
        self.crop_size = crop_size
    
    def __call__(self, sample):
        img = sample['image']
        bg  = sample['label']
        
        # random scales
        s = np.random.uniform(0.6, 1 / 0.6, 1)[0]
        img = skimage.transform.resize(img, (int(img.shape[0] * s), int(img.shape[1] * s)), mode='constant', order=0, preserve_range=True)
        bg  = skimage.transform.resize(bg, (int(bg.shape[0] * s), int(bg.shape[1] * s)), mode='constant', order=0, preserve_range=True)
        
        if np.random.rand() > 0.5:
            img = np.flipud(img)
            bg  = np.flipud(bg)

        if np.random.rand() > 0.5:
            img = np.fliplr(img)
            bg  = np.fliplr(bg)
        
        # if the input image is too small, use it withouth agumentation
        if (img.shape[0] < self.crop_size) or (img.shape[1] < self.crop_size):
            img_crop = skimage.transform.resize(img, (self.crop_size, self.crop_size), mode='constant', order=0, preserve_range=True)
            bg_crop = skimage.transform.resize(bg, (self.crop_size, self.crop_size), mode='constant', order=0, preserve_range=True)
            
        # if the input image is enough large, randomly crop a image from the original image
        else:
            # randomly select an angle for rotation
            theta = np.random.randint(0, 90, 1)[0]
            
            # calculate the width (height) for cropping
            crop_width = int(np.ceil(self.crop_size * np.cos(np.radians(theta)) + self.crop_size * np.sin(np.radians(theta))))
            
            # if images is too small to rotate, then do not rotate
            if (img.shape[0] < crop_width + 10) or (img.shape[1] < crop_width + 10):
                crop_width = self.crop_size
                theta = 0
            
            # calculate coordinates for cropping
            w = np.random.randint(0, img.shape[0] - crop_width)
            h = np.random.randint(0, img.shape[1] - crop_width)

            # crop image
            img_crop = img[w:(w + crop_width + 5), h:(h + crop_width + 5)]
            bg_crop  = bg[w:(w + crop_width + 5), h:(h + crop_width + 5)]
            
            # rotate the cropped images
            if theta != 0:
                rotate_center = (int(img_crop.shape[0] / 2), int(img_crop.shape[1] / 2))
                img_crop = skimage.transform.rotate(img_crop, theta, resize=False, center=None)
                bg_crop  = skimage.transform.rotate(bg_crop, theta, resize=False, center=None)
        
            img_crop = self.__center_crop(img_crop, self.crop_size, self.crop_size)
            bg_crop = self.__center_crop(bg_crop, self.crop_size, self.crop_size)
        
        return {'image': img_crop, 'label': bg_crop}

       

    def __center_crop(self, img, crop_width, crop_height):
        w = img.shape[0]
        h = img.shape[0]
        img_crop = img[int((w - crop_width) // 2):int((w + crop_width) // 2),
                       int((h - crop_height) // 2):int((h + crop_height) // 2)]
        return img_crop
        






class U2Net(ModuleTemplate):
    """U\ :sup:`2`-Net class for model training and object segmentation
    
    The :class:`U2Net <justdeepit.models.sod.U2Net>` class implements basic methods
    to train U\ :sup:`2`-Net and perform object segmentation.
    """


    def __init__(self, model_weight=None, workspace=None):
        # load model
        self.model = U2NetArch(3, 1)
        if model_weight is not None:
            self.model.load_state_dict(torch.load(model_weight, map_location='cpu'))
        
        # workspace
        self.tempd = None
        if workspace is None:
            self.tempd = tempfile.TemporaryDirectory()
            self.workspace = self.tempd.name
        else:
            self.workspace = workspace
        logger.info('Set workspace at `{}`.'.format(self.workspace))

        
        # transform
        self.transform_train = None
        self.transform_valid = None
    
    
    def __del__(self):
        try:
            if self.tempd is not None:
                self.tempd.cleanup()
        except:
            pass
         
    
    def __get_device(self, gpu=1):
        device = 'cpu'
        if gpu > 0:
            if torch.cuda.is_available():
                device = 'cuda:0'
        device = torch.device(device)
        return device
     
       
    def __normPRED(self, d):
        ma = torch.max(d)
        mi = torch.min(d)
        dn = (d-mi)/(ma-mi)
        return dn
    
 
    
    def __set_optimizer(self, optimizer):
        if optimizer is None:
            optimizer = torch.optim.Adam(self.model.parameters(), lr=0.01, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
        else:
            _prefix = []
            if 'torch.' not in optimizer:
                _prefix.append('torch')
            if 'optim.' not in optimizer:
                _prefix.append('optim')
            optimizer = '.'.join(_prefix) + '.' + optimizer
            optimizer_lf = optimizer.split('(', 1)[0]
            optimizer_rg = optimizer.split(',', 1)[1]
            optimizer = optimizer_lf + '(self.model.parameters(),' + optimizer_rg
            optimizer = eval(optimizer)
        return optimizer
    
    
    def __set_scheduler(self, optimizer, scheduler):
        if scheduler is None:
            return None
        elif scheduler.replace(' ', '') == '':
            return None
        else:
            _prefix = []
            if 'torch.' not in scheduler:
                _prefix.append('torch')
            if 'optim.' not in scheduler:
                _prefix.append('optim')
            if 'lr_scheduler.' not in scheduler:
                _prefix.append('lr_scheduler')
            scheduler = '.'.join(_prefix) + '.' + scheduler
            scheduler_lf = scheduler.split('(', 1)[0]
            scheduler_rg = scheduler.split(',', 1)[1]
            scheduler = scheduler_lf + '(optimizer, ' + scheduler_rg
            scheduler = eval(scheduler)
        return scheduler
    
    
    def train(self, train_data_fpath,
              optimizer=None, scheduler=None,
              batchsize=8, epoch=100,
              cpu=4, gpu=1,
              strategy='resizing', window_size=320):
        
        if not torch.cuda.is_available():
            gpu = 0
        if torch.cuda.device_count() < gpu:
            gpu = torch.cuda.device_count()
        
        device = self.__get_device(gpu)
        self.model = self.model.to(device)
        
        
        transform = None
        if strategy[0:4] == 'rand':
            transform = torchvision.transforms.Compose([
                RandomScaledCrop(window_size),
                Resize(288),
                Normalize(),
                ToTensor()
            ])
        elif strategy[0:5] == 'resiz':
            transform = torchvision.transforms.Compose([
                Resize(288),
                Normalize(),
                ToTensor()
            ])
        else:
            raise ValueError('Undefined strategy for transforming training images.')
        
        train_dataset = TrainDatasetLoader(train_data_fpath, transform=transform)
        train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batchsize, shuffle=True, num_workers=cpu)
        
        optimizer = self.__set_optimizer(optimizer)
        scheduler = self.__set_scheduler(optimizer, scheduler)
       
        for i in range(1, epoch + 1):
            self.model.train()
            
            running_loss = running_tar_loss = 0
            
            for data in train_dataloader:

                inputs, labels = data['image'], data['label']
                
                if False:
                    # to show image for debugging, comment out the ToTensor and Normalize in transforms
                    logger.debug(inputs.to('cpu').detach().numpy().copy())
                    logger.debug(labels.to('cpu').detach().numpy().copy())
                    logger.debug(np.max(labels.to('cpu').detach().numpy().copy()))
                    import matplotlib.pyplot as plt
                    fig = plt.figure()
                    ax1 = fig.add_subplot(1, 2, 1)
                    ax2 = fig.add_subplot(1, 2, 2)
                    ax1.imshow(inputs.to('cpu').detach().numpy().copy()[0,:,:,:].astype(np.uint8))
                    ax2.imshow(labels.to('cpu').detach().numpy().copy()[0,:,:,:].astype(np.uint8))
                    plt.show()
                
                inputs = inputs.to(device)
                labels = labels.to(device)
                inputs_v = torch.autograd.Variable(inputs, requires_grad=False)
                labels_v = torch.autograd.Variable(labels, requires_grad=False)
                optimizer.zero_grad()
                
                d0, d1, d2, d3, d4, d5, d6 = self.model(inputs_v)
                loss2, loss = self.model.bce_loss(d0, d1, d2, d3, d4, d5, d6, labels_v)
                loss.backward()
                optimizer.step()

                running_loss += loss.data.item()
                running_tar_loss += loss2.data.item()
                
                del d0, d1, d2, d3, d4, d5, d6, loss2, loss
            
            if scheduler is not None:
                scheduler.step()
            
            if i % 20 == 0 or i == epoch:
                logger.info('{:s} - [epoch: {:3d}/{:3d}] train loss: {:3f}, tar loss: {:3f} '.format(
                            datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'), i, epoch,
                            running_loss / len(train_dataset), running_tar_loss / len(train_dataset)))
            
            if self.workspace is not None:
                if i % 100 == 0:
                    torch.save(self.model.state_dict(),
                               os.path.join(self.workspace, 'u2net_weights_chk.{:010}.pth'.format(i)))
        
        self.model = self.model.to(torch.device('cpu'))



   

 
    def save(self, weight_fpath):
        torch.save(self.model.to('cpu').state_dict(), weight_fpath)

    

   
    
    def inference(self, image_path, strategy='resizing', batchsize=8, cpu=4, gpu=1,
                  window_size=320,
                  score_cutoff=0.5, image_opening_kernel=0, image_closing_kernel=0):
        """Object Segmentation
        
        Method :func:`inference` performs
        salient object detection through *resizing* or *sliding* approach.
        The *resizing* approach resizes the original image to 288 × 288 pixels for model training.
        On the other hand, *sliding* crops adjacent square areas from the original image
        for input to the model,
        and the outputs are merged into a single image.
        The size of the square areas can be specified by the user,
        but it is recommended to use the value specified by ``window_size`` for training.
        
        Args:
            image_path (str): A path to a image file, a list of image files,
                              or a path to a directory which contains multiple images.
            strategy (str): Strategy for model trainig. One of ``resizing`` or ``slide`` can be specified.
            output_type (str): Output format. 
            batchsize (int): Number of batch size. Note that a large number of
                              batch size may cause out of memory error.
            epoch (int): Number of epochs.
            cpu (int): Number of CPUs are used for prerpocessing training images.
            gpu (int): Number of GPUs are used for object segmentation.
            window_size (int): The width of images should be cropped from the original images
                                       when ``slide`` srategy was selected.
            score_cutoff (float): A threshold to cutoff U2Net outputs. Values higher than this threshold
                              are considering as detected objects.
            image_opening_kernel (int): The kernel size for image closing
                                        to remove the noise that detected as object.
            image_closing_kernel (int): The kernel size for image opening
                                        to remove the small bubbles in object.

        Returns:
            array: mask image or mask annotations.
        
        """
        # cpu/gpu
        if not torch.cuda.is_available():
            gpu = 0
        if torch.cuda.device_count() < gpu:
            gpu = torch.cuda.device_count()
        
        # strategy
        if strategy == 'auto':
            if (image.shape[0] < 640) or (image.shape[0] < 640):
                strategy = 'resizing'
            else:
                strategy = 'sliding'
        
        # query images
        images_fpath = []
        if isinstance(image_path, list):
            images_fpath = image_path
        elif os.path.isfile(image_path):
            images_fpath = [image_path]
        else:
            for f in glob.glob(os.path.join(image_path, '*')):
                if os.path.splitext(f)[1].lower() in ['.jpg', '.jpeg', '.png', '.tiff']:
                    images_fpath.append(f)
        
        # model
        device = self.__get_device(gpu)
        self.model = self.model.to(device)
        self.model.eval()
        
        # transform
        transform = torchvision.transforms.Compose([
                Resize(288),
                Normalize(),
                ToTensor()
        ])
        
        # detection
        pred_masks = []
        if strategy[0:5] == 'resiz':
            pred_masks = self.__inference_subset(images_fpath, transform, device, batchsize, cpu, score_cutoff)
        elif strategy[0:4] == 'slid':
            # perform prediction one-by-one
            for image_fpath in tqdm.tqdm(images_fpath, desc='Processed images: ', leave=True):
                tqdm_desc = ''
                image_blocks, blocks_info = self.__slice_image(image_fpath, window_size)
                pred_mask_blocks = self.__inference_subset(image_blocks, transform, device, batchsize, cpu, score_cutoff, 'Inferencing sliced blocks', False)
                pred_masks.append(self.__merge_sliced_images(pred_mask_blocks, blocks_info))
        else:
            raise NotImplementedError('Only resizing and sliding approaches can be used during detection.')
                
        
        self.model = self.model.to(torch.device('cpu'))
        
        if cpu == 1:
            imganns = []
            for i in tqdm.tqdm(range(len(pred_masks)), desc='Post-processed images: '):
                imganns.append(self._inference_post_process(
                    images_fpath[i], pred_masks[i], score_cutoff, image_opening_kernel, image_closing_kernel))
        else:
            imganns = joblib.Parallel(n_jobs=cpu)(
                joblib.delayed(self._inference_post_process)(
                    images_fpath[i], pred_masks[i], score_cutoff, image_opening_kernel, image_closing_kernel
                ) for i in tqdm.tqdm(range(len(pred_masks)), desc='Post-processed images: '))
        
        return justdeepit.utils.ImageAnnotations(imganns)
        
        
 
    def _inference_post_process(self, image_fpath, pred_mask, score_cutoff, image_opening_kernel, image_closing_kernel):
        input_image_shape = skimage.io.imread(image_fpath).shape
        pred_mask = skimage.transform.resize(pred_mask, (input_image_shape[0], input_image_shape[1]),
                                             mode='constant', order=0, preserve_range=True)
        pred_mask[(pred_mask > score_cutoff)] = 1
        
        # opening and closing preocess
        if image_opening_kernel > 0:
            pred_mask = skimage.morphology.opening(pred_mask,
                                skimage.morphology.square(image_opening_kernel))
        if image_closing_kernel > 0:
            pred_mask = skimage.morphology.closing(pred_mask,
                                skimage.morphology.square(image_closing_kernel))
        
        return justdeepit.utils.ImageAnnotation(image_fpath, pred_mask, 'array')
        


    
    
    def __slice_image(self, image_fpath, window_size):
        input_image = skimage.io.imread(image_fpath)
        
        images = []
        blocks_info = {'image_shape': input_image.shape,
                       'slide_window_size': window_size,
                       'xy': []}
        
        # generate small blocks from the original image
        step_size = window_size - math.ceil(window_size / 5)
        n_w_steps = math.ceil(input_image.shape[0] / step_size)
        n_h_steps = math.ceil(input_image.shape[1] / step_size)
        for w in range(n_w_steps):
            for h in range(n_h_steps):
                w_from = int(step_size * w)
                w_to   = w_from + window_size
                h_from = int(step_size * h)
                h_to   = h_from + window_size
                if w_to > input_image.shape[0]:
                    w_from = input_image.shape[0] - window_size
                    w_to   = input_image.shape[0]
                if h_to > input_image.shape[1]:
                    h_from = input_image.shape[1] - window_size
                    h_to   = input_image.shape[1]
               
                images.append(input_image[w_from:w_to, h_from:h_to].copy())
                blocks_info['xy'].append([w_from, w_to, h_from, h_to])
        
        return images, blocks_info
        
    
    
    def __merge_sliced_images(self, pred_mask_blocks, blocks_info):
        pred_mask = np.zeros((blocks_info['image_shape'][0], blocks_info['image_shape'][1]))
        for pred_grid, xy in zip(pred_mask_blocks, blocks_info['xy']):
            pred_grid = skimage.transform.resize(pred_grid,
                         (blocks_info['slide_window_size'], blocks_info['slide_window_size']),
                         mode='constant', order=0, preserve_range=True)
            pred_mask[xy[0]:xy[1], xy[2]:xy[3]] = np.maximum(pred_grid, pred_mask[xy[0]:xy[1], xy[2]:xy[3]])
        
        return  pred_mask
        
       
    
    
    def __inference_subset(self, images_fpath, transform, device, batchsize, cpu, score_cutoff,
                           tqdm_desc='Processed batches: ', tqdm_leave=True):
        valid_image = InferenceDatasetLoader(images_fpath, transform)
        valid_dataloader = torch.utils.data.DataLoader(valid_image, batch_size=batchsize, num_workers=cpu)
        
        pred_masks = []
        for data in tqdm.tqdm(valid_dataloader, desc=tqdm_desc, leave=tqdm_leave):
            inputs, labels = data['image'], data['label']
            inputs = inputs.to(device)
            inputs_v = torch.autograd.Variable(inputs, requires_grad=False)
            d1,d2,d3,d4,d5,d6,d7 = self.model(inputs_v)
            pred = self.__normPRED(d1[:, 0, :, :])
            pred = pred.cpu().data.numpy()
            pred_masks.extend(pred)

        return pred_masks

    
   
    
