import os
import json
import shutil
import glob
import pprint
import unittest
import skimage
import skimage.io
from justdeepit.utils import ImageAnnotation, ImageAnnotations







class TestBbox(unittest.TestCase):
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.output_dpath = 'outputs/utils/od_coco'
        
        os.makedirs(self.output_dpath, exist_ok=True)
        shutil.copytree('sample_datasets/od',
                        os.path.join(self.output_dpath, 'data'), dirs_exist_ok=True)
        
        self.coco = [os.path.join(self.output_dpath, 'data', 'images'),
                     os.path.join(self.output_dpath, 'data', 'annotations', 'COCO', 'instances_default.json')]
        self.voc = [os.path.join(self.output_dpath, 'data', 'images'),
                    os.path.join(self.output_dpath, 'data', 'annotations', 'PascalVOC')]
    
    
    def test_coco2x(self):
        pref = os.path.join(self.output_dpath, 'test_coco2x_')
        
        img_fpath = os.path.join(self.coco[0], 'e1.png')
        ann_fpath = self.coco[1]
        imgann = ImageAnnotation(img_fpath, ann_fpath)
        # pprint.pprint(imgann.regions)
        
        
        # coco to annotation file
        imgann.format('base', pref + 'base.json')
        imgann.format('coco', pref + 'coco.json')
        imgann.format('voc',  pref + 'voc.xml')
        
        # coco to image
        imgann.draw('mask',    pref + 'mask.png')
        imgann.draw('mask',    pref + 'maskalpha05.png', alpha=0.5)
        imgann.draw('masked',  pref + 'masked.png')
        imgann.draw('bbox',    pref + 'bbox.png')
        
        # coco to image with label
        imgann.draw('mask',    pref + 'mask.v2.png', label=True, score=True)
        imgann.draw('mask',    pref + 'maskalpha05.v2.png', label=True, score=True, alpha=0.5)
        imgann.draw('masked',  pref + 'masked.v2.png', label=True, score=True)
        imgann.draw('bbox',    pref + 'bbox.v2.png', label=True, score=True)
        
    
    def  test_voc2x(self):
        pref = os.path.join(self.output_dpath, 'test_voc2x_')
        
        anns = ImageAnnotations()
        for i, img_fpath in enumerate(glob.glob(os.path.join(self.voc[0], '*'))):
            ann_fpath = os.path.join(self.voc[1], os.path.splitext(os.path.basename(img_fpath))[0] + '.xml')
            ann = ImageAnnotation(img_fpath, ann_fpath, 'voc')
            
            if i == 0:
                ann.draw('mask', pref + 'mask.png')
                ann.draw('mask',    pref + 'maskalpha05.png', alpha=0.5)
                ann.draw('masked',  pref + 'masked.png')
                ann.draw('bbox',    pref + 'bbox.png')
            
            anns.append(ann)
        anns.format('coco', pref + 'coco.json')


class TestSegmentation(unittest.TestCase):
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.output_dpath = 'outputs/utils/os_coco'
        
        os.makedirs(self.output_dpath, exist_ok=True)
        shutil.copytree('sample_datasets/os',
                        os.path.join(self.output_dpath, 'data'), dirs_exist_ok=True)
        
        self.coco = [os.path.join(self.output_dpath, 'data', 'images'),
                     os.path.join(self.output_dpath, 'data', 'annotations', 'COCO', 'instances_default.json')]
    
    
    
    def test_coco2x(self):
        pref = os.path.join(self.output_dpath, 'test_coco2x_')
        
        img_fpath = os.path.join(self.coco[0], 'e1.png')
        ann_fpath = self.coco[1]
        imgann = ImageAnnotation(img_fpath, ann_fpath)
        # pprint.pprint(imgann.regions)
        
        
        # coco to annotation file
        imgann.format('base', pref + 'base.json')
        imgann.format('coco', pref + 'coco.json')
        imgann.format('voc',  pref + 'voc.xml')
        
        # coco to image
        imgann.draw('mask',    pref + 'mask.png')
        imgann.draw('mask',    pref + 'maskalpha05.png', alpha=0.5)
        imgann.draw('masked',  pref + 'masked.png')
        imgann.draw('bbox',    pref + 'bbox.png')
        imgann.draw('contour', pref + 'contour.png')
        
        # coco to image with label
        imgann.draw('mask',    pref + 'mask.v2.png', label=True, score=True)
        imgann.draw('mask',    pref + 'maskalpha05.v2.png', label=True, score=True, alpha=0.5)
        imgann.draw('masked',  pref + 'masked.v2.png', label=True, score=True)
        imgann.draw('bbox',    pref + 'bbox.v2.png', label=True, score=True)
        imgann.draw('contour', pref + 'contour.v2.png', label=True, score=True)
    
    
    
    def test_rgbmask2coco(self):
        pref = os.path.join(self.output_dpath, 'test_rgbmask2coco_')
        imganns = []
        for mask_i in ['e1.png', 'e2.png']:
            
            img_fpath = os.path.join('sample_datasets', 'mask', 'images', mask_i)
            ann_fpath = os.path.join('sample_datasets', 'mask', 'annotations', mask_i)
            
            if mask_i == 'e1.png':
                imgann = ImageAnnotation(img_fpath, ann_fpath)
            else:
                rgb2class = {'255,0,0': 'flower', '0,0,255': 'leaf', '0,255,0': 'tree'}
                imgann = ImageAnnotation(img_fpath, ann_fpath, rgb2class=rgb2class)
            imgann.draw('mask',    pref + mask_i + '_mask.png', label=True, score=True)
            imgann.draw('mask',    pref + mask_i + '_maskalpha05.png', label=True, score=True, alpha=0.5)
            imgann.draw('masked',  pref + mask_i + '_masked.png', label=True, score=True)
            imgann.draw('bbox',    pref + mask_i + '_bbox.png', label=True, score=True)
            imgann.draw('contour', pref + mask_i + '_contour.png', label=True, score=True)
            imgann.draw('bbox+contour', pref + mask_i + '_bbboxcontour.png', label=True, score=True)
        
        anns = ImageAnnotations(imganns)
        anns.format('coco', pref + 'coco.json')




class TestImageAnnotations(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.output_dpath = 'outputs/utils/imanns'
        
        os.makedirs(self.output_dpath, exist_ok=True)
        shutil.copytree('sample_datasets/os',
                        os.path.join(self.output_dpath, 'data'), dirs_exist_ok=True)
        
        self.coco = [os.path.join(self.output_dpath, 'data', 'images'),
                     os.path.join(self.output_dpath, 'data', 'annotations', 'COCO', 'instances_default.json')]
    
    
    def test_coco(self):
        
        imanns = ImageAnnotations()
        for image_fpath in glob.glob(os.path.join(self.coco[0], '*.png')):
            imann = ImageAnnotation(image_fpath, self.coco[1])
            imanns.append(imann)
        
        imanns.format('coco', os.path.join(self.output_dpath, 'coco.json'))
    






if __name__ == '__main__':
    unittest.main()


