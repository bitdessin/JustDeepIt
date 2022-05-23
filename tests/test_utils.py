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
        
        if not os.path.exists(self.output_dpath):
            os.makedirs(self.output_dpath, exist_ok=True)
            shutil.copytree('sample_datasets/od/COCO',
                            os.path.join(self.output_dpath, 'data'))
        
        self.coco = [os.path.join(self.output_dpath, 'data', 'images'),
                     os.path.join(self.output_dpath, 'data', 'annotations', 'instances_default.json')]
    
    
    
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
        




class TestSegmentation(unittest.TestCase):
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.output_dpath = 'outputs/utils/os_coco'
        
        if not os.path.exists(self.output_dpath):
            os.makedirs(self.output_dpath, exist_ok=True)
            shutil.copytree('sample_datasets/os/COCO',
                            os.path.join(self.output_dpath, 'data'))
        
        self.coco = [os.path.join(self.output_dpath, 'data', 'images'),
                     os.path.join(self.output_dpath, 'data', 'annotations', 'instances_default.json')]
    
    
    
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
        




class TestImageAnnotations(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.output_dpath = 'outputs/utils/imanns'
        
        if not os.path.exists(self.output_dpath):
            os.makedirs(self.output_dpath, exist_ok=True)
            shutil.copytree('sample_datasets/os/COCO',
                            os.path.join(self.output_dpath, 'data'))
        
        self.coco = [os.path.join(self.output_dpath, 'data', 'images'),
                     os.path.join(self.output_dpath, 'data', 'annotations', 'instances_default.json')]
    
    
    def test_coco(self):
        
        imanns = ImageAnnotations()
        for image_fpath in glob.glob(os.path.join(self.coco[0], '*.png')):
            imann = ImageAnnotation(image_fpath, self.coco[1])
            imanns.append(imann)
        
        imanns.format('coco', os.path.join(self.output_dpath, 'coco.json'))
    



class TestImageOrientation(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        super(TestImageOrientation, self).__init__(*args, **kwargs)

        self.images = {
        #    '1': [['inputs/orient/ori_1_01.jpg', 'inputs/orient/ori_1_01.json'],
        #          ['inputs/orient/ori_1_02.jpg', 'inputs/orient/ori_1_02.json']],
        #    '3': [['inputs/orient/ori_3_01.jpg', 'inputs/orient/ori_3_01.json'],
        #          ['inputs/orient/ori_3_02.jpg', 'inputs/orient/ori_3_02.json']],
        #    '6': [['inputs/orient/ori_6_01.jpg', 'inputs/orient/ori_6_01.json'],
        #          ['inputs/orient/ori_6_02.jpg', 'inputs/orient/ori_6_02.json']],
        #    '8': [['inputs/orient/ori_8_01.jpg', 'inputs/orient/ori_8_01.json'],
        #          ['inputs/orient/ori_8_02.jpg', 'inputs/orient/ori_8_02.json']],
        }
        self.output_dpath = 'outputs/utils/imageorientation'


    def test_image_orientation(self):
        pass
        #for ori, images_fpath in self.images.items():
        #    for fpath in images_fpath:
        #        imann = ImageAnnotation(fpath[0], fpath[1], 'vott')
        #        imann.draw('contour', os.path.join(self.output_dpath,
        #                    os.path.splitext(os.path.basename(fpath[0]))[0] + '.contour.png'))






if __name__ == '__main__':
    unittest.main()


