import os
import datetime
import random
import pathlib
import shutil
import glob
import copy
import tqdm
import logging
import multiprocessing
import joblib
import traceback
import numpy as np
import pandas as pd
import PIL
import PIL.ExifTags
import cv2
import skimage.io
import skimage.color
import skimage.exposure
import torch
import tkinter
import tkinter.filedialog
import ttkbootstrap as ttk
import agrolens


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)       
cv2.setNumThreads(1)




class SOD(agrolens.app.AppBase):
    """Application for salient object detection
    
    :code:`SOD` class offers fundamental methods to perform salient object detection (SOD) 
    with U2Net. In contrast to :code:`U2Net` class has simple :code:`train` and :code:`inference`
    methods to perform SOD, :code:`SOD` class offers some useful methods for sorting training
    images, aligning images by position for time-series images, sumarizing objects information
    such as area, colors, etc.
    """
    def __init__(self, workspace=None):
        super().__init__('SOD', workspace)


    
    def sort_train_images(self, image_dpath=None, annotation_fpath=None, annotation_format='mask', image_suffix=None, mask_suffix=None):
        """Sort training images

        This function checks images in 'inputs/train_images' directory of workspace,
        summarises images with the given suffixes.
        Training images should have both image and mask image, and both are the same
        prefix but different suffix.
        This function checks all images has mask or not, if not, discard the images from training.

        Args:
            image_suffix (str): Suffix of image data.
            mask_suffix (str): Suffix of mask image.
            image_path (str): A path to a directory which contains training images.
            annotation_path (str): A path to a file or directory which contains image annotations.
            annotation_format (str): A string to specify annotation format.
        Returs:
            (str): A status of running result.
        """
        
        run_status = True
        self.set_jobstatus('TRAIN', 'SORT', 'BEGIN')
        
        try:
            # generate mask images from annotation files if mask images are not given.
            if annotation_format == 'coco':
                ws_image_dpath = os.path.join(self.workspace, 'train_dataset')
                os.makedirs(ws_image_dpath, exist_ok=True)
                for image_fpath in sorted(glob.glob(os.path.join(image_dpath, '**'), recursive=True)):
                    image_fname, image_fext = os.path.splitext(image_fpath)
                    if image_fext.lower() in self.image_ext:
                        imgann = agrolens.utils.ImageAnnotation(image_fpath, annotation_fpath, annotation_format)
                        shutil.copy2(image_fpath, os.path.join(ws_image_dpath, os.path.basename(image_fname) + '.img' + image_fext))
                        imgann.draw('bimask', os.path.join(ws_image_dpath, os.path.basename(image_fname) + '.mask.png'))
                image_dpath = ws_image_dpath
                image_suffix = '.img' + image_fext
                mask_suffix = '.mask.png'
                        
            elif annotation_format == 'mask':
                pass
            else:
                raise ValueError('Unsupportted annotation for training SOD model.')
            
            
            n_images = 0
            with open(os.path.join(self.workspace, 'train_dataset', 'train_images.txt'), 'w') as outfh:
                for image_fpath in sorted(glob.glob(os.path.join(image_dpath, '*' + image_suffix))):
                    mask_fpath = image_fpath[:-len(image_suffix)] + mask_suffix
                    if os.path.exists(image_fpath) and os.path.exists(mask_fpath):
                        outfh.write('{}\t{}\n'.format(image_fpath, mask_fpath))
                        n_images += 1
            logger.info('There are {} images are valid for model training.'.format(n_images))
    

        except BaseException as e:
            traceback.print_exc()
            self.set_jobstatus('TRAIN', 'SORT', 'ERROR', str(e))
            run_status = False
        else:
            self.set_jobstatus('TRAIN', 'SORT', 'COMPLETED')

        return run_status
    
    
    
    def train_model(self, weight, batchsize, epoch, cpu, gpu, strategy, window_size):
        """Train model
        
        Train model with the given paramters. The images for model training are
        obtained from 'inputs/train_images.txt' directory of workspace.
        The check point is stored by every 100 epochs automatically.

        Args:
            weight (str): A path to save the trained weight.
            batchsize (int): Batch size for model trainig.
            cpu (int): Number of CPUs for image preprocessings.
            strategy (int): A strategy to treat training images.
                            `resize` or `randomcrop` can be specified.
            window_size (int): Width for random crop strategy.

        Returs:
            (str): A status of running result.
        """

        run_status = True
        
        self.set_jobstatus('TRAIN', 'TRAIN', 'BEGIN')
        try:
            
            tmp_dpath = os.path.join(self.workspace, 'tmp', 'train')
            logger.info('The check points will be saved as {} every 100 epochs.'.format(tmp_dpath))
            
            if os.path.exists(weight):
                model = agrolens.models.SOD(model_weight=weight, workspace=tmp_dpath)
            else:
                model = agrolens.models.SOD(workspace=tmp_dpath)
            model.train(os.path.join(self.workspace, 'train_dataset', 'train_images.txt'),
                        batch_size=batchsize, epoch=epoch, cpu=cpu, gpu=gpu, strategy=strategy, window_size=window_size)
            model.save(weight)

        except BaseException as e:
            traceback.print_exc()
            self.set_jobstatus('TRAIN', 'TRAIN', 'ERROR', str(e))
            run_status = False
        else:
            self.set_jobstatus('TRAIN', 'TRAIN', 'COMPLETED', 'Params: batchsize {}; epoch {}; strategy {}; window_size {}; cpu {}; gpu {}'.format(batchsize, epoch, strategy, window_size, cpu, gpu))
        
        return run_status
    
    
    
    
    
    def seek_images(self):
        """List up images for analysis

        This function loads query_images.txt file and list up all images.
        """
        self.images = []
        with open(os.path.join(self.workspace, 'query_dataset', 'query_images.txt'), 'r') as infh:
            for _image in infh:
                self.images.append(_image.replace('\n', '').split('\t'))
    
    
    
    def __get_bfmatches_spots(self, img1, img2):
        if isinstance(img1, str):
            img1 = cv2.imread(img1, cv2.IMREAD_GRAYSCALE)
        if isinstance(img2, str):
            img2 = cv2.imread(img2, cv2.IMREAD_GRAYSCALE)
        
        try:
            # if two images are too different, it might give error in bf.knnMatch
            akaze = cv2.AKAZE_create()
            kp1, des1 = akaze.detectAndCompute(img1, None)
            kp2, des2 = akaze.detectAndCompute(img2, None)
        
            bf = cv2.BFMatcher()
            matches = bf.knnMatch(des1, des2, k=2)
            
            good_matches = []
            for m, n in matches:
                if m.distance < 0.6 * n.distance:
                    good_matches.append([m])
            
        except:
            kp1, kp2, des1, des2, good_matches = None, None, None, None, []
        
        return kp1, kp2, des1, des2, good_matches
    
    

    #def __check_alignable_images(self, images, cpu):
    #    image_files = None
    #    repeat_align = True
    #    
    #    def __get_n_bfmatches_spots(img1, img2):
    #        img1 = cv2.imread(img1, cv2.IMREAD_GRAYSCALE)
    #        img2 = cv2.imread(img2, cv2.IMREAD_GRAYSCALE)
    #        good_matches = []
    #        try:
    #            akaze = cv2.AKAZE_create()
    #            kp1, des1 = akaze.detectAndCompute(img1, None)
    #            kp2, des2 = akaze.detectAndCompute(img2, None)            
    #            bf = cv2.BFMatcher()
    #            matches = bf.knnMatch(des1, des2, k=2)
    #            good_matches = []
    #            for m, n in matches:
    #                if m.distance < 0.6 * n.distance:
    #                    good_matches.append([m])
    #        except:
    #            good_matches = []    
    #        
    #        return len(good_matches)
    #
    #    # in some cases, it might be good to repeated the following steps to remove the unnatural images
    #    # but considering the time-series images, run once should be good.
    #    while repeat_align:
    #        repeat_align = False
    #        
    #        image_files = []
    #        n_bf_matched_spots = []
    #        
    #        logger.info('Checking images are valid or not for image alignment using {} CPUs.'.format(cpu))
    #        n_bf_matched_spots = joblib.Parallel(n_jobs=cpu)(
    #            joblib.delayed(__get_n_bfmatches_spots)(images[i], images[i + 1]) for i in range(len(images) - 1)
    #        )
    #        
    #        n_bf_matched_spots = np.log10(np.array(n_bf_matched_spots))
    #        n_bf_matched_spots_mu = np.mean(n_bf_matched_spots[n_bf_matched_spots > 0])
    #        n_bf_matched_spots_sd = np.std(n_bf_matched_spots[n_bf_matched_spots > 0])
    #        image_files.append(images[0])
    #        for i in range(len(n_bf_matched_spots)):
    #            if n_bf_matched_spots[i] > (n_bf_matched_spots_mu - n_bf_matched_spots_sd * 3):
    #                image_files.append(images[i + 1])
    #            else:
    #                logger.info('Image {} is not suitable for alignment since Brute-Force Matcher in OpenCV only gave a few matched features between image {} and the previous image {}.'.format(images[i + 1], images[i + 1], images[i]))
    #                # repeat_align = True
    #        images = image_files
    #    
    #    return image_files
        
    
    def __align_image(self, img1_fpath, img2_fpath):
    
        img1 = cv2.imread(img1_fpath, cv2.IMREAD_GRAYSCALE)
        img2 = cv2.imread(img2_fpath, cv2.IMREAD_GRAYSCALE)
        
        kp1, kp2, des1, des2, good_matches = self.__get_bfmatches_spots(img1, img2)
        
        try:
            ref_matched_kpts = np.float32([kp1[m[0].queryIdx].pt for m in good_matches])
            sensed_matched_kpts = np.float32([kp2[m[0].trainIdx].pt for m in good_matches])
        
            H, status = cv2.findHomography(sensed_matched_kpts, ref_matched_kpts, cv2.RANSAC, 5.0)
        
            img = cv2.imread(img2_fpath)
            aligned_img = cv2.warpPerspective(img, H, (img.shape[1], img.shape[0]))
        except:
            logger.warning('An error occurred while align {} to the reference {}.'.format(img1_fpath, img2_fpath))
            traceback.print_exc()
            aligned_img = None
        
        return aligned_img
    
     

    def align_images(self, image_files):
        """Align images
    
        The method used to align images that saved in `self.images` attribute.
        Since not all images are valid for alignment, this function internally
        calls other functions to check the quality and then perform alignemnt
        for th valid images.
        """
    
        run_status = True
        aligned_image_files = []
        aligned_image_files_status = []
        
        try:
            output_dpath = os.path.join(self.workspace, 'aligned_images')
            if not os.path.exists(output_dpath):
                os.makedirs(output_dpath)
            
            # copy the first image
            aligned_image_files.append(os.path.join(output_dpath, os.path.basename(image_files[0])))
            aligned_image_files_status.append(os.path.join(output_dpath, os.path.basename(image_files[0])))
            shutil.copy(image_files[0], aligned_image_files[-1])
            
            # align the next image according to the current image
            for i in tqdm.tqdm(range(0, len(image_files)), desc='aligning images'):
                if i == 0:
                    continue
                aligned_image = self.__align_image(aligned_image_files[-1], image_files[i])
                if aligned_image is not None:
                    aligned_image_files.append(os.path.join(output_dpath, os.path.basename(image_files[i])))
                    aligned_image_files_status.append(os.path.join(output_dpath, os.path.basename(image_files[i])))
                    cv2.imwrite(aligned_image_files[-1], aligned_image)
                else:
                    aligned_image_files_status.append(None)
                    logger.warning('OpenCV cannot handle this image {} for position alignment. The image may be too much different from the previous image.'.format(image_files[i]))
        
        except BaseException as e:
            traceback.print_exc()
            self.set_jobstatus('EVAL', 'ALIGN', 'ERROR', str(e))
            run_status = False
        else:
            self.set_jobstatus('EVAL', 'ALIGN', 'COMPLETED')
            pass
    
        return aligned_image_files_status
    
    
       

    def sort_query_images(self, image_dpath, align_images=False):
        """Sort images for analysis

        This function checks images in :file:`image_dpath`.
        lists up all .png, .jpg images, and save path for these images into 
        'inputs/query_images.txt' file. Images listed in this file are used for 
        object segmentation and other analysis.
        
        image_dpath (str): A path to an folder which contains test images.
        align_images (bool): Align images by position if ``True`` is set.
        
        Returns:
            (str): A status of running result.
        """

        run_status = True
        
        self.set_jobstatus('EVAL', 'SORT', 'BEGIN')
        
        def __get_exif_datetime(fpath):
            im = PIL.Image.open(fpath)
            dt = ''
            exif = im._getexif()
            if exif is not None:
                for tag_id, value in exif.items():
                    tag = PIL.ExifTags.TAGS.get(tag_id, tag_id)
                    if tag == 'DateTimeOriginal':
                        dt = value.replace(':', '').replace(' ', '')
                        break
            im.close()
            return dt

        try:
            # seek all the input images
            image_files = []
            for f in sorted(glob.glob(os.path.join(image_dpath, '**'), recursive=True)):
                if os.path.splitext(f)[1].lower() in self.image_ext:
                    image_files.append([f, __get_exif_datetime(f)])
            
            # alignment image first if required
            if align_images:
                aligned_image_files = []
                _ = self.align_images([_[0] for _ in image_files])
                for i in range(len(image_files)):
                    if _[i] is not None:
                        aligned_image_files.append(image_files[i])
                        aligned_image_files[-1][0] = _[i]
                image_files = aligned_image_files
            
            with open(os.path.join(self.workspace, 'query_dataset', 'query_images.txt'), 'w') as outfh:
                for image_file in image_files:
                    outfh.write('{}\n'.format('\t'.join(image_file)))
        
        
        except BaseException as e:
            traceback.print_exc()
            self.set_jobstatus('EVAL', 'SORT', 'ERROR', str(e))
            run_status = False
        else:
            self.set_jobstatus('EVAL', 'SORT', 'COMPLETED', 'Params: align_images {}'.format(str(align_images)))
        
        return run_status



   
    
    def detect_objects(self, weight, batchsize,
                       strategy, u_cutoff, image_opening, image_closing, window_size,
                       cpu, gpu):
        """Object segmentation

        Object segmentaion.


        Args:
            weight (str): A path to save the trained weight.
            batchsize (int): Batch size for model trainig.
            strategy (str): strategy.
            u_cutoff (float): Cut off.
            image_opening (int): Kernel size.
            image_closing (int): Kernel size.
            window_size (int): slide size
            aligned_images (bool): Use aligned images or not.
            cpu (int): Number of CPUs.

        Returs:
            (str): A status of running result.

        """

        run_status = True
        
        self.set_jobstatus('EVAL', 'DETECT', 'BEGIN')
        self.seek_images()
        try:
            model = agrolens.models.SOD(model_weight=weight)
            for image in self.images:
                image_fpath = image[0]
                logger.info('Processing {} ...'.format(image_fpath))
                image_name = os.path.splitext(os.path.basename(image_fpath))[0]
                
                mask_obj = model.inference(image_fpath, strategy, batchsize, cpu, gpu,
                                           window_size, u_cutoff, image_opening, image_closing)
                
                # save mask as images
                mask_obj.draw('mask', os.path.join(self.workspace, 'detection_results', image_name + '.mask.png'))
                mask_obj.draw('masked', os.path.join(self.workspace, 'detection_results', image_name + '.crop.png'))
                mask_obj.draw('contour', os.path.join(self.workspace, 'detection_results', image_name + '.outline.png'))
                # save mask as COCO
                #mask_obj.save('coco', os.path.join(self.workspace, 'detection_results', image_name + '.xml'))
                
                
        except BaseException as e:
            traceback.print_exc()
            self.set_jobstatus('EVAL', 'DETECT', 'ERROR', str(e))
            run_status = False
        else:
            self.set_jobstatus('EVAL', 'DETECT', 'COMPLETED', 'Params: batchsize {}; ucutoff {}; strategy {}; window_size {}; image_opening {}; image_closing {}; cpu {}; gpu {}'.format(batchsize, u_cutoff, strategy, window_size, image_opening, image_closing, cpu, gpu))
    
        return run_status



       
  
    def summarize_objects(self, cpu, is_series=False, image_opening_kernel=0, image_closing_kernel=0):
        
        run_status = True
        self.set_jobstatus('EVAL', 'SUMMARIZE', 'BEGIN')
       
        self.seek_images()
        
        def RGB(regionmask, intensity): 
            r = np.median(intensity[:, :, 0][regionmask[:, :, 0]])
            g = np.median(intensity[:, :, 1][regionmask[:, :, 1]])
            b = np.median(intensity[:, :, 2][regionmask[:, :, 2]])
            return (r, g, b)
        
        def HSV(regionmask, intensity): 
            intensity = skimage.color.rgb2hsv(intensity)
            h = np.median(intensity[:, :, 0][regionmask[:, :, 0]])
            s = np.median(intensity[:, :, 1][regionmask[:, :, 1]])
            v = np.median(intensity[:, :, 2][regionmask[:, :, 2]])
            return (h, s, v)
        
        def Lab(regionmask, intensity): 
            intensity = skimage.color.rgb2lab(intensity)
            l = np.median(intensity[:, :, 0][regionmask[:, :, 0]])
            a = np.median(intensity[:, :, 1][regionmask[:, :, 1]])
            b = np.median(intensity[:, :, 2][regionmask[:, :, 2]])
            return (l, a, b)
        
        
        def __summarize_objects(image_fpath, mask_fpath, output_fpath, image_opening_kernel=0, image_closing_kernel=0, master_mask=None):
            image = skimage.io.imread(image_fpath)
            mask = skimage.io.imread(mask_fpath)
            labeled_mask = None
            if master_mask is None:
                mask4label = mask[:, :, 0].copy()
                mask4label[mask4label > 0] = 1
                
                # remove small bright spots and connect small dark cracks
                if image_opening_kernel > 0:
                    mask4label = skimage.morphology.opening(mask4label,
                                        skimage.morphology.square(image_opening_kernel))
                mask4label[mask4label > 0] = 1
                
                #  remove small dark spots and connect small bright cracks
                if image_closing_kernel > 0:
                    mask4label = skimage.morphology.closing(mask4label,
                                        skimage.morphology.square(image_closing_kernel))
                mask4label[mask4label > 0] = 1

                #if erosion_size > 0:
                #    # shrinks bright regions and enlarges dark regions
                #    mask4label = skimage.morphology.erosion(mask4label, skimage.morphology.square(erosion_size))
                
                #if dilation_size > 0:
                #    # enlarges bright regions and shrinks dark regions
                #    mask4label = skimage.morphology.dilation(mask4label, skimage.morphology.square(dilation_size))
                
                labeled_mask = skimage.measure.label(mask4label, background=0)
            else:
                labeled_mask = np.load(master_mask)['label']
            
            # object features calculations
            labeled_mask_ = np.zeros(image.shape)
            for ch in range(image.shape[2]):
                labeled_mask_[:, :, ch][(mask[:, :, ch] > 0)] = labeled_mask[(mask[:, :, ch] > 0)].copy()
            labeled_mask_ = labeled_mask_.astype(int)
            labeled_mask = labeled_mask_[:, :, 0].copy()
            
            image_ = image.astype(int)
            obj_textures = skimage.measure.regionprops_table(
                        labeled_mask_, image_,
                        properties=('label', 'centroid', 'area', 'convex_area',
                                    'major_axis_length', 'minor_axis_length'),
                        separator=';', cache=True,
                        extra_properties=(RGB, HSV, Lab))
            
            # create a labeled mask image
            for i in range(1, np.max(labeled_mask_)):
                random.seed(i)
                obj_area = (labeled_mask_ == i)
                mask[:, :, 0][obj_area[:, :, 0]] = random.randint(10, 255)
                mask[:, :, 1][obj_area[:, :, 1]] = random.randint(10, 255)
                mask[:, :, 2][obj_area[:, :, 2]] = random.randint(10, 255)
            
            for i, props in enumerate(skimage.measure.regionprops(labeled_mask)):
                y0, x0 = props.centroid
                cv2.putText(mask, str(i + 1), (int(x0), int(y0)), cv2.FONT_HERSHEY_SIMPLEX,
                            3, (255,255,255), 2,  cv2.LINE_AA)
            
            # save the labeled mask as numpy matrix and image
            labeled_mask = labeled_mask.astype(np.uint32)
            np.savez_compressed(output_fpath + '.npz', label=labeled_mask)
            skimage.io.imsave(output_fpath + '.png', mask, check_contrast=False)
            pd.DataFrame(obj_textures).to_csv(output_fpath + '.txt', header=True, index=False, sep='\t')
        
        try:
            logger.info('Finding objects and calculate the summary data using {} CPUs.'.format(cpu))
            
            valid_images = []
            for _img in self.images:
                    valid_images.append([
                        os.path.join(self.workspace, 'detection_results',
                                     os.path.splitext(os.path.basename(_img[0]))[0] + '.crop.png'),
                        os.path.join(self.workspace, 'detection_results',
                                     os.path.splitext(os.path.basename(_img[0]))[0] + '.mask.png'),
                        os.path.join(self.workspace, 'detection_results',
                                     os.path.splitext(os.path.basename(_img[0]))[0] + '.objects')
                    ])
            
            if not is_series:
                joblib.Parallel(n_jobs=cpu)(           
                    joblib.delayed(__summarize_objects)(
                            valid_images[i][0],
                            valid_images[i][1],
                            valid_images[i][2],
                            image_opening_kernel, image_closing_kernel, None
                    ) for i in range(len(valid_images)))
            
            else:
                # pile up all aligned images and then detect object areas
                mask = 0.0
                for image_fpath, mask_fpath, output_fpath in valid_images:
                    mask = skimage.io.imread(mask_fpath) + mask
                mask = mask / np.max(mask) * 255
                
                # save the time-series master
                ts_master_mask_fpath = os.path.join(self.workspace, 'detection_results', 'timeseries_master.mask.png')
                skimage.io.imsave(ts_master_mask_fpath, mask.astype(np.uint8), check_contrast=False)
                
                ts_master_labeledmask_fpath = os.path.join(self.workspace, 'detection_results', 'timeseries_master.objects')
                __summarize_objects(ts_master_mask_fpath,
                                    ts_master_mask_fpath,
                                    ts_master_labeledmask_fpath, 
                                    image_opening_kernel, image_closing_kernel, None)
                
                # generate labeled masks for each image with the time-series master mask
                joblib.Parallel(n_jobs=cpu)(           
                    joblib.delayed(__summarize_objects)(
                            valid_images[i][0],
                            valid_images[i][1],
                            valid_images[i][2],
                            image_opening_kernel, image_closing_kernel, ts_master_labeledmask_fpath + '.npz'
                    ) for i in range(len(valid_images)))
                

        except BaseException as e:
            traceback.print_exc()
            self.set_jobstatus('EVAL', 'SUMMARIZE', 'ERROR', str(e))
            run_status = False
        else:
            self.set_jobstatus('EVAL', 'SUMMARIZE', 'COMPLETED')

        return run_status

    

    
    def generate_movie(self, fps=10.0, scale=1.0, fourcc='mp4v', ext='.mp4'):
        """Video Generation

        Create video.


        Args:
            fps (int): fps
            scale (int): scale
            fourcc (bool): four cc

        Returs:
            (str): A status of running result.
        """

        run_status = True
        
        self.set_jobstatus('EVAL', 'MOVIE', 'BEGIN')
        self.seek_images()
        
        _ = cv2.imread(self.images[0][0])
        _ = cv2.resize(_, dsize=None, fx=scale, fy=scale)
        size_w = _.shape[1]
        size_h = _.shape[0]
        try:
            
            for output_type in [['detection_results', '.mask'], ['detection_results', '.crop'], ['detection_results', '.outline']]:
                fourcc_ = cv2.VideoWriter_fourcc(*fourcc)
                video = cv2.VideoWriter(os.path.join(self.workspace, 'video' + output_type[1] + ext),
                                        fourcc_, fps, (size_w, size_h))
            
                for image in tqdm.tqdm(self.images, desc='writing movie'):
                    image_name = os.path.splitext(os.path.basename(image[0]))[0]
                    image_path = os.path.join(self.workspace, output_type[0], image_name + output_type[1] + '.png')
                
                    # some images in self.images might be failure in image alignmnet steps
                    # it need to check the successfully aligned images in self.images.
                    if os.path.exists(image_path):
                        img = cv2.imread(image_path)
                
                        if img is not None:
                            img = cv2.resize(img, dsize=(size_w, size_h))
                            video.write(img)
                        else:
                            logger.warning('Image {} cannot be loaded by OpenCV since the alignment step was failure for this image.'.format(image_name))
                cv2.destroyAllWindows()
                video.release()
        
        except BaseException as e:
            traceback.print_exc()
            self.set_jobstatus('EVAL', 'MOVIE', 'ERROR', str(e))
            run_status = False
        else:
            self.set_jobstatus('EVAL', 'MOVIE', 'COMPLETED')
    
        return run_status





class SODGUI(SOD):
    """Base implementation of AgroLens GUI
    
    GUIBase class implements fundamental parts of GUI for AgroLens.
    
    
    Attributes:
       window (ThemedTk): A main window of GUI generated by tkinter.
       tasb (Notebook): A tab manager of GUI application window.
       config (dictionary): A dictionary to store inputs from GUI.
    """
    
    
    def __init__(self):
        super().__init__()
        
        
        # common configs
        self.__status_style = {
            'UNLOADED':   'secondary.TLabel',
            'UNCOMPLETE': 'TLabel',
            'COMPLETED':  'success.TLabel',
            'RUNNING':    'warning.TLabel',
            'ERROR':      'danger.TLabel',
        }
        
        
        # app window
        self.window = ttk.Window(themename='lumen')
        self.window.title(r'AgroLens - Salient Object Detection')
        self.window_style = tkinter.ttk.Style()
        self.window_style.configure('.', font=('System', 10))
        
        # app tabs (3 tabs)
        tabs = [['config', 'Preferences',        tkinter.NORMAL],
                ['train',  'Model Training',   tkinter.DISABLED],
                ['eval',   'Image Analysis', tkinter.DISABLED]]
        self.tabs = ttk.Notebook(self.window, name='tab')
        self.tabs.grid(padx=10, pady=10)
        for name_, label_, state_ in tabs:
            self.tabs.add(ttk.Frame(self.tabs, name=name_),
                          text=label_, state=state_)
        
        self.__gui_flag = {'config': '', 'train': '', 'eval': ''}
        
        # modules
        self.config = {}
        self.setup_module('config')
        self.setup_module('train')
        self.setup_module('eval')
        
        # job runing status
        self.thread_job = None
        self.thread_module_name = None
        
        
        # startup modules
        self.locate_module('config')
        self.locate_module('train')
        self.locate_module('eval')


    
    
    def setup_module(self, module_name):
        if module_name == 'config':
            self.__setup_module_config()
        elif module_name == 'train':
            self.__setup_module_train()
        elif module_name == 'eval':
            self.__setup_module_eval()
        else:
            raise ValueError('Unsupported module name `{}`.'.format(module_name))
    
    
    
    def __setup_module_config(self):
        # main frame
        frame = ttk.Frame(self.tabs.nametowidget('config'), padding=10, name='module')
        
        # subframes
        subframe_desc   = ttk.Frame(frame, name='desc')
        subframe_params = ttk.LabelFrame(frame, padding=5, name='params', text='  Settings  ')
        subframe_pref   = ttk.Frame(subframe_params, name='preference')
        subframe_ws     = ttk.Frame(subframe_params, name='workspace')
        
        # subframe - app description
        app_title   = ttk.Label(subframe_desc, name='appTitle',
                                        width=40, text='AgroLens',
                                        style='TLabel', font=('System', 0, 'bold'))
        app_version = ttk.Label(subframe_desc, name='appVersion',
                                        width=10, text='',
                                        #width=10, text='v{}'.format(agrolens.__version__),
                                        style='TLabel')
        app_desc    = ttk.Label(subframe_desc, name='appDesc',
                                        width=80, text='Salient Object Detection',
                                        style='TLabel')
        hr = ttk.Separator(subframe_desc, name='separator', orient='horizontal')
        
        # subframe - preferences
        cpu = tkinter.IntVar()
        cpu.set(multiprocessing.cpu_count())
        cpu_label = ttk.Label(subframe_pref, name='cpuLabel',
                                      text='CPU', width=5)
        cpu_input = ttk.Entry(subframe_pref, name='cpuInput',
                                      textvariable=cpu, width=5, justify=tkinter.RIGHT)
        gpu = tkinter.IntVar()
        gpu_input_state = tkinter.NORMAL
        gpu.set(1)
        if not torch.cuda.is_available():
            gpu.set(0)
            gpu_input_state = tkinter.DISABLED
        gpu_label = ttk.Label(subframe_pref, name='gpuLabel',
                                      text='GPU', width=5)
        gpu_input = ttk.Entry(subframe_pref, name='gpuInput',
                                      textvariable=gpu, width=5, justify=tkinter.RIGHT, state=gpu_input_state)
        
        # submodule - workspace
        workspace = tkinter.StringVar()
        workspace_label = ttk.Label(subframe_ws, name='wsLabel',
                                            width=10, text='Workspace')
        workspace_input = ttk.Entry(subframe_ws, name='wsInput',
                                            width=45, textvariable=workspace)
        workspace_button = ttk.Button(subframe_ws, name='wsSelectButton',
                                              width=18, text='Select',
                                              command=lambda: self.__filedialog('config', workspace, 'opendir_ws'))
        loadworkspace_button = ttk.Button(subframe_ws, width=18, text='Load Workspace', 
                                                  name='loadButton',
                                                  state=tkinter.DISABLED,
                                                  command=lambda: self.load_workspace())
        
        self.config.update({
            'workspace': workspace,
            'cpu':       cpu,
            'gpu':       gpu,
        })
    
    
    
    def __setup_module_train(self):
        # main frame
        frame = ttk.Frame(self.tabs.nametowidget('train'), padding=10, name='module')
        
        # sub frames
        subframe = ttk.LabelFrame(frame, padding=5, name='params', text='  Training Settings  ')
        subframe_sort = ttk.Frame(subframe, name='sort')
        subframe_weight = ttk.Frame(subframe, name='weight')
        subframe_params = ttk.Frame(subframe, name='trainparams')
        subframe_method = ttk.Frame(subframe, name='strategy')
        

        # subframe - image sorting
        imagesuffix = tkinter.StringVar()
        imagesuffix.set('.jpg')
        imagesuffix_label = ttk.Label(subframe_sort, name='imagesuffixLabel',
                                              text='image suffix', width=12)
        imagesuffix_input = ttk.Entry(subframe_sort, name='imagesuffixInput',
                                              textvariable=imagesuffix, width=20)
        labelsuffix = tkinter.StringVar()
        labelsuffix.set('.mask.png')
        labelsuffix_label = ttk.Label(subframe_sort, name='labelsuffixLabel',
                                              text='label suffix', width=11)
        labelsuffix_input = ttk.Entry(subframe_sort, name='labelsuffixInput',
                                              textvariable=labelsuffix, width=20)
        
        # subframe - model training / weight
        weight = tkinter.StringVar()
        weight_label  = ttk.Label(subframe_weight, name='weightLabel',
                                          width=12, text='model weight')
        weight_input  = ttk.Entry(subframe_weight, name='weightInput',
                                          width=44, textvariable=weight)
        weight_button = ttk.Button(subframe_weight, name='weightSelectButton',
                                           text='Select', width=18,
                                           command=lambda: self.__filedialog('train', weight, 'savefile_weight'))

        imagesdpath = tkinter.StringVar()
        imagesdpath_label  = ttk.Label(subframe_weight, name='imagesdpathLabel',
                                               width=12, text='Image folder')
        imagesdpath_input  = ttk.Entry(subframe_weight, name='imagesdpathInput',
                                               width=44, textvariable=imagesdpath)
        imagesdpath_button = ttk.Button(subframe_weight, name='imagesdpathSelectButton',
                                                text='Select', width=18,
                                                command=lambda: self.__filedialog('train', imagesdpath, 'opendir_trainimages'))

       
        # subframe - model training / params
        batchsize = tkinter.IntVar()
        batchsize.set(16)
        batchsize_label = ttk.Label(subframe_params, name='batchsizeLabel',
                                            text='batch size', width=10)
        batchsize_input = ttk.Entry(subframe_params, name='batchsizeInput',
                                            textvariable=batchsize, width=6, justify=tkinter.RIGHT)
        epoch = tkinter.IntVar()
        epoch.set(10000)
        epoch_label = ttk.Label(subframe_params, name='epochLabel',
                                        text='epochs', width=10)
        epoch_input = ttk.Entry(subframe_params, name='epochInput',
                                        textvariable=epoch, width=6, justify=tkinter.RIGHT)
        
        # subframe - model trianing / strategy
        strategy = tkinter.StringVar()
        strategy.set('resize')
        strategy_label = ttk.Label(subframe_method, name='strategyLabel',
                                           text='strategy', width=10)
        strategy_input = ttk.Combobox(subframe_method, name='strategyInput',
                                              textvariable=strategy, values=['resize', 'randomcrop'],
                                              width=15)
        windowsize = tkinter.IntVar()
        windowsize.set(320)
        windowsize_label = ttk.Label(subframe_method, name='windowsizeLabel',
                                                 text='random crop size', width=15)
        windowsize_input = ttk.Entry(subframe_method, name='windowsizeInput',
                                                 textvariable=windowsize, width=6,
                                                 justify=tkinter.RIGHT,
                                                 state=tkinter.NORMAL)
        
        # subframe job panel
        self.setup_panel_jobs('train')
        
        self.config.update({
            'train__weight'   : weight,
            'train__batchsize': batchsize,
            'train__epoch'    : epoch,
            'train__imagedpath' : imagesdpath,
            'train__imagesuffix' : imagesuffix,
            'train__labelsuffix' : labelsuffix,
            'train__strategy'    : strategy,
            'train__windowsize'  : windowsize,
        })
          
    
       
    def __setup_module_eval(self):
        # main frame
        frame = ttk.Frame(self.tabs.nametowidget('eval'), padding=10, name='module')

        # sub frames
        subframe = ttk.LabelFrame(frame, padding=5, name='params', text='   Detection Settings   ')
        subframe_weight = ttk.Frame(subframe, name='weight')
        subframe_params = ttk.Frame(subframe, name='evalparams')
        subframe_method = ttk.Frame(subframe, name='strategy')
        subframe_objsum = ttk.LabelFrame(frame, name='objsum', text='   Object Summarization   ')
        
        
        # subframe - detection / weight
        weight = tkinter.StringVar()
        weight_label  = ttk.Label(subframe_weight, name='weightLabel',
                                          width=12, text='Model weight')
        weight_input  = ttk.Entry(subframe_weight, name='weightInput',
                                          width=44, textvariable=weight)
        weight_button = ttk.Button(subframe_weight, name='weightSelectButton',
                                           text='Select', width=18,
                                           command=lambda: self.__filedialog('eval', weight, 'openfile_weight'))

        imagesdpath = tkinter.StringVar()
        imagesdpath_label  = ttk.Label(subframe_weight, name='imagesdpathLabel',
                                               width=12, text='Image folder')
        imagesdpath_input  = ttk.Entry(subframe_weight, name='imagesdpathInput',
                                               width=44, textvariable=imagesdpath)
        imagesdpath_button = ttk.Button(subframe_weight, name='imagesdpathSelectButton',
                                                text='Select', width=18,
                                                command=lambda: self.__filedialog('eval', imagesdpath, 'opendir_queryimages'))


        # subframe - detection / params
        batchsize = tkinter.IntVar()
        batchsize.set(16)
        batchsize_label = ttk.Label(subframe_params, name='batchsizeLabel',
                                            text='batch size', width=10)
        batchsize_input = ttk.Entry(subframe_params, name='batchsizeInput',
                                            textvariable=batchsize, width=6, justify=tkinter.RIGHT)
        ucutoff = tkinter.DoubleVar()
        ucutoff.set(0.5)
        ucutoff_label = ttk.Label(subframe_params, name='ucutoffLabel',
                                          text='U cutoff', width=10)
        ucutoff_input = ttk.Entry(subframe_params, name='ucutoffInput',
                                          textvariable=ucutoff, width=6)
       
        
        # subframe - detection / strategy
        strategy = tkinter.StringVar()
        strategy.set('resize')
        strategy_label = ttk.Label(subframe_method, name='strategyLabel',
                                           text='strategy', width=10)
        strategy_input = ttk.Combobox(subframe_method, name='strategyInput',
                                              textvariable=strategy, values=['resize', 'slide'],
                                              width=15)
        windowsize = tkinter.IntVar()
        windowsize.set(320)
        windowsize_label = ttk.Label(subframe_method, name='windowsizeLabel',
                                            text='slide window size', width=18)
        windowsize_input = ttk.Entry(subframe_method, name='windowsizeInput',
                                            textvariable=windowsize, width=6,
                                            justify=tkinter.RIGHT,
                                            state=tkinter.NORMAL)
        
        # subframe - summarization
        imageerosion = tkinter.IntVar()
        imageerosion.set(0)
        imageerosion_label = ttk.Label(subframe_objsum, name='imageerosionLabel',
                                               text='image erosion kernel size', width=22)
        imageerosion_input = ttk.Entry(subframe_objsum, name='imageerosionInput',
                                               textvariable=imageerosion, width=6)
        imagedilation = tkinter.IntVar()
        imagedilation.set(0)
        imagedilation_label = ttk.Label(subframe_objsum, name='imagedilationLabel',
                                                text='image dilation kernel size', width=22)
        imagedilation_input = ttk.Entry(subframe_objsum, name='imagedilationInput',
                                                textvariable=imagedilation, width=6)
 
        imageopening = tkinter.IntVar()
        imageopening.set(0)
        imageopening_label = ttk.Label(subframe_objsum, name='imageopeningLabel',
                                               text='image opening kernel size', width=22)
        imageopening_input = ttk.Entry(subframe_objsum, name='imageopeningInput',
                                               textvariable=imageopening, width=6,
                                               justify=tkinter.RIGHT)
        imageclosing = tkinter.IntVar()
        imageclosing.set(0)
        imageclosing_label = ttk.Label(subframe_objsum, name='imageclosingLabel',
                                               text='image closing kernel size', width=22)
        imageclosing_input = ttk.Entry(subframe_objsum, name='imageclosingInput',
                                               textvariable=imageclosing, width=6,
                                               justify=tkinter.RIGHT)
 
        isseries = tkinter.BooleanVar()
        isseries.set(False)
        isseries_chkbox = ttk.Checkbutton(subframe_objsum, name='isseriesChkbox',
                                                  text='time series',
                                                  style='Roundtoggle.Toolbutton',
                                                  width=15, variable=isseries)
        
        alignimages = tkinter.BooleanVar()
        alignimages.set(False)
        alignimages_chkbox = ttk.Checkbutton(subframe_objsum, name='alignimagesChkbox',
                                                  text='align images',
                                                  style='Roundtoggle.Toolbutton',
                                                  width=15, variable=alignimages)
        
        
        
        self.setup_panel_jobs('eval')
            
        self.config.update({
            'eval__imagedpath': imagesdpath,
            'eval__weight'   : weight,
            'eval__strategy' : strategy,
            'eval__windowsize': windowsize,
            'eval__batchsize': batchsize,
            'eval__imageopening': imageopening,
            'eval__imageclosing': imageclosing,
            'eval__imageerosion': imageerosion,
            'eval__imagedilation': imagedilation,
            'eval__ucutoff': ucutoff,
            'eval__isseries': isseries,
            'eval__alignimages': isseries,
        })
    
        
    
    def setup_panel_jobs(self, module_name, init_panel=False):
        if self._job_status_fpath is None:
            return False
        
        self.refresh_jobstatus()
        job_status = self.job_status[module_name.upper()]
        
        frame = self.tabs.nametowidget(module_name + '.module')
        
        jobpanel_subframe = ttk.LabelFrame(frame, padding=(5, 5), name='jobpanel', text='  Job Panel  ')
        jobpanel_subframe.grid(pady=(20, 5), sticky=tkinter.W + tkinter.E)
        
        job_headers = ['Job', 'Status', 'Datetime', 'Run']
        for i, job_header in enumerate(job_headers):
            job_header = ttk.Label(jobpanel_subframe, text=job_header, padding=0)
            job_header.grid(row=0, column=i)
        
        jobpanel_hr = ttk.Separator(jobpanel_subframe, name='separator', orient='horizontal')
        jobpanel_hr.grid(row=1, column=0, columnspan=len(job_headers), pady=(5, 10), sticky=tkinter.E + tkinter.W)
        
        # job items
        prev_job_status = 'COMPLETED'
        for _job_code, _job_status in sorted(job_status.items(), key=lambda kv: kv[1]['id']):
            if _job_code == 'INIT':
                continue
            if _job_status['status'] == 'BEGIN':
                _job_status['status'] = 'UNCOMPLETE'
            
            enable_run_button = tkinter.DISABLED
            if prev_job_status == 'COMPLETED':
                if module_name == 'train':
                    if len(self.config[module_name + '__weight'].get()) > 0:
                        enable_run_button = tkinter.NORMAL
                elif module_name == 'eval':
                    if os.path.exists(self.config[module_name + '__weight'].get()):
                        enable_run_button = tkinter.NORMAL
                if init_panel:
                    enable_run_button = tkinter.DISABLED
            
            jobtitle_label  = ttk.Label(jobpanel_subframe, name=_job_code.lower() + 'Label',
                                                text=_job_status['title'], width=21, anchor='center')
            jobstatus_label = ttk.Label(jobpanel_subframe, name=_job_code.lower() + 'Status',
                                                text=_job_status['status'], width=17, anchor='center', padding=(2, 5),
                                                style=self.__status_style[_job_status['status']])
            jobendtime_label = ttk.Label(jobpanel_subframe, name=_job_code.lower() + 'Datetime',
                                                 text=_job_status['datetime'], width=22, anchor='center')
            jobrun_button = ttk.Button(jobpanel_subframe, text='RUN', name=_job_code.lower() + 'RunButton',
                                               width=14, state=enable_run_button,
                                               command=lambda jc = _job_code: self.run_job(module_name, jc))
            
            jobtitle_label.grid(row=_job_status['id'] + 1,   column=0, pady=1, sticky=tkinter.E + tkinter.W)
            jobstatus_label.grid(row=_job_status['id'] + 1,  column=1, pady=1, sticky=tkinter.E + tkinter.W)
            jobendtime_label.grid(row=_job_status['id'] + 1, column=2, pady=1, sticky=tkinter.E + tkinter.W)
            jobrun_button.grid(row=_job_status['id'] + 1,    column=3, pady=1, sticky=tkinter.E + tkinter.W)
            
            prev_job_status = _job_status['status']
 


    
    def load_workspace(self):
        self.init_workspace(self.config['workspace'].get())
        self.setup_panel_jobs('train', init_panel=True)
        self.tabs.tab(1, state=tkinter.NORMAL)
        self.setup_panel_jobs('eval', init_panel=True)
        self.tabs.tab(2, state=tkinter.NORMAL)
        
    
    def startup(self):
        self.window.mainloop()
    


    def __filedialog(self, module_name, tkvar, mode='openfile_none'):
        open_mode, open_ftype = mode.split('_')
        
        root_dpath = os.getcwd()
        if 'workspace' in self.config and self.config['workspace'].get():
            root_dpath = self.config['workspace'].get()
            
        if open_mode == 'openfile':
            fp = tkinter.filedialog.askopenfilename(filetypes=[('', '*')], initialdir=root_dpath)
        elif open_mode == 'opendir':
            fp = tkinter.filedialog.askdirectory(initialdir=root_dpath)
        elif open_mode == 'savefile':
            fp = tkinter.filedialog.asksaveasfilename(defaultextension='.pth', initialdir=root_dpath)
        else:
            raise ValueError('Unexpected mode.')

        if len(fp) > 0:
            tkvar.set(fp)
            self.__gui_flag[module_name] += open_ftype + ';'
            self.__enable_loadwsbutton()
            self.__enable_jobpanel()


    def __enable_loadwsbutton(self):
        if 'ws' in self.__gui_flag['config']:
            self.tabs.nametowidget('config.module.params.workspace.loadButton').config(state=tkinter.NORMAL)


    def __enable_jobpanel(self):
        if 'weight' in  self.__gui_flag['train'] and 'trainimages' in self.__gui_flag['train']:
            self.setup_panel_jobs('train')
        if 'weight' in self.__gui_flag['eval'] and 'queryimages' in self.__gui_flag['eval']:
            self.setup_panel_jobs('eval')


         
    
    def run_job(self, module_name, job_code):
        # disable tabs during job runing
        for i, tab_name in enumerate(self.tabs.tabs()):
            if tab_name != '.tab.' + module_name:
                self.tabs.tab(i, state=tkinter.DISABLED)
        
        for w in self.tabs.nametowidget(module_name + '.module.jobpanel').winfo_children():
            if 'RunButton' in str(w):
                w.config(state=tkinter.DISABLED)
        
        self.tabs.nametowidget(module_name + '.module.jobpanel.' + job_code.lower() + 'Status').config(text='RUNNING',
                               style=self.__status_style['RUNNING'])
        self.tabs.nametowidget(module_name + '.module.jobpanel.' + job_code.lower() + 'RunButton').config(text='RUNNING',
                                        state=tkinter.DISABLED)
        ##self.tabs.nametowidget(module_name + '.module.jobpanel.' + job_code.lower() + 'RunButton').config(text='STOP',
        ##                                state=tkinter.NORMAL, command=lambda: self.stop_job())
        self.tabs.nametowidget(module_name + '.module.jobpanel.' + job_code.lower() + 'Datetime').config(text='')
        self.tabs.nametowidget(module_name + '.module.jobpanel.' + job_code.lower() + 'RunButton').update()
        self.tabs.nametowidget(module_name + '.module.jobpanel.' + job_code.lower() + 'Status').update()
        self.tabs.nametowidget(module_name + '.module.jobpanel.' + job_code.lower() + 'Datetime').update()
        
        if module_name == 'train':
            if job_code == 'SORT':
                ##self.thread_job = multiprocessing.Process(target=self.sort_train_images,
                ##        args=(self.config['train__imagedpath'].get(),
                ##              None,
                ##              'mask',
                ##              self.config['train__imagesuffix'].get(),
                ##              self.config['train__labelsuffix'].get(),))
                self.sort_train_images(
                          self.config['train__imagedpath'].get(),
                           None,
                           'mask',
                           self.config['train__imagesuffix'].get(),
                           self.config['train__labelsuffix'].get()
                )
        
            elif job_code == 'TRAIN':
                ##self.thread_job = multiprocessing.Process(target=self.train_model,
                ##    args=(self.config['train__weight'].get(),
                ##          self.config['train__batchsize'].get(),
                ##          self.config['train__epoch'].get(),
                ##          self.config['cpu'].get(),
                ##          self.config['gpu'].get(),
                ##          self.config['train__strategy'].get(),
                ##          self.config['train__windowsize'].get(),))
                self.train_model(
                          self.config['train__weight'].get(),
                          self.config['train__batchsize'].get(),
                          self.config['train__epoch'].get(),
                          self.config['cpu'].get(),
                          self.config['gpu'].get(),
                          self.config['train__strategy'].get(),
                          self.config['train__windowsize'].get()
                )
        

        elif module_name == 'eval':
            if job_code == 'SORT':
                ##self.thread_job = multiprocessing.Process(target=self.sort_query_images,
                ##    args=(self.config['eval__imagedpath'].get(),
                ##          self.config['eval__alignimages'].get(),))
                self.sort_query_images(
                          self.config['eval__imagedpath'].get(),
                          self.config['eval__alignimages'].get()
                )
                
                
            elif job_code == 'DETECT':
                ##self.thread_job = multiprocessing.Process(target=self.detect_objects,
                ##    args=(self.config['eval__weight'].get(),
                ##          self.config['eval__batchsize'].get(),
                ##          self.config['eval__strategy'].get(),
                ##          self.config['eval__ucutoff'].get(),
                ##          self.config['eval__imageopening'].get(),
                ##          self.config['eval__imageclosing'].get(),
                ##          self.config['eval__windowsize'].get(),
                ##          self.config['cpu'].get(),
                ##          self.config['gpu'].get(), ))
                self.detect_objects(
                        self.config['eval__weight'].get(),
                        self.config['eval__batchsize'].get(),
                        self.config['eval__strategy'].get(),
                        self.config['eval__ucutoff'].get(),
                        self.config['eval__imageopening'].get(),
                        self.config['eval__imageclosing'].get(),
                        self.config['eval__windowsize'].get(),
                        self.config['cpu'].get(),
                        self.config['gpu'].get()
                )

            elif job_code == 'SUMMARIZE':
                ##self.thread_job = multiprocessing.Process(target=self.summarize_objects,
                ##    args=(self.config['cpu'].get(),
                ##          self.config['eval__isseries'].get(), 
                ##          self.config['eval__imageopening'].get(),
                ##          self.config['eval__imageclosing'].get(), ))
                self.summarize_objects(
                    self.config['cpu'].get(),
                    self.config['eval__isseries'].get(),
                    self.config['eval__imageopening'].get(),
                    self.config['eval__imageclosing'].get()
                )
            
            elif job_code == 'MOVIE':
                ##self.thread_job = multiprocessing.Process(target=self.generate_movie,
                ##    args=(10.0, 1.0, 'mp4v', '.mp4', ))
                self.generate_movie(10.0, 1.0, 'mp4v', '.mp4')
        
        ## self.thread_job.start()
        self.thread_module_name = module_name
        self.check_thread_job()
        
 
    
    def stop_job(self):
        if self.thread_job is not None and self.thread_job.is_alive():
            self.thread_job.terminate()
    
    
    def check_thread_job(self):
        
        if self.thread_job is not None and self.thread_job.is_alive():
            self.window.after(1000, self.check_thread_job)
        
        else:
            if self.thread_job is not None:
                self.thread_job.join()
            self.setup_panel_jobs(self.thread_module_name)
            
            for i, tab_name in enumerate(self.tabs.tabs()):
                self.tabs.tab(i, state=tkinter.NORMAL)
            
            self.thread_job = None
            self.thread_module_name = None




    def locate_module(self, module_name):
        if module_name == 'config':
            self.__locate_module_config()
        elif module_name == 'train':
            self.__locate_module_train()
        elif module_name == 'eval':
            self.__locate_module_eval()
        else:
            raise ValueError('Unsupported module name.')
    

    def __locate_module_config(self):
        self.tabs.nametowidget('config.module').grid(sticky=tkinter.W + tkinter.E)
        self.tabs.nametowidget('config.module.desc').grid(pady=5, sticky=tkinter.W + tkinter.E)
        if True:
            self.tabs.nametowidget('config.module.desc.appTitle').grid(row=0, column=0, pady=5, sticky=tkinter.W)
            self.tabs.nametowidget('config.module.desc.appVersion').grid(row=0, column=1, pady=5, sticky=tkinter.E)
            self.tabs.nametowidget('config.module.desc.appDesc').grid(row=1, column=0, columnspan=2, pady=5, sticky=tkinter.E + tkinter.W)
            self.tabs.nametowidget('config.module.desc.separator').grid(row=2, column=0, columnspan=2, pady=5, sticky=tkinter.W + tkinter.E)
        self.tabs.nametowidget('config.module.params').grid(sticky=tkinter.W + tkinter.E)
        self.tabs.nametowidget('config.module.params.preference').grid(pady=5, sticky=tkinter.W + tkinter.E)
        if True:
            self.tabs.nametowidget('config.module.params.preference.cpuLabel').grid(row=0, column=0, sticky=tkinter.W)
            self.tabs.nametowidget('config.module.params.preference.cpuInput').grid(row=0, column=1, sticky=tkinter.W)
            self.tabs.nametowidget('config.module.params.preference.gpuLabel').grid(row=0, column=2, padx=(20, 0), sticky=tkinter.W)
            self.tabs.nametowidget('config.module.params.preference.gpuInput').grid(row=0, column=3, sticky=tkinter.W)
        self.tabs.nametowidget('config.module.params.workspace').grid(pady=5, sticky=tkinter.W + tkinter.E)
        if True:
            self.tabs.nametowidget('config.module.params.workspace.wsLabel').grid(row=0, column=0, padx=(0, 5), pady=5, sticky=tkinter.W + tkinter.E)
            self.tabs.nametowidget('config.module.params.workspace.wsInput').grid(row=0, column=1, pady=5, sticky=tkinter.W + tkinter.E)
            self.tabs.nametowidget('config.module.params.workspace.wsSelectButton').grid(row=0, column=2, padx=(10, 0), pady=5, sticky=tkinter.W + tkinter.E)
            self.tabs.nametowidget('config.module.params.workspace.loadButton').grid(row=1, column=2, padx=(10, 0), pady=5, sticky=tkinter.W + tkinter.E)
    
    
    
    def __locate_module_train(self):
        self.tabs.nametowidget('train.module').grid(sticky=tkinter.W + tkinter.E)
        self.tabs.nametowidget('train.module.params').grid(sticky=tkinter.W + tkinter.E)
        self.tabs.nametowidget('train.module.params.weight').grid(pady=5, sticky=tkinter.W + tkinter.E)
        if True:
            self.tabs.nametowidget('train.module.params.weight.weightLabel').grid(row=0, column=0, pady=5, sticky=tkinter.W + tkinter.E)
            self.tabs.nametowidget('train.module.params.weight.weightInput').grid(row=0, column=1, pady=5, sticky=tkinter.W + tkinter.E)
            self.tabs.nametowidget('train.module.params.weight.weightSelectButton').grid(row=0, column=2, padx=(10, 0), pady=5, sticky=tkinter.W + tkinter.E)
            self.tabs.nametowidget('train.module.params.weight.imagesdpathLabel').grid(row=1, column=0, pady=5, sticky=tkinter.W + tkinter.E)
            self.tabs.nametowidget('train.module.params.weight.imagesdpathInput').grid(row=1, column=1, pady=5, sticky=tkinter.W + tkinter.E)
            self.tabs.nametowidget('train.module.params.weight.imagesdpathSelectButton').grid(row=1, column=2, padx=(10, 0), pady=5, sticky=tkinter.W + tkinter.E)

        self.tabs.nametowidget('train.module.params.sort').grid(pady=5, sticky=tkinter.W + tkinter.E)
        if True:
            self.tabs.nametowidget('train.module.params.sort.imagesuffixLabel').grid(row=0, column=0, sticky=tkinter.W)
            self.tabs.nametowidget('train.module.params.sort.imagesuffixInput').grid(row=0, column=1, sticky=tkinter.W)
            self.tabs.nametowidget('train.module.params.sort.labelsuffixLabel').grid(row=0, column=2, padx=(50, 0), sticky=tkinter.W)
            self.tabs.nametowidget('train.module.params.sort.labelsuffixInput').grid(row=0, column=3, sticky=tkinter.W)
        self.tabs.nametowidget('train.module.params.trainparams').grid(pady=5, sticky=tkinter.W + tkinter.E)
        if True:
            self.tabs.nametowidget('train.module.params.trainparams.batchsizeLabel').grid(row=0, column=0, sticky=tkinter.W)
            self.tabs.nametowidget('train.module.params.trainparams.batchsizeInput').grid(row=0, column=1, sticky=tkinter.W)
            self.tabs.nametowidget('train.module.params.trainparams.epochLabel').grid(row=0, column=2, padx=(50, 0), sticky=tkinter.W)
            self.tabs.nametowidget('train.module.params.trainparams.epochInput').grid(row=0, column=3, sticky=tkinter.W)
        self.tabs.nametowidget('train.module.params.strategy').grid(pady=5, sticky=tkinter.W + tkinter.E)
        if True:
            self.tabs.nametowidget('train.module.params.strategy.strategyLabel').grid(row=0, column=0, sticky=tkinter.W)
            self.tabs.nametowidget('train.module.params.strategy.strategyInput').grid(row=0, column=1, sticky=tkinter.W)
            self.tabs.nametowidget('train.module.params.strategy.windowsizeLabel').grid(row=0, column=2, padx=(50, 0), sticky=tkinter.W)
            self.tabs.nametowidget('train.module.params.strategy.windowsizeInput').grid(row=0, column=3, sticky=tkinter.W)
            
    
    
       
    def __locate_module_eval(self):
        self.tabs.nametowidget('eval.module').grid(sticky=tkinter.W + tkinter.E)
        self.tabs.nametowidget('eval.module.params').grid(sticky=tkinter.W + tkinter.E)
        self.tabs.nametowidget('eval.module.params.weight').grid(pady=5, sticky=tkinter.W + tkinter.E)
        if True:
            self.tabs.nametowidget('eval.module.params.weight.weightLabel').grid(row=0, column=0, pady=5, sticky=tkinter.W + tkinter.E)
            self.tabs.nametowidget('eval.module.params.weight.weightInput').grid(row=0, column=1, pady=5, sticky=tkinter.W + tkinter.E)
            self.tabs.nametowidget('eval.module.params.weight.weightSelectButton').grid(row=0, column=2, padx=(10, 0), pady=5, sticky=tkinter.W + tkinter.E)
            self.tabs.nametowidget('eval.module.params.weight.imagesdpathLabel').grid(row=1, column=0, pady=5, sticky=tkinter.W + tkinter.E)
            self.tabs.nametowidget('eval.module.params.weight.imagesdpathInput').grid(row=1, column=1, pady=5, sticky=tkinter.W + tkinter.E)
            self.tabs.nametowidget('eval.module.params.weight.imagesdpathSelectButton').grid(row=1, column=2, padx=(10, 0), pady=5, sticky=tkinter.W + tkinter.E)

        self.tabs.nametowidget('eval.module.params.evalparams').grid(pady=5, sticky=tkinter.W + tkinter.E)
        if True:
            self.tabs.nametowidget('eval.module.params.evalparams.batchsizeLabel').grid(row=0, column=0, pady=5, sticky=tkinter.W)
            self.tabs.nametowidget('eval.module.params.evalparams.batchsizeInput').grid(row=0, column=1, pady=5, sticky=tkinter.W)
            self.tabs.nametowidget('eval.module.params.evalparams.ucutoffLabel').grid(row=0, column=3, pady=5, padx=(10, 0), sticky=tkinter.W)
            self.tabs.nametowidget('eval.module.params.evalparams.ucutoffInput').grid(row=0, column=4, pady=5, sticky=tkinter.W)
           
        self.tabs.nametowidget('eval.module.params.strategy').grid(pady=5, sticky=tkinter.W + tkinter.E)
        if True:
            self.tabs.nametowidget('eval.module.params.strategy.strategyLabel').grid(row=0, column=0, pady=5, sticky=tkinter.W)
            self.tabs.nametowidget('eval.module.params.strategy.strategyInput').grid(row=0, column=1, pady=5, sticky=tkinter.W)
            self.tabs.nametowidget('eval.module.params.strategy.windowsizeLabel').grid(row=0, column=2, padx=(50, 0), pady=5, sticky=tkinter.W)
            self.tabs.nametowidget('eval.module.params.strategy.windowsizeInput').grid(row=0, column=3, pady=5, sticky=tkinter.W)
        
        self.tabs.nametowidget('eval.module.objsum').grid(pady=5, sticky=tkinter.W + tkinter.E)
        if True:
            #self.tabs.nametowidget('eval.module.objsum.imageerosionLabel').grid(row=1, column=0, pady=5, padx=(5, 5), sticky=tkinter.W)
            #self.tabs.nametowidget('eval.module.objsum.imageerosionInput').grid(row=1, column=1, pady=5, padx=(5, 5), sticky=tkinter.W)
            #self.tabs.nametowidget('eval.module.objsum.imagedilationLabel').grid(row=1, column=2, pady=5, padx=(5, 5), sticky=tkinter.W)
            #self.tabs.nametowidget('eval.module.objsum.imagedilationInput').grid(row=1, column=3, pady=5, padx=(5, 5), sticky=tkinter.W)
            self.tabs.nametowidget('eval.module.objsum.isseriesChkbox').grid(row=0, column=0, columnspan=2, pady=5, padx=(5, 5), sticky=tkinter.W)
            self.tabs.nametowidget('eval.module.objsum.alignimagesChkbox').grid(row=0, column=1, columnspan=2, pady=5, padx=(5, 5), sticky=tkinter.W)
            self.tabs.nametowidget('eval.module.objsum.imageopeningLabel').grid(row=1, column=0, pady=5, padx=(5, 5), sticky=tkinter.W)
            self.tabs.nametowidget('eval.module.objsum.imageopeningInput').grid(row=1, column=1, pady=5, padx=(5, 5), sticky=tkinter.W)
            self.tabs.nametowidget('eval.module.objsum.imageclosingLabel').grid(row=1, column=2, pady=5, padx=(5, 5), sticky=tkinter.W)
            self.tabs.nametowidget('eval.module.objsum.imageclosingInput').grid(row=1, column=3, pady=5, padx=(5, 5), sticky=tkinter.W)
 





