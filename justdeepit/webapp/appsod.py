import os
import datetime
import random
import pathlib
import shutil
import glob
import copy
import tqdm
import logging
import threading
import multiprocessing
import joblib
import traceback
import inspect
import ctypes
import numpy as np
import pandas as pd
import PIL
import PIL.ExifTags
import cv2
import skimage.io
import skimage.color
import skimage.exposure
import torch
import justdeepit
from justdeepit.webapp import AppBase


logger = logging.getLogger(__name__)       


class SOD(AppBase):
    def __init__(self, workspace):
        super().__init__(workspace)
        self.app = 'SOD'
        self.images = []
    
    
    
    def __build_model(self, model_arch, model_weight, ws):
        if os.path.exists(model_weight):
            model = justdeepit.models.SOD(model_arch, model_weight, workspace=ws)
        else:
            model = justdeepit.models.SOD(model_arch, workspace=ws)
        return model
 
    
    
    def train_model(self, image_dpath, annotation_path, annotation_format,
                    model_arch, model_weight,
                    optimizer, scheduler, 
                    batchsize, epoch,
                    cpu, gpu, strategy, window_size):
        
        job_status = self.set_jobstatus(self.code.TRAINING, self.code.JOB__TRAIN_MODEL, self.code.STARTED, '')

        try:
            self.check_training_images(image_dpath, annotation_path, annotation_format)

            tmp_dpath = os.path.join(self.workspace_, 'tmp')
            logger.info('The check points will be saved as {} every 100 epochs.'.format(tmp_dpath))
            
            model = self.__build_model(model_arch, model_weight, tmp_dpath)
            model.train(image_dpath, annotation_path, annotation_format,
                        optimizer, scheduler,
                        batchsize, epoch, cpu, gpu,
                        strategy=strategy, window_size=window_size)
            model.save(model_weight)
                
        except KeyboardInterrupt:
            job_status = self.set_jobstatus(self.code.TRAINING, self.code.JOB__TRAIN_MODEL, self.code.INTERRUPT, '')
            
        except BaseException as e:
            traceback.print_exc()
            job_status = self.set_jobstatus(self.code.TRAINING, self.code.JOB__TRAIN_MODEL, self.code.ERROR, str(e))
        else:
            job_status = self.set_jobstatus(self.code.TRAINING, self.code.JOB__TRAIN_MODEL, self.code.COMPLETED,
                                            'Params: batchsize {}; epoch {}; optimizer: {}; scheduler: {}.'.format(batchsize, epoch, optimizer, scheduler))

        return job_status    
    
    
    
    
    
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
    
     

    def __align_images(self, image_files):
    
        aligned_image_files = []
        aligned_image_files_status = []
        
        output_dpath = os.path.join(self.workspace_, 'tmp', 'aligned_images')
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
        
        return aligned_image_files_status
    
    
       

    def sort_query_images(self, image_dpath, align_images=False):
        
        job_status = self.set_jobstatus(self.code.INFERENCE, self.code.JOB__SORT_IMAGES, self.code.STARTED, '')
        
        def __get_exif_datetime(fpath):
            im = PIL.Image.open(fpath)
            dt = ''
            if hasattr(im, '_getexif'):
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
            image_files = []
            for f in sorted(glob.glob(os.path.join(image_dpath, '**'), recursive=True)):
                if os.path.splitext(f)[1].lower() in self.image_ext:
                    image_files.append([f, __get_exif_datetime(f)])
            
            # alignment image first if required
            if align_images:
                aligned_image_files = []
                _ = self.__align_images([_[0] for _ in image_files])
                for i in range(len(image_files)):
                    if _[i] is not None:
                        aligned_image_files.append(image_files[i])
                        aligned_image_files[-1][0] = _[i]
                image_files = aligned_image_files
            
            with open(os.path.join(self.workspace_, 'data', 'query', 'query_images.txt'), 'w') as outfh:
                for image_file in image_files:
                    outfh.write('{}\n'.format('\t'.join(image_file)))
        
            job_status = self.set_jobstatus(self.code.INFERENCE, self.code.JOB__SORT_IMAGES, self.code.FINISHED, '')
        except KeyboardInterrupt:
            job_status = self.set_jobstatus(self.code.INFERENCE, self.code.JOB__SORT_IMAGES, self.code.INTERRUPT, '')
            
        except BaseException as e:
            traceback.print_exc()
            job_status = self.set_jobstatus(self.code.INFERENCE, self.code.JOB__SORT_IMAGES, self.code.ERROR, str(e))
        else:
            job_status = self.set_jobstatus(self.code.INFERENCE, self.code.JOB__SORT_IMAGES, self.code.COMPLETED, '')

        return job_status


   
    
    def detect_objects(self, model_arch, model_weight, batchsize,
                       strategy, score_cutoff, image_opening, image_closing, window_size,
                       cpu, gpu):
        
        def __save_outputs(ws, image_fpath, output):
            image_name = os.path.splitext(os.path.basename(image_fpath))[0]
            output.draw('mask', os.path.join(ws, 'outputs', image_name + '.mask.png'))
            output.draw('masked', os.path.join(ws, 'outputs', image_name + '.crop.png'))
            output.draw('contour', os.path.join(ws, 'outputs', image_name + '.outline.png'))
            
        
        job_status = self.set_jobstatus(self.code.INFERENCE, self.code.JOB__INFER, self.code.STARTED, '')

        try:
            self.seek_query_images()
            model = justdeepit.models.SOD(model_arch, model_weight)
            outputs = model.inference(self.images,
                                      strategy, batchsize, cpu, gpu,
                                      window_size, score_cutoff, image_opening, image_closing)
            
            joblib.Parallel(n_jobs=cpu)(
                joblib.delayed(__save_outputs)(self.workspace_, self.images[i], outputs[i]) for i in range(len(self.images)))

                
        except KeyboardInterrupt:
            job_status = self.set_jobstatus(self.code.INFERENCE, self.code.JOB__INFER, self.code.INTERRUPT, '')
            
        except BaseException as e:
            traceback.print_exc()
            job_status = self.set_jobstatus(self.code.INFERENCE, self.code.JOB__INFER, self.code.ERROR, str(e))
        else:
            job_status = self.set_jobstatus(self.code.INFERENCE, self.code.JOB__INFER, self.code.COMPLETED, '')

        return job_status

       
  
    def summarize_objects(self, cpu, is_series=False, image_opening_kernel=0, image_closing_kernel=0):
        
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
        
        

        job_status = self.set_jobstatus(self.code.INFERENCE, self.code.JOB__SUMMARIZE, self.code.STARTED, '')

        self.seek_query_images()
        try:
            logger.info('Finding objects and calculate the summary data using {} CPUs.'.format(cpu))
            
            valid_images = []
            for _img in self.images:
                    valid_images.append([
                        os.path.join(self.workspace_, 'outputs',
                                     os.path.splitext(os.path.basename(_img))[0] + '.crop.png'),
                        os.path.join(self.workspace_, 'outputs',
                                     os.path.splitext(os.path.basename(_img))[0] + '.mask.png'),
                        os.path.join(self.workspace_, 'outputs',
                                     os.path.splitext(os.path.basename(_img))[0] + '.objects')
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
                ts_master_mask_fpath = os.path.join(self.workspace_, 'outputs', 'timeseries_master.mask.png')
                skimage.io.imsave(ts_master_mask_fpath, mask.astype(np.uint8), check_contrast=False)
            
                ts_master_labeledmask_fpath = os.path.join(self.workspace_, 'outputs', 'timeseries_master.objects')
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



            job_status = self.set_jobstatus(self.code.INFERENCE, self.code.JOB__SUMMARIZE, self.code.FINISHED, '')
        except KeyboardInterrupt:
            job_status = self.set_jobstatus(self.code.INFERENCE, self.code.JOB__SUMMARIZE, self.code.INTERRUPT, '')
        except BaseException as e:
            traceback.print_exc()
            job_status = self.set_jobstatus(self.code.INFERENCE, self.code.JOB__SUMMARIZE, self.code.ERROR, str(e))
        else:
            job_status = self.set_jobstatus(self.code.INFERENCE, self.code.JOB__SUMMARIZE, self.code.COMPLETED, '')

        return job_status

    

    
    def generate_movie(self, fps=10.0, scale=1.0, fourcc='mp4v', ext='.mp4'):

        
        self.seek_query_images()
        
        _ = cv2.imread(self.images[0][0])
        _ = cv2.resize(_, dsize=None, fx=scale, fy=scale)
        size_w = _.shape[1]
        size_h = _.shape[0]
        try:
           
            for output_type in [['outputs', '.mask'], ['outputs', '.crop'], ['outputs', '.outline']]:
                fourcc_ = cv2.VideoWriter_fourcc(*fourcc)
                video = cv2.VideoWriter(os.path.join(self.workspace_, 'outputs' + 'video' + output_type[1] + ext),
                                        fourcc_, fps, (size_w, size_h))
          
                for image in tqdm.tqdm(self.images, desc='writing movie'):
                    image_name = os.path.splitext(os.path.basename(image[0]))[0]
                    image_path = os.path.join(self.workspace_, output_type[0], image_name + output_type[1] + '.png')
                
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
        else:
            pass
    
        return job_status





