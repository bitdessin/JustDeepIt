==========================
Crop and weed segmentation
==========================

Weeds are a major agricultural problem affecting crop yield and quality,
especially in large fields.
Image recognition using deep learning technology
can be applied accurately and efficiently to process large number of images acquired
from drones, field-robots, etc.
Thus, there is much demand
to develop efficient weed detection systems using deep learning technology
to prevent agricultural losses.
In this tutorial, we show an example to detect/segment crops and weeds
using Mask R-CNN\ [#maskrcnn]_ through JustDeepIt.



Dataset preparation
===================

SugarBeets2016\ [#sugarbeet]_ dataset has 11,552 RGB images captured under the fields,
and each image has a annotation for sugar beets and weeds.
The dataset can be downloaded from the
`StachnissLab <https://www.ipb.uni-bonn.de/data/sugarbeets2016/>`_ website.
By clicking the `Complete Dataset <https://www.ipb.uni-bonn.de/datasets_IJRR2017/>`_ link
on the `StachnissLab <https://www.ipb.uni-bonn.de/data/sugarbeets2016/>`_ website,
the download page is displayed.
Next, click on the `annotations` link and then the `cropweed` link
to go to the detailed data download page.
In this page, we download all 11 files named as ijrr_sugarbeets_2016_annotations.part*.rar
where * is digits from 01 to 11.

After downloading the 11 files, we decompress them with the corresponding software.
Then, :file:`ijrr_sugarbeets_2016_annotations` folder is generated after the decompression
and this folder contains :file:`CRA_16....` folders which contains images and annotations (RBG mask).

Next, we randomly selected 5,000 and 1,000 images
and save them into :file:`train` and :file:`test` folders for training and test, respectively.
In addition, since JustDeepIt requires annotations in the COCO format,
we need convert the RGB masks into a COCO format file (:file:`train.json`).
Python script :file:`convert_rgbmask2coco.py` stored in GitHub
(`JustDeepIt/tutorials/SugarWeeds2016/scripts <https://github.com/biunit/JustDeepIt/tree/main/tutorials/SugarWeeds2016/scripts>`_)
can be used for random sampling and format conversion at the same time.

In addition, JustDeepIt requires a text file containing class names.
We create a file :file:`class_label.txt` containing 
"sugarbeets" on the first line and "weeds" on the second line,
as the SugarBeets2016 dataset has two classes, namely, sugar beets and weeds.

The above dataset preparation can be performed manually or automatically
using the following shell scripts:



.. code-block:: sh

    # download tutorials/SugarBeets2016 or clone JustDeepIt repository from https://github.com/biunit/JustDeepIt
    git clone https://github.com/biunit/JustDeepIt
    cd JustDeepIt/tutorials/SugarBeets2016

    # download image data
    wget https://www.ipb.uni-bonn.de/datasets_IJRR2017/annotations/cropweed/ijrr_sugarbeets_2016_annotations.part01.rar
    wget https://www.ipb.uni-bonn.de/datasets_IJRR2017/annotations/cropweed/ijrr_sugarbeets_2016_annotations.part02.rar
    wget https://www.ipb.uni-bonn.de/datasets_IJRR2017/annotations/cropweed/ijrr_sugarbeets_2016_annotations.part03.rar
    wget https://www.ipb.uni-bonn.de/datasets_IJRR2017/annotations/cropweed/ijrr_sugarbeets_2016_annotations.part04.rar
    wget https://www.ipb.uni-bonn.de/datasets_IJRR2017/annotations/cropweed/ijrr_sugarbeets_2016_annotations.part05.rar
    wget https://www.ipb.uni-bonn.de/datasets_IJRR2017/annotations/cropweed/ijrr_sugarbeets_2016_annotations.part06.rar
    wget https://www.ipb.uni-bonn.de/datasets_IJRR2017/annotations/cropweed/ijrr_sugarbeets_2016_annotations.part07.rar
    wget https://www.ipb.uni-bonn.de/datasets_IJRR2017/annotations/cropweed/ijrr_sugarbeets_2016_annotations.part08.rar
    wget https://www.ipb.uni-bonn.de/datasets_IJRR2017/annotations/cropweed/ijrr_sugarbeets_2016_annotations.part09.rar
    wget https://www.ipb.uni-bonn.de/datasets_IJRR2017/annotations/cropweed/ijrr_sugarbeets_2016_annotations.part10.rar
    wget https://www.ipb.uni-bonn.de/datasets_IJRR2017/annotations/cropweed/ijrr_sugarbeets_2016_annotations.part11.rar

    # decompress data
    unrar x -e ijrr_sugarbeets_2016_annotations.part01.rar

    # generate training and test images and annotation in COCO format
    python scripts/convert_rgbmask2coco.py ijrr_sugarbeets_2016_annotations .






Settings
========



To start JustDeepIt, we open the terminal and run the following command.
Then, we open the web browser, access to \http://127.0.0.1:8000,
and start "Instance Segmentation" mode.


.. code-block:: sh

    justdeepit
    # INFO:uvicorn.error:Started server process [61]
    # INFO:uvicorn.error:Waiting for application startup.
    # INFO:uvicorn.error:Application startup complete.
    # INFO:uvicorn.error:Uvicorn running on http://127.0.0.1:8000 (Press CTRL+C to quit)



We set the **architecture** to Mask R-CNN,
the **workspace** to the location containing folder :file:`train` and file :file:`train.json`,
and the other parameters as shown in the screenshot below.
Note that the value of **workspace** may be different from the screenshot
depending on user's environment.
Then, we press button **Load Workspace**.


.. image:: ../_static/tutorials_SugarBeets2016_pref.png
    :align: center



Once the workspace is set,
the functions of **Training** and **Inference** become available.


Training
========


To train the model,
we select tab **Training**
and specify the **model weight** as the location storing the training weights,
**image folder** as the folder containing training images (i.e., :file:`train`),
**annotation** format as the format of the annotation file (COCO in this case),
and **annotation** as the file of image annotations (i.e., :file:`train.json`).
The other parameters are set as shown in screenshot below.
Note that the values of **model weight**, **image folder**, and **annotation** may be
different from the screenshot depending on user's environment.
Then, we press button **Start Training** for model training.



.. image:: ../_static/tutorials_SugarBeets2016_train.png
    :align: center


Training takes about 20 hours, and it depends on the computer hardware.



Inference
=========

In tab **Inference**, the **model weight** is specified to the training weights,
whose file extension is :file:`.pth` in general.
Then, we specify **image folder** to the folder containing the images for detection
(i.e., :file:`test`),
and other parameters as shown in screenshot below.
Note that the values of **model weight** and **image folder** may be
different from the screenshot depending on user's environment.
Next, we press button **Start Inference** for object detection.


.. image:: ../_static/tutorials_SugarBeets2016_eval.png
    :align: center


The detection results will be stored in folder :file:`justdeepitws/outputs` of the workspace
as images with bounding boxes and contours
and a JSON file in the COCO format (:file:`annotation.json`).
Examples of wheat head detection results are shown in the figure below.

.. image:: ../_static/tutorials_SugarBeets2016_inference_output.jpg
    :align: center






API
====


Model training and object detection can be performed using the JustDeepIt API.
Python script :file:`run_justdeepit.py` stored in GitHub
(`JustDeepIt/tutorials/SugarBeets2016/scripts <https://github.com/biunit/JustDeepIt/tree/main/tutorials/SugarBeets2016/scripts>`_) can be used for this purpose.



.. code-block:: sh

    cd JustDeepIt/tutorials/SugarBeets2016
    
    # run instance segmentation with Detectron2 backend
    python run_justdeepit.py train
    python run_justdeepit.py test
    
    # run instance segmentation with MMDetection backend
    python scripts/run_justdeepit.py train mmdetection
    python scripts/run_justdeepit.py test mmdetection






References
==========

.. [#maskrcnn] He, K., Gkioxari, G., Dollár, P., and Girshick, R. (2017). Mask R-CNN. http://arxiv.org/abs/1703.06870
.. [#sugarbeet] Chebrolu, N., Lottes, P., Schaefer, A., Winterhalter, W., Burgard, W., and Stachniss, C. (2017). Agricultural robot dataset for plant classification, localization and mapping on sugar beet fields. Int. J. Rob. Res. 36(10). doi: 10.1177/0278364917720510. 



