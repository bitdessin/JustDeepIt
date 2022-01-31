===================================
Wheat Head Detection (Faster R-CNN)
===================================


Heading is a key phenological stage during growth of most crops
because it reflects the transition from the vegetative growth stage to the reproductive stage.
Monitoring heading helps researchers understand the interactions between growth stages and environments.
Furthermore, it supports farmers in management activities,
such as making decisions about a treatment to be applied.
Detecting crop heads from images captured by fixed-point cameras or drones
allows to conduct high-throughput phenotyping and efficiently run large farms.
Deep learning technology has become more common for various detection tasks.
For instance, in this tutorial, we used AgroLens to train Faster R-CNN\ [#fasterrcnn]_ for wheat head detection.



Dataset Preparation
===================


The global wheat head detection (GWHD) dataset is a large-scale dataset used for wheat head detection\ [#gwhd]_.
The images in the GWHD dataset were taken from various cultivars growing in different environments worldwide.
Detailed descriptions of the GWHD dataset and instructions for downloading are available on the Global Wheat website.

Following the dataset instructions, we download files :file:`train.zip` and :file:`test.zip`.
By decompressing file :file:`train.zip`, we obtaine folder :file:`train` and file :file:`train.csv`.
Folder :file:`train` contains images of wheat heads,
and file :file:`train.csv` contains the bounding-box coordinates of wheat heads
for each image in folder :file:`train`.

As AgroLens requires annotations in the COCO format,
we first convert file :file:`train.csv` into a file in the COCO format (:file:`train.json`).
Python script :file:`gwhd2coco.py` stored in 
`GitHub:AgroLens/tutorials/GWHD/scripts <https://github.com/biunit/AgroLens/>`_ can be used for format conversion.
In addition, AgroLens requires a text file containing class names.
We create file :file:`class_label.txt` containing only “spike” on the first line,
as the GWHD dataset only has one class, namely, wheat head.

The above dataset preparation can be performed manually or automatically using the following shell scripts:


.. include:: ../../../tutorials/GWHD/README.rst
    :start-after: .. <dataset>
    :end-before: .. </dataset>




AgroLens Settings
=================


To initialize AgroLens for object detection, we open terminal and run the following command:


.. code-block:: sh
    
    agrolens od


We set the **architecture** to Faster R-CNN,
the **workspace** to the location containing folder :file:`train` and file :file:`train.json`,
and the other parameters as shown in the screenshot below.
Then, we press button **Load Workspace**.


.. image:: ../_static/tutorials_GWHD_config.png
    :align: center



Once the workspace is set, the functions of model training and object detection become available.


Model Training
==============


To train the model,
we select tab **model training**
and specify the **model weight** as the location storing the training weights,
**image folder** as the folder containing training images (i.e., :file:`train`),
**annotation format** as the format of the annotation file (COCO in this case),
and **annotation** as the file of image annotations (i.e., :file:`train.json`).
Then, we press buttons **RUN** for image sorting and model training.



.. image:: ../_static/tutorials_GWHD_train.png
    :align: center


Training takes 3-4 days, and it depends on the computer hardware.



Object Detection
================

In tab **image analysis**, the model weight is specified to the training weights,
whose file extension is pth in general.
We specify **image folder** to the folder containing the images for detection,
and other parameters as shown in screenshot below.
Next, we decompress file :file:`test.zip` of the GWHD dataset
and used the images in folder :file:`test` as test images.
Then, we press buttons **RUN** for image sorting and object detection.


.. image:: ../_static/tutorials_GWHD_inference.png
    :align: center


The detection results will be stored in folder :file:`outputs/detection_results` of the workspace
as images with bounding boxes and a JSON file in the COCO format (:file:`inference_result.json`).



Results
=======   

Examples of wheat head detection results are shown in the figure below.

.. image:: ../_static/tutorials_GWHD_inference_output.jpg
    :align: center






API
====


Model training and object detection can be performed using the AgroLens API.
Python script :file:`run_mmdet.py` stored in `GitHub:AgroLens/tutorials/GWHD/scripts <https://github.com/biunit/AgroLens>`_ can be used for this purpose.
See `GitHub:AgroLens/tutorials/GWHD/ <https://github.com/biunit/AgroLens>`_ for detailed information.


.. include:: ../../../tutorials/GWHD/README.rst
    :start-after: .. <script>
    :end-before: .. </script>
    







References
==========

.. [#fasterrcnn] Ren S, He K, Girshick R, Sun J. Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks. https://arxiv.org/abs/1506.01497
.. [#gwhd] David E, Madec S, Sadeghi-Tehran P, Aasen H, Zheng B, Liu S, Kirchgessner N, Ishikawa G, Nagasawa K, Badhon M A, Pozniak C, Solan B, Hund A, Chapman S C, Baret F, Stavness I, Guo W. Global Wheat Head Detection (GWHD) Dataset: A Large and Diverse Dataset of High-Resolution RGB-Labelled Images to Develop and Benchmark Wheat Head Detection Methods. https://doi.org/10.34133/2020/3521852




