===============
Leaf extraction
===============

The background may affect the performance of object classification and detection.
Thus, extracting a salient object by removing the background
may improve the performance of these tasks.
However, training the model requires several annotated images.
Usually, annotation is costly and time-consuming.
In this tutorial, we illustrate the use of JustDeepIt
to train U\ :sup:`2`-Net\ [#u2net]_ for background removal from unannotated images.



Dataset preparation
===================


The `AI Pest Image (AIPI) dataset <https://www.naro.affrc.go.jp/org/niaes/damage/>`_
contains images of over 400 thousands
cucumbers, eggplants, strawberries, and tomatoes labeled according to crop tissue and disease.
However, no annotations are available at the leaf level.
To simplify the tutorial, we only use the leaf tissue of cucumber
to train U\ :sup:`2`-Net for leaf extraction.

Images of cucumber leaves can be downloaded manually or using wget commands.
There are 29,045 cucumber leaf images in the AIPI dataset.
We store the images into folder :file:`data/images` and then use the JustDeepIt API
to train U2-Net for leaf extraction.
Note that, :file:`data/images` has already contained some images that were downloaded from the AIPI database.

.. code-block:: sh
    
    git clone https://github.com/biunit/JustDeepIt
    cd JustDeepIt/tutorials/AIPI
    
    ls data/images
    # 079906_20180207110627_01.jpeg 112102_20180605160755_01.jpeg
    # 079908_20180306131723_01.jpeg 200001_20171023154452_01.jpeg
    # 079908_20180306132320_01.jpeg 239906_20171218134824_01.jpeg
    # 111200_20171110094949_01.jpeg 450001_20171222112431_01.jpeg
    # 111202_20171023145701_01.jpeg README.rst
    # 112101_20180613085423_01.jpeg




Model training and leaf extraction
==================================

Training is performed as follow steps:

    1.	We download the weights of U\ :sup:`2`-Net (0\ :sup:`th`-trained U\ :sup:`2`-Net) pretrained on the DUTS dataset from the corresponding GitHub repository (`xuebinqin/U-2-Net u2net.pth <https://github.com/xuebinqin/U-2-Net>`_) and use the 0\ :sup:`th`-trained U\ :sup:`2`-Net for leaf segmentation on the images in folder :file:`AIPI/data/images`.
    2.	After detection, we validate the results. The images in which the entire area is detected as a salient object (e.g., image 3) or no detection occurred (e.g., image 4) are discarded.
    3.	We use the remaining images and detection results (i.e., mask images) to train U\ :sup:`2`-Net.
    4.	We use the trained U\ :sup:`2`-Net from step 3 for salient object detection on the cucumber leaf images in folder :file:`AIPI/data/images`.
    5.	We repeat steps 2-4 to train U\ :sup:`2`-Net five times, finally obtaining the 5\ :sup:`th`-trained U\ :sup:`2`-Net.


.. image:: ../_static/u2net-iterative-training-process.png
    :width: 80%
    :align: center


As most steps are repeated five times,
we use the JustDeepIt API to efficiently repeat them automatically.
The executable Python scripts :file:`iterative_u2net.py` can be obtained from GitHub
(`JustDeepIt/tutorials/AIPI/scripts <https://github.com/biunit/JustDeepIt/tree/main/tutorials/AIPI/scripts>`_.
Training in this case study takes 4-5 days, and it depends on the computer hardware.


.. code-block:: sh
    
    # donwload images and put them into JustDeepIt/tutorials/AIPI/data/images folder
    # (user can test the following script without downloading
    #  since there are 10 sample images in the image folder.)
    python scripts/iterative_u2net.py




The results of leaf segmentation by the 5\ :sup:`th`-trained U\ :sup:`2`-Net
are stored in folder :file:`AIPI/data/images_5`
after running the :file:`iterative_u2net.py`.   
Examples of leaf segmentation of the 0\ :sup:`th`- and 5\ :sup:`th`-trained U\ :sup:`2`-Net
are shown in the figure below. 

.. image:: ../_static/tutorials_AIPI_output.jpg
    :width: 80%
    :align: center



References
==========

.. [#u2net] Qin X, Zhang Z, Huang C, Dehghan M, Zaiane O R, Jagersand M. U2-Net: Going Deeper with Nested U-Structure for Salient Object Detection. https://doi.org/10.1016/j.patcog.2020.107404

