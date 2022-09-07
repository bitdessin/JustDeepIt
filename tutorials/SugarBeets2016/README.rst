==============
SugarBeets2016
==============

Dataset
=======


.. <dataset>

.. code-block:: sh
    
    # download tutorials/SugarBeets2016 or clone JustDeepIt repository from https://github.com/jsun/JustDeepIt
    git clone https://github.com/jsun/JustDeepIt
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
    
    
.. </dataset>






Model training and validation
=============================


Detectron2
----------

.. <script>

.. code-block:: sh

    python scripts/run_justdeepit.py train

    python scripts/run_justdeepit.py test


.. </script>



MMDetection
-----------


.. code-block:: sh

    python scripts/run_justdeepit.py train mmdetection

    python scripts/run_justdeepit.py test mmdetection





