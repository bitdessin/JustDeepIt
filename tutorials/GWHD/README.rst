====
GWHD
====


Dataset
=======

.. <dataset>

.. code-block:: sh
    
    # download tutorials/GWHD or clone AgroLens repository from https://github.com/biunit/AgroLens
    git clone https://github.com/biunit/AgroLens
    cd AgroLens/tutorials/GWHD
    
    # download image data (train.zip) from http://www.global-wheat.com/ manually
    # and put train.zip in AgroLens/tutorials/GWHD
    
    # decompress data
    unzip train.zip
    unzip test.zip
    
    # generate COCO format annotation from GWHD format (CSV format) annotation
    python scripts/gwhd2coco.py ./train train.csv train.json
    
    # make class label
    echo "spike" > class_label.txt


.. </dataset>



Model training and validation
=============================

Detectron2
----------

.. code-block:: sh
    
    python scripts/run_detectron2.py train
    
    python scripts/run_detectron2.py valid



MMDetection
-----------

.. <script>

.. code-block:: sh
    
    python scripts/run_mmdet.py train
    
    python scripts/run_mmdet.py test
    

.. </script>



