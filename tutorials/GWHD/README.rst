====
GWHD
====


Dataset
=======

.. <dataset>

.. code-block:: sh
    
    # download tutorials/GWHD or clone JustDeepIt repository from https://github.com/jsun/JustDeepIt
    git clone https://github.com/jsun/JustDeepIt
    cd JustDeepIt/tutorials/GWHD
    
    # download image data (train.zip and test.zip) from http://www.global-wheat.com/ manually
    # and put train.zip and test.zip in JustDeepIt/tutorials/GWHD
    
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
    
    python scripts/run_justdeepit.py train 
    
    python scripts/run_justdeepit.py test






MMDetection
-----------


.. code-block:: sh
    
    python scripts/run_justdeepit.py train mmdetection
    
    python scripts/run_justdeepit.py test mmdetection
    




