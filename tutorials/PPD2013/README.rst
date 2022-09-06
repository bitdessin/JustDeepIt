=======
PPD2013
=======

Dataset
=======

.. <dataset>

.. code-block:: sh
    
    # download tutorials/PPD2013 or clone JustDeepIt repository from https://github.com/jsun/JustDeepIt
    git clone https://github.com/jsun/JustDeepIt
    cd JustDeepIt/tutorials/PPD2013

    # download dataset (Plant_Phenotyping_Datasets.zip) from http://www.plant-phenotyping.org/datasets
    # and put Plant_Phenotyping_Datasets.zip in JustDeepIt/tutorials/PPD2013
    
    # decompress data
    unzip Plant_Phenotyping_Datasets.zip
    
    # create folders to store images for training and inference
    mkdir -p trains
    mkdir -p masks
    mkdir -p tests

    # select 4 images and the corresponding mask images for training
    cp Plant_Phenotyping_Datasets/Tray/Ara2013-Canon/ara2013_tray01_rgb.png trains/ara2013_tray01.png
    cp Plant_Phenotyping_Datasets/Tray/Ara2013-Canon/ara2013_tray09_rgb.png trains/ara2013_tray09.png
    cp Plant_Phenotyping_Datasets/Tray/Ara2013-Canon/ara2013_tray18_rgb.png trains/ara2013_tray18.png
    cp Plant_Phenotyping_Datasets/Tray/Ara2013-Canon/ara2013_tray27_rgb.png trains/ara2013_tray27.png
    cp Plant_Phenotyping_Datasets/Tray/Ara2013-Canon/ara2013_tray01_fg.png masks/ara2013_tray01.png
    cp Plant_Phenotyping_Datasets/Tray/Ara2013-Canon/ara2013_tray09_fg.png masks/ara2013_tray09.png
    cp Plant_Phenotyping_Datasets/Tray/Ara2013-Canon/ara2013_tray18_fg.png masks/ara2013_tray18.png
    cp Plant_Phenotyping_Datasets/Tray/Ara2013-Canon/ara2013_tray27_fg.png masks/ara2013_tray27.png
    
    # use all images for inference
    cp Plant_Phenotyping_Datasets/Tray/Ara2013-Canon/*_rgb.png tests/



.. </dataset>



