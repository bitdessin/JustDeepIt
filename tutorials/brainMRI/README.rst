===============
Tumor Detection
===============


Dataset
=======

.. <dataset>

.. code-block:: sh
    
    git clone https://github.com/jsun/JustDeepIt
    cd JustDeepIt/tutorials/brainMRI
    
    # download dataset from https://www.kaggle.com/mateuszbuda/lgg-mri-segmentation
    # and put archive.zip in JustDeepIt/tutorials/brainMRI
    unzip archive.zip
    
    mkdir -p trains
    mkdir -p masks
    mkdir -p tests
    
    # randomly select 90% patients for training and 10% for validation
    python scripts/split_train_and_valid.py kaggle_3m .


.. </dataset>




