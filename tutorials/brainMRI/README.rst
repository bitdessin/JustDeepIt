===============
Tumor Detection
===============


Dataset
=======

.. <dataset>

.. code-block:: sh
    
    git clone https://github.com/biunit/AgroLens
    cd AgroLens/tutorials/TCGAbrainMRI
    
    # download dataset from https://www.kaggle.com/mateuszbuda/lgg-mri-segmentation
    # and put archive.zip in AgroLens/tutorials/TCGAbrainMRI
    unzip archive.zip

    bash make_mask.sh
    
    mkdir -p inputs/train_images
    mkdir -p inputs/query_imagse

    mv kaggle_3m/TCGA_DU_*/*.png inputs/train_images/

    mv inputs/train_images/TCGA_DU_5*_image.png inputs/query_images/
    rm inputs/train_images/TCGA_DU_5*_mask.png
    


.. </dataset>




