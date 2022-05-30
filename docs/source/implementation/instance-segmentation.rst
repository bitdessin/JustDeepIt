=====================
Instance Segmentation
=====================


Instance segmentation determines the pixelwise mask for each object in an image.
JustDeepIt internally calls the MMDetection or Detectron2 library
to build instance segmentation models and perform model training and image segmentation.
The latest version of JustDeepIt supports Mask R-CNN\ [#maskrcnn]_.




GUI
===


The GUI window for instance segmentation consists of three tabs:
**Preferences**, **Training**, and **Inference**.
These tabs are used for setting common parameters,
training models,
and inference (i.e., instance segmentation) from the test
images using the trained model, respectively.
Tabs **Training** and **Inference** are disabled
until the settings in tab **Preferences** are defined.



Preferences
-----------

Tab **Preferences** is used for setting common parameters,
such as the architecture of the segmentation model,
number of CPUs and GPUs to be used,
and the location (i.e., directory path) to the workspace
which is used to save intermediate and final results.
Detailed descriptions of the arguments are provided in the following table.



.. image:: ../_static/app_is_pref.png
    :align: center



.. csv-table::
    :header: "Argument", "Description"
    
    "**backend**", "The backend to build an instance segmentation model.
    The current version of JustDeepIt supports MMDetection and Detectron2 as a backend."
    "**architecture**", "Architecture of instance segmentation model."
    "**config**", "A path to a configuration file of MMDetection or Detectron2.
    If the path is not given, then use the default configuration file defined in JustDeepIt."
    "**class label**", "A path to a text file which contains class labels.
    The file should be multiple rows with one column,
    and string in each row represents a class label
    (e.g., `class_label.txt <https://github.com/biunit/JustDeepIt/blob/main/tutorials/IS/data/class_label.txt>`_)."
    "**CPU**", "Number of CPUs."
    "**GPU**", "Number of GPUs."
    "**workspace**", "Workspace to store intermediate and final results."
 

Once the parameters are set and the workspace is loaded,
the initial configuration file is stored in the workspace
(:file:`justdeepitws/config`).
The configuration file is named
:file:`default.py` when using MMDetection as a backend
or :file:`default.yaml` when using Detectron2 as a backend.
The configuration file can be manually modified as needed
prior to training.


Training
--------

Tab **Training** is used to train the model for instance segmentation.
It allows users to set general parameters of training,
such as the learning rate, batch size, and number of epochs.
Detailed descriptions of the arguments are provided in the following table.
Furthermore, to set detailed parameters, such as model architectures and loss functions,
users can directly modify the configuration file in the workspace,
which is generated when setting the arguments in tab **Preferences**
(i.e., :file:`default.py` or :file:`default.yaml`),
prior to training as necessary.
However, actually, according to our experience,
we recommend that users increase the number of training images
rather than changing such complex parameters to improve the inference accuracy.


.. image:: ../_static/app_is_train.png
    :align: center




.. csv-table::
    :header: "Argument", "Description"
    
    "**model weight**", "A path to store the model weight.
    If the file is exists, then resume training from the given weight."
    "**image folder**", "A path to a folder which contains training images."
    "**annotation format**", "Annotation format."
    "**annotation**", "A path to a file (COCO format)."
    "**batch size**", "Batch size."
    "**learning rate**", "Initial learning rate."
    "**epochs**", "Number of epochs."
    "**cutoff**", "Cutoff of confidence score for training."



Inference
---------

Tab **Inference** is used for instance segmentation from test images using the trained model.
It allows the user to set the confidence score of instance segmentation results and batch size.


.. image:: ../_static/app_is_eval.png
    :align: center


.. csv-table::
    :header: "Argument", "Description"
    
    "**model weight**", "A path to a trained model weight."
    "**image folder**", "A path to a folder contained multiple test images."
    "**batch size**", "Batch size."
    "**cutoff**", "Cutoff of confidence score for inference (i.e., instance segmentation)."
    




CUI
===


JustDeepIt implements three simple methods,
:meth:`train <justdeepit.models.IS.train>`,
:meth:`save <justdeepit.models.IS.save>`,
and :meth:`inference <justdeepit.models.IS.inference>`.
:meth:`train <justdeepit.models.IS.train>` is used for training the models,
while :meth:`save <justdeepit.models.IS.save>` is used for saving the trained weight,
and :meth:`inference <justdeepit.models.IS.inference>` is used for instance segmentation in test images.
Detailed descriptions of these functions are provided below.


Architectures
-------------

To initialize a neural network architecture for instance segmentation,
class :class:`justdeepit.models.IS <justdeepit.models.IS>` with
the corresponding arguments can be used.
For example, to initialize a Mask R-CNN\ [#maskrcnn]_ architecture with random initial weight,
MMDetection (``mmdetection``) or Detectron2 (``detectron2``)
can be used as the backend for building the model architecture.
Currently, only Mask R-CNN is supported.


.. code-block:: py

    from justdeepit.models import IS

    model = IS('./class_label.txt', model_arch='maskrcnn')


To initialize a Mask R-CNN architecture with the specified trained weight
(e.g., the weight pre-trained by COCO dataset),
users can use argument ``model_weight`` during initialization.
Note that, the pre-trained weight file (:file:`.pth`)
can be downloaded from the GitHub repositories of
`MMDetection <https://github.com/open-mmlab/mmdetection/tree/master/configs>`_
or `Detectron2 <https://github.com/facebookresearch/detectron2/tree/main/configs>`_.


.. code-block:: py

    from justdeepit.models import IS

    weight_fpath = '/path/to/pretrained_weight.pth'
    model = IS('./class_label.txt', model_arch='maskrcnn', model_weight=weight_fpath)


The available architectures for instance segmentation
can be checked by executing the following code.


.. code-block:: py

    from justdeepit.models import IS
    model = IS()
    print(model.available_architectures)




Training
--------

Method :meth:`train <justdeepit.models.IS.train>` is used for the model training
and requires at least two arguments
to specify the annotations and folder containing the training images.
Annotations can be specified in a single file in the COCO format.
Training process requires a GPU environment if MMDetection is chosen as the backend
because it only supports GPU training.
Refer to the API documentation of :meth:`train <justdeepit.models.IS.train>`
for detailed usage.



.. code-block:: py

    from justdeepit.models import IS

    coco_fmt = '/path/to/coco/annotation.json'
    train_images_dpath = '/path/to/folder/images'

    model = IS('./class_label.txt', model_arch='maskrcnn')

    model.train(coco_fmt, train_images_dpath)




The trained weight can be saved using method :meth:`save <justdeepit.models.IS.save>`,
which simultaneously stores the trained weight (extension :file:`.pth`)
and model configuration file (extensions :file:`.py` for MMDetection backend and :file:`.yaml` for Detectron2 backend).
The users can apply the weight and configuration file as needed
for generating a model using the MMDetection or Detectron2 library directly.
Refer to the API documentation of :meth:`save <justdeepit.models.IS.save>`
for detailed usage.


.. code-block:: py

    model.save('trained_weight.pth')





Inference
---------

Method :meth:`inference <justdeepit.models.IS.inference>`
is used to perform instance segmentation against the test images using the trained model.
This method requires at least one argument to specify a single image,
list of images, or folder containing multiple images.
The segmentation results are returned as class object
:class:`justdeepit.utils.ImageAnnotations <justdeepit.utils.ImageAnnotations>`,
which is a list of class objects :class:`justdeepit.utils.ImageAnnotation <justdeepit.utils.ImageAnnotation>`.


To save the results in the COCO format,
we can use method :meth:`format <justdeepit.utils.ImageAnnotations.format>`
implemented in class :class:`justdeepit.utils.ImageAnnotations <justdeepit.utils.ImageAnnotations>` to represent a JSON file in the COCO format.



.. code-block:: py

    from justdeepit.models import IS

    test_images = ['sample1.jpg', 'sample2.jpg', 'sample3.jpg']

    model = IS('./class_label.txt', model_arch='maskrcnn', model_weight='trained_weight.pth')
    outputs = model.inference(test_images)

    outputs.format('coco', './predicted_outputs.coco.json')




To save the segmentation results as images, for example,
showing the detected contours and bounding boxes on the images, method :meth:`draw <justdeepit.utils.ImageAnnotation.draw>`
implemented in class :class:`justdeepit.utils.ImageAnnotation <justdeepit.utils.ImageAnnotation>` can be used.



.. code-block:: py
    
    for output in outputs:
        output.draw('bbox+contour', os.path.join('./predicted_outputs', os.path.basename(output.image_path)))


Refer to the corresponding API documentation of
:meth:`inference <justdeepit.models.IS.inference>`,
:meth:`format <justdeepit.utils.ImageAnnotations.format>`, and
:meth:`draw <justdeepit.utils.ImageAnnotation.draw>`,
for the detailed usage.




References
===========

.. [#maskrcnn] He K, Gkioxari G, Dollár P, Girshick R. Mask R-CNN. https://arxiv.org/abs/1703.06870



