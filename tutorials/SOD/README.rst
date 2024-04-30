========================
Salient object detection
========================

JustDeepIt supports users to perform object detection, instance segmentation,
and salient object detection with GUI or CUI.
In this tutorial, to overview of functions for salient object detection,
we showed the usage of JustDeepIt for salient object detection with an artificial dataset.


Dataset
=======


The artificial dataset used for this quick start is stored in
GitHub (`JustDeepIt/tutorials/SOD <https://github.com/biunit/JustDeepIt/tree/main/tutorials/SOD>`_).
The :file:`data` folder contains :file:`images` and :file:`masks` folders.
The :file:`images` folder contains training images,
and the :file:`masks` folder contains mask images (annotation).
The user can use :code:`git` command to download the dataset from GitHub with the following script.


.. code-block:: sh
    
    git clone https://github.com/biunit/JustDeepIt

    ls JustDeepIt/tutorials/SOD
    # data run_u2net.py

    ls JustDeepIt/tutorials/SOD/data
    # images masks


.. image:: ../_static/quickstart_sod_data.jpg
    :align: center




Settings
========



To start JustDeepIt, we open the terminal,
as the following commands,
change the current directory to :file:`JustDeepIt/tutorials/SOD`,
and run :code:`justdeepit` command.



.. code-block:: sh

    cd JustDeepIt/tutorials/SOD

    justdeepit
    # INFO:uvicorn.error:Started server process [61]
    # INFO:uvicorn.error:Waiting for application startup.
    # INFO:uvicorn.error:Application startup complete.
    # INFO:uvicorn.error:Uvicorn running on http://127.0.0.1:8000 (Press CTRL+C to quit)


Then, we open the web browser and accesss to \http://127.0.0.1:8000.
At the startup screen, we press "Salient Object Detection" button
to start salient object detection mode.



.. image:: ../_static/app_main.png
    :width: 70%
    :align: center


Next, at the **Preferences** screen,
we set parameters as shown in the screenshot below.
The **workspace** will be automatically set as the path of the current folder
(e.g., :file:`JustDeepIt/tutorials/SOD`, depending on the user's environment).
Then, we press button **Load Workspace**.


.. image:: ../_static/quickstart_sod_pref.png
    :align: center

Once the **Preferences** is set,
the functions of **Training** and **Inference** become available.



Training
========


To train the model,
we select tab **Training**
and specify the **model weight** as the location storing the training weight,
**image folder** as the folder containing training images (i.e., :file:`data/images`),
**annotation** as the folder containing mask images (i.e., :file:`data/masks`),
and **annotation format** as ``mask``.
The other parameters are set as shown in screenshot below.
Note that the values of **model weight**, **image folder**, and **annotation** may be
different from the screenshot depending on user's environment.
Then, we press button **Start Training** for model training.



.. image:: ../_static/quickstart_sod_train.png
    :align: center


Training takes 5-20 minutes, and it depends on the computer hardware.


Inference
=========

In tab **Inference**, the **model weight** is specified to the training weights,
whose file extension is :file:`.pth` in general.
We specify **image folder** to the folder
containing the images for inference.
Here, for convenience, we use the training images (e.g., :file:`data/images`) for inference.
Note that the values of **model weight** and **image folder** may be
different from the screenshot depending on user's environment.
Then, we press button **Start Inference** for inference.


.. image:: ../_static/quickstart_sod_eval.png
    :align: center


The inference results will be stored in the folder :file:`justdeepitws/outputs` of the workspace.
Examples of inference results are shown in the figure below.
Black background indicates that there is no objects.


.. image:: ../_static/quickstart_sod_inference_output.png
    :width: 70%
    :align: center




API
====


Training and inference can be performed using the JustDeepIt API.
Python script :file:`run_justdeepit.py` stored in GitHub
(`JustDeepIt/tutorials/SOD <https://github.com/biunit/JustDeepIt/tree/main/tutorials/SOD>`_)
can be used for this purpose.


.. code-block:: sh

    cd JustDeepIt/tutorials/SOD

    python scripts/run_justdeepit.py



