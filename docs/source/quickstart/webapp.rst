===============
Web Application
===============

One of JustDeepIt’s unique features is its support for both GUI and CUI.
For example, a researcher who is unfamiliar with programming
can create models for object detection in GUI.
The researcher can then work with a systems engineer
to build a web application for object detection on the server
which usually only supports CUI.
This tutorial demonstrates how to build a simple web application using Flask.


.. warning::

    Source code shown in this section is only an example;
    measures to address various security issues must be taken
    to build a full-fledged web application.
    


Preparation
===========


Source code
-----------

The source code for this tutorial is available on GitHub
(`JustDeepIt/tutorials/WebApp <https://github.com/biunit/JustDeepIt/tree/main/tutorials/WebApp>`_)
and can be downloaded using the following command as necessary.

.. code-block:: sh
    
    git clone https://github.com/biunit/JustDeepIt
    
    ls JustDeepIt/tutorials/WebApp
    # app.py  class_label.txt  templates  uploads



Package installation
--------------------

Flask is a Python package for building web application systems.
It can be installed using the following command.

.. code-block:: sh
    
    pip install flask



Model
-----


We refer to quick start for :doc:`od` to train the model,
then we obtain the :file:`class_label.txt` file and :file:`fasterrcnn.pth` file.
To skip this step, we run the following command to download the trained model
(i.e., :file:`fasterrcnn.pth`) into the tutorial folder.


.. code-block:: sh
    
    # git clone https://github.com/biunit/JustDeepIt
    cd JustDeepIt/tutorials/WebApp
    wget https://biunit.dev/src/justdeepit.fasterrcnn.pth -O fasterrcnn.pth




Web Application
===============

For simplicity, we create a web application with minimal functionality,
allowing users to upload images
perform object detection, and show the results on the web page.
We create an :file:`app.py` file in the same folder as :file:`class_label.txt`
and write the following code.


.. literalinclude:: ../../../tutorials/WebApp/app.py


The :code:`detection_app` function first provides an image upload form.
Once the image is uploaded,
the function performs object detection on the image using the :code:`model`,
saves the detection result, and displays the result on the web page.

We then create an HTML template :file:`index.html` into :file:`templates` folder
for uploading images and showing the detection results as follows.


.. literalinclude:: ../../../tutorials/WebApp/templates/index.html


Finally, we run the following command to launch the web application.


.. code-block:: sh
    
    FLASK_APP=app.py flask run





