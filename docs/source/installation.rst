============
Installation
============

JustDeepIt application is written in Python.
The stable releases of JustDeepIt have been tested
on the following OS under Python 3.8 and 3.9 environments:

- Ubuntu 20.04
- Debian 11.4
- macOS 12.3 (Intel / M2)

The source code and built package of JustDeepIt
are available through GitHub (`biunit/JustDeepIt <https://github.com/biunit/JustDeepIt>`_)
and Python Package Index (`PyPI: JustDeepIt <https://pypi.org/project/JustDeepIt/>`_), respectively.
Users can install JustDeepIt into a platform directly by using source code or PyPI package.


Installation via PyPI 
---------------------

JustDeepIt requires one of
`MMDetection <https://mmdetection.readthedocs.io/en/latest/>`_ or
`Detectron2 <https://detectron2.readthedocs.io/en/latest/>`_
to build object detection model.
Thus, before the installation of JustDeepIt,
we recommend installing MMDetection and Detectron2
(and also `PyTorch <https://pytorch.org/>`_) first with
the recommended procedures depending on CUDA version.
The following is an example of installing
PyTorch, MMDetection, and Detectron2 under a CUDA 11.3 environment.


.. code-block:: bash
    
    pip install --upgrade pip
    
    # PyTorch
    pip install torch==1.11.0 torchvision --extra-index-url https://download.pytorch.org/whl/cu113
    
    # MMDetection
    pip install mmcv-full
    git clone https://github.com/open-mmlab/mmdetection.git mmdetection
    pip install -r ./mmdetection/requirements/build.txt
    pip install ./mmdetection
    
    # Detectron2
    pip install 'git+https://github.com/facebookresearch/fvcore'
    git clone https://github.com/facebookresearch/detectron2 detectron2
    pip install ./detectron2 


Then, install the latest stable version of JustDeepIt from PyPI with :code:`pip` command.


.. code-block:: bash
    
    pip install justdeepit




Installation via source code
----------------------------

To install JustDeepIt from source, first install PyTroch,
MMDetection, and Detectron2 following the above description,
and then run the following commands to download and install JustDeepIt with its dependencies.

.. code-block:: bash
    
    git clone https://github.com/biunit/JustDeepIt.git JustDeepIt
    cd JustDeepIt
    
    pip install -r requirements.txt
    pip install .





