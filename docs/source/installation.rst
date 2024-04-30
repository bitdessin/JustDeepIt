============
Installation
============

JustDeepIt application is written in Python.
The stable releases of JustDeepIt have been tested
on the following OS under Python 3.11 environment:

- Ubuntu 22.04
- macOS 12.3 (Intel / M2)

The source code and built package of JustDeepIt
are available through GitHub (`biunit/JustDeepIt <https://github.com/biunit/JustDeepIt>`_)
and Python Package Index (`PyPI: JustDeepIt <https://pypi.org/project/JustDeepIt/>`_), respectively.

JustDeepIt requires
`PyTorch <https://pytorch.org/>`_ and
`MMDetection <https://mmdetection.readthedocs.io/en/latest/>`_
to build object detection model.
Thus, before the installation of JustDeepIt,
we recommend installing PyTorch and MMDetection first with
the recommended procedures depending on CUDA version.
The following is an example of installing
PyTorch and MMDetection under a CUDA 11.8 environment.


.. code-block:: bash
    
    pip install --upgrade pip
    
    # PyTorch (under CUDA 11.8 environment)
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

    # MMDetection (mmcv, mmdet)
    pip install -U openmim
    mim install mmengine
    mim install "mmcv>=2.0.0"
    mim install "mmdet>=3.0.0"


Then, install the latest stable version of JustDeepIt from PyPI with :code:`pip` command.


.. code-block:: bash
    
    pip install justdeepit





