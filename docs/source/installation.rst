============
Installation
============

AgroLens application is written in Python programming language.
The stable releases of AgroLens have been tested on the following system
under Python 3.7 and 3.8 environments:

* Ubuntu 12.04
* macOS 11.3 (Intel)

The stable releases of AgroLens are available through
Python Package Index (`PyPI: AgroLens <https://pypi.org/project/AgroLens/>`_).
Before the installation of AgroLens,
it is recommended to install `PyTorch <https://pytorch.org/>`_,
`MMDetection <https://mmdetection.readthedocs.io/>`_,
`Detectron2 <https://detectron2.readthedocs.io/>`_ first
with the recommended procedures according to CUDA version, for example:

.. code-block:: bash
    
    # PyTorch
    pip install torch torchvision
    
    # MMDetection
    pip install openmim
    mim install mmdet
    
    # Detectron2
    git clone https://github.com/facebookresearch/detectron2.git
    python -m pip install -e detectron2


Note that, AgroLens requires one of MMDetection or Detectron2 to
build object detection model.
Thus, user can install prefer one instead of installtion of both packages.

Then, to install the latest stable version from PyPI with :code:`pip` command.

.. code-block:: bash
    
    pip install agrolens


The source code of the devel version of AgroLens is available on GitHub
`biunit/AgroLens <https://github.com/biunit/AgroLens>`_.
To install the devel version from source,
run the following commands to download and install AgroLens with its dependencies.


.. code-block:: bash
    
    git clone git@github.com:biunit/AgroLens.git
    cd AgroLens
    
    pip install -r requirements.txt
    
    python setup.py build
    python setup.py install


