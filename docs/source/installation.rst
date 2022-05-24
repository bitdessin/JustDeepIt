============
Installation
============

JustDeepIt application is written in Python programming language.
The stable releases of JustDeepIt have been tested
on the following OS under Python 3.8 and 3.9 environments:

- Ubuntu 20.04
- macOS 12.3 (Intel)

The source code and built package of JustDeepIt
are available through GitHub (`biunit/JustDeepIt <https://github.com/biunit/JustDeepIt>`_)
and Python Package Index (`PyPI: JustDeepIt <https://pypi.org/project/JustDeepIt/>`_), respectively.
A Dockerfile is also provided to make it easier
for users to set up the environment for JustDeepIt.
Users can install JustDeepIt into a platform directly by using source code
or PyPI package or install into virtual container using Dockerfile.


Installation via PyPI 
---------------------

JustDeepIt requires one of
`MMDetection <https://mmdetection.readthedocs.io/en/latest/>`_ or
`Detectron2 <https://detectron2.readthedocs.io/en/latest/>`_
to build object detection model.
Thus, we recommend installing MMDetection and Detectron2
(and also `PyTorch <https://pytorch.org/>`_) first with
the recommended procedures according to CUDA version, before the installation of JustDeepIt.
The following is an example of installing the latest versions of
PyTorch, MMDetection, and Detectron2 under a CUDA 11.3 environment.


.. code-block:: bash
    
    pip install --upgrade pip
    
    # PyTorch
    pip install torch==1.11.0 torchvision --extra-index-url https://download.pytorch.org/whl/cu113
    
    # MMDetection
    pip install mmcv-full -f https://download.openmmlab.com/mmcv/dist/cu113/torch1.11.0/index.html
    git clone https://github.com/open-mmlab/mmdetection.git mmdetection
    pip install -r ./mmdetection/requirements/build.txt
    pip install ./mmdetection
    
    # Detectron2
    pip install 'git+https://github.com/facebookresearch/fvcore'
    git clone https://github.com/facebookresearch/detectron2 detectron2
    pip install ./detectron2 


.. note::
    
    JustDeepIt requires one of MMDetection or Detectron2
    to build object detection model.
    Thus, user can also install one instead of both packages.
    Currently, MMDetection supports more architectures than Detectron2
    and supports GPUs training.
    Detectron2 supports training with both CPUs and GPUs.
   

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



Installation via Docker
------------------------

Dockerfile for building a Docker image of JustDeepIt is available on
GitHub (`biunit/JustDeepIt/docker <https://github.com/biunit/JustDeepIt/tree/main/docker>`_)
and can be downloaded to build the image using the following commands:

.. code-block:: bash
    
    docker build -t justdeepit:local --no-cache  \
        --build-arg cuda=11.6.2 \
        --build-arg cudnn=8 \
        --build-arg platform=ubuntu20.04 .


Note that the versions of :code:`cuda`, :code:`cudnn`, and :code:`platform`
should be changed according to the individual platform environments.
Installation takes approximately 20 minutes,
depending on the computer hardware and network speed.

The version of :code:`cuda` can be checked using the following command.
In this case, the version of :code:`cuda` is 11.6.
The user can check the latest driver version for cuda 11.6 from
`Dockerhub:nvdia/cuda <https://hub.docker.com/r/nvidia/cuda>`_.
Currently, the latest driver version is 11.6.2.

.. code-block:: bash
    
    nvcc -V
    # nvcc: NVIDIA (R) Cuda compiler driver
    # Copyright (c) 2005-2022 NVIDIA Corporation
    # Built on Tue_Mar__8_18:18:20_PST_2022
    # Cuda compilation tools, release 11.6, V11.6.124
    # Build cuda_11.6.r11.6/compiler.31057947_0


The version of :code:`cudnn` can be checked using the following command.
In this case, the major version of :code:`cudnn` is 8.

.. code-block:: bash
    
    dpkg -l | grep "cudnn"
    # ii  cudnn-local-repo-ubuntu2004-8.4.0.27 1.0-1 amd64
    
The version of :code:`platform` can be checked using the following command.
In this case, the version of :code:`platform` is Ubuntu 20.04 (ubuntu20.04).

.. code-block:: bash
    
    uname -v 
    # 44~20.04.1-Ubuntu SMP Thu Mar 24 16:43:35 UTC 2022


After building the Docker image,
the Docker container generated from the image
can be started by running the following command.

.. code-block:: bash
    
    docker image ls
    # REPOSITORY    TAG                               IMAGE ID       CREATED        SIZE
    # justdeepit    local                             19bcfd96c278   1 hours ago    14.7GB
    # nvidia/cuda   11.6.2-cudnn8-devel-ubuntu20.04   d64238d69fda   3 weeks ago    7.7GB
    
    docker run --gpus all -v $(pwd):/home/appuser -p 8000:8000 --rm -it justdeepit:local


Then, JustDeepIt can be started by executing the folloiwng command on the Docker container.
In this case, JustDeepIt can be accessed via web browser at \http://0.0.0.0:8000.

.. code-block:: bash
    
    justdeepit --host 0.0.0.0 --port 8000
    # INFO:uvicorn.error:Started server process [61]
    # INFO:uvicorn.error:Waiting for application startup.
    # INFO:uvicorn.error:Application startup complete.
    # INFO:uvicorn.error:Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)



