==============
Implementation
==============


JustDeepIt implements a graphical user interface (GUI) and character user interface (CUI)
to solve various image analysis problems using deep learning technologies
including for object detection, instance segmentation, and salient object detection.
The GUI is implemented using the `FastAPI <https://fastapi.tiangolo.com/>`_.
It allows users to perform image analysis using deep learning 
with simple mouse and keyboard operations.
The CUI can be used via application programming interfaces
which contains a few straightforward functions to simplify its usage.
The CUI is assumed to be used on computer clusters that only support this environment,
The detailed CUI and GUI implementations for each task are presented in the following sections.



.. toctree::
    :maxdepth: 1

    implementation/object-detection
    implementation/instance-segmentation
    implementation/salient-object-detection


