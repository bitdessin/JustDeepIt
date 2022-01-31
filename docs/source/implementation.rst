==============
Implementation
==============


AgroLens implements a character user interface (CUI)
and graphical user interface (GUI) for the user to build deep learning models
for object detection, instance segmentation, and salient object detection.
The CUI is assumed to be used on computer clusters that only support this environment,
and it can be used via simple and intelligible application programming interfaces (APIs).
The GUI is interactively manipulated through the mouse and keyboard.
The basic GUI configuration is implemented using the `tkinter <https://docs.python.org/3/library/tkinter.html>`_ library,
and the user interface is drawn using the `ttkbootstrap <https://ttkbootstrap.readthedocs.io/en/latest/>`_ package.
The detailed CUI and GUI implementations for each task are presented in the following sections.



.. toctree::
    :maxdepth: 1

    implementation/od
    implementation/is
    implementation/sod


