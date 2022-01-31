====
AIPI
====


Dataset
=======

Dataset can be downloaded from `病害虫被害画像データベース <https://www.naro.affrc.go.jp/org/niaes/damage/>`_
by manual or by using :command:`wget` command.
Actually, in this tutorial, we only download images from the following two pages.

- `Cucumber leaf surface <https://www.naro.affrc.go.jp/org/niaes/damage/image_db/03_%E3%82%AD%E3%83%A5%E3%82%A6%E3%83%AA-%E8%91%89%EF%BC%88%E8%A1%A8%EF%BC%89.html>`_
- `Cucumber leaf abaxial side <https://www.naro.affrc.go.jp/org/niaes/damage/image_db/03_%E3%82%AD%E3%83%A5%E3%82%A6%E3%83%AA-%E8%91%89%EF%BC%88%E8%A3%8F%EF%BC%89-%E3%83%AF%E3%82%BF%E3%82%A2%E3%83%96%E3%83%A9%E3%83%A0%E3%82%B7.html>`_

Put all images into :file:`data/images`.

Then download the pre-trained U2-Net weight (`Google Drive <https://drive.google.com/file/d/1ao1ovG1Qtx4b7EoskHXmi2E9rp5CHLcZ/view?usp=sharing>`_) shown on `GitHub <https://github.com/xuebinqin/U-2-Net>`_ into :file:`weights` named :file`u2net.0.pth`.



Training
========

Iteratively train U2-Net to detect leaves by using AgroLens API.

.. code-block:: sh
    
    python iterative_u2net.py




