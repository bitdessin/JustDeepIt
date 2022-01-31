from setuptools import setup, find_packages
import os
import locale




package_desc = 'Deep learning has been applied to solve various problems, especially in image recognition, '\
    'across many fields including the life sciences and agriculture. '\
    'Many studies have reported the use of deep learning to identify plant and insect species, '\
    'detect flowers and fruits, segment plant individuals from fixed-point observation cameras or drone images, '\
    'and other applications. '\
    'Programming languages such as Python and its libraries including PyTorch, MMDetection, and Detectron2 ' \
    'have made deep learning easier and more accessible to researchers. '\
    'However, it remains difficult for many researchers without advanced programming skills '\
    'to use deep learning and environments such as the character user interface (CUI) through keyboard input. '\
    'AgroLens aims to support researchers by facilitating the use of deep learning for object detection and segmentation '\
    'by providing both graphical user interface (GUI) and CUI operations. '\
    'AgroLens can be used for plant detection, pest detection, '\
    'and a variety of tasks in life sciences, agriculture, and other fields.'


with open(os.path.join(os.path.dirname(__file__), 'agrolens', '__init__.py'), encoding='utf-8') as fh:
    for buf in fh:
        if buf.startswith('__version__'):
            exec(buf)
            break


setup(
    name        = 'AgroLens',
    version     = __version__,
    description = 'a GUI tool for object detection and segmentation based on deep learning',
    classifiers = [
        'Development Status :: 5 - Production/Stable',
        'Environment :: Console',
        'Environment :: X11 Applications',
        'Intended Audience :: Science/Research',
        'Natural Language :: English',
        'Operating System :: Unix',
        'Programming Language :: Python :: 3',
        'Topic :: Scientific/Engineering :: Image Recognition',
    ],
    keywords     = 'object detection, object segmentation',
    author       = 'Jianqiang Sun',
    author_email = 'jsun@aabbdd.jp',
    url          = 'https://github.com/biunit/AgroLens',
    license      = 'MIT',
    packages     = find_packages(),
    package_dir  = {'agrolens': 'agrolens'},
    entry_points={'console_scripts': [
                        'agrolens=agrolens.bin.app:main',
                    ]},
    include_package_data = True,
    zip_safe = True,
    long_description = package_desc,
    install_requires = [
        'tqdm', 'joblib', 'ttkbootstrap',
        'numpy', 'pandas',
        'pillow', 'opencv-python>=4.5.1', 'scikit-image>=0.18.1',
        'torch>=1.8', 'torchvision'
    ],
    
)

