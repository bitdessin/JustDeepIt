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
    'JustDeepIt aims to support researchers by facilitating the use of deep learning for object detection and segmentation '\
    'by providing both graphical user interface (GUI) and CUI operations. '\
    'JustDeepIt can be used for plant detection, pest detection, '\
    'and a variety of tasks in life sciences, agriculture, and other fields.'



with open(os.path.join(os.path.dirname(__file__), 'justdeepit', '__init__.py'), encoding='utf-8') as fh:
    for line in fh:
        if line.startswith('__version__'):
            exec(line)
            break


install_requirements = []
with open('requirements.txt') as fh:
    install_requirements = fh.read().splitlines()


setup(
    name        = 'JustDeepIt',
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
    author_email = 'sun@biunit.dev',
    url          = 'https://github.com/biunit/JustDeepIt',
    license      = 'MIT',
    packages     = find_packages(),
    entry_points={'console_scripts': [
                        'justdeepit=justdeepit.webapp.app:run_app',
                    ]},
    include_package_data = True,
    zip_safe = True,
    long_description = package_desc,
    install_requires = install_requirements,
)

