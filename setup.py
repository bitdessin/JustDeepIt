import os
import setuptools

def get_version():
    with open(os.path.join(os.path.dirname(__file__), 'src', 'justdeepit', '__init__.py'), encoding='utf-8') as fh:
        for line in fh:
            if line.startswith('__version__'):
                return line.split('=')[-1].strip().strip("'")

setuptools.setup(
    version=get_version(),
)

