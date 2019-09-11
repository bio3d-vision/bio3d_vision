"""
setup.py
Set up the installation of the `bio3d_vision` package using Distutils.

Matthew Guay <matthew.guay@nih.gov>
"""
from distutils.core import setup
from setuptools import find_packages

# Get README text
with open('README.md') as f:
    readme = f.read()

# Get license text
with open('LICENSE') as f:
    the_license = f.read()

setup(
    name='bio3d_vision',
    version='0.1',
    description='Download and manipulate datasets from the bio3d-vision collection.',
    long_description=readme,
    author='Matthew Guay',
    author_email='matthew.guay@nih.gov',
    url='https://github.com/bio3d-vision/bio3d_vision',
    license=the_license,
    packages=find_packages(exclude=('examples')),
    install_requires=['matplotlib', 'numpy', 'scipy', 'tifffile']
)
