#!/usr/bin/env python

import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name='arabidopsis-rosette-analysis',
    version='0.3.0',
    description='Extract traits from top-view images of Arabidopsis plants. ',
    long_description=long_description,
    long_description_content_type="text/markdown",
    author='Suxing Liu',
    author_email='suxing.liu@uga.edu',
    license='BSD-3-Clause',
    url='https://github.com/Computational-Plant-Science/arabidopsis-rosette-analysis',
    packages=setuptools.find_packages(),
    include_package_data=True,
    entry_points={
        'console_scripts': [
            'arabidopsis = core.cli:cli'
        ]
    },
    python_requires='>=3.6.8',
    install_requires=[
        'click==7.1.2',
        'psutil==5.8.0',
        'numba==0.53.1',
        'pandas==1.2.4',
        'networkx==2.5.1',
        'skan==0.9',
        'tabulate==0.8.9',
        'imutils==0.5.3',
        'python-magic',
        'seaborn==0.11.1',
        'openpyxl==3.0.5',
        'opencv-python==4.4.0.46',
        'matplotlib==3.3.3',
        'scikit-learn==0.24.0',
        'scikit-image==0.18.1',
        'scikit-build==0.11.1',
        'scipy==1.5.2',
        'Pillow==8.1.1'
    ],
    setup_requires=['wheel'],
    tests_require=['pytest', 'coveralls'])
