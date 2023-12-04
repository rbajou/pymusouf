#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from setuptools import setup, find_packages
import glob
from pathlib import Path

REQUIREMENTS = [
    'pandas',
    'numpy',
    'scipy',
    'matplotlib',
    'scikit-learn',
    'scikit-image',
    'pyjson',
    'pylandau',  #used in 'utils/functions.py'
    'iminuit', #used in 'utils/functions.py'
    'pillow', #for animation
    'pyyaml',
    'palettable', 
    'requests', 
    'bs4', 
    'cx_Freeze', #make executable files
    'mat73', #read v7.3 mat files
]

tel_files_path = Path(__file__).parents[0] / 'files' / 'telescopes'
tel_list = [t.split('/')[-1] for t in  glob.glob(str(tel_files_path) + '/*')]

setup(
    name='pymusouf',
    version='0.1.0',
    description="",
    author="Raphaël Bajou",
    author_email='r.bajou2@gmail.com',
    url='https://github.com/rbajou/pymusouf.git',
    packages=find_packages(), #['muon_tracking'],#
    package_dir={
        'pymusouf': [
                     'exe', 
                     'files', 
                     'forwardsolver', 
                     'modeling3d', 
                     'muo2d',  
                     'raypath', 
                     'reco',
                     'scripts', 
                     'telescope', 
                     'test', 
                     'tracking',
                     'utils']
    },  
    package_data={
          'files': [ f'{str(tel_files_path)}/{tel}/**' for tel in tel_list],
      },
    include_package_data=False,
    install_requires=REQUIREMENTS,
    zip_safe=False,
    keywords='Muography Soufrière',
    classifiers=[
        'Programming Language :: Python :: 3.7',
    ]
)

