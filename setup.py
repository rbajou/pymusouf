#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from setuptools import setup, find_packages
import glob
from pathlib import Path
import sys


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
    'pyyaml',
    'palettable', 
    # 'requests', 
    # 'bs4', 
    # 'cx_Freeze', #make executable files
    'mat73', #read v7.3 mat files
    'pillow', #for .gif animation
]

try : 
    survey_dir = Path(__file__).parent / 'files' / 'survey'
    list_avail_survey = [name.split('/')[-1] for name in glob.glob(str(survey_dir) + "/*")]
    files_list=[]
    for survey in list_avail_survey:  
        survey_path = survey_dir / survey
        tel_files_path = survey_path / 'telescope'
        tel_list = [t.split('/')[-1] for t in  glob.glob(str(tel_files_path) + '/*')]
        files_list.extend([ f'{str(tel_files_path)}/{tel}/**' for tel in tel_list ])
        # if survey == "soufriere" : 
        #     flux_path = survey_path /  'flux' / 'corsika' / 'muons'
        #     dem_path = survey_path / 'dem'
        #     files_list.extend([ f'{str(flux_path)}/**', f'{dem_path}/soufriereStructure_2.npy', f'{dem_path}/volcanoCenter.txt' ])
        # elif survey == "copahue":
        flux_path = survey_path /  'flux' / 'corsika' / 'muons'
        dem_path = survey_path / 'dem'
        files_list.extend([ f'{str(flux_path)}/**', f'{dem_path}/*Structure*', f'{dem_path}/*Center*' ])
        # else : 
        #     print(f"No dem file found for survey : {survey}")
except :
    raise FileExistsError("Files list was not properly set.")


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
                     'config',
                     'exe', 
                     'files', 
                     'forwardsolver', 
                     'modeling3d', 
                     'muo2d',  
                     'raypath', 
                     'reco',
                     'scripts',
                     'survey',
                     'telescope', 
                     'test', 
                     'timeserie',
                     'tracking',
                     'utils'
                     ]
    },  
    package_data={
          'files': files_list,
      },
    include_package_data=True,
    install_requires=REQUIREMENTS,
    keywords='Muography x Scintillator',
    classifiers=[
        'Programming Language :: Python :: 3.7',
    ]
)


