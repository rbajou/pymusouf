#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path

from setuptools import find_packages, setup

ROOT = Path(__file__).parent


REQUIREMENTS = [
    # 'iminuit', #fit in 'utils/functions.py'
    'matplotlib',
    'mat73', #read v7.3 mat files
    'numba', #for multiprocessing
    'numpy',
    'palettable', # nice color maps
    'pandas', #dataframe
    'psutil', #memory check
    'pyjson', #json files
    'pyvista',
    'pyyaml', #yaml files
    'scikit-learn', #ml library
    'scikit-image', #ransac
    'scipy', 
    'torch', #machine learning
    'tqdm', #process monitoring 
    'seaborn',#nice plot templates
    # 'uproot', #to read, update, and write .root file
    'vtk', #3d model rendering
]


def list_package_files(package_name: str, patterns: list[str]) -> list[str]:
    package_root = ROOT / package_name
    if not package_root.exists():
        return []

    collected = []
    for pattern in patterns:
        collected.extend(
            str(path.relative_to(package_root))
            for path in package_root.rglob(pattern)
            if path.is_file()
        )
    return sorted(set(collected))


def build_data_files(folder_name: str) -> list[tuple[str, list[str]]]:
    base = ROOT / folder_name
    if not base.exists():
        return []

    grouped: dict[str, list[str]] = {}
    for path in base.rglob('*'):
        if path.is_file():
            target = str(path.parent.relative_to(ROOT))
            rel_file = str(path.relative_to(ROOT))
            grouped.setdefault(target, []).append(rel_file)

    return sorted((target, sorted(files)) for target, files in grouped.items())


PACKAGE_DATA = {
    'config': list_package_files('config', ['*.yaml']),
    'sample': list_package_files('sample', ['**/*']),
    'survey': list_package_files('survey', ['*.yaml']),
    'telescope': list_package_files('telescope', ['*.yaml', 'resources/**/*.json']),
}


setup(
    name='pymusouf',
    version='0.2.0',
    description="This package is for processing and analyzing muography data " \
                "recorded at Soufrière de Guadeloupe/Copahue volcanoes " \
                "with scintillator-based telescopes developed at IP2I, Lyon",
    author="Raphaël Bajou",
    author_email='r.bajou2@gmail.com',
    url='https://github.com/rbajou/pymusouf.git',
    packages=find_packages(exclude=['struct_link', 'struct_link.*']),
    include_package_data=True,
    package_data=PACKAGE_DATA,
    install_requires=REQUIREMENTS,
    keywords=['Muons', 'Tomography', 'Volcano', 'Scintillator'],
    classifiers=[
        'Programming Language :: Python :: 3.11',
    ],
)


