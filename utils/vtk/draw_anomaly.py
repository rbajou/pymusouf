#!/usr/bin/python3
# -*- coding: utf-8 -*-
from datetime import datetime
import numpy as np 
from pathlib import Path
from tqdm import tqdm
import vtk
from vtk.util import numpy_support
#package module(s)    
from config import STRUCT_DIR
from utils.tools import print_file_datetime


dir_survey = STRUCT_DIR / "soufriere"
dir_dem = dir_survey/"dem"
dir_voxel =dir_survey /"voxel"
vs = 8  #voxel size in m
input_file = dir_voxel / f"topo_voi_vox{vs}m.vtu"
print_file_datetime(input_file)
reader = vtk.vtkXMLUnstructuredGridReader()
reader.SetFileName(input_file)
reader.Update()
ugrid = reader.GetOutput()

# cell_locator = vtk.vtkCellLocator()
cell_locator = vtk.vtkStaticCellLocator()
cell_locator.SetDataSet(ugrid)
cell_locator.BuildLocator()

ncells = ugrid.GetNumberOfCells()
bboxes = np.zeros((ncells, 6))  # xmin,xmax,ymin,ymax,zmin,zmax

for i in tqdm(range(ncells), desc="Cells "):
    cell = ugrid.GetCell(i)
    pts = cell.GetPoints()
    xs, ys, zs = [], [], []
    for j in range(pts.GetNumberOfPoints()):
        x,y,z = pts.GetPoint(j)
        xs.append(x); ys.append(y); zs.append(z)
    bboxes[i] = [min(xs), max(xs), min(ys), max(ys), min(zs), max(zs)]

centers = 0.5 * (bboxes[:,[0,2,4]] + bboxes[:,[1,3,5]])

xmin, xmax, ymin, ymax, zmin, zmax = ugrid.GetBounds()

# Tirage aléatoire de sous-volumes
margin_xy = (xmax - xmin)/2
margin_z  = (zmax - zmin)/2

valid_centers = centers[
    (centers[:,0] > xmin + margin_xy) &
    (centers[:,0] < xmax - margin_xy) &
    (centers[:,1] > ymin + margin_xy) &
    (centers[:,1] < ymax - margin_xy) &
    (centers[:,2] > zmin + margin_z) &
    (centers[:,2] < zmax - margin_z)
]
def random_subvolume(n_samples):
    samples = []

    for _ in range(n_samples):
        c = valid_centers[np.random.randint(len(valid_centers))]
        Z0 = np.random.uniform(zmin, zmax - (zmax-zmin))
        samples.append((c[0], c[1], Z0, Z0+(zmax-zmin)))

    return samples

dataset = []
# for Xc, Yc, Z0, Z1 in random_subvolume(1000):
#     # réutilise la sélection déterministe
#     dataset.append(extract_subvolume(Xc, Yc, Z0, Z1))
