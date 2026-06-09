#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
Convert a voxelized model (from a source UnstructuredGrid) into:
- vtkUnstructuredGrid of hexahedral cells (.vtu)
- vtkStructuredGrid (.vts)
- vtkImageData (.vti)

Each output contains a cell-data array `cell_id` giving the
ID of the source (input) cell that contains the voxel center, or -1.

This allows selecting the same source-cells when switching grid types
in ParaView (use the `cell_id` cell array).
"""
import numpy as np
from pathlib import Path
from tqdm import tqdm
import vtk
# package module(s)
from config import STRUCT_DIR
from utils.tools import print_file_datetime
from survey import CURRENT_SURVEY

def weighted_center_of_mass(ugrid):
    points = ugrid.GetPoints()
    npts = points.GetNumberOfPoints()

    coords = np.zeros((npts, 3))
    for i in range(npts):
        coords[i] = points.GetPoint(i)

    z = coords[:, 2]
    zmin = z.min()
    weights = z - zmin

    # Sécurité : éviter division par zéro
    if np.all(weights == 0):
        raise ValueError("Poids nuls : problème avec les élévations")

    center = np.average(coords, axis=0, weights=weights)
    return center

def peak_point(ugrid):
    points = ugrid.GetPoints()
    npts = points.GetNumberOfPoints()

    max_z = -np.inf
    peak = None

    for i in range(npts):
        x, y, z = points.GetPoint(i)
        if z > max_z:
            max_z = z
            peak = np.array([x, y, z])

    return peak

dir_survey = STRUCT_DIR / "soufriere"
dir_dem = dir_survey / "dem"
dir_voxel = dir_survey / "voxel"
dir_model = dir_survey / "model"
vs = 8  # voxel size in m (edge length)
# input_file1 = dir_voxel / f"topo_voi_vox{vs}m.vtu"
input_file1 = dir_dem / f"topo_voi.vts"

input_file2 = dir_model / "ElecCond_CentralCube.vtk"
basename2 = input_file2.stem
print_file_datetime(input_file2)

reader = vtk.vtkXMLStructuredGridReader()
reader.SetFileName(str(input_file1))
reader.Update()
ugrid1 = reader.GetOutput()
print(type(ugrid1))

reader2 = vtk.vtkUnstructuredGridReader()
reader2.SetFileName(str(input_file2))
reader2.Update()
ugrid2 = reader2.GetOutput()

C1 = peak_point(ugrid1)  # grille UTM
C2 = peak_point(ugrid2)  # grille locale

print("Centre grille 1 (UTM) :", C1)
print("Centre grille 2 (local) :", C2)

translation = C1 - C2

transform = vtk.vtkTransform()
transform.Translate(translation)

tf = vtk.vtkTransformFilter()
tf.SetTransform(transform)
tf.SetInputData(ugrid2)
tf.Update()

ugrid2_aligned = tf.GetOutput()

file_out = dir_model / f"{basename2}_aligned.vtu"
writer = vtk.vtkXMLUnstructuredGridWriter()
writer.SetFileName(str(file_out))
writer.SetInputData(ugrid2_aligned)
writer.Write()
print(f"Saved aligned unstructured grid: {file_out}")