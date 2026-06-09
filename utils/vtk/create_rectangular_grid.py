#!/usr/bin/python3
# -*- coding: utf-8 -*-
import numpy as np 
from pathlib import Path
import sys
from tqdm import tqdm
import vtk
from vtk.util import numpy_support
#package module(s)    
from config import STRUCT_DIR
from utils.tools import print_file_datetime

dir_survey = STRUCT_DIR / "soufriere"
dir_dem = dir_survey/"dem"
dir_voxel =dir_survey /"voxel"
# file = dir_voxel / "topo_voi_vox8m.vtu"
vs = int(sys.argv[1]) if len(sys.argv) >1 else 16  # voxel size in m (edge length)
input_file = dir_voxel / f"topo_voi_vox{vs}m.vtu"
print_file_datetime(input_file)
reader = vtk.vtkXMLUnstructuredGridReader()
reader.SetFileName(input_file)
reader.Update()
ugrid = reader.GetOutput()
print(ugrid.GetNumberOfCells())
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

xmin, xmax, ymin, ymax, zmin, zmax=ugrid.GetBounds()

# nombre de cellules structurées (arrondi supérieur pour couvrir toute l’emprise)
nx = int(np.ceil((xmax - xmin) / vs))
ny = int(np.ceil((ymax - ymin) / vs))
nz = int(np.ceil((zmax - zmin) / vs))

# Coordonnées des points (pas = vs)
x = xmin + (np.arange(nx)) * vs
y = ymin + (np.arange(ny)) * vs
z = zmin + (np.arange(nz)) * vs

# grille 3D des centres
X, Y, Z = np.meshgrid(x, y, z, indexing="ij")
points = np.column_stack([
    X.ravel(order="F"),
    Y.ravel(order="F"),
    Z.ravel(order="F")
])

print(f"Structured grid dimensions (cells): {nx} x {ny} x {nz}")
print(f"Total structured cells: {points.shape[0]-1}")

# --- Création vtkStructuredGrid ---
sgrid = vtk.vtkStructuredGrid()
sgrid.SetDimensions(nx, ny, nz)

vtk_points = vtk.vtkPoints()
vtk_points.SetData(numpy_support.numpy_to_vtk(points, deep=True))
sgrid.SetPoints(vtk_points)

npoints = len(points)
print("npoints:",npoints)
nvx, nvy, nvz = (nx-1), (ny-1), (nz-1)
ncells = nvx * nvy * nvz
print("ncells:", ncells)
# --- Transfert de l'array "voxel_volume" vers la grille structurée ---

# Récupération de l'array d'origine (cell data)
vtk_cell_ids = vtk.vtkIntArray()
vtk_cell_ids.SetName("cell_id")
vtk_cell_ids.SetNumberOfComponents(1)
vtk_cell_ids.SetNumberOfTuples(nvx* nvy* nvz)

vol_array = ugrid.GetCellData().GetArray("voxel_volume")
if vol_array is None:
    raise RuntimeError("Array 'voxel_volume' introuvable dans le VTU.")
vol_np = numpy_support.vtk_to_numpy(vol_array)
voxel_volumes = np.zeros(ncells)
cnt = 0
for k in range(nvz):
    for j in range(nvy):
        for i in range(nvx):
            cx = xmin + (i + 0.5) * vs
            cy = ymin + (j + 0.5) * vs
            cz = zmin + (k + 0.5) * vs
            cell_id = cell_locator.FindCell((cx, cy, cz))
            vtk_cell_ids.SetValue(cnt, int(cell_id) if cell_id >= 0 else -1)
            if cz <= zmin+250:
                voxel_volumes[cnt] = vs**3
            else:
                voxel_volumes[cnt] = 0.0

            cnt += 1

sgrid.GetCellData().AddArray(vtk_cell_ids)
# Conversion numpy → vtk
vtk_voxel_volumes = numpy_support.numpy_to_vtk(
    voxel_volumes, deep=True)
vtk_voxel_volumes.SetName("voxel_volume")
# Ajout en CellData (IMPORTANT : array cellule)
sgrid.GetCellData().AddArray(vtk_voxel_volumes)


# --- Écriture fichier VTS ---
output_file = dir_voxel / f"rect_grid_vox{vs}m.vts"

writer = vtk.vtkXMLStructuredGridWriter()
writer.SetFileName(str(output_file))
writer.SetInputData(sgrid)
writer.Write()

print(f"Structured grid saved to: {output_file}")
print(f"Dimensions (cells): {nx} x {ny} x {nz}")
