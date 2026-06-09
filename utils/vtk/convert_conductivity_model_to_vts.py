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
import sys
import vtk
# package module(s)
from config import STRUCT_DIR
from utils.tools import print_file_datetime

def conductivity_to_density(sigma):
    """
    Convertit la conductivité électrique en densité basé sur Rosas-Carbajal et al. (2016)
    
    Paramètres:
    -----------
    sigma : float ou array
        Conductivité électrique en S/m, dans l'intervalle [0.001, 1.0]
        
    Retourne:
    ---------
    density : float ou array
        Densité estimée en g/cm³ dans l'intervalle [1.8, 2.7]
    """
    # Coefficients basés sur l'interprétation géologique de l'article
    if sigma==0 : return 0
    a = -0.267
    b = 1.9
    # Relation logarithmique
    density = a * np.log10(sigma) + b
    
    # Clip pour s'assurer qu'on reste dans les bornes
    density = np.clip(density, 1.8, 2.7)
    return density



dir_survey = STRUCT_DIR / "soufriere"
dir_dem = dir_survey / "dem"
dir_voxel = dir_survey / "voxel"
dir_model = dir_survey / "model"
vs = int(sys.argv[1]) if len(sys.argv) > 1 else 32  # voxel size in m (edge length)
topo_file = dir_voxel / f"topo_voi_vox{vs}m.vts"
topo_reader = vtk.vtkXMLStructuredGridReader()
topo_reader.SetFileName(str(topo_file))
topo_reader.Update()
topo_grid = topo_reader.GetOutput()

topo_voxel_volume_array = topo_grid.GetCellData().GetArray("voxel_volume")
topo_cell_locator = vtk.vtkStaticCellLocator()
topo_cell_locator.SetDataSet(topo_grid)
topo_cell_locator.BuildLocator()

ert_file = dir_model / "ElecCond_CentralCube_aligned.vtu"
print_file_datetime(ert_file)


# reader = vtk.vtkUnstructuredGridReader()
ert_reader = vtk.vtkXMLUnstructuredGridReader()
ert_reader.SetFileName(str(ert_file))
ert_reader.Update()
ert_ugrid = ert_reader.GetOutput()

# build a locator to query which source cell contains a point
ert_cell_locator = vtk.vtkStaticCellLocator()
ert_cell_locator.SetDataSet(ert_ugrid)
ert_cell_locator.BuildLocator()

# extract  array from source grid
sigma_array = ert_ugrid.GetCellData().GetArray("sigma.16")
if sigma_array is None:
    raise ValueError("Source grid does not contain 'sigma.16' cell array")

# xyz_center = np.array([642960,1774280,1466])

# bounds and voxel counts
xmin, xmax, ymin, ymax, zmin, zmax = ert_ugrid.GetBounds()
print(f"ERT grid bounds: x[{xmin}, {xmax}], y[{ymin}, {ymax}], z[{zmin}, {zmax}]")

xmin, xmax, ymin, ymax, zmin, zmax = topo_grid.GetBounds()
print(f"VOI grid bounds: x[{xmin}, {xmax}], y[{ymin}, {ymax}], z[{zmin}, {zmax}]")

# number of voxels along each axis (round to nearest integer)
nvox_x = int(round((xmax - xmin) / vs))
nvox_y = int(round((ymax - ymin) / vs))
nvox_z = int(round((zmax - zmin) / vs))

if nvox_x <= 0 or nvox_y <= 0 or nvox_z <= 0:
    raise RuntimeError("Computed non-positive voxel counts; check bounds and vs")

# auxiliary counters
nxp = nvox_x + 1  # number of grid points along x
nyp = nvox_y + 1
nzp = nvox_z + 1

print(f"Voxels: {nvox_x} x {nvox_y} x {nvox_z} (voxel size {vs} m)")

#############################
# Create a shared points grid (corners for all voxels)
#############################
points = vtk.vtkPoints()
points.SetNumberOfPoints(nxp * nyp * nzp)

def pid(ix, iy, iz):
    return ix + iy * nxp + iz * (nxp * nyp)

for iz in range(nzp):
    z = zmin + iz * vs 
    for iy in range(nyp):
        y = ymin + iy * vs 
        for ix in range(nxp):
            x = xmin + ix * vs 
            points.SetPoint(pid(ix, iy, iz), x, y, z)

#############################
# Build StructuredGrid
#############################
# Copy topo grid structure exactly
sgrid = vtk.vtkStructuredGrid()
sgrid.DeepCopy(topo_grid)

n_cells = topo_grid.GetNumberOfCells()

sg_cell_ids = vtk.vtkIntArray()
sg_cell_ids.SetName("cell_id")
sg_cell_ids.SetNumberOfComponents(1)
sg_cell_ids.SetNumberOfTuples(n_cells)

sg_sigma = vtk.vtkFloatArray()
sg_sigma.SetName("sigma.16")
sg_sigma.SetNumberOfComponents(1)
sg_sigma.SetNumberOfTuples(n_cells)

sg_density = vtk.vtkFloatArray()
sg_density.SetName("density")
sg_density.SetNumberOfComponents(1)
sg_density.SetNumberOfTuples(n_cells)

sg_volume = vtk.vtkFloatArray()
sg_volume.SetName("voxel_volume")
sg_volume.SetNumberOfComponents(1)
sg_volume.SetNumberOfTuples(n_cells)
cell = vtk.vtkGenericCell()
closest_point = [0.0, 0.0, 0.0]
dist2 = vtk.mutable(0.0)
subId = vtk.mutable(0)

for cid in tqdm(range(n_cells), desc="Mapping cells"):

    topo_grid.GetCell(cid, cell)
    bounds = cell.GetBounds()

    cx = 0.5 * (bounds[0] + bounds[1])
    cy = 0.5 * (bounds[2] + bounds[3])
    cz = 0.5 * (bounds[4] + bounds[5])

    topo_cell_id = cid

    vol = topo_voxel_volume_array.GetValue(cid)

    sg_volume.SetValue(cid, vol)
    sg_cell_ids.SetValue(cid, topo_cell_id)

    ert_cell_id = ert_cell_locator.FindCell((cx, cy, cz))

    if topo_cell_id >=0 :  
        # fallback : cellule la plus proche
        if ert_cell_id < 0:
            closest_cell = vtk.mutable(0)
            ert_cell_locator.FindClosestPoint(
                (cx, cy, cz),
                closest_point,
                cell,
                closest_cell,
                subId,
                dist2
            )
            ert_cell_id = closest_cell
        sigma = sigma_array.GetValue(int(ert_cell_id))
    else: 
        sigma = 0.
    sg_sigma.SetValue(cid, sigma)

    density = conductivity_to_density(sigma)
    sg_density.SetValue(cid, density)

sgrid.GetCellData().AddArray(sg_cell_ids)
sgrid.GetCellData().AddArray(sg_sigma)
sgrid.GetCellData().AddArray(sg_density)
sgrid.GetCellData().AddArray(sg_volume)

sgrid.GetCellData().SetActiveScalars("density")

basename = topo_file.stem
# out_vts = dir_model / f"{basename}_vox{vs}m.vts"
out_vts = dir_model / f"ElecCond_{basename}.vts"
writer_vts = vtk.vtkXMLStructuredGridWriter()
writer_vts.SetFileName(str(out_vts))
writer_vts.SetInputData(sgrid)
writer_vts.Write()
print(f"Saved structured grid: {out_vts}")
