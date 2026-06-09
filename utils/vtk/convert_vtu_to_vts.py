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

from pathlib import Path
from tqdm import tqdm
import sys
import vtk
from config import STRUCT_DIR

# package module(s)
from utils.tools import print_file_datetime


dir_survey = STRUCT_DIR / "soufriere"
dir_dem = dir_survey / "dem"
dir_voxel = dir_survey / "voxel"
dir_model = dir_survey / "model"
vs = int(sys.argv[1]) if len(sys.argv) > 1 else 16  # voxel size in m (edge length)
input_file = dir_voxel / f"topo_voi_vox{vs}m.vtu"
# input_file = dir_voxel / f"topo_center_anom_voi_vox{vs}m.vtu"
# input_file = dir_voxel / f"topo_bulge_voi_vox{vs}m.vtu"
basename = input_file.stem
print_file_datetime(input_file)

# read source unstructured grid (the original model to reference)
reader = vtk.vtkXMLUnstructuredGridReader()
reader.SetFileName(str(input_file))
reader.Update()
src_ugrid = reader.GetOutput()

# build a locator to query which source cell contains a point
cell_locator = vtk.vtkStaticCellLocator()
cell_locator.SetDataSet(src_ugrid)
cell_locator.BuildLocator()

# extract voxel_volume array from source grid
voxel_volume_array = src_ugrid.GetCellData().GetArray("voxel_volume")
if voxel_volume_array is None:
    raise ValueError("Source grid does not contain 'voxel_volume' cell array")
fdens = True
voxel_density_array = src_ugrid.GetCellData().GetArray("density")
if voxel_density_array is None:
    fdens = False
    # raise ValueError("Source grid does not contain 'density' cell array")


# bounds and voxel counts
xmin, xmax, ymin, ymax, zmin, zmax = src_ugrid.GetBounds()
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
sgrid = vtk.vtkStructuredGrid()
sgrid.SetDimensions(nxp, nyp, nzp)
sgrid.SetPoints(points)

# for structured grid, cells are (nvox_x * nvox_y * nvox_z)
sg_cell_ids = vtk.vtkIntArray()
sg_cell_ids.SetName("cell_id")
sg_cell_ids.SetNumberOfComponents(1)
sg_cell_ids.SetNumberOfTuples(nvox_x * nvox_y * nvox_z)

sg_voxel_volumes = vtk.vtkFloatArray()
sg_voxel_volumes.SetName("voxel_volume")
sg_voxel_volumes.SetNumberOfComponents(1)
sg_voxel_volumes.SetNumberOfTuples(nvox_x * nvox_y * nvox_z)

if fdens : 
    sg_voxel_densities = vtk.vtkFloatArray()
    sg_voxel_densities.SetName("density")
    sg_voxel_densities.SetNumberOfComponents(1)
    sg_voxel_densities.SetNumberOfTuples(nvox_x * nvox_y * nvox_z)

cnt = 0
for k in tqdm(range(nvox_z), desc="Structured grid cell mapping"):
    for j in range(nvox_y):
        for i in range(nvox_x):
            cx = xmin + (i + 0.5) * vs
            cy = ymin + (j + 0.5) * vs
            cz = zmin + (k + 0.5) * vs
            src_cell_id = cell_locator.FindCell((cx, cy, cz))
            sg_cell_ids.SetValue(cnt, int(src_cell_id) if src_cell_id >= 0 else -1)
            if src_cell_id >= 0:
                vol = voxel_volume_array.GetValue(int(src_cell_id))
                sg_voxel_volumes.SetValue(cnt, vol)
                if fdens :
                    dens = voxel_density_array.GetValue(int(src_cell_id))
                    sg_voxel_densities.SetValue(cnt, dens)
            else:
                sg_voxel_volumes.SetValue(cnt, 0.0)
                if fdens : sg_voxel_densities.SetValue(cnt, 0.0)
            cnt += 1

sgrid.GetCellData().AddArray(sg_cell_ids)
sgrid.GetCellData().AddArray(sg_voxel_volumes)
if fdens : sgrid.GetCellData().AddArray(sg_voxel_densities)
sgrid.GetCellData().SetScalars(sg_cell_ids)

out_vts = dir_voxel / f"{basename}.vts"
writer_vts = vtk.vtkXMLStructuredGridWriter()
writer_vts.SetFileName(str(out_vts))
writer_vts.SetInputData(sgrid)
writer_vts.Write()
print(f"Saved structured grid: {out_vts}")



#############################
# Build ImageData (VTI) with cell-data cell_id
#############################
'''
image = vtk.vtkImageData()
# For vtkImageData, dimensions are number of points -> points = voxels+1
image.SetDimensions(nxp, nyp, nzp)
image.SetSpacing(vs, vs, vs)
image.SetOrigin(xmin, ymin, zmin)

# create cell data arrays (one value per voxel)
img_cell_ids = vtk.vtkIntArray()
img_cell_ids.SetName("cell_id")
img_cell_ids.SetNumberOfComponents(1)
img_cell_ids.SetNumberOfTuples(nvox_x * nvox_y * nvox_z)

img_voxel_volumes = vtk.vtkFloatArray()
img_voxel_volumes.SetName("voxel_volume")
img_voxel_volumes.SetNumberOfComponents(1)
img_voxel_volumes.SetNumberOfTuples(nvox_x * nvox_y * nvox_z)

cnt = 0
for k in tqdm(range(nvox_z), desc="ImageData cell mapping"):
    for j in range(nvox_y):
        for i in range(nvox_x):
            cx = xmin + (i + 0.5) * vs
            cy = ymin + (j + 0.5) * vs
            cz = zmin + (k + 0.5) * vs
            src_cell_id = cell_locator.FindCell((cx, cy, cz))
            img_cell_ids.SetValue(cnt, int(src_cell_id) if src_cell_id >= 0 else -1)
            if src_cell_id >= 0:
                vol = voxel_volume_array.GetValue(int(src_cell_id))
                img_voxel_volumes.SetValue(cnt, vol)
            else:
                img_voxel_volumes.SetValue(cnt, 0.0)
            cnt += 1

image.GetCellData().AddArray(img_cell_ids)
image.GetCellData().AddArray(img_voxel_volumes)
image.GetCellData().SetScalars(img_cell_ids)

out_vti = dir_voxel / f"{basename}.vti"
writer_vti = vtk.vtkXMLImageDataWriter()
writer_vti.SetFileName(str(out_vti))
writer_vti.SetInputData(image)
writer_vti.Write()
print(f"Saved image data: {out_vti}")

print("All conversions done. Load any of the outputs in ParaView and use the 'cell_id' cell array to select corresponding cells from the source model.")
'''
