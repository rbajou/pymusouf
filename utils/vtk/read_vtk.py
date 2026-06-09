#!/usr/bin/python3
# -*- coding: utf-8 -*-
import pyvista as pv
import numpy as np 
from pathlib import Path
import vtk
#package module(s)    
from config import STRUCT_DIR
from utils.tools import print_file_datetime

dir_dem = STRUCT_DIR / "soufriere" / "dem"
dir_model = STRUCT_DIR / "soufriere" / "models"
# file = dir_dem / "topo_roi.vts"
# file = dir_dem / "topo_voi_vox8m.vtu"
file = dir_model / "ElecCond_CentralCube.vtk"
# file = dir_dem / "topo_voi_vox64m.vtu"
basename = file.stem
grid = pv.read(file)
print_file_datetime(file)
print(basename)
grid = np.asarray(grid.points)
print(type(grid))
print(grid.shape)
exit()
shp0 = int(np.sqrt(grid.shape[0]))
grid = grid.reshape((-1, shp0, 3))
print(grid.shape)
file_out = dir_dem / f"{basename}.npy"
np.save (file_out, grid)
print(f"Save {file_out}")