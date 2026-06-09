#!/usr/bin/python3
# -*- coding: utf-8 -*-
from datetime import datetime
import numpy as np 
from pathlib import Path
import vtk
#package module(s)
from config import STRUCT_DIR
from utils.tools import print_file_datetime

if __name__ == "__main__":

    dir_path = STRUCT_DIR / "soufriere" / "dem"
    file = dir_path / "soufriere_dome_surface_1m.npy"
    basename = file.stem
    print_file_datetime(file)
    surface = np.load(file)
    nx, ny, _ = surface.shape
    print(nx, ny)
    
    # Créer la grille structurée
    grid = vtk.vtkStructuredGrid()
    grid.SetDimensions(nx, ny, 1)
    # Créer les points
    points = vtk.vtkPoints()
    points.SetNumberOfPoints(nx * ny)
    
    k = 0
    for j in range(ny):
        for i in range(nx):
            x = surface[i, j, 0]  # Easting
            y = surface[i, j, 1]  # Northing
            z = surface[i, j, 2]  # Elevation
            points.SetPoint(k, x, y, z)
            k += 1

    grid.SetPoints(points)

    # Elévation comme champ scalaire
    elevation = vtk.vtkFloatArray()
    elevation.SetName("Elevation")
    elevation.SetNumberOfValues(nx * ny)

    k = 0
    for j in range(ny):
        for i in range(nx):
            elevation.SetValue(k, surface[i, j, 2])
            k += 1

    grid.GetPointData().SetScalars(elevation)

    # Écriture du fichier VTK
    writer = vtk.vtkStructuredGridWriter()
    file_out = dir_path / f"{basename}.vtk"
    writer.SetFileName(file_out)
    print(f"Save {file_out}")
    writer.SetInputData(grid)
    writer.Write()