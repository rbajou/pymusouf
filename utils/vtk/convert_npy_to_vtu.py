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

    dir_path = STRUCT_DIR / "soufriere" / "voxel"
    input_file = dir_path / "vox_matrix_res64m.npy"
    output_file = dir_path / f"vox_matrix_res64m.vtu"
    print_file_datetime(input_file)
    voxels = np.load(input_file) #(nvoxels, nfaces, ncorners, ncoords)
    print(voxels.shape)
    nv, nf, nc, _ = voxels.shape
    # Créer la grille non-structurée
    points = vtk.vtkPoints()
    ugrid = vtk.vtkUnstructuredGrid()

    point_id_map = {}
    current_point_id = 0

    for v in range(nv):
        # Récupération des 24 sommets
        coords = voxels[v].reshape(-1, 3)
        # Sommets uniques
        unique_pts = np.unique(coords, axis=0)
        if unique_pts.shape[0] != 8:
            raise ValueError("Voxel non hexaédrique")
        # Tri spatial pour imposer l’ordre VTK
        center = unique_pts.mean(axis=0)
        rel = unique_pts - center

        # z puis y puis x → ordre stable
        order = np.lexsort((rel[:,0], rel[:,1], rel[:,2]))
        ordered_pts = unique_pts[order]

        hex_cell = vtk.vtkHexahedron()

        for i, p in enumerate(ordered_pts):
            key = tuple(p)
            if key not in point_id_map:
                points.InsertNextPoint(p)
                point_id_map[key] = current_point_id
                current_point_id += 1

            hex_cell.GetPointIds().SetId(i, point_id_map[key])

        ugrid.InsertNextCell(hex_cell.GetCellType(),
                            hex_cell.GetPointIds())

        ugrid.SetPoints(points)

    writer = vtk.vtkXMLUnstructuredGridWriter()
    writer.SetFileName(output_file)
    print(f'Save {output_file}')
    writer.SetInputData(ugrid)
    writer.Write()
