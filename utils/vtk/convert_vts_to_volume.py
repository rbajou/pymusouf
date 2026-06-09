import numpy as np
from pathlib import Path
import vtk
from vtk.util import numpy_support

from config import STRUCT_DIR

# ----------------------------
# Paramètres
# ----------------------------
dir_dem = STRUCT_DIR / "soufriere" / "dem"
input_vts = dir_dem / "topo_roi.vts"
output_vts =  dir_dem / "topo_voi.vts"

dz = 2.0  # résolution verticale (m)

# ----------------------------
# Lecture de la grille surface
# ----------------------------
reader = vtk.vtkXMLStructuredGridReader()
reader.SetFileName(input_vts)
reader.Update()

surf = reader.GetOutput()
xmin, xmax, ymin, ymax, zmin, zmax = surf.GetExtent()
nx, ny, nz = xmax-xmin+1, ymax-ymin+1, zmax-zmin+1
assert nz == 1, "La grille d'entrée doit être 2D (nz=1)"

points = surf.GetPoints()

# ----------------------------
# Extraction topo
# ----------------------------
X = np.zeros((nx, ny))
Y = np.zeros((nx, ny))
Ztop = np.zeros((nx, ny))

idx = 0
for j in range(ny):
    for i in range(nx):
        x, y, z = points.GetPoint(idx)
        X[i, j] = x
        Y[i, j] = y
        Ztop[i, j] = z
        idx += 1

z_min = Ztop.min()
z_max = Ztop.max()

z_levels = np.arange(z_min, z_max + dz, dz)
nz_vol = len(z_levels)

# ----------------------------
# Création points + scalaire elevation
# ----------------------------
vol_points = vtk.vtkPoints()
vol_points.SetNumberOfPoints(nx * ny * nz_vol)

elevation = np.zeros(nx * ny * nz_vol)

idx = 0
for k, z in enumerate(z_levels):
    for j in range(ny):
        for i in range(nx):
            zc = min(z, Ztop[i, j])
            vol_points.SetPoint(idx, X[i, j], Y[i, j], zc)
            elevation[idx] = zc
            idx += 1

# ----------------------------
# Grille volumique
# ----------------------------
vol_grid = vtk.vtkStructuredGrid()
vol_grid.SetDimensions(nx, ny, nz_vol)
vol_grid.SetPoints(vol_points)

# ----------------------------
# Ajout du champ scalaire
# ----------------------------
elev_vtk = numpy_support.numpy_to_vtk(
    elevation,
    deep=True,
    array_type=vtk.VTK_DOUBLE
)
elev_vtk.SetName("elevation")

vol_grid.GetPointData().AddArray(elev_vtk)
vol_grid.GetPointData().SetActiveScalars("elevation")

# ----------------------------
# Écriture fichier
# ----------------------------
writer = vtk.vtkXMLStructuredGridWriter()
writer.SetFileName(output_vts)
writer.SetInputData(vol_grid)
writer.Write()

print(f"Volume sauvegardé : {output_vts}")
print(f"Dimensions : {nx} x {ny} x {nz_vol}")
print(f"Champ scalaire : elevation")