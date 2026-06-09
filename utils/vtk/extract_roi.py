import numpy as np
from pathlib import Path
import vtk
from config import STRUCT_DIR

# ----------------------------
# Paramètres
# ----------------------------
dir_dem = STRUCT_DIR / "soufriere" / "dem"
input_file = dir_dem / "topo.npy"
output_file = dir_dem / f"topo_roi.vts"

center_x = 642969.0
center_y = 1774280.0

resolution = 2.0          # m
roi_size = 1200.0          # m
half_size = roi_size / 2  # 200 m


# ----------------------------
# Chargement des données
# ----------------------------
data = np.load(input_file)  # shape (nx, ny, 3)

X = data[:, :, 0]
Y = data[:, :, 1]
Z = data[:, :, 2]

nx, ny = X.shape

# ----------------------------
# Trouver l'indice du point central
# ----------------------------
# On suppose une grille régulière
x0 = X[0, 0]
y0 = Y[0, 0]

ix_center = int(round((center_x - x0) / resolution))
iy_center = int(round((center_y - y0) / resolution))

# Taille de la ROI en nombre de points
roi_pts = int(roi_size / resolution)  # 400 / 2 = 200 points

half_pts = roi_pts // 2

ix_min = ix_center - half_pts
ix_max = ix_center + half_pts
iy_min = iy_center - half_pts
iy_max = iy_center + half_pts

# Sécurité bornes
if ix_min < 0 or iy_min < 0 or ix_max > nx or iy_max > ny:
    raise ValueError("La région d'intérêt sort des limites du modèle")

# ----------------------------
# Extraction ROI
# ----------------------------
X_roi = X[ix_min:ix_max, iy_min:iy_max]
Y_roi = Y[ix_min:ix_max, iy_min:iy_max]
Z_roi = Z[ix_min:ix_max, iy_min:iy_max]

nx_roi, ny_roi = X_roi.shape

# ----------------------------
# Création du Structured Grid VTK
# ----------------------------
points = vtk.vtkPoints()
points.SetNumberOfPoints(nx_roi * ny_roi)

idx = 0
for j in range(ny_roi):
    for i in range(nx_roi):
        points.SetPoint(idx, X_roi[i, j], Y_roi[i, j], Z_roi[i, j])
        idx += 1

grid = vtk.vtkStructuredGrid()
grid.SetDimensions(nx_roi, ny_roi, 1)
grid.SetPoints(points)

# ----------------------------
# Écriture .vts
# ----------------------------
writer = vtk.vtkXMLStructuredGridWriter()
writer.SetFileName(output_file)
writer.SetInputData(grid)
writer.Write()

print(f"ROI sauvegardée : {output_file}")