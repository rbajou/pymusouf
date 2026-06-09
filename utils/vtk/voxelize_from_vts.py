import numpy as np
from pathlib import Path
import sys
import vtk
from vtk.util import numpy_support
from tqdm import tqdm

from config import STRUCT_DIR



def bilinear_z(x, y, X0, Y0, dx, dy, Z):
    ix = (x - X0) / dx
    iy = (y - Y0) / dy

    i0 = int(np.floor(ix))
    j0 = int(np.floor(iy))
    i0 = np.clip(i0, 0, nx-1)
    j0 = np.clip(j0, 0, ny-1)
    tx = ix - i0
    ty = iy - j0

    i1 = min(i0 + 1, Z.shape[0] - 1)
    j1 = min(j0 + 1, Z.shape[1] - 1)
    
    z00 = Z[i0, j0]
    z10 = Z[i1, j0]
    z01 = Z[i0, j1]
    z11 = Z[i1, j1]

    return (
        (1 - tx) * (1 - ty) * z00 +
        tx * (1 - ty) * z10 +
        (1 - tx) * ty * z01 +
        tx * ty * z11
    )
  
def truncated_voxel_volume(x0, y0, z0, L, topo_func):
    # z topo aux 4 coins
    z = np.array([
        topo_func(x0,     y0),
        topo_func(x0+L,   y0),
        topo_func(x0+L,   y0+L),
        topo_func(x0,     y0+L)
    ])

    z_clamped = np.clip(z, z0, z0 + L)
    h = z_clamped - z0

    h_mean = h.mean()
    return L * L * h_mean


def refine_voxel(x, y, z, vs, topo, tol_z, vz_min, voxels):
    z00 = topo(x, y)
    z10 = topo(x+vs, y)
    z11 = topo(x+vs, y+vs)
    z01 = topo(x, y+vs)
    zc  = topo(x+vs/2, y+vs/2)
    z_mean = 0.25 * (z00 + z10 + z11 + z01)
    error = abs(zc - z_mean)
    z_top = max(z00, z10, z11, z01)
    # voxel entièrement au-dessus
    if z >= z_top:
        return
    # voxel entièrement sous la topo
    if z + vs <= min(z00, z10, z11, z01):
        voxels.append((x, y, z, vs))
        return
    # voxel intersectant
    if error < tol_z or vs/2 < vz_min:
        voxels.append((x, y, z, vs))
        return
    # subdivision verticale
    refine_voxel(x, y, z,     vs/2, topo, tol_z, vz_min, voxels)
    refine_voxel(x, y, z+vs/2, vs/2, topo, tol_z, vz_min, voxels)



# ----------------------------
# Paramètres
# ----------------------------
dir_dem = STRUCT_DIR / "soufriere" / "dem"
input_vts = dir_dem / "topo_roi.vts"
vs = int(sys.argv[1]) if len(sys.argv) > 1 else 64  # in m
print(f"Voxel size : {vs} m")
# ----------------------------
# Lecture surface topo
# ----------------------------
reader = vtk.vtkXMLStructuredGridReader()
reader.SetFileName(input_vts)
reader.Update()
surf = reader.GetOutput()
extent = surf.GetExtent()
xmin, xmax, ymin, ymax, zmin, zmax = extent
nx, ny, nz = xmax-xmin+1, ymax-ymin+1, zmax-zmin+1

pts = surf.GetPoints()
# ----------------------------
# Résolution topographie
# ----------------------------
p00 = np.array(pts.GetPoint(0))
p10 = np.array(pts.GetPoint(1))          # (i=1, j=0)
p01 = np.array(pts.GetPoint(nx))         # (i=0, j=1)
res_x = np.linalg.norm(p10[:2] - p00[:2])
res_y = np.linalg.norm(p01[:2] - p00[:2])
if not np.isclose(res_x, res_y, rtol=1e-6):
    print(f"Attention : résolution anisotrope res_x={res_x}, res_y={res_y}")
else:
    print(f"Resolution : {res_x} m")

X = np.zeros((nx, ny))
Y = np.zeros((nx, ny))
Ztop = np.zeros((nx, ny))

idx = 0
for j in tqdm(range(ny), total=nx, desc="Points "):
    for i in range(nx):
        x, y, z = pts.GetPoint(idx)
        X[i, j] = x
        Y[i, j] = y
        Ztop[i, j] = z
        idx += 1


x_min, x_max = X.min(), X.max()
y_min, y_max = Y.min(), Y.max()
z_min, z_max = Ztop.min(), Ztop.max()

# ----------------------------
# Interpolateur bilinéaire
# ----------------------------
def topo(x,y):
    return bilinear_z(x, y, X0=x_min, Y0=y_min, dx=res_x, dy=res_y, Z=Ztop)

# ----------------------------
# Grille voxel régulière
# ----------------------------
xs = np.arange(x_min, x_max, vs)
ys = np.arange(y_min, y_max, vs)
zs = np.arange(z_min, z_max, vs)

ugrid = vtk.vtkUnstructuredGrid()
points = vtk.vtkPoints()

point_id = {}
pid = 0

def get_pid(p):
    global pid
    if p not in point_id:
        point_id[p] = pid
        points.InsertNextPoint(*p)
        pid += 1
    return point_id[p]

# ----------------------------
# Création des voxels
# ----------------------------

voxels = []
volumes = []

for x in tqdm(xs, total=len(xs), desc="Voxels "):
    for y in ys:
        z_surface = topo(x + vs/2, y + vs/2)
        for z in zs:
            refine_voxel(
                x, y, z,
                vs,
                topo,
                tol_z=0.3,
                vz_min=1.0,
                voxels=voxels
            )

volumes = []
for x, y, z, vz in voxels:
    z_top = min(z + vz, topo(x+vs/2, y+vs/2))
    corners = [
        (x, y, z),
        (x+vs, y, z),
        (x+vs, y+vs, z),
        (x, y+vs, z),
        (x, y, z_top),
        (x+vs, y, z_top),
        (x+vs, y+vs, z_top),
        (x, y+vs, z_top),
    ]

    ids = [get_pid(p) for p in corners]

    hex = vtk.vtkHexahedron()
    for i in range(8):
        hex.GetPointIds().SetId(i, ids[i])

    ugrid.InsertNextCell(hex.GetCellType(), hex.GetPointIds())
    volumes.append(truncated_voxel_volume(x, y, z, vz, topo))

# ----------------------------
# Ajout volume scalaire
# ----------------------------
ugrid.SetPoints(points)

vol_arr = numpy_support.numpy_to_vtk(
    np.array(volumes),
    deep=True,
    array_type=vtk.VTK_DOUBLE
)
vol_arr.SetName("voxel_volume")

ugrid.GetCellData().AddArray(vol_arr)
ugrid.GetCellData().SetActiveScalars("voxel_volume")

# ----------------------------
# Écriture
# ----------------------------
writer = vtk.vtkXMLUnstructuredGridWriter()
output_vtu =  dir_dem.parent/"voxel"/ f"topo_voi_vox{int(vs)}m.vtu"
writer.SetFileName(output_vtu)
writer.SetInputData(ugrid)
writer.Write()

print("Voxelisation terminée :", output_vtu)