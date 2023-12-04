import numpy as np
from pathlib import Path
import time
from scipy.interpolate import RegularGridInterpolator, LinearNDInterpolator, NearestNDInterpolator
#personal modules
from raypath import RayPath
from telescope import dict_tel
#from modeling3d.voxel import Voxel, DirectProblem 

main_path = Path.cwd()
files_path = main_path/ 'files'
dem_path = files_path / "dem"
filename2 = "soufriereStructure_2.npy"
surface_grid = np.load(dem_path / filename2)
surface_center = np.loadtxt(dem_path / "volcanoCenter.txt").T
tel = dict_tel['SNJ']
tel_files_path = files_path / 'telescopes'  / tel.name
dout_ray = tel_files_path / 'raypath'/ f'az{tel.azimuth:.1f}_elev{tel.elevation:.1f}' 
raypath = RayPath(telescope=tel,
                       surface_grid=surface_grid,)

#tel.compute_angle_matrix()
tx, ty, tz = tel.utm
SX, SY, SZ = surface_grid 
sort = np.argsort(SX)
#sx_sort, sy_sort, SZ_sort = np.unique(SX[sort]).flatten(), np.unique(SY[sort]).flatten(), SZ[sort]
#reg = RegularGridInterpolator((sx_sort, sy_sort), SZ_sort)#, method='linear')
sx_un_flat, sy_un_flat = np.unique(SX).flatten(), np.unique(SY).flatten()
sx_flat, sy_flat, sz_flat = SX.flatten(), SY.flatten(), SZ.flatten()
reg = RegularGridInterpolator((sx_un_flat, sy_un_flat), SZ, method='linear')
res_reg = reg((tx, ty ))
print(f'z_interp_reg = {res_reg}')
print(f'tel.altitude = {tz}')
t0=time.time()
lin = LinearNDInterpolator((sx_flat, sy_flat), sz_flat)
res_lin = lin((tx, ty ))
print(f'z_interp_lin = {res_lin} ({time.time()-t0:.2f} s)')
time.sleep(1)
print(f'z_interp_lin = {res_lin} ({time.time()-t0:.2f} s)')
t0=time.time()
near = NearestNDInterpolator(list(zip(sx_flat, sy_flat)), sz_flat)
res_near = near((tx, ty ))
print(f'z_interp_near = {res_near} ({time.time()-t0:.2f} s)')
from scipy.interpolate import griddata
points, values = np.array([SX.flatten(), SY.flatten()]).T, SZ.flatten()
t0=time.time()
res_grid = griddata(points, values, (tx, ty), method='linear')
print(f'z_interp_grid = {res_grid} ({time.time()-t0:.2f} s)')
