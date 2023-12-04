# -*- coding: utf-8 -*-
#!/usr/bin/env python3

#%%
import numpy as np
from pathlib import Path
import time
from scipy.io import loadmat
import pickle
import argparse
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")
#personal modules
from telescope import dict_tel, str2telescope
from raypath import RayPath
from acqvars import AcqVars


main_path = Path(__file__).parents[1]
files_path = main_path / 'files'

parser=argparse.ArgumentParser(description='''Plot event rate, hit, flux, opacity and density maps''', epilog="""All is well that ends well.""")
parser.add_argument('--telescope', '-tel',  default='SNJ', help='Telescope name (e.g SNJ). It provides the associated configuration.',  type=str2telescope) #required=True,
#parser.add_argument('--surface_grid', '-sg', default='')
args = parser.parse_args()
tel = args.telescope
#%%
#filename = "soufriereStructure.npy" #5m
filename = "soufriereStructure_2.npy" #5m cover larger surface than 'soufriereStructure' grid
#filename = "soufriere_1m.npy"
structname = filename.split('.')[0] #"demStruct"
dem_path = files_path / "dem"
if filename.endswith(".mat"):
    objstruct= loadmat(str(dem_path / f"{filename}"))
    east, north, grid_alt = objstruct[structname][0][0][0].flatten(), objstruct[structname][0][0][1].flatten(), objstruct[structname][0][0][2]
    print(east.shape, north.shape, grid_alt.shape)
    grid_east, grid_north = np.meshgrid(east, north)
    surface_grid = np.ones((3,grid_alt.shape[0], grid_alt.shape[1]))
    surface_grid = grid_east, grid_north, grid_alt
    np.save(dem_path/structname, surface_grid)
elif filename.endswith(f".npy"):
    surface_grid = np.load(dem_path/filename)
else : print("Wrong format")

#%%
tel_path = files_path / "telescopes"  / tel.name
raypath = RayPath(telescope=tel,
                    surface_grid=surface_grid,)
import time
t0 = time.time()
print(f'compute_interceptions_distance() start --- dt= {time.time() - t0:.3e} s')
fout = tel_path / 'raypath'/ f'az{tel.azimuth:.1f}_elev{tel.elevation:.1f}' / 'raypath'
raypath(file=fout, max_range=1500)

#%%
acqVarDir = tel_path / "acqvars" / f"az{tel.azimuth:.1f}ze{tel.zenith:.1f}"
acqVars = AcqVars(telescope=tel, 
                    dir=acqVarDir,
                    mat_files=None,
                    tomo=True)

#%%
#comparison
# az0, ze0 = 5.181999e+01, 7.124958e+01
# azM, zeM = acqVars.az_tomo[conf], acqVars.ze_tomo[conf]
# eps= 0.01
# m = (((1-eps)*az0 < azM) & (azM < (1+eps)*az0 )) & (((1-eps)*ze0 < zeM) & (zeM < (1+eps)*ze0 ))
# print(m.shape)
# idx = np.argwhere(m)[0,:]
# print(f"[az;ze] = [{az0:.2f};{ze0:.2f}]°")
# print(f"aqst.thickness[{idx}] = {a0[idx]}\n acqVars.thickness['3p1'][{idx}] = {a1[idx]}")

#plot
conf ='3p1'
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
fig = plt.figure(figsize= (12,8))
gs = GridSpec(1, 2)#, left=0.04, right=0.99, wspace=0.1, hspace=0.5)
ax0 = fig.add_subplot(gs[0,0],aspect="equal")
#A, Z = np.meshgrid(np.linspace(a.min(), a.max(), 31  )  , np.linspace(z.min(), z.max(), 31  ))
tel.compute_angle_matrix()
az0, ze0 = tel.azimuthMatrix[conf]*180/np.pi, tel.zenithMatrix[conf]*180/np.pi
Z0 = raypath.raypath[conf]['thickness']
c = ax0.pcolor(az0, ze0, Z0, cmap='jet', shading='auto', vmin=0 , vmax=1500 )
ax0.invert_yaxis()
cbar = fig.colorbar(c, ax=ax0, shrink=0.75, format='%.0e', orientation="horizontal")
cbar.ax.tick_params(labelsize=8)
cbar.set_label(label=u'thickness [m]', size=12)
ax0.set_ylabel('zenith $\\theta$ [deg]', fontsize=12)
ax0.set_xlabel('azimuth $\\varphi$ [deg]', fontsize=12)
ax0.set(frame_on=False)
ax1 = fig.add_subplot(gs[0,1],aspect="equal")
az1, ze1 = acqVars.az_tomo[conf], acqVars.ze_tomo[conf]
Z1 = acqVars.thickness[conf]
c = ax1.pcolor(az1, ze1, Z1, cmap='jet', shading='auto', vmin=0 , vmax=1500 )
ax1.invert_yaxis()
cbar = fig.colorbar(c, ax=ax1, shrink=0.75, format='%.0e', orientation="horizontal")
cbar.ax.tick_params(labelsize=8)
cbar.set_label(label=u'thickness [m]', size=12)
ax1.set_ylabel('zenith $\\theta$ [deg]', fontsize=12)
ax1.set_xlabel('azimuth $\\varphi$ [deg]', fontsize=12)
ax1.set(frame_on=False)

gs.tight_layout(fig)
plt.show()
