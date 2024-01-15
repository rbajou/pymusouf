# -*- coding: utf-8 -*-
#!/usr/bin/env python3

#%%
import numpy as np
from pathlib import Path
from scipy.io import loadmat
import argparse
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore", category=UserWarning)#, module="matplotlib")

#personal modules
from config import MAIN_PATH, FILES_DIR
from telescope import str2telescope
from raypath import RayPath
from survey import CURRENT_SURVEY

params = {'legend.fontsize': 'xx-large',
         'axes.labelsize': 'xx-large',
         'axes.titlesize':'xx-large',
         'xtick.labelsize':'xx-large',
         'ytick.labelsize':'xx-large',
         'axes.labelpad':10,
         'mathtext.fontset': 'stix',
         'font.family': 'STIXGeneral',
         'axes.grid' : True, 
         'grid.color' : 'black',
         'grid.linewidth' : '0.3',
         'grid.linestyle' : 'dotted',
         }

plt.rcParams.update(params)

survey = CURRENT_SURVEY

main_path = MAIN_PATH #Path(__file__).parents[1]
files_path = FILES_DIR

parser=argparse.ArgumentParser(description='''Compute ray path apparent thickness for given telescope''', epilog="""All is well that ends well.""")
parser.add_argument('--telescope', '-tel', required=True, help='Telescope name (e.g SNJ). It provides the associated configuration.',  type=str2telescope) #required=True,
parser.add_argument('--out_dir', '-o', default=None, type=str)
#parser.add_argument('--surface_grid', '-sg', default='')

args = parser.parse_args()
tel = args.telescope
#%%
dem_file = survey.dem_file
filename = dem_file.parts[-1]
structname = filename.split('.')[0] #"demStruct"
if filename.endswith(".mat"):
    objstruct= loadmat(str(dem_file))
    east, north, grid_alt = objstruct[structname][0][0][0].flatten(), objstruct[structname][0][0][1].flatten(), objstruct[structname][0][0][2]
    print(east.shape, north.shape, grid_alt.shape)
    grid_east, grid_north = np.meshgrid(east, north)
    surface_grid = np.ones((3,grid_alt.shape[0], grid_alt.shape[1]))
    surface_grid = grid_east, grid_north, grid_alt
    np.save(dem_file.parent / structname, surface_grid)
    print(f'Save {dem_file.parent / structname}.npy')
elif filename.endswith(f".npy"):
    surface_grid = np.load(dem_file)
    print(f'Load {dem_file}')
else : print("Wrong format")


#%%
tel_path = survey.path / "telescope"  / tel.name

raypath = RayPath(telescope=tel,
                    surface_grid=surface_grid,)

fout = tel_path / 'raypath'/ f'az{tel.azimuth:.1f}_elev{tel.elevation:.1f}' / 'raypath'
raypath(file=fout, max_range=1500)

dict_config = args.telescope.configurations.items()

for key, conf in dict_config:
    fig, ax = plt.subplots(figsize= (12,8))
    tel.compute_angle_matrix()
    az0, ze0 = tel.azimuthMatrix[key]*180/np.pi, tel.zenithMatrix[key]*180/np.pi
    Z0 = raypath.raypath[key]['thickness']
    profile_topo = raypath.raypath[key]['profile_topo']
    ax.plot(profile_topo[:,0], profile_topo[:,1], linewidth=3, color='black')
    Z0[np.isnan(Z0)] = 0
    im = ax.pcolormesh(az0, ze0, Z0, cmap='jet', vmin=0 , vmax=1500, shading="auto" )
    arr = im.get_array()
    arr[arr==0] = np.nan
    ax.invert_yaxis()
    cbar = fig.colorbar(im, ax=ax, shrink=0.75, format='%.0e', orientation="vertical")
    cbar.set_label(label=u'thickness [m]')
    ax.set_ylabel('Zenith $\\theta$ [deg]')
    ax.set_xlabel('Azimuth $\\varphi$ [deg]')
    fig.tight_layout()
    # plt.show()
    if args.out_dir is not None:
        od = args.out_dir
        od = Path(od)
        od.mkdir(parents=True, exist_ok=True)
        if len(dict_config) > 1 : fout = od / f"apparent_thickness_{key}.png"
        else: fout = od / f"apparent_thickness.png"
        fig.savefig(fout)
        print(f"Save {fout}")
