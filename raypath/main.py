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
from telescope import str2telescope
from raypath import RayPath


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


main_path = Path(__file__).parents[1]
files_path = main_path / 'files'

parser=argparse.ArgumentParser(description='''Compute ray path apparent thickness for given telescope''', epilog="""All is well that ends well.""")
parser.add_argument('--telescope', '-tel',  default='SNJ', help='Telescope name (e.g SNJ). It provides the associated configuration.',  type=str2telescope) #required=True,
parser.add_argument('--out_dir', '-o', default=None, type=str)
#parser.add_argument('--surface_grid', '-sg', default='')

args = parser.parse_args()
tel = args.telescope
#%%
filename = "soufriereStructure_2.npy" #5m cover larger surface than 'soufriereStructure' grid
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

fout = tel_path / 'raypath'/ f'az{tel.azimuth:.1f}_elev{tel.elevation:.1f}' / 'raypath'
raypath(file=fout, max_range=1500)

dict_config = args.telescope.configurations.items()


for conf, _ in dict_config:
    fig, ax = plt.subplots(figsize= (12,8))
    tel.compute_angle_matrix()
    az0, ze0 = tel.azimuthMatrix[conf]*180/np.pi, tel.zenithMatrix[conf]*180/np.pi
    Z0 = raypath.raypath[conf]['thickness']
    profile_topo = raypath.raypath[conf]['profile_topo']
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
        if len(dict_config) > 1 : fout = od / f"apparent_thickness_{conf}.png"
        else: fout = od / f"apparent_thickness.png"
        fig.savefig(fout)
        print(f"Save {fout}")
