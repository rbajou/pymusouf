# -*- coding: utf-8 -*-
#!/usr/bin/env python3

#%%
import numpy as np
from pathlib import Path
import time
from scipy.io import loadmat
import pickle
import argparse
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")
#personal modules
from raypath import RayPath
from survey import CURRENT_SURVEY, DICT_SURVEY

if __name__ == "__main__":

    survey = CURRENT_SURVEY
    main_path = Path(__file__).parents[1]
    # files_path = main_path / 'files'
    # dem_path = files_path / "dem"
    # filename1 = "soufriereStructure.mat" #5m
    # filename2 = "soufriereStructure_2.npy" #5m
    # filename3 = "soufriere_1m.npy" #1m
    # structname = filename1.split('.')[0]
    # objstruct= loadmat(str(dem_path / f"{filename1}"))
    # east, north, grid_alt = objstruct[structname][0][0][0].flatten(), objstruct[structname][0][0][1].flatten(), objstruct[structname][0][0][2]
    # print(east.shape, north.shape, grid_alt.shape)
    # grid_east, grid_north = np.meshgrid(east, north)
    # X1, Y1, Z1 = grid_east, grid_north, grid_alt
    # X2, Y2, Z2  = np.load(dem_path/filename2)
    # X3, Y3, Z3 = np.load(dem_path/filename3)
    fig = plt.figure(figsize=(12,9))
    ax = fig.add_subplot(projection='3d')
    X1, Y1, Z1 = survey.surface_grid
    ax.plot_surface(X1, Y1, Z1, alpha=0.2, color='blue', label='structure') 
    # ax.plot_surface(X2,Y2,Z2, alpha=0.2, color='green', label='soufriereStructure_2') 
   # ax.plot_surface(X3,Y3,Z3, alpha=0.2, color='orange', label='soufriere_1m') 
    ax.legend()
    rmax = 1500

    ###thickness color scale
    import palettable
    import matplotlib.colors as cm
    import matplotlib.colors as colors
    #from mpl_toolkits.mplot3d.art3d import PolyCollection, Poly3DCollection, Line3DCollection
    cmap_thick = palettable.scientific.sequential.Batlow_20.mpl_colormap #'jet' #palettable.scientific.diverging.Roma_20
    L_min, L_max= 0, 1500 
    range_thick = np.linspace(0,1500, 100) 
    norm_r = cm.Normalize(vmin=L_min, vmax=L_max)(range_thick)
    color_scale_thick =  cmap_thick(norm_r)
    

    for name, tel in survey.telescopes.items():
        x,y,z = tel.utm
        ax.scatter(x,y,z, color=tel.color, label=name)
        raypath = RayPath(telescope=tel,
                    surface_grid=survey.surface_grid)
        pickle_file = survey.path/ 'telescope'/ tel.name/'raypath'/f'az{tel.azimuth}_elev{tel.elevation}'/'raypath'
        raypath( pickle_file , max_range=1500) 
        thick = raypath.raypath['3p1']['thickness'].flatten()
        arg_col =  [np.argmin(abs(range_thick-v))for v in thick]   
        color_values = color_scale_thick[arg_col] 
        mask = (np.isnan(thick)) 
        tel.plot_ray_values(ax, color_values=color_values, front_panel=tel.panels[0], rear_panel=tel.panels[-2], mask =mask.flatten(), rmax=rmax )
        tel.plot_ray_paths(ax, front_panel=tel.panels[0], rear_panel=tel.panels[-2], mask =mask.flatten(), rmax=rmax, c="grey", linewidth=0.5 )
    plt.show()

    