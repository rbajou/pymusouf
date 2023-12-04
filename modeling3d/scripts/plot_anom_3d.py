#!/usr/bin/env python3
# -*- coding: utf-8 -*-



from dataclasses import dataclass, field
from textwrap import fill
from tkinter import E
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as cm

from matplotlib.colors import LogNorm
from mpl_toolkits.mplot3d.art3d import PolyCollection, Poly3DCollection, Line3DCollection
from matplotlib.colors import Normalize
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.cm import ScalarMappable
from scipy.interpolate import griddata
import sys
import os
import scipy.io as sio
from pathlib import Path
import inspect
import time
from datetime import datetime, date, timezone
import logging
import glob
import yaml
import pickle
import json
import warnings
import pandas as pd
import mat73 #read v7.3 mat files
import palettable
import utm
import shapefile
#personal modules
from configuration import str2telescope, Telescope
from analysis import AcqVars

@dataclass
class VoxeledDome:
    resolution : float
    matrix : np.ndarray
    def __post_init__(self):
        nvoxels = len(self.matrix)
        object.__setattr__(self, 'nvoxels',  nvoxels )
        #### 
        x = self.matrix[:,1:5]
        y = self.matrix[:,9:13]
        z = self.matrix[:,17:25]
        self.barycenter =  self.matrix[:, 25:28] #shape=(nvox,3)
        #### https://stackoverflow.com/questions/42611342/representing-voxels-with-matplotlib
        self.cubes = np.array( [
            [  [x[:,0], y[:,3], z[:,0]], [x[:,0], y[:,0], z[:,0]],[x[:,1], y[:,0], z[:,0]], [x[:,1], y[:,3], z[:,0]] ],
            [  [x[:,0], y[:,0], z[:,0]], [x[:,0], y[:,0], z[:,4]],[x[:,1], y[:,0], z[:,5]], [x[:,1], y[:,0], z[:,0]] ],
            [  [x[:,1], y[:,0], z[:,5]], [x[:,1], y[:,0], z[:,0]],[x[:,1], y[:,3], z[:,0]], [x[:,1], y[:,3], z[:,6]] ],
            [  [x[:,0], y[:,0], z[:,4]], [x[:,0], y[:,0], z[:,0]],[x[:,0], y[:,3], z[:,0]], [x[:,0], y[:,3], z[:,7]] ],
            [  [x[:,0], y[:,3], z[:,0]], [x[:,0], y[:,3], z[:,7]],[x[:,1], y[:,3], z[:,6]], [x[:,1], y[:,3], z[:,0]] ],
            [  [x[:,0], y[:,3], z[:,7]], [x[:,0], y[:,0], z[:,4]],[x[:,1], y[:,0], z[:,5]], [x[:,1], y[:,3], z[:,6]] ]
        ] ) 

    
        self.cubes = np.swapaxes(self.cubes.T,1,-1)
        
        self.x = x
        self.y = y
        self.z = z 
        self.xyz_down = np.array([self.matrix[:,1:5].T, self.matrix[:,9:13].T, self.matrix[:,17:21].T]).T #shape=(nvoxels,4,3)
        self.xyz_up = np.array([self.matrix[:,5:9].T, self.matrix[:,14:18].T, self.matrix[:,21:25].T]).T #shape=(nvoxels,4,3)


    def get_distance_matrix(self):
        '''
        Matrix featuring distance between each voxel couple (i,j): (d)_i,j
        '''
        nvoxels = self.barycenter.shape[0]
        self.d = np.zeros(shape=(nvoxels,nvoxels))
        #print(f"get_distance_matrix()\ndome.d = {self.d.shape}")
        #print(nvoxels)
        #print(self.barycenter.shape)
        for j in range(nvoxels):
            for i in range(nvoxels):
                self.d[i,j] = np.linalg.norm(self.barycenter[i]-self.barycenter[j]) 

def frho_to_strength(rho):
    strength = 1.326*1e-25 *(rho*1e3)**(9.67) * 1e-6
    return strength


if __name__=="__main__":
    start_time = time.time()
    print("Start: ", time.strftime("%H:%M:%S", time.localtime()))#start time
            
  
    
    param_dir = Path.home() / 'muon_code_v2_0' / 'AcquisitionParams'
    print(time.strftime("%H%M%S-%d%m%y"))
    res= int(sys.argv[1]) #m
    ml_dir = Path.home()  / "MatLab" / "MuonCompute_Volumes_mnt" 
    fGmatrix_dome = ml_dir /  "matrices" / f"cubesMatrix_{res}.mat"
    fVolcanoCenter = ml_dir / "matrices" / "volcanoCenter.mat"

    ####
    ###telescopes
    ltel_n = ["SB", "SNJ", "BR", "OM"]
    #ltel_n = ["SNJ",]#, "OM"]
    str_tel =  "_".join(ltel_n)
    ltel_coord = np.array([ str2telescope(tel).utm[:-1] for tel in ltel_n])
    ltel_color = np.array([ str2telescope(tel).color for tel in ltel_n])
    data_dir = Path.home()/"data"
    run = "real" #"real"
    tag_inv = "smoothing"
    datestr = sys.argv[2]
    out_dir = data_dir / "inversion" / run / str_tel / tag_inv / datestr #"23022023_1410"
    logging.basicConfig(filename=str(out_dir/f'best_params.log'), level=logging.INFO)#, filemode='w')
    timestr = time.strftime("%d%m%Y-%H%M%S")
    logging.info(timestr)        
            

    mat_dome = mat73.loadmat(str(fGmatrix_dome))['cubesMatrix'] #shape=(N_cubes, N_par_cubes=31)
    vol_center = sio.loadmat(str(fVolcanoCenter))['volcanoCenter'][0]
    vc_x = vol_center[0]
    vc_y = vol_center[1]
    dtc = np.sqrt( (mat_dome[:,25]-vc_x)**2 + (mat_dome[:,26]-vc_y)**2 ) 
    sv_center = (dtc <= 375) 
    
    ###
    R = 375+res#m
    sv_cyl  = (dtc <= R) ###mask anomaly: cylinder with radius = R, height = height_dome 
 
    ####MASK DOME VOXELS
    is_1tel = False
    is_overlap = False
    is_4tel = True
    if is_1tel:
        print("loading NJ region mask...")
        tel = "NJ"
        fmask = ml_dir / "CubesMuonDependances" /  f"Resol_{res}" / tel / f"mask_cubes_{tel}_3p_{res}m.mat"
        mask_cubes = sio.loadmat(str(fmask))['mask'].T[0]
        mask_cubes = mask_cubes != 0 #convert to bool type
    if  is_overlap : 
        print("loading overlap region (SB & SNJ & BR) mask...")
        fmask = ml_dir / "CubesMuonDependances" /  f"Resol_{res}" / "overlap" / "NJ_SB_BR" / f"mask_cubes_overlap_{res}m.mat"
        mask_cubes = sio.loadmat(str(fmask))['mask'].T[0]
        mask_cubes = mask_cubes != 0 #convert to bool type
    if is_4tel : 
        print("loading union 4 telescope regions mask...")
        union_mask_tel = np.ones((4, mat_dome.shape[0]))
        for i, t in enumerate(ltel_n) : 
            if t== "SNJ": 
                t="NJ"
                fmask = ml_dir / "CubesMuonDependances" /  f"Resol_{res}" / t / f"mask_cubes_{t}_3p_{res}m.mat"
            else: fmask = ml_dir / "CubesMuonDependances" /  f"Resol_{res}" / t / f"mask_cubes_{t}_{res}m.mat"
            mask_cubes = sio.loadmat(str(fmask))['mask'].T[0]
            union_mask_tel[i] = mask_cubes
        union_mask_tel = np.sum(union_mask_tel, axis=0)
        mask_cubes = union_mask_tel != 0 #convert to bool type
    
    mask_cubes = mask_cubes != 0 #convert to bool type
    mask_cyl = mask_cubes != 0
    mask_cubes = mask_cubes & sv_center
    mask_cyl = mask_cyl & sv_cyl
    
    
    dome = VoxeledDome(resolution=res, matrix=mat_dome)
    nvox = len(mask_cubes[mask_cubes==True])
    dome.cubes = dome.cubes[mask_cubes] #select voxels of interest
    dome.barycenter = dome.barycenter[mask_cubes]
    dome.xyz_up = dome.xyz_up[mask_cubes]
    dome.xyz_down = dome.xyz_down[mask_cubes]
    #dome.get_distance_matrix()
    all_cubes= dome.cubes
    
    print(f"all_cubes = {all_cubes.shape}, {all_cubes[0,:]}")
    msouth  = all_cubes[:,0, 0, 0] - vc_x < 0
    #####fumeroless
    ffumerole = Path.home()/"QGis_Elements/dome/fumerole.shp"
    shps2 = shapefile.Reader(str(ffumerole)).shapes()
    coord_fumerole = [643042, 1774205]
    xy_fum =np.zeros(shape=(len(shps2), 2))
    # for s, shape in enumerate(shps2):
    #     points = shape.points
    #     long, lat = points[0]
    #     x, y = utm.from_latlon(lat,long)[0],  utm.from_latlon(lat,long)[1]
    #     xy_fum[s,:] = np.array([x,y])
        #ax1.scatter(x-vc_x, y-vc_y,  s=35, marker="^", color="purple")

    #np.argmin()
    coord_center_anom = coord_fumerole#[vc_x+100, vc_y-50 ]
    dtc = np.sqrt( (all_cubes[:,0, 0, 0]-coord_center_anom[0])**2 + (all_cubes[:,0, 0, 1]-coord_center_anom[1])**2 )  
    radius_anom = 100+res
    
    manomsouth = (dtc <= radius_anom)
    #manomsouth = ( 0 < all_cubes[:,0, 0, 0] - vc_x )&  (all_cubes[:,0, 0, 0] - vc_x < 150)
    #manomsouth = manomsouth & ( -100 < all_cubes[:,0, 0, 1] - vc_y )&  (all_cubes[:,0, 0, 1] - vc_y < 10)
    cubes_south =all_cubes[msouth]
    
    
    
    
    print(f"cubes_south = {cubes_south.shape}")
    print(f"msouth = {msouth.shape}")
    print("ok mask cubes")
    #exit()
    
    #ilos_anom = dome.cubes[mask_cubes & mask_cyl]
    
    xrange = [np.min(dome.x), np.max(dome.x)]
    yrange = [np.min(dome.y), np.max(dome.y)]
    zrange = [np.min(dome.z), np.max(dome.z)]
   
    ndome_cubes = dome.cubes.shape[0]

    lc_dome = Line3DCollection(np.concatenate(dome.cubes), colors='red', edgecolor="k", alpha=0.25, linewidths=0.4, linestyles='dotted') 
   
    xrange = [642128+200, 643792-200]
    yrange = [1773448+200, 1775112-200]
    zrange = [1.0494e+03, 1.4658e+03 + 50]
    #zrange = [1.3e3-res/2, 1.3e3+res/2]
    fig = plt.figure(figsize=(12,8))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlim(xrange)
    ax.set_ylim(yrange)
    ax.set_zlim(zrange)
    ax.set_aspect('auto') #'equal'
    ax.grid()
    
    
    
  
    fin = out_dir /  f"rho_post_res{res}m.npy"         
    

    print(fin.exists())
    with open(str(fin), 'rb') as f:
        arr_rho_post = np.load(f)
    
    
    rho_vox_post = arr_rho_post[0,0,0]
    strength_vox_post = frho_to_strength(rho_vox_post)
    
    ####Plot density model
    cmap_rho = palettable.scientific.sequential.Batlow_20.mpl_colormap #'jet' #palettable.scientific.diverging.Roma_20
    rho_min, rho_max, n = .8, 2.7, 100
    range_val = np.linspace(rho_min, rho_max, n)
    norm_r = cm.Normalize(vmin=rho_min, vmax=rho_max)(range_val)
    color_scale_rho =  cmap_rho(norm_r)
    vmin_r, vmax_r = rho_min, rho_max
    
    ####strength model scale
    cmap_strength = palettable.scientific.diverging.Roma_20.mpl_colormap
    strength_min, strength_max = 1e-1, frho_to_strength(rho_max)
    range_val_s = np.logspace(np.log10(strength_min), np.log10(strength_max), n)
    #norm_s = cm.Normalize(vmin=strength_min, vmax=strength_max)
    lognorm_s = cm.LogNorm(vmin=strength_min, vmax=strength_max)
    vmin_s, vmax_s = strength_min, strength_max
    #color_scale_strength = cmap_strength(norm_s(range_val_s))
    color_scale_strength = cmap_strength(lognorm_s(range_val_s))

    
    arg_col_rho =  [np.argmin(abs(range_val-v))for v in rho_vox_post]  
    arg_col_strength =  [np.argmin(abs(range_val_s-v))for v in strength_vox_post]     
    #rho_vox_post = rho_vox_post[mask_slice]
    color_vox_rho = color_scale_rho[arg_col_rho]
    color_vox_strength = color_scale_strength[arg_col_strength]

    
    #mask_density = (rho_vox_post> 2)   
    #color_vox_rho[mask_density ] = np.array([1, 0, 0, 0.5])
    #color_vox_strength[manomsouth] = np.array([1, 0, 0, 0.5])
    vox_anom = strength_vox_post[manomsouth]
    
    Vtot_tel = 1.5e7 #m3
    Vtot_dome = 5e7 #m3
    
    mhighstrength = strength_vox_post >= 10 
    nvox = len( all_cubes[mhighstrength])
    vvox = nvox *res**3 
    vtot_tmp = np.sum(vvox)
    print(f"Volume involved all high strength (sig>= 10 MPa)= {vtot_tmp:.3e} m^3, fV_tel = {vtot_tmp/Vtot_tel*100:.3f}%, fV_dome = {vtot_tmp/Vtot_dome*100:.3f}%")
    
    
    mmidstrength = (10 > strength_vox_post) & (strength_vox_post >= 1 )
    nvox = len( all_cubes[mmidstrength])
    vvox = nvox *res**3 
    vtot_tmp = np.sum(vvox)
    print(f"Volume involved all mid strength ( 10 > sig >= 1 MPa)= {vtot_tmp:.3e} m^3, , fV_tel = {vtot_tmp/Vtot_tel*100:.3f}%, fV_dome = {vtot_tmp/Vtot_dome*100:.3f}%")
    
    
    for strength_thres in [1, 3, 5, 10]:
        mask_strength = vox_anom < strength_thres  #MPa
        nvox = len(vox_anom[mask_strength==True])
        vvox = nvox *res**3 
        vtot_tmp = np.sum(vvox)
        print(f"Volume involved strength anomaly (<{strength_thres} MPa)= {vtot_tmp:.3e} m^3, , fV_tel = {vtot_tmp/Vtot_tel*100:.3f}%, fV_dome = {vtot_tmp/Vtot_dome*100:.3f}%")
        
    strength_thres = 5
    mask_strength = (1 < vox_anom) &  (vox_anom < 10)  #MPa
    nvox = len(vox_anom[mask_strength==True])
    vvox = nvox*res**3     
    vtot_tmp = np.sum(vvox)
    print(f"Volume involved strength anomaly (1<sig<10 MPa)= {vtot_tmp:.3e} m^3, , fV_tel = {vtot_tmp/Vtot_tel*100:.3f}%, fV_dome = {vtot_tmp/Vtot_dome*100:.3f}%")
    
    #green
    m = manomsouth & ( strength_vox_post < strength_thres)
    #color_vox_strength[m] = np.array([0,1,0,0.5])
    
    color_vox_strength[~m] = np.nan
    # pc_sel = Poly3DCollection(np.concatenate(all_cubes),  
    #                             cmap=cmap_rho,
    #                             alpha=0.3,
    #                             facecolors=np.repeat(color_vox_rho,6,axis=0),
    #                             edgecolors=np.clip(color_vox_rho - 0.5, 0, 1),  # brighter 
    #                             norm=Normalize(clip=True),
    #                             linewidths=0.3)
    
    exit()
    
    pc_sel = Poly3DCollection(np.concatenate(all_cubes[m]),  
                                cmap=cmap_strength,
                                alpha=0.3,
                                facecolors=np.repeat(color_vox_strength[m],6,axis=0),
                                edgecolors=np.clip(color_vox_strength[m] - 0.5, 0, 1),  # brighter 
                                norm=lognorm_s,#Normalize(clip=True),
                                linewidths=0.3)
    
    #vmin, vmax = rho_min, rho_max#np.nanmin(val_cubes), np.nanmax(val_cubes)
    #print(f'vmin, vmax = {vmin}, {vmax}')
    #pc_sel.set_clim(vmin, vmax)
    print(vmin_s, vmax_s)
    pc_sel.set_clim(vmin_s, vmax_s)
    #cb = plt.colorbar(pc_sel, orientation='vertical', ax=ax, label="relative density $\\Delta\\overline{\\rho}$ [%]")#[g.cm$^{-3}$]")
    cax = fig.add_axes([ax.get_position().x1+0.12,ax.get_position().y0,0.03, ax.get_position().y1])
    cb = plt.colorbar(ScalarMappable(norm=lognorm_s, cmap=cmap_strength), orientation='vertical', ax=ax, cax=cax, alpha=1., label="unixial compressive strength [MPa]")
    fontsize='x-large'
    #divider = make_axes_locatable(ax)
    #cax = divider.append_axes("right", size="5%", pad=0.05)
    #cb = fig.colorbar(ScalarMappable(norm=lognorm_s, cmap=cmap_strength), cax=cax)    
    #cb.set_label(label="unixial compressive strength [MPa]", size=fontsize)#, labelpad=1.)
    cb.ax.tick_params(which="both",labelsize=fontsize)
    
    
    fig.subplots_adjust(left=0.05, bottom=0.1, right=0.87, top=0.95, wspace=0.15, hspace=0.15)
    
    #ax.add_collection3d(lc_dome)
    #ax.add_collection3d(pc_tel)
    ax.add_collection3d(pc_sel)
    #ax.view_init(90, -90)
    ax.dist = 8    # define perspective (default=10)
    # ax2 = fig.add_subplot(122, projection='3d')
    # ax2.update_from(ax)
    #ax.view_init(30, -120)
    print(f"{time.time()-start_time:0f}s")
    ax.view_init(90, 270)
    fout = out_dir / "dome_anom_vox_bird.png"
    fig.savefig(fout)
    print(f"save {fout}")
    
    ax.view_init(30, -60)
    ax.dist = 8    # define perspective (default=10)
  
    
    ####DEM

    fout = "/Users/raphael/mesh_souf_models/mesh/soufriere_1m_cut.txt"
    #dem = rio.open(dem_file)
    x, y, z  = np.loadtxt(fout, delimiter="\t").T
    #print(np.min(y), np.max(y))
    #exit()
    npts = 1000
    xn, yn = np.linspace(np.nanmin(x), np.nanmax(x), npts),  np.linspace(np.nanmin(y), np.nanmax(y), npts)
    points = np.zeros((len(x),2))
    points[:,0] = x
    points[:,1] = y
    values = z
    X, Y = np.meshgrid(xn, yn)
    # ana_dir = Path.home() /  "data" / tel.name /run /  "ana" / tag_ana
    # rho_dir =  ana_dir / "density" / tag_rho
    fout = out_dir / "grid_dem_tmp.txt"
    if not fout.exists():
        Z = griddata(points, values, (X, Y))
        np.savetxt(fout, Z)
        print(f"save {fout}")
    else: Z = np.loadtxt(fout)
    #kwargs_topo = dict ( alpha=0.2, color='greenyellow', edgecolor='turquoise' )
    kwargs_topo = dict ( alpha=0.2, color='lightgrey', edgecolor='grey' )
    mask_topo = (xrange[0] <X) & (X < xrange[1]) & (yrange[0] <Y) & (Y < yrange[1])
    Z[~mask_topo] = np.nan
    ax.plot_surface(X,Y,Z,**kwargs_topo)
    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    ax.set_zlabel("Z [m]")
    
    fout = out_dir / "dome_anom_vox_z30_eastside.png"
    fig.savefig(fout, transparent=True)
    print(f"save {fout}")
    
    exit()
    ax.view_init(30, 270)
    fout = out_dir / "dome_anom_vox_z30.png"
    fig.savefig(fout)
    print(f"save {fout}")
    
    ax.view_init(50, 270)
    fout = out_dir / "dome_anom_vox_z50.png"
    fig.savefig(fout)
    print(f"save {fout}")
    
    ax.view_init(0, 270)
    fout = out_dir / "dome_anom_vox_z0.png"
    fig.savefig(fout)
    print(f"save {fout}")

    #plt.show()
    
    


