#!/usr/bin/env python3
# -*- coding: utf-8 -*-



import numpy as np
import matplotlib.pyplot as plt
from matplotlib.markers import MarkerStyle
from mpl_toolkits.axes_grid1 import make_axes_locatable
import sys
import os
import scipy.io as sio
from pathlib import Path
import time
import mat73 #read v7.3 mat files
import palettable

#package module(s)
from config import MAIN_PATH
from modeling3d.inversion import Inversion
from modeling3d.voxel import Voxel
from raypath import RayPathSoufriere
from survey import CURRENT_SURVEY
from telescope import str2telescope

if __name__=="__main__":
    t0= time.time()

    survey = CURRENT_SURVEY
    
    main_path = MAIN_PATH #Path(__file__).parents[2] 

    surface_grid = survey.surface_grid
    surface_center = survey.surface_center
    res_vox = 64

    voxel = Voxel(  surface_grid = surface_grid,
                    surface_center = surface_center, 
                    res_vox = res_vox)
    
    dout_vox_struct = survey.path / "voxel"
    dout_vox_struct.mkdir(parents=True, exist_ok=True)
    fout_vox_struct = dout_vox_struct / f"voxMatrix_res{res_vox}m.npy"
    if fout_vox_struct.exists(): 
        print(f"Load {fout_vox_struct}") #.relative_to(main_path)
        vox_matrix = np.load(fout_vox_struct)
        voxel.vox_matrix = vox_matrix
        voxel.barycenters =  voxel.vox_matrix[:,25:28]

    else : 
        print(f"generateMesh() start")
        voxel.generateMesh()
        vox_matrix = voxel.vox_matrix
        np.save(fout_vox_struct, vox_matrix)

    print(f"generateMesh() end --- {time.time() - t0:.1f} s")

    vox_dist_to_center = np.sqrt( (voxel.barycenters[:,0] - surface_center[0]) **2 + (voxel.barycenters[:,1] - surface_center[1]) **2 )
    
    radius_max = 375 #m
    mask_center = (vox_dist_to_center <= radius_max)
    
    radius_cyl = 2*res_vox #m
    mask_cyl = (vox_dist_to_center <= radius_cyl)

    
    tel = survey.telescopes['SNJ']

    tel_files_path = survey.path / 'telescope' / tel.name 

    dir_ray = tel_files_path / "raypath" / f"az{tel.azimuth:.1f}_elev{tel.elevation:.1f}" 
   
    pklfile_voxray = dir_ray / "voxel" / f"voxray_res{res_vox}m.pkl"

    import pickle

    with open(pklfile_voxray, 'rb') as f : 
        voxray_matrix = pickle.load(f)


    conf = list(tel.configurations.keys())[0]

    G = voxray_matrix[conf]
    print(f"voxray_matrix = {G.shape}")
    mask_vox = np.isnan(G)
    #print(np.any(np.isnan(G), axis = 1))

    #print(np.any(G==0., axis = 1))
    mask_vox = ~np.any(np.isnan(G), axis = 0)

    G[np.isnan(G)] = 0
    G[G != 0] = 1
    nvox_per_ray = np.count_nonzero(G, axis=1) # (nray,)
    nz = nvox_per_ray != 0
    G[nz] = np.divide(G[nz].T, nvox_per_ray[nz]).T
    
    vox_val = np.ones(shape=G.shape[1]) * 2.0   # (nvox,)
    vox_val[mask_cyl] = 1.0 
    vox_unc = vox_val * 0.1 # (nvox,)
    ray_val = G @ vox_val # (nray,) 
    unc = G @ vox_unc # (nray,)
    print(f"nvox_per_ray.shape = {nvox_per_ray.shape}")
    mask_nvox_nnan = ~np.all(np.isnan(G), axis=1) # (nray,)
    print(f"mask_nvox_nnan.shape = {mask_nvox_nnan.shape}")
    # print(np.all(np.isnan(G), axis=1))
    mask_rays = ( nvox_per_ray != 0 ) & mask_nvox_nnan
    print(f"mask_rays.shape = {mask_rays.shape}")
    print(np.any(np.all(vox_matrix==0, axis=1)==True))
   
    #data_obs[mask] = data_obs[mask] / nvox_per_ray[mask]
    #data_unc[mask] = data_unc[mask] / nvox_per_ray[mask]
    data_gaus = np.zeros(ray_val.shape)
   
    data_gaus[mask_rays] = np.random.normal(loc=ray_val[mask_rays], scale=unc[mask_rays])
    print(voxel.vox_matrix.shape, data_gaus[mask_rays].shape, unc[mask_rays].shape, G[mask_rays].shape )
    inversion = Inversion(voxel = voxel, 
                          data = data_gaus[mask_rays], 
                          unc = unc[mask_rays], 
                          voxray_matrix = G[mask_rays], )

    voxel.getVoxelDistances()
    mat_voxdist = voxel.vox_distances

    ###test 
    l_rho0 = [2.0, ]#, 2.0]#[1.8, 2.0]
    l_sig = [3e-1, ] # np.linspace(0.1, 0.8, 8)#[1e-1, 2e-1, 3e-1, 4e-1]#[2e-1, 3e-1] #[1e-1, 2e-1, 3e-1, 4e-1, 5e-1, 6e-1] # 7e-1, 8e-1, 9e-1, 1]
    l_lc = [res_vox, ] # np.linspace(50, 400, 8)#[1e2, 1.1e-1, 1.2e2, 1.3e-1, 1.4e2, 1.5e2, 1.6e2, 1.7e2, 1.8e2] #1e1, 2e1, 4e1, 6e1, 8e1, 
    nvox = G.shape[1]
    mat_rho_post = np.ones(shape=(len(l_rho0), len(l_sig), len(l_lc), nvox) )
    #mat_chi2 = np.ones(shape=(len(l_rho0), len(l_sig), len(l_lc), nvox) )
    mat_std_dev = np.ones(shape=(len(l_rho0), len(l_sig), len(l_lc), nvox) )
    mat_misfit = np.ones(shape=(len(l_rho0), len(l_sig), len(l_lc)) )

    for i, rho0 in enumerate(l_rho0):
    
        for j, sig in enumerate(l_sig):
    
            for k, lc in enumerate(l_lc): 
    
                inversion.smoothing(err_prior=sig, distance=mat_voxdist, length=lc)#vec_scaling) #damping=mat_scaling               
                #inv.gaus_smoothing(err_prior=sig, d=d, l=lc,damping=None) #diag_weight=1e1 damping=mat_scaling
                print(f"smoothing: (rho0, err_prior, correlation length) = ({rho0:.1f} g/cm^3, {sig:.3e} g/cm^3, {lc:.3e} m) --- {(time.time() - t0):.3f}  s ---") 
                vec_rho0 = rho0*np.ones(nvox)
                vec_sig  = sig*np.ones(nvox)
                inversion.get_model_post(rho0=vec_rho0)
                mat_rho_post[i,j,k] = inversion.rho_post

                print(f"rho_post = {mat_rho_post[i,j,k]} , shape={mat_rho_post[i,j,k].shape}")


    cmap_rho = palettable.scientific.sequential.Batlow_20.mpl_colormap #'jet' #palettable.scientific.diverging.Roma_20
    rho_min, rho_max, n = .5, 2.5, 100
    range_val_rho = np.linspace(rho_min, rho_max, n)
    import matplotlib.colors as mplc
    from matplotlib.cm import ScalarMappable
    norm_r = mplc.Normalize(vmin=rho_min, vmax=rho_max)(range_val_rho)
    color_scale_rho =  cmap_rho(norm_r)
    vmin_r, vmax_r = rho_min, rho_max

    rho_post = mat_rho_post[0,0,0]

    print(f"rho_post_cyl = {rho_post[mask_cyl]}")
    print(f"rho_post = {rho_post[~mask_cyl]}")

    print(f"voxel.vox_xyz.shape = {voxel.vox_xyz.shape}")


    post_col = [np.argmin(abs(range_val_rho - v)) for v in vox_val]    #rho_post

    color_vox = color_scale_rho[post_col]
    
    ##plot model
    fig = plt.figure()
    ax = fig.add_subplot(projection = "3d")
    voxel.getVoxels()
    #color_vox = np.array([[0.98039216, 0.8,  0.98039216, 1.        ]])
    kwargs_mesh = dict(facecolor=color_vox, edgecolor='grey', alpha=0.2)
    voxel.plot3Dmesh(ax=ax, vox_xyz=voxel.vox_xyz[~mask_cyl], **kwargs_mesh)#color_vox=color_vox)
    kwargs_mesh['alpha'] = 0.5
    voxel.plot3Dmesh(ax=ax, vox_xyz=voxel.vox_xyz[mask_cyl], **kwargs_mesh)
    kwargs_topo = dict(color='lightgrey', edgecolor='grey',  alpha=0.2 )
    voxel.plotTopography(ax, **kwargs_topo)
    dx=1000
    xrange = [surface_center[0]-dx, surface_center[0]+dx]
    yrange = [surface_center[1]-dx, surface_center[1]+dx]
    zrange = [1.0494e+03, 1.4658e+03 + 50]
    ax.set_xlim(xrange)
    ax.set_ylim(yrange)
    ax.set_zlim(zrange)
    ax.set_aspect('auto') #'equal'
    ax.grid()
    ax.view_init(30, -60)
    ax.dist = 8    # define perspective (default=10)
    xstart, xend = ax.get_xlim()
    ax.xaxis.set_ticks(np.arange(xstart, xend, 5e2))
    ystart, yend = ax.get_ylim()
    ax.yaxis.set_ticks(np.arange(ystart, yend, 5e2))
    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    ltel_n = ["SB", "SNJ", "BR", "OM"]
    str_tel =  "_".join(ltel_n)
    ltel_coord = np.array([ str2telescope(tel).utm for tel in ltel_n])
    ltel_color = np.array([ str2telescope(tel).color for tel in ltel_n])
    ax.scatter(ltel_coord[:,0], ltel_coord[:,1], ltel_coord[:,-1], c=ltel_color, s=30,marker='s',)

    # raypath = RayPathSoufriere[tel.name]
    # thickness = {key: ray['thickness'] for key, ray in raypath.raypath.items() }
    # mask = np.isnan(thickness['3p1']).flatten()
    # conf = tel.configurations['3p1']
    # front, rear = conf.panels[0], conf.panels[-1]
    # tel.plot_ray_paths(ax=ax, front_panel=front, rear_panel=rear, mask=mask, rmax=1500,  color='grey', linewidth=0.3 )#

    plt.show()

    exit()

        