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
from matplotlib.ticker import MultipleLocator,EngFormatter, ScalarFormatter
from matplotlib.offsetbox import AnchoredText
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.gridspec import GridSpec
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

#personal modules
from configuration import str2telescope, Telescope
from analysis import AcqVars, Observable

from plot_density_3d import VoxeledDome


from inversion import  DataSynth
from configuration import dict_tel
import warnings
warnings.filterwarnings("ignore")

import palettable

def scatter_rhopost(fig, ax, title, xyz, colors, **kwargs):
    x, y, z = xyz.T
    ax.scatter(x, y, z, color=colors, **kwargs)
    ax.set_title(title)
    


if __name__=="__main__":
    start_time = time.time()
    print("Start: ", time.strftime("%H:%M:%S", time.localtime()))#start time
            
    conf='3p1'
    param_dir = Path.home() / 'muon_code_v2_0' / 'AcquisitionParams'
    print(time.strftime("%H%M%S-%d%m%y"))
    res= int(sys.argv[1]) #m
    ml_dir = Path.home()  / "MatLab" / "MuonCompute_Volumes_mnt" 
    fGmatrix_dome = ml_dir /  "matrices" / f"cubesMatrix_{res}.mat"
    fVolcanoCenter = ml_dir / "matrices" / "volcanoCenter.mat"
    
    
    
    ####
    ###coordinates
    ltel = ["SNJ", "SB", "BR", "OM"]
    ltel_coord = np.array([ str2telescope(tel).utm[:-1] for tel in ltel])
    ltel_color = np.array([ str2telescope(tel).color for tel in ltel])

    mat_dome = mat73.loadmat(str(fGmatrix_dome))['cubesMatrix'] #shape=(N_cubes, N_par_cubes=31)
    vol_center = sio.loadmat(str(fVolcanoCenter))['volcanoCenter'][0]
    vc_x = vol_center[0]
    vc_y = vol_center[1]
    dtc = np.sqrt( (mat_dome[:,25]-vc_x)**2 + (mat_dome[:,26]-vc_y)**2 ) 
    sv_center = (dtc <= 375) 
    
    ###
    R = 100#m
    sv_cyl  = (dtc <= R) ###mask anomaly: cylinder with radius = R, height = height_dome 
    
    #######telescope sight cone :
    ####MASK DOME VOXELS
    is_1tel = False
    is_overlap = False
    is_4tel = True
    ltel_n = ["SB", "SNJ", "BR", "OM"]
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
    #print(f"mask_cubes = , shape = {mask_cubes.shape}")
    #print(f"sv_center = {sv_center}, shape={sv_center.shape}")
    mask_cubes = mask_cubes & sv_center
    #print(f"mask_cubes &  sv_center = {mask_cubes}, shape={mask_cubes[mask_cubes==1].shape}")
    mask_cyl = mask_cyl & sv_cyl


    data_dir = Path.home()/"data"
    run = "synthetic" #"real"
    
    tag_inv = "smoothing"
    datestr = sys.argv[2]
    out_dir = data_dir / "inversion" / run / tag_inv / datestr #"23022023_1410"
    logging.basicConfig(filename=str(out_dir/f'best_params.log'), level=logging.INFO)#, filemode='w')
    timestr = time.strftime("%d%m%Y-%H%M%S")
    logging.info(timestr)
    
    dome = VoxeledDome(resolution=res, matrix=mat_dome)
    nvox = len(mask_cubes[mask_cubes==True])
    mask_dome = (mask_cubes==True)
    dome.cubes = dome.cubes[mask_dome] #select voxels of interest
    dome.barycenter = dome.barycenter[mask_dome]
    dome.xyz_up = dome.xyz_up[mask_dome]
    dome.xyz_down = dome.xyz_down[mask_dome]
    dome.get_distance_matrix()

    ####test smoothing matrix    
    '''
    sigma, l = 1e-3, 32
    diag_weight = 10
    smoothing_matrix = sigma**2 *np.exp(-dome.d/l) * np.exp(diag_weight * np.diag(np.ones(shape=nvox)))
    fig, ax = plt.subplots(figsize=(10,10))
    vmin, vmax = np.nanmin(smoothing_matrix[smoothing_matrix != 0]), np.nanmax(smoothing_matrix)
    im = ax.imshow(smoothing_matrix, norm=LogNorm(vmin=vmin, vmax=vmax))
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = fig.colorbar(im, cax=cax, orientation="vertical")
    cbar.set_label(label="smoothing [g.cm$^{-3}$]",size=12)
    ax.set_title("Smoothing matrix $C_{\\rho}$" + f" with $\\sigma$ = {sigma} g.cm$^{-3}$; $l$ = {l:.1f} m", fontsize=12)
    fig.savefig(out_dir/f"smoothing_matrix_res{res}m.png")
    '''
    ####
    #exit()
    
    rho_vox_true = np.load(out_dir /f"rho_true_res{res}m.npy")
    #unc_vox_true = np.load(out_dir /f"unc_true_res{res}m.npy")
    
    #rho_vox_true = np.ones(shape = mask_d ome.shape)*np.nan
    #rho_vox_true[mask_dome] = 2.0
    #rho_vox_true[mask_cyl] = 1.0
    #rho_vox_true = rho_vox_true[~np.isnan(rho_vox_true)]
    #print(f"rho_vox_true[~np.isnan(rho_vox_true)] = {rho_vox_true.shape}")
    
    
    Gmat = np.loadtxt(out_dir/ f"Gmat_all_res{res}m.txt")#[mask_dome]
    data_syn = np.loadtxt(out_dir/ f"data_syn_all_res{res}m.txt")#[mask_dome]
    unc_syn = np.loadtxt(out_dir/ f"unc_syn_all_res{res}m.txt")#[mask_dome]

    
    ####Define slice mask
    zslice=1.3e3 #m
    ilos_vox = dome.cubes
    mask_slice =  (zslice - res <= dome.barycenter[:,-1]) & (dome.barycenter[:,-1] <= zslice )
    ilos_vox  = ilos_vox[mask_slice]
    nvox_slice = ilos_vox.shape[0]
    ilos_bar = dome.barycenter[mask_slice]
    ilos_xyzup = dome.xyz_up[mask_slice]
    ilos_xyzdown = dome.xyz_down[mask_slice]
    
    '''
    dome.get_distance_matrix()
    fig, ax = plt.subplots()
    Z= dome.d
    im = ax.imshow(Z, cmap="viridis")
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = fig.colorbar(im, cax=cax, orientation="vertical")
    cbar.set_label(label="distance voxel $d$ [m]",size=12)
    fout = out_dir/f"distance_voxel_res{res}m"
    fig.savefig(f"{fout}.png")    
    print(f"save {fout}.png")
    '''




    dx = 700
    xmin, xmax = vc_x-dx, vc_x+dx
    #xmin, xmax = 642128+200, 643792-200
    xrange = [xmin, xmax]
    dy = dx
    ymin, ymax = vc_y-dy, vc_y+dy
    #ymin, ymax = 1773448+200, 1775112-200
    yrange = [ymin, ymax]
    zmin, zmax = 1.0494e+03, 1.4658e+03 + 50
    zrange = [zmin, zmax]
    elev, azim = 90, -90
    #####
  
    ####
    mat_rho0 = np.loadtxt(out_dir/ "rho0.txt",delimiter="\t")
    if mat_rho0.shape == ():  mat_rho0 = np.array([mat_rho0.mean()])
    mat_sigma = np.loadtxt(out_dir/ "sigma_prior.txt",delimiter="\t")
    if mat_sigma.shape == ():  mat_sigma = np.array([[mat_sigma.mean()]])
    mat_length = np.loadtxt(out_dir/ "correlation_length.txt", delimiter="\t")
    if mat_length.shape == ():  mat_length = np.array([[mat_length.mean()]])
    
    
    fin = out_dir /  f"rho_post_res{res}m.npy"         
    print(fin.exists())
    with open(str(fin), 'rb') as f:
        arr_rho_post = np.load(f)
    
    fin = out_dir /  f"std_dev_res{res}m.npy"         
    print(fin.exists())
    with open(str(fin), 'rb') as f:
        arr_std_post = np.load(f)
        
    ####Plot density model
    cmap = palettable.scientific.sequential.Batlow_20.mpl_colormap #'jet' #palettable.scientific.diverging.Roma_20
    rho_min, rho_max, n = .1, 2.0, 100
    range_val = np.linspace(rho_min, rho_max, n)
    norm = cm.Normalize(vmin=rho_min, vmax=rho_max)(range_val)
    std_min, std_max, n = .02, .2, 100
    range_val_std = np.linspace(std_min, std_max, n)
    norm_std = cm.Normalize(vmin=std_min, vmax=std_max)(range_val_std)
    color_scale_std =  cmap(norm_std)
    vmin_std, vmax_std = std_min, std_max
    #color_scale =  plt.colormaps[cmap](norm) #cmap here is a string 
    color_scale =  cmap(norm)
    vmin, vmax = rho_min, rho_max
    rho_vox_true_slice = rho_vox_true[mask_slice]
    arg_col_true =  [np.argmin(abs(range_val-v))for v in rho_vox_true_slice]
    color_vox_true = color_scale[arg_col_true]
    verts = ilos_xyzdown
    #exit()
    
    
    nrho = len(mat_rho0)
    nrow = mat_sigma.shape[0]
    ncol = mat_length.shape[1]
    print(f"nrow, ncols={nrow},{ncol}")

    misfit_d = np.ones(shape=(nrho, nrow, ncol) )
    misfit_m = np.ones(shape=(nrho, nrow, ncol) )
    nlos, nvox = Gmat.shape
    nvox_per_los = np.count_nonzero(Gmat, axis=1)
    print(f'Gmat={Gmat[Gmat!=0]}')
    nvox_per_los = np.ones(nlos)
    ####Plot mosaic 
    ilos_vox = dome.cubes

    lzslice = [1.25e3, 1.3e3, 1.35e3, 1.4e3]

    fig_best_data = plt.figure(figsize=(10,8))#, constrained_layout=True)
    fig_best_model = plt.figure(figsize=(10,8))#, constrained_layout=True)
    fig_best_std = plt.figure(figsize=(10,8))
    # gs_best = GridSpec(1, len(lzslice),
    #             wspace=0.0, hspace=0.0, 
    #             top=0.97-0.5/(nrow+1), bottom=0.5/(nrow+1), 
    #             left=0.5/(ncol+1), right=0.90-0.5/(ncol+1))
    ax_best_data = fig_best_data.subplot_mosaic(
            [lzslice[:2],   lzslice[2:]], #[lzslice],#
            sharex=True, sharey=True
        )
    ax_best_model = fig_best_model.subplot_mosaic(
            [lzslice[:2],   lzslice[2:]], #[lzslice],#
            sharex=True, sharey=True
        )
    ax_best_std = fig_best_std.subplot_mosaic(
            [lzslice[:2],   lzslice[2:]], #[lzslice],#
            sharex=True, sharey=True
        )
    #plt.gcf().text(0.4, 0.93, f"$\\rho_{0}$ = {rho0}"+" g.cm$^{-3}$,"+f"\nz = {zslice} m a.s.l", fontsize=fontsize+4)

    

    for ix_z, zslice in enumerate(lzslice):
        ####Define slice mask
        mask_slice =  (zslice - res <= dome.barycenter[:,-1]) & (dome.barycenter[:,-1] <= zslice )
        ilos_vox_s  = ilos_vox[mask_slice]
        nvox_slice = ilos_vox_s.shape[0]
        ilos_bar = dome.barycenter[mask_slice]
        ilos_xyzup = dome.xyz_up[mask_slice]
        ilos_xyzdown = dome.xyz_down[mask_slice]
        
        rho_vox_true_slice = rho_vox_true[mask_slice]
        arg_col_true =  [np.argmin(abs(range_val-v))for v in rho_vox_true_slice]
        color_vox_true = color_scale[arg_col_true]
        verts = ilos_xyzdown
        #verts[:, :, :-1] = verts[:, :, :-1] - np.array([xmin, ymin])
        for ix_rho0 in range(nrho): #
            rho0 = mat_rho0[ix_rho0]
            fig = plt.figure(figsize=(ncol+2, nrow+2)) 
            fig_std = plt.figure(figsize=(ncol+2, nrow+2)) 
            # gs = GridSpec(nrow, ncol,
            #     wspace=0.0, hspace=0.0, 
            #     top=0.97-0.5/(nrow+1), bottom=0.5/(nrow+1), 
            #     left=0.08+0.5/(ncol+1), right=.98-0.5/(ncol+1))
            gs = GridSpec(nrow, ncol,
                wspace=0.0, hspace=0.0, 
                top=0.97-0.5/(nrow+1), bottom=0.5/(nrow+1), 
                left=0.5/(ncol+1), right=0.90-0.5/(ncol+1))
            gs_std = GridSpec(nrow, ncol,
                wspace=0.0, hspace=0.0, 
                top=0.97-0.5/(nrow+1), bottom=0.5/(nrow+1), 
                left=0.5/(ncol+1), right=0.90-0.5/(ncol+1))
            fontsize = 8
            plt.gcf().text(0.4, 0.93, f"$\\rho_{0}$ = {rho0}"+" g.cm$^{-3}$,"+f"\nz = {zslice} m a.s.l", fontsize=fontsize+4)
            ####
            
        
            Nd = len(data_syn)
            print(f"Nd={Nd}")
            lphi = []
            c=0
            
            for i in range(nrow):
                #if i > 0 : break
                for j in range(ncol):
                    sigma, length= mat_sigma[i, j], mat_length[i,j]
                    key = f"{sigma:.3f},{length:.3f}"
                    #apply mask for given slice
                    rho_vox_post = arr_rho_post[ix_rho0,i,j]
                    std_vox_post = arr_std_post[ix_rho0,i,j]
                    data_post = Gmat @ rho_vox_post #/ nvox_per_los
                    misfit_m[ix_rho0,i,j] = np.linalg.norm(rho_vox_post - rho_vox_true)**2
                    rho_vox_post = rho_vox_post[mask_slice]
                    std_vox_post = std_vox_post[mask_slice]
                    misfit_d[ix_rho0,i,j] = 1/Nd *  np.sum( (data_syn - data_post)**2 / unc_syn**2) 
                
                    
                    arg_col =  [np.argmin(abs(range_val-v))for v in rho_vox_post] 
                    color_vox = color_scale[arg_col]
                    x,y,z = ilos_bar.T
                    arg_col_std=  [np.argmin(abs(range_val-v))for v in std_vox_post]    
                    
                    
                    ax= plt.subplot(gs[i,j])
                    ax.annotate(f"{c}", xy=(0.5, .5), xycoords='axes fraction', xytext=(.85, .05), fontsize=6)
                    if i== 0 and j==0: 
                        ax.annotate('$\\sigma$ [g.cm$^{-3}$]', rotation='vertical', xy=(-0.4, 0.1), xycoords='axes fraction', xytext=(-0.4, 0.9), 
                                ha="center", va="center", arrowprops=dict(arrowstyle="->", color='k'))
                        ax.annotate('$l$ [m]', xy=(0.9, 1.4), xycoords='axes fraction', xytext=(0.1, 1.4), 
                                ha="center", va="center",arrowprops=dict(arrowstyle="->", color='k'))
                    ax.set_xticklabels([])
                    ax.set_yticklabels([])
                    ax.tick_params(axis="y",direction="in", pad=-22)
                    ax.tick_params(axis="x",direction="in", pad=-15)
                    ax.tick_params(which="both", bottom=True, top=True, left=True, right=True)
                    if i==0: 
                        ax.set_title(f"{length:.3e}", fontsize=8)
                
                    if j==0: 
                        ax.set_ylabel(f"{sigma:.3e}", fontsize=8)
                    
                    for l, v in enumerate(verts):
                        pc = PolyCollection([v[:,:-1]],  
                                                cmap=cmap,
                                                #alpha=0.3,
                                                facecolors=color_vox[l], #np.repeat(color_vox,axis=0),
                                                edgecolors=np.clip(color_vox[l] - 0.5, 0, 1),  # brighter 
                                                norm=Normalize(clip=True),
                                                linewidths=0.3)
                        pc.set_clim(vmin, vmax)
                        ax.add_collection(pc)
                        pcstd = PolyCollection([v[:,:-1]],  
                                                cmap=cmap,
                                                #alpha=0.3,
                                                facecolors=color_vox_std[l], #np.repeat(color_vox,axis=0),
                                                edgecolors=np.clip(color_vox_std[l] - 0.5, 0, 1),  # brighter 
                                                norm=Normalize(clip=True),
                                                linewidths=0.3)
                        pc.set_clim(vmin, vmax)
                        ax.add_collection(pc)
                    
                    ax.scatter(ltel_coord[:,0], ltel_coord[:,1], c=ltel_color, s=10,marker='s',)
                    s = "$\\phi_{d}$"+f"={misfit_d[ix_rho0,i,j]:.2e}\n"+"$\\phi_{m}$"+f"={misfit_m[ix_rho0,i,j]:.2e}"
                    ax.annotate(s, xy=(0.5, .5), xycoords='axes fraction', xytext=(0.1, .75), fontsize=fontsize-2)
                    #anchored_text = AnchoredText(s, loc="upper left", frameon=False, prop=dict(fontsize=fontsize-2))
                    ax.set_xlim(xrange)
                    ax.set_ylim(yrange)
                    ax.set_aspect('auto') #'equal'
                    ax.grid(True, which='both',linestyle='-', linewidth="0.3 ", color='grey')
                    
                    c+=1
                    
            ###select model for a given set of parameters
            #sig_sel, lc_sel = 0.3, 160 #manuscrit
            sig_sel, lc_sel = 0.2, 100
            ###find the best data misfit_d i.e closer to one, and minimal model misfit_m  (~squared residuals)
            phi_d = misfit_d[ix_rho0]
            a=abs(phi_d-1)
            try : 
                ij_d = (np.unravel_index(np.argmin(abs(mat_sigma-sig_sel)), a.shape)[0], np.unravel_index(np.argmin(abs(mat_length-lc_sel)[1]), a.shape)[1])
            except : 
                ij_d = np.unravel_index(np.argmin(abs(mat_sigma-sig_sel)), a.shape)
           
            best_misfit_d = phi_d[ij_d]
            sigma_d, length_d = mat_sigma[ij_d], mat_length[ij_d]
            best_par_d =f"$\\rho_{0}$ = {rho0}"+" g.cm$^{-3}$, " +f"$\\sigma$ = {sigma_d}"+" g.cm$^{-3}$, " +f"$l$ = {length_d}"+" m\n"
            
            
            phi_m = misfit_m[ix_rho0]
            #ij_m = np.unravel_index(np.argmin(phi_m, axis=None), phi_m.shape) #np.argmin(abs(misfit_d[ix_rho0]-1), axis=1)
            try : 
                ij_m = (np.unravel_index(np.argmin(abs(mat_sigma-sig_sel)), a.shape)[0], np.unravel_index(np.argmin(abs(mat_length-lc_sel)[1]), a.shape)[1])
            except : 
                ij_m = np.unravel_index(np.argmin(abs(mat_sigma-sig_sel)), a.shape)
            best_misfit_m = phi_m[ij_m]
            sigma_m, length_m = mat_sigma[ij_m], mat_length[ij_m]
            best_par_m =f"$\\rho_{0}$ = {rho0}"+" g.cm$^{-3}$, " +f"$\\sigma$ = {sigma_m}"+" g.cm$^{-3}$, " +f"$l$ = {length_m}"+" m\n"
            str_best = "best regularization params:\n", "data:\n$\\phi_{d}$"+f" = {best_misfit_d:.2e} for " +best_par_d + "model:\n$\\phi_{m}$"+f" = {best_misfit_m:.2e} for " +best_par_m 
            print(str_best)
            logging.info(str_best)
            
            
            print(fig.get_axes())
          
            axes = np.asarray(fig.get_axes()).reshape(nrow, ncol)
            #axes[ij_d].plot()
            axes[ij_d].annotate("**", xy=(0.5, .5), xycoords='axes fraction', xytext=(.85, .9), fontsize=fontsize+2, color="red")
            axes[ij_m].annotate("**", xy=(0.5, .5), xycoords='axes fraction', xytext=(.85, .8), fontsize=fontsize+2, color="green")
            ####Add color bar for density
            ax1 = plt.subplot(gs[nrow-1, ncol-1])
            try : 
                ax2 = plt.subplot(gs[nrow-4, ncol-1])
            except : 
                ax2 = plt.subplot(gs[0, 0])
                
            cax = fig.add_axes([ax1.get_position().x1+0.03,ax1.get_position().y0,0.02,ax2.get_position().y0])
            cb = fig.colorbar(pc,  cax=cax)
            #ticks = np.linspace(0.2, 2.0, 10)
            #cb.ax.set_yticklabels([f"{i:.1f}" for i in ticks])
            locator = MultipleLocator(2e-1)
            cb.ax.yaxis.set_major_locator(locator)
            cb.set_label(label="density $\\~{\\rho}$ [g.cm$^{-3}$]", size=fontsize+2, labelpad=1.)
            cb.ax.tick_params(which="both",labelsize=fontsize)
            ####plot model (upper right mosaic)
            ax1 = plt.subplot(gs[0, ncol-1])
            width = ax1.get_position().width
            x0 = ax1.get_position().x0+width+0.03 #left
            y0 = ax1.get_position().y0 #bottom
            height = ax1.get_position().height
            ax = fig.add_axes([x0, y0, width, height])
            ax.set_title("model", fontsize=10)
            for i, v in enumerate(verts):
                pc = PolyCollection([v[:,:-1]],  
                                        cmap=cmap,
                                        #alpha=0.3,
                                        facecolors=color_vox_true[i], #np.repeat(color_vox,axis=0),
                                        edgecolors=np.clip(color_vox_true[i] - 0.5, 0, 1),  # brighter 
                                        norm=Normalize(clip=True),
                                        linewidths=0.3)
                ax.add_collection(pc)
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.tick_params(axis="y",direction="in", pad=-22)
            ax.tick_params(axis="x",direction="in", pad=-15)
            ax.tick_params(which="both", bottom=True, top=True, left=True, right=True)
            ax.set_xlim(xrange)
            ax.set_ylim(yrange)
            ax.set_aspect('auto') #'equal'
            ax.grid(True, which='both',linestyle='-', linewidth="0.3 ", color='grey')        
            
            fout = out_dir / f"rho_post_rho0{rho0}_z{int(zslice)}m_res{res}m"
            fig.savefig(f"{str(fout)}.png")   
            print(f"save {fout}.png")
            print("End : ", time.strftime("%H:%M:%S", time.localtime()))#start time
            
             
            #####plot best misfit data inversion
            #ax_best = fig_best.add_subplot(gs_best[ix_z])
            dict_ax={'data':ax_best_data[zslice] , 'model':ax_best_model[zslice], 'std':ax_best_std[zslice]}
            for (key, axb) in zip(dict_ax.items()):
                #axb = ax_best[zslice] 
                sigma, length= mat_sigma[ij_d], mat_length[ij_d]
                #apply mask for given slice
                rho_vox_post = arr_rho_post[ix_rho0, ij_d[0], ij_d[1]]
                rho_vox_post = rho_vox_post[mask_slice]
                arg_col =  [np.argmin(abs(range_val-v))for v in rho_vox_post]   
                color_vox = color_scale[arg_col]
                std_vox_post = arr_std_post[ix_rho0, ij_d[0], ij_d[1]]
                std_vox_post = std_vox_post[mask_slice]
                arg_col_std =  [np.argmin(abs(range_val_std-v))for v in std_vox_post]    
                color_vox_std = color_scale[arg_col_std]
                for l, v in enumerate(verts):
                    pc = PolyCollection([v[:,:-1]-np.array([vc_x, vc_y])],  
                                            cmap=cmap,
                                            #alpha=0.3,
                                            facecolors=color_vox[l], #np.repeat(color_vox,axis=0),
                                            edgecolors=np.clip(color_vox[l] - 0.5, 0, 1),  # brighter 
                                            norm=Normalize(clip=True),
                                linewidths=0.3)
                    pc.set_clim(vmin, vmax)
                    axb.add_collection(pc)
                    # pcstd = PolyCollection([v[:,:-1]-np.array([vc_x, vc_y])],  
                    #                         cmap=cmap,
                    #                         #alpha=0.3,
                    #                         facecolors=color_vox_std[l], #np.repeat(color_vox,axis=0),
                    #                         edgecolors=np.clip(color_vox_std[l] - 0.5, 0, 1),  # brighter 
                    #                         norm=Normalize(clip=True),
                    #             linewidths=0.3)
                    # pcstd.set_clim(vmin, vmax)
                    # axb.add_collection(pcstd)

                locator = MultipleLocator(2e2)
                axb.xaxis.set_major_locator(locator)
                axb.yaxis.set_major_locator(locator)
                axb.scatter(ltel_coord[:,0]-vc_x, ltel_coord[:,1]-vc_y, c=ltel_color, s=40,marker='s',)
                #axb.annotate(f"\nz = {zslice:.0f} m a.s.l", 
                #             xy=(0.5, .5), xycoords='axes fraction', xytext=(.05, .85), fontsize=14) #xytext=(.25, .85) #<- centered
                anchored_text = AnchoredText(f"z = {zslice:.0f} m a.s.l", loc="upper left", frameon=True, prop=dict(fontsize=12))
                axb.add_artist(anchored_text)
                #axb.annotate("$\\phi_{d}$"+f"={misfit_d[ix_rho0,i,j]:.2e}", xy=(0.5, .5), xycoords='axes fraction', xytext=(0.15, .85), fontsize=fontsize-2)
                axb.set_xlim([-dx, +dx])
                axb.set_ylim([-dy, +dy])
                axb.set_aspect('auto') #'equal'
                axb.grid(True, which='both',linestyle='dotted', linewidth="0.3 ", color='grey')

                axbin = inset_axes(axb,
                        width="22.5%", # width = 30% of parent_bbox
                        height=.75, # height : 1 inch
                        loc='upper right')
                
                #axbin.set_title("model", fontsize=10)
                #axbin.annotate(f"model", xy=(0.5, .5), xycoords='axes fraction', xytext=(.4, .9), fontsize=8)
                #anchored_text = AnchoredText("model", loc="lower right",  frameon=False, prop=dict(fontsize=6))#fontweight="bold"
                #axbin.add_artist(anchored_text)
                for i, v in enumerate(verts):
                
                    pc = PolyCollection([v[:,:-1]],  
                                            cmap=cmap,
                                            #alpha=0.3,
                                            facecolors=color_vox_true[i], #np.repeat(color_vox,axis=0),
                                            edgecolors=np.clip(color_vox_true[i] - 0.5, 0, 1),  # brighter 
                                            norm=Normalize(clip=True),
                                            linewidths=0.3)
                    pc.set_clim(vmin, vmax)
                    axbin.add_collection(pc)
                #axbin.set_xticklabels([])
                #axbin.set_yticklabels([])
                
                axbin.set_xlim([xrange[0]+200,xrange[1]-200])
                axbin.set_ylim([yrange[0]+200,yrange[1]-200])
                #axbin.axis('off')
                axbin.tick_params(
                axis='both',          # changes apply to the x-axis
                which='both',      # both major and minor ticks are affected
                top=False,         # ticks along the top edge are off
                bottom=False,      # ticks along the bottom edge are off
                left=False,      # ticks along the bottom edge are off
                right=False, 
                labeltop=False,
                labelbottom=False, # labels along the bottom edge are off
                labelleft=False,
                labelright=False)
            #####

      
    ####Add color bar for density scale
    for (key,fig),(_, ax)  in zip({'data':fig_best_data, 'model':fig_best_model, 'std':fig_best_std }.items(), {'data':ax_best_data, 'model':ax_best_model, 'std':ax_best_std}.items()):
        ax1 = ax[lzslice[-1]] 
        ax2 = ax[lzslice[1]] 
        cax = fig.add_axes([ax1.get_position().x1+0.02,ax1.get_position().y0,0.02,ax2.get_position().y0])
        cb = fig.colorbar(pc,  cax=cax)
        ax[lzslice[-2]].set_xlabel("X [m]")
        ax[lzslice[-1]].set_xlabel("X [m]")
        ax[lzslice[0]].set_ylabel("Y [m]")
        ax[lzslice[-2]].set_ylabel("Y [m]")
        #ticks = np.linspace(0.2, 2.0, 10)
        #cb.ax.set_yticklabels([f"{i:.1f}" for i in ticks])
        locator = MultipleLocator(2e-1)
        cb.ax.yaxis.set_major_locator(locator)
        cb.set_label(label="density $\\~{\\rho}$ [g.cm$^{-3}$]", size=14)#, labelpad=1.)
        cb.ax.tick_params(which="both",labelsize=fontsize)
        sigma, length= mat_sigma[ix], mat_length[ix]
        fig.text(.3, .9, s=f"$\\rho_{0}$ = {rho0}"+" g.cm$^{-3}$, "
                    +f"$\\sigma$ = {sigma}"+" g.cm$^{-3}$, " 
                    +f"$l$ = {length}"+" m\n" , fontsize=14)

        fout = out_dir / f"best_misfit_{key}_rho_post_rho0{rho0}_res{res}m"
        fig.savefig(f"{str(fout)}.png", transparent=True)    
        print(f"save {fout}.png")
    
    #print("End : ", time.strftime("%H:%M:%S", time.localtime()))#start time
    print(f"End --- {(time.time() - start_time):.3f}  s ---") 
    


        

        









    '''
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    postp, rho0p, sigp, lp = arr_rho_post[-1,-1,-1], mat_rho0[-1], mat_sigma[-1], mat_length[-1]
    colors = color_scale[[np.argmin(abs(range_val-v))for v in postp]]
    scatter_rhopost(fig, ax, title="", xyz=dome.barycenter, colors=colors)
    xt, yt, zt = tel.utm.T
    ax.scatter(xt, yt, zt, color=tel.color)

    '''
    
    
    '''
    fig = plt.figure()
    #####COLOR SCALE -> COLOR VOXELS
    x,y,z = ilos_bar.T

    ax = fig.add_subplot(111)
    for i, v in enumerate(verts):
        pc = PolyCollection([v[:,:-1]],  
                                cmap=cmap,
                                #alpha=0.3,
                                facecolors=color_vox_true[i], #np.repeat(color_vox,axis=0),
                                edgecolors=np.clip(color_vox_true[i] - 0.5, 0, 1),  # brighter 
                                #norm=Normalize(clip=True),
                                linewidths=0.3)
        
        ax.add_collection(pc)
    #ax.scatter(xsort, ysort, marker='o', s=10)#, zs=np.nanmin(zsort)-100)#*np.ones(xsort.shape))
    ax.set_title("density model")
    ax.set_xlim(xrange)
    ax.set_ylim(yrange)
    ax.set_aspect('auto') #'equal'
    locator = MultipleLocator(2e2)
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.get_major_formatter().set_useOffset(True)
    ax.xaxis.get_major_formatter().set_scientific(True)
    # ax.xaxis.set_major_formatter(ScalarFormatter(useOffset=True,useMathText=True))
    locator = MultipleLocator(2e2)
    ax.yaxis.set_major_locator(locator)
    ax.yaxis.get_major_formatter().set_useOffset(True)#set_major_formatter(ScalarFormatter(useOffset=True,useMathText=True))
    ax.yaxis.get_major_formatter().set_scientific(True)
    ax.grid(True, which='both',linestyle='-', linewidth="0.3 ", color='grey')
    pc.set_clim(vmin, vmax)
    cb = plt.colorbar(pc, orientation='vertical', ax=ax, )
    cb.set_label(label="mean density $\\overline{\\rho}$ [g.cm$^{-3}$]", labelpad=1.)
    locator = MultipleLocator(2e-1)
    cb.ax.yaxis.set_major_locator(locator)
    cb.ax.tick_params(labelsize=12)
    fout  = out_dir / f"rho_true_z{int(zslice)}m_res{res}m.png"
    print(f"save {fout}")
    plt.savefig(fout)
    '''
