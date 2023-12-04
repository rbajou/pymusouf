#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib.cm import ScalarMappable
from mpl_toolkits.mplot3d.art3d import PolyCollection, Poly3DCollection, Line3DCollection
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
from scipy.spatial import ConvexHull
from scipy.spatial import Delaunay


#personal modules
from configuration import str2telescope, Telescope
from analysis import AcqVars, Observable

from plot_density_3d import VoxeledDome


from inversion import  DataSynth
from configuration import dict_tel
import warnings
warnings.filterwarnings("ignore")
#warnings.filterwarnings( "ignore", module = "matplotlib\..*" )

import palettable

def scatter_rhopost(fig, ax, title, xyz, colors, **kwargs):
    x, y, z = xyz.T
    ax.scatter(x, y, z, color=colors, **kwargs)
    ax.set_title(title)
    
def frho_to_strength(rho):
    strength = 1.326*1e-25 *(rho*1e3)**(9.67) * 1e-6
    return strength


params = {'legend.fontsize': 'xx-large',
            'axes.labelsize': 'xx-large',
            'axes.titlesize':'xx-large',
            'xtick.labelsize':'xx-large',
            'ytick.labelsize':'xx-large',
            'axes.labelpad':10}
plt.rcParams.update(params)  


if __name__=="__main__":
    
    
    #print(f'tick strength = {frho_to_strength(4e-1)}')
    #exit()
    start_time = time.time()
    print("Start: ", time.strftime("%H:%M:%S", time.localtime()))#start time
            
    param_dir = Path.home() / 'muon_code_v2_0' / 'AcquisitionParams'
    #print(time.strftime("%H%M%S-%d%m%y"))
    res= int(sys.argv[1]) #m
    ml_dir = Path.home()  / "MatLab" / "MuonCompute_Volumes_mnt" 
    fGmatrix_dome = ml_dir /  "matrices" / f"cubesMatrix_{res}.mat"
    fVolcanoCenter = ml_dir / "matrices" / "volcanoCenter.mat"
    
    
    
    #####anomalies ERT 
    out_ert_60 = Path.home()/ "data/ERT/slices/df_contours_2960.csv"
    df_ert_60= pd.read_csv(out_ert_60, sep="\t", index_col=0)
    A1 = df_ert_60['sigma'] == 0.1
    A2 = df_ert_60['sigma'] == 1.
    xa1, ya1, za1 = df_ert_60[A1]["X"].to_numpy(), df_ert_60[A1]["Y"].to_numpy(), df_ert_60[A1]["Z"].to_numpy()
    xa2, ya2, za2 = df_ert_60[A2]["X"].to_numpy(), df_ert_60[A2]["Y"].to_numpy(), df_ert_60[A2]["Z"].to_numpy()
   
 
    ####
    ###coordinates
    ltel_n = ["SB", "SNJ", "BR", "OM"]
    #ltel_n = ["SNJ",]#, "OM"]
    str_tel =  "_".join(ltel_n)
    ltel_coord = np.array([ str2telescope(tel).utm[:-1] for tel in ltel_n])


    mat_dome = mat73.loadmat(str(fGmatrix_dome))['cubesMatrix'] #shape=(N_cubes, N_par_cubes=31)
    vol_center = sio.loadmat(str(fVolcanoCenter))['volcanoCenter'][0]
  
    vc_x = vol_center[0]
    vc_y = vol_center[1]
    dtc = np.sqrt( (mat_dome[:,25]-vc_x)**2 + (mat_dome[:,26]-vc_y)**2 ) 
    sv_center = (dtc <= 375) 
    
    ###
    R = 100#m
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
        
    
    mask_cubes = mask_cubes & sv_center

    data_dir = Path.home()/"data"
    run = "real" #"real"
    
    tag_inv = "smoothing"
    datestr = sys.argv[2]
    out_dir = data_dir / "inversion" / run / str_tel / tag_inv / datestr #"23022023_1410"
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
    #dome.get_distance_matrix()

    Gmat = np.loadtxt(out_dir/ f"Gmat_all_res{res}m.txt")#[mask_dome]
    data_real = np.loadtxt(out_dir/ f"data_all_res{res}m.txt")#[mask_dome]
    unc_real = np.loadtxt(out_dir/ f"unc_all_res{res}m.txt")#[mask_dome]


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




   
    #####
  
    ####
    mat_rho0 = np.loadtxt(out_dir/ "rho0.txt",delimiter="\t")
    if mat_rho0.shape == ():  mat_rho0 = np.array([mat_rho0.mean()])
    mat_sigma = np.loadtxt(out_dir/ "sigma_prior.txt",delimiter="\t")
    if mat_sigma.shape == ():  mat_sigma = np.array([[mat_sigma.mean()]])
    mat_length = np.loadtxt(out_dir/ "correlation_length.txt", delimiter="\t")
    if mat_length.shape == ():  mat_length = np.array([[mat_length.mean()]])
    
    
    
   

    fin = out_dir /  f"rho_post_res{res}m.npy"         
    with open(str(fin), 'rb') as f:
        arr_rho_post = np.load(f)
    fin = out_dir /  f"std_dev_res{res}m.npy"         
    with open(str(fin), 'rb') as f:
        arr_std = np.load(f)
    
    ####Plot density model
    cmap_rho = palettable.scientific.sequential.Batlow_20.mpl_colormap #'jet' #palettable.scientific.diverging.Roma_20
    rho_min, rho_max, n = .8, 2.7, 100
    range_val = np.linspace(rho_min, rho_max, n)
    norm_r = colors.Normalize(vmin=rho_min, vmax=rho_max)(range_val)
    color_scale_rho =  cmap_rho(norm_r)
    vmin_r, vmax_r = rho_min, rho_max
    
    
    ####strength model scale
    cmap_strength = palettable.scientific.diverging.Roma_20.mpl_colormap 
    strength_min, strength_max = 1e-1, frho_to_strength(rho_max)
    range_val_s = np.logspace(np.log10(strength_min), np.log10(strength_max), n)
    norm_s = colors.Normalize(vmin=strength_min, vmax=strength_max)
    lognorm_s = colors.LogNorm(vmin=strength_min, vmax=strength_max)
    vmin_s, vmax_s = strength_min, strength_max
    #color_scale_strength = cmap_strength(norm_s(range_val_s))
    color_scale_strength = cmap_strength(lognorm_s(range_val_s))

    std_min, std_max, n = .02, .2, 100
    range_val_std = np.linspace(std_min, std_max, n)
    norm_std = colors.Normalize(vmin=std_min, vmax=std_max)(range_val_std)
    color_scale_std =  cmap_rho(norm_std)
    vmin_std, vmax_std = std_min, std_max
    
    
    nrho = len(mat_rho0)
    nrow = mat_sigma.shape[0]
    ncol = mat_length.shape[1]

    misfit_d = np.ones(shape=(nrho, nrow, ncol) )
    misfit_m = np.ones(shape=(nrho, nrow, ncol) )
    nlos, nvox = Gmat.shape
    nvox_per_los = np.count_nonzero(Gmat, axis=1)
    print(f'Gmat={Gmat[Gmat!=0]}')
    nvox_per_los = np.ones(nlos)
    ####Plot mosaic 
    ilos_vox = dome.cubes


    ####crater 
    TAR = np.array([6.429728210000000e5, 1.774237168000000e6, 1450])#, 1.5e3])
    CSS = np.array([6.430440754723654e5, 1.774125703493994e6, 1450])#, 1.5e3])
    # BLK = np.array([643217.61, 1774144.31, 1420]) #1350
    # G56 = np.array([643094.02, 1774205.92]) #1418
    # FNO = np.array([642958.52, 1774500.03])
    # NAP = np.array([642994.49, 1774232.5])


    
    for xslice in [vc_x, vc_x+50, vc_x+100]:
    
        ysmin, ysmax = np.nanmin(dome.barycenter[:1]), np.nanmax(dome.barycenter[:1])
        dy = 500
        ysmin, ysmax = 1773900,1774650 #vc_y - dy,  vc_y + dy
        ####Define slice mask
        mask_slice =  ((xslice - 3*res) <= dome.barycenter[:,0]) & (dome.barycenter[:,0] <= (xslice ))
        mask_slice = mask_slice & (ysmin <= dome.barycenter[:,1]) & (dome.barycenter[:,1] <= ysmax )
        ilos_vox_s  = ilos_vox[mask_slice]
        nvox_slice = ilos_vox_s.shape[0]
        ilos_bar = dome.barycenter[mask_slice]
        ilos_xyzup = dome.xyz_up[mask_slice]
        ilos_xyzdown = dome.xyz_down[mask_slice]
        

        
        ####
        ####plot limits (y,z)=(northing, altitude)    
        yrange = [ ysmin-vc_y, ysmax-vc_y]
        zmin, zmax = 1100, 1500 #altitude
        zrange = [zmin, zmax]
        dz = 200
        elev, azim = 90, -90

        #print(f"yrange, zrange = {yrange}, {zrange}")

        verts = ilos_xyzdown
        
        #print(f"verts={len(ilos_xyzdown)}, {verts[:1,:]}")
        
        for ix_rho0 in range(nrho): #
            rho0 = mat_rho0[ix_rho0]
            fig = plt.figure(figsize=(ncol+2, nrow+2)) 
            # gs = GridSpec(nrow, ncol,
            #     wspace=0.0, hspace=0.0, 
            #     top=0.97-0.5/(nrow+1), bottom=0.5/(nrow+1), 
            #     left=0.08+0.5/(ncol+1), right=.98-0.5/(ncol+1))
            gs = GridSpec(nrow, ncol,
                wspace=0.0, hspace=0.0, 
                top=0.97-0.5/(nrow+1), bottom=0.5/(nrow+1), 
                left=0.5/(ncol+1), right=0.90-0.5/(ncol+1))
            fontsize = 8
            plt.gcf().text(0.4, 0.93, f"$\\rho_{0}$ = {rho0}"+" g.cm$^{-3}$")#, fontsize=fontsize+4)
            ####
            
        
            Nd = len(data_real)
            print(f"Nd={Nd}")
            lphi = []
            c=0
            
            for i in range(nrow):
                #if i > 0 : break
                for j in range(ncol):
                    sigma, length= mat_sigma[i, j], mat_length[i,j]
                    print(f" (rho0, err_prior, correlation length) = ({rho0:.1f} g/cm^3, {sigma:.3e} g/cm^3, {length:.3e} m) --- {(time.time() - start_time):.3f}  s ---") 
                    key = f"{sigma:.3f},{length:.3f}"
                    #apply mask for given slice
                    #print("arr_rho_post.shape=",arr_rho_post.shape)
                    
                    rho_vox_post = arr_rho_post[ix_rho0,i,j]
                    data_post = Gmat @ rho_vox_post #/ nvox_per_los
                    rho_vox_post = rho_vox_post[mask_slice]
                    std_dev = arr_std[ix_rho0,i,j][mask_slice]
                    misfit_d[ix_rho0,i,j] = 1/Nd *  np.sum( (data_real - data_post)**2 / unc_real**2) 
                    print('rho_vox_post = ', len(rho_vox_post), len(rho_vox_post[rho_vox_post!=0]))
                    print(f'rho_vox_post = {np.nanmean(rho_vox_post):.3f} +/- {np.nanstd(rho_vox_post):.3f} g/cm^3')
                    print(f'std_dev = {np.nanmean(std_dev):.3f} +/- {np.nanstd(std_dev):.3f} g/cm^3')
                    if len(rho_vox_post) == 0 : continue
                    
                    
                    #strength_post = frho_to_strength(rho_vox_post)
                    
                    
                    
                    arg_col_rho =  [np.argmin(abs(range_val-v))for v in rho_vox_post]    
                    
                    color_vox_rho = color_scale_rho[arg_col_rho]
                    x,y,z = ilos_bar.T
                    ax= plt.subplot(gs[i,j])
                    
                    '''
                    ax.annotate(f"{c}", xy=(0.5, .5), xycoords='axes fraction', xytext=(.85, .05), fontsize=6)
                    if i== 0 and j==0: 
                        ax.annotate('$\\sigma$ [g.cm$^{-3}$]', rotation='vertical', xy=(-0.4, 0.1), xycoords='axes fraction', xytext=(-0.4, 0.9), 
                                ha="center", va="center", arrowprops=dict(arrowstyle="->", color='k'))
                        ax.annotate('$l$ [m]', xy=(0.9, 1.4), xycoords='axes fraction', xytext=(0.1, 1.4), 
                                ha="center", va="center",arrowprops=dict(arrowstyle="->", color='k'))
                    # ax.set_xticklabels([])
                    # ax.set_yticklabels([])
                    ax.tick_params(axis="y",direction="in", pad=-22)
                    ax.tick_params(axis="x",direction="in", pad=-15)
                    ax.tick_params(which="both", bottom=True, top=True, left=True, right=True)
                    if i==0: 
                        ax.set_title(f"{length:.3e}", fontsize=8)
                
                    if j==0: 
                        ax.set_ylabel(f"{sigma:.3e}", fontsize=8)
                    ax.grid(False)
                    #ax.grid(True, which='both',linestyle='-', linewidth="0.3 ", color='grey')
                    for l, (vd, vu) in enumerate(zip(ilos_xyzdown, ilos_xyzup)):
                        #print(f"here{l}")
                        #if l > 100: exit()
                        #print(f"verts_down_{l}, verts_up_{l}= {vd[:,1:]}, {vu[:,1:]}",)
                        #print(f"verts_{l}=",v[:,1:])
                        v= [vd[0,1:], vd[2,1:], vu[1,1:], vu[0,1:]]
                        pc = PolyCollection( [v],  
                                                cmap=cmap_rho,
                                                #alpha=0.3,
                                                facecolors=color_vox_rho[l], #np.repeat(color_vox,axis=0),
                                                edgecolors=np.clip(color_vox_rho[l] - 0.5, 0, 1),  # brighter 
                                                norm=colors.Normalize(clip=True),
                                                linewidths=0.3)
                        pc.set_clim(vmin_r, vmax_r)
                        ax.add_collection(pc)
                    ax.scatter(ltel_coord[:,0], ltel_coord[:,1], c='magenta', s=10,marker='.',)
                    s = "$\\phi_{d}$"+f"={misfit_d[ix_rho0,i,j]:.2e}"
                    ax.annotate(s, xy=(0.5, .5), xycoords='axes fraction', xytext=(0.1, .75), fontsize=fontsize-2)
                    #anchored_text = AnchoredText(s, loc="upper left", frameon=False, prop=dict(fontsize=fontsize-2))
                    ax.set_xlim(yrange)
                    ax.set_ylim(zrange)
                    ax.set_aspect('auto') #'equal'
                    '''
                    c+=1
            
        
        
            
            ###find the best data misfit_d i.e closer to one, and minimal model misfit_m  (~squared residuals)
            phi_d = misfit_d[ix_rho0]
            a=abs(phi_d-1)
            ij_d = np.unravel_index(np.argmin(a, axis=None), a.shape) #np.argmin(abs(misfit_d[ix_rho0]-1), axis=1)
            
            best_misfit_d = phi_d[ij_d]
            sigma_d, length_d = mat_sigma[ij_d], mat_length[ij_d]
            best_par_d =f"$\\rho_{0}$ = {rho0}"+" g.cm$^{-3}$, " +f"$\\sigma$ = {sigma_d}"+" g.cm$^{-3}$, " +f"$l$ = {length_d}"+" m\n"
            
            
            ###select model for a given set of parameters
            
            
            sig_sel, lc_sel = 0.3, 160
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
            

            
            # axes = np.asarray(fig.get_axes()).reshape(nrow, ncol)
            # #axes[ij_d].plot()
            # axes[ij_d].annotate("**", xy=(0.5, .5), xycoords='axes fraction', xytext=(.85, .9), fontsize=fontsize+2, color="red")
            # axes[ij_m].annotate("**", xy=(0.5, .5), xycoords='axes fraction', xytext=(.85, .8), fontsize=fontsize+2, color="green")
            # ####Add color bar for density
            # ax1 = plt.subplot(gs[nrow-1, ncol-1])
            # try : 
            #     ax2 = plt.subplot(gs[nrow-4, ncol-1])
            # except : 
            #     ax2 = plt.subplot(gs[0, 0])
            # cax = fig.add_axes([ax1.get_position().x1+0.03,ax1.get_position().y0,0.02,ax2.get_position().y0])
            # cb = fig.colorbar(pc,  cax=cax)
            # #ticks = np.linspace(0.2, 2.0, 10)
            # #cb.ax.set_yticklabels([f"{i:.1f}" for i in ticks])
            # locator = MultipleLocator(4e-1)
            # cb.ax.yaxis.set_major_locator(locator)
            # cb.set_label(label="mean density $\\overline{\\rho}$ [g.cm$^{-3}$]", size=fontsize+2, labelpad=1.)
            # cb.ax.tick_params(which="both",labelsize=fontsize)


            # fout = out_dir / f"rho_post_rho0{rho0}_res{res}m_vert_slice"
            # fig.savefig(f"{str(fout)}.png")   
            # plt.close()
            # print(f"save {fout}.png")
                
            
            #####plot best misfit data inversion
            fig_best_rho, axr = plt.subplots(figsize=(12,9))
            fig_best_strength, axs = plt.subplots(figsize=(12,9))
            fig_std, ax_std = plt.subplots(figsize=(12,9))#, constrained_layout=True)   
            #fig_std.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.95, wspace=0.20, hspace=0.20)
        
            
            fontsize=16
            sigma, length= mat_sigma[ij_d], mat_length[ij_d]
            #apply mask for given slice
            rho_vox_post = arr_rho_post[ix_rho0, ij_d[0], ij_d[1]]
            rho_vox_post = rho_vox_post[mask_slice]
            std_dev_post = arr_std[ix_rho0,i,j][mask_slice]
            strength_vox_post = frho_to_strength(rho_vox_post)
            arg_col_rho =  [np.argmin(abs(range_val-v))for v in rho_vox_post]   
            arg_col_strength =  [np.argmin(abs(range_val_s-v))for v in strength_vox_post]     
            color_vox_rho = color_scale_rho[arg_col_rho]
            color_vox_strength = color_scale_strength[arg_col_strength]
            arg_col_std =  [np.argmin(abs(range_val_std-v))for v in std_dev_post]      
            color_vox_std = color_scale_std[arg_col_std]
            axr.set_axisbelow(True) #grid behind the plot
            axs.set_axisbelow(True)
            axr.grid(False)
            axr.grid(True, which='both',linestyle='-', linewidth="0.3", color='grey')
            axs.grid(False)
            axs.grid(True, which='both',linestyle='-', linewidth="0.3", color='grey')
            ax_std.grid(False)
            ax_std.grid(True, which='both',linestyle='-', linewidth="0.3", color='grey')
            #for l, v in enumerate(verts):
            for l, (vd, vu) in enumerate(zip(ilos_xyzdown, ilos_xyzup)):
                #print(f"here{l}")
                #print(f"verts_down_{l}, verts_up_{l}= {vd[:,1:]}, {vu[:,1:]}",)
                #print(f"verts_{l}=",v[:,1:])

               # v= [vd[0,1:], vd[2,1:], vu[1,1:], vu[0,1:]]#-np.array([vc_y, 0])
                v = [ vd[2,1:], vd[0,1:], vu[0,1:], vu[2,1:]]-np.array([vc_y, 0])   
                pcr = PolyCollection( [v] ,  
                                        cmap=cmap_rho,
                                        #alpha=0.3,
                                        facecolors=color_vox_rho[l], #np.repeat(color_vox,axis=0),
                                        edgecolors=np.clip(color_vox_rho[l] - 0.5, 0, 1),  # brighter 
                                        norm=colors.Normalize(clip=True),
                            linewidths=0.3)
                pcr.set_clim(vmin_r, vmax_r)
                axr.add_collection(pcr)
                pcs = PolyCollection( [v] ,  
                                        cmap=cmap_strength,
                                        #alpha=0.3,
                                        facecolors=color_vox_strength[l], #np.repeat(color_vox,axis=0),
                                        edgecolors=np.clip(color_vox_strength[l] - 0.5, 0, 1),  # brighter 
                                        norm=lognorm_s,
                            linewidths=0.3)
                pcs.set_clim(vmin_s, vmax_s)
                axs.add_collection(pcs)
                pcstd = PolyCollection( [v] ,  
                                        cmap=cmap_rho,
                                        #alpha=0.3,
                                        facecolors=color_vox_std[l], #np.repeat(color_vox,axis=0),
                                        edgecolors=np.clip(color_vox_std[l] - 0.5, 0, 1),  # brighter 
                                        norm=colors.Normalize(clip=True),
                            linewidths=0.3)
                pcstd.set_clim(vmin_std, vmax_std)
                ax_std.add_collection(pcstd)


            print(f"cubes --- {(time.time() - start_time):.0f}  s ---") 
        
            
            for a in [axr, axs, ax_std] : 
                a.xaxis.set_major_locator(MultipleLocator(1e2))
                a.yaxis.set_major_locator(MultipleLocator(5e1))
                a.set_xlim(yrange)
                a.set_ylim([zmin, zmax])
                a.tick_params(axis="both", which="both", bottom=True, labelbottom=True, left=True, labelleft=True, size=fontsize-4)
                a.set_ylabel("Altitude Z [m]")#, fontsize=fontsize)
                a.set_xlabel("Northing Y [m]")#, fontsize=fontsize)
               
                m = (1774400 > ya1) & (ya1 > 1773900)
                # pointsa1 = np.vstack((ya1[m]-vc_y,za1[m]-vc_y)).T
                # print(pointsa1.shape)
                a.scatter(ya1-vc_y, za1, color="orange", marker="o",  s=40., alpha=0.5 )
                a.scatter(ya2-vc_y, za2, color="red", marker="o",  s=40., alpha=0.5 )

            
                      
                anchored_text = AnchoredText(f"X = {xslice-vc_x:.0f} m ", loc="upper left", frameon=True, prop=dict(fontsize='x-large'))
                a.add_artist(anchored_text)
                ####North arrow
                x, y, arrow_length = 0.96, 0.9, 0.05
                a.annotate('N', xy=(x, y), xycoords='axes fraction', xytext=(x-arrow_length, y),
                    arrowprops=dict(facecolor='black', width=5, headwidth=15, alpha=0.4),
                    ha="center", va="center", fontsize='x-large')      
                #plt.plot(pointsa1[:,0], pointsa1[:,1], 'o')
                
                #tria1 = Delaunay(pointsa1)
                # for s, simplex in enumerate(tria1.simplices):
                #     ysi, zsi  = pointsa1[simplex, 0], pointsa1[simplex, 1]
                #     d = np.sqrt((np.max(ysi)-np.min(ysi))**2 + (np.max(zsi)-np.min(zsi))**2)
                #     if d < 500:
                #         plt.fill(pointsa1[simplex, 0], pointsa1[simplex, 1], color="orange", alpha=0.5)#'k-')
                # m = ya2 > 1773900
                # points = np.vstack((ya2[m],za2[m])).T   
                # tria2 = Delaunay(points)
                # for s, simplex in enumerate(tria2.simplices):
                #     ysi, zsi  = points[simplex, 0], points[simplex, 1]
                #     d = np.sqrt((np.max(ysi)-np.min(ysi))**2 + (np.max(zsi)-np.min(zsi))**2)
                #     if d < 150:
                #         plt.fill(points[simplex, 0], points[simplex, 1], color="red")#'k-')
                
                kwargs=dict( s=40, marker="D", color="deepskyblue", edgecolor="black")
                a.scatter(TAR[1]-vc_y, TAR[-1],**kwargs) #"cyan"
                a.annotate(f"TAR", xy =(TAR[1]-vc_y, TAR[-1]+10), fontsize='x-large',
                             xycoords='data', annotation_clip=False ,alpha=0.8,fontweight='normal')  
                a.scatter(CSS[1]-vc_y, CSS[-1],**kwargs) #"cyan"
                a.annotate(f"CSS", xy =(CSS[1]-vc_y, CSS[-1]+10), fontsize='x-large',
                             xycoords='data', annotation_clip=False , alpha=0.8,fontweight='normal')  
                # ax1.scatter(FNO[0]-vc_x, FNO[1]-vc_y,**kwargs) #"cyan"
                # ax1.scatter(BLK[0]-vc_x, BLK[1]-vc_y,**kwargs) #"cyan"
                # ax1.scatter(G56[0]-vc_x, G56[1]-vc_y,**kwargs) #"cyan"
                                
                
                ####Add color bar for density scale
                divider = make_axes_locatable(a)
                cax = divider.append_axes("right", size="5%", pad=0.05)
                #a.set_title(f"X = {xslice} m", fontsize=fontsize)
                if a == axr: 
                    cb = fig_best_rho.colorbar(pcr,  cax=cax, orientation="vertical")
                    cb.ax.yaxis.set_major_locator( MultipleLocator(4e-1))
                    cb.set_label(label="density $\\~{\\rho}$ [g.cm$^{-3}$]")#, size=fontsize)#, labelpad=1.)
                if a == axs: 
                    cb = fig_best_strength.colorbar(pcs,  cax=cax, orientation="vertical")
                    cb = fig.colorbar(ScalarMappable(norm=lognorm_s, cmap=cmap_strength), cax=cax)
                    cb.set_label(label="unixial compressive strength [MPa]")#, size=fontsize)#, labelpad=1.)
                if a == ax_std: 
                    cb = fig_best_strength.colorbar(pcstd,  cax=cax, orientation="vertical")
                    #cb = fig.colorbar(ScalarMappable(norm=lognorm_s, cmap=cmap_strength), cax=cax)
                    cb.set_label(label="standard deviation $\\~{\\sigma}$ [g.cm$^{-3}$]")#, size=fontsize)#, labelpad=1.)
                
                
                cb.ax.tick_params(which="both")#,labelsize=fontsize-4)
            
            fout = out_dir / f"best_misfit_data_rho_post_rho0{rho0}_res{res}m_x{int(xslice)}_yz"
            fig_best_rho.savefig(f"{str(fout)}.png", transparent=True)
            print(f"save {fout}.png")
            fout = out_dir / f"best_misfit_strength_rho0{rho0}_res{res}m_x{int(xslice)}_yz"
            fig_best_strength.savefig(f"{str(fout)}.png", transparent=True)  
            print(f"save {fout}.png")
            
            fout = out_dir / f"std_dev_rho0{rho0}_res{res}m_x{int(xslice)}_yz"
            fig_std.savefig(f"{str(fout)}.png", dpi=300,  transparent=True)    
            print(f"save {fout}.png")
                    
            plt.close()
            
            
            print(f"End --- {(time.time() - start_time):.0f}  s ---") 
            
            







