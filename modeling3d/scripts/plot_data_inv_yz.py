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


if __name__=="__main__":
    
    
    #print(f'tick strength = {frho_to_strength(4e-1)}')
    #exit()
    start_time = time.time()
    print("Start: ", time.strftime("%H:%M:%S", time.localtime()))#start time
            
    param_dir = Path.home() / 'muon_code_v2_0' / 'AcquisitionParams'
    print(time.strftime("%H%M%S-%d%m%y"))
    res= int(sys.argv[1]) #m
    ml_dir = Path.home()  / "MatLab" / "MuonCompute_Volumes_mnt" 
    fGmatrix_dome = ml_dir /  "matrices" / f"cubesMatrix_{res}.mat"
    fVolcanoCenter = ml_dir / "matrices" / "volcanoCenter.mat"
    
    
    
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
        
    mask_cyl = mask_cubes != 0
    mask_cyl = mask_cyl & sv_cyl
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
    
    
   

    print(fin.exists())
    with open(str(fin), 'rb') as f:
        arr_rho_post = np.load(f)
    
    ####Plot density model
    cmap_rho = palettable.scientific.sequential.Batlow_20.mpl_colormap #'jet' #palettable.scientific.diverging.Roma_20
    rho_min, rho_max, n = .8, 2.7, 100
    range_val = np.linspace(rho_min, rho_max, n)
    norm_r = colors.Normalize(vmin=rho_min, vmax=rho_max)(range_val)
    color_scale_rho =  cmap_rho(norm_r)
    vmin_r, vmax_r = rho_min, rho_max
    
    
    ####strength model scale
    cmap_strength = palettable.scientific.diverging.Roma_20.mpl_colormap 
    strength_min, strength_max = 1e-1, 1e3#frho_to_strength(rho_max)
    range_val_s = np.logspace(np.log10(strength_min), np.log10(strength_max), n)
    norm_s = colors.Normalize(vmin=strength_min, vmax=strength_max)
    lognorm_s = colors.LogNorm(vmin=strength_min, vmax=strength_max)
    vmin_s, vmax_s = strength_min, strength_max
    #color_scale_strength = cmap_strength(norm_s(range_val_s))
    color_scale_strength = cmap_strength(lognorm_s(range_val_s))

    
    
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


    lxslice = [vc_x, vc_x+50, vc_x+100, vc_x+150]
    l_ipan = ["a", "b", "c", "d"]
    
    for ix_rho0 in range(nrho): #
        rho0 = mat_rho0[ix_rho0]
        rho0 = mat_rho0[ix_rho0]
        
        fig_best_rho = plt.figure(figsize=(12,9))#, constrained_layout=True)
        # gs_best = GridSpec(1, len(lxslice),
        #             wspace=0.0, hspace=0.0, 
        #             top=0.97-0.5/(nrow+1), bottom=0.5/(nrow+1), 
        #             left=0.5/(ncol+1), right=0.90-0.5/(ncol+1))
        fig_best_rho.subplots_adjust(left=0.08, bottom=0.1, right=0.88, top=0.95, wspace=0.20, hspace=0.20)
        ax_best_rho = fig_best_rho.subplot_mosaic(
                [lxslice[:2],   lxslice[2:]], #[lxslice],#
                sharex=True, sharey=True
            )
        fig_best_strength = plt.figure(figsize=(12,9))#, constrained_layout=True)
        ax_best_strength = fig_best_strength.subplot_mosaic(
                [lxslice[:2],   lxslice[2:]], #[lxslice],#
                sharex=True, sharey=True
            )
        fig_best_strength.subplots_adjust(left=0.08, bottom=0.1, right=0.88, top=0.95, wspace=0.20, hspace=0.20)
        #plt.gcf().text(0.4, 0.93, f"$\\rho_{0}$ = {rho0}"+" g.cm$^{-3}$,"+f"\nz = {xslice} m a.s.l", fontsize=fontsize+4)

    
    
        for ix_x, xslice in enumerate(lxslice):
            
            ysmin, ysmax = np.nanmin(dome.barycenter[:1]), np.nanmax(dome.barycenter[:1])
            dy = 500
            ysmin, ysmax = 1773900,1774650 #vc_y - dy,  vc_y + dy
            ####Define slice mask
            #im = np.argmin(abs(dome.barycenter[:,0] - xslice))
            #mask =(dome.barycenter[:,0] == dome.barycenter[:,im])
            mask_slice =  ((xslice - 3*res) <= dome.barycenter[:,0]) & (dome.barycenter[:,0] <= (xslice))
            mask_slice = mask_slice & (ysmin-res <= dome.barycenter[:,1]) & (dome.barycenter[:,1] <= ysmax + res )
            ilos_vox_s  = ilos_vox[mask_slice]
            nvox_slice = ilos_vox_s.shape[0]
            ilos_bar = dome.barycenter[mask_slice]
            ilos_xyzup = dome.xyz_up[mask_slice]
            ilos_xyzdown = dome.xyz_down[mask_slice]
            

            
            ####
            ####plot limits (y,z)=(northing, altitude)    
            yrange = [ ysmin-vc_y, ysmax-vc_y] 
            zmin, zmax = 1150, 1500 #altitude
            zrange = [zmin, zmax]
            dz = 200
            elev, azim = 90, -90


            verts = ilos_xyzdown
                    

            
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
            plt.gcf().text(0.4, 0.93, f"$\\rho_{0}$ = {rho0}"+" g.cm$^{-3}$", fontsize=fontsize+4)
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
                    print("arr_rho_post.shape=",arr_rho_post.shape)
                    
                    rho_vox_post = arr_rho_post[ix_rho0,i,j]
                    data_post = Gmat @ rho_vox_post #/ nvox_per_los
                    rho_vox_post = rho_vox_post[mask_slice]
                    misfit_d[ix_rho0,i,j] = 1/Nd *  np.sum( (data_real - data_post)**2 / unc_real**2) 
                    print('rho_vox_post = ', len(rho_vox_post), len(rho_vox_post[rho_vox_post!=0]))
                    print(f'rho_vox_post = {np.nanmean(rho_vox_post):.3f} +/- {np.nanstd(rho_vox_post):.3f} g/cm^3')
                    if len(rho_vox_post) == 0 : continue
                    
                    
                    #strength_post = frho_to_strength(rho_vox_post)
                    
                    
                    
                    arg_col_rho =  [np.argmin(abs(range_val-v))for v in rho_vox_post]    
                    
                    color_vox_rho = color_scale_rho[arg_col_rho]
                    x,y,z = ilos_bar.T
                    ax= plt.subplot(gs[i,j])
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
                    '''
                    for l, (vd, vu) in enumerate(zip(ilos_xyzdown, ilos_xyzup)):
                        #print(f"here{l}")
                        #if l > 100: exit()
                        #print(f"verts_down_{l}, verts_up_{l}= {vd[:,1:]}, {vu[:,1:]}",)
                        #print(f"verts_{l}=",v[:,1:])
                        v = [vd[0,1:], vd[2,1:], vu[1,1:], vu[0,1:]] -np.array([vc_y, 0])
                        pc = PolyCollection( [np.fliplr(v)],  
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
            
            sig_sel, lc_sel = 0.2, 100
            #sig_sel, lc_sel = 0.3, 160
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
            fontsize=16
            sigma, length= mat_sigma[ij_d], mat_length[ij_d]
            #apply mask for given slice
            rho_vox_post = arr_rho_post[ix_rho0, ij_d[0], ij_d[1]]
            rho_vox_post = rho_vox_post[mask_slice]
            strength_vox_post = frho_to_strength(rho_vox_post)
            arg_col_rho =  [np.argmin(abs(range_val-v))for v in rho_vox_post]   
            arg_col_strength =  [np.argmin(abs(range_val_s-v))for v in strength_vox_post]     
            color_vox_rho = color_scale_rho[arg_col_rho]
            color_vox_strength = color_scale_strength[arg_col_strength]
            ax_best_rho[xslice].set_axisbelow(True) #grid behind the plot
            ax_best_strength[xslice].set_axisbelow(True)
            ax_best_rho[xslice].grid(True, which='both',linestyle='-', linewidth="0.5 ", color='grey')
            ax_best_strength[xslice].grid(True, which='both',linestyle='-', linewidth="0.5 ", color='grey')
            
            
            #for l, v in enumerate(verts):
            for l, (vd, vu) in enumerate(zip(ilos_xyzdown, ilos_xyzup)):
                #print(f"here{l}")
                #print(f"verts_down_{l}, verts_up_{l}= {vd[:,1:]}, {vu[:,1:]}",)
                #print(f"verts_{l}=",v[:,1:])

                #v = [vd[0,1:], vd[2,1:], vu[1,1:], vu[0,1:]]-np.array([vc_y, 0])
                #v= [vd[2,1:], vd[0,1:], vu[1,1:], vu[0,1:]]-np.array([vc_y, 0])
                v = [ vd[2,1:], vd[0,1:], vu[0,1:], vu[2,1:]]-np.array([vc_y, 0])
                pcr = PolyCollection( [v] ,  
                                        cmap=cmap_rho,
                                        #alpha=0.3,
                                        facecolors=color_vox_rho[l], #np.repeat(color_vox,axis=0),
                                        edgecolors=np.clip(color_vox_rho[l] - 0.5, 0, 1),  # brighter 
                                        norm=colors.Normalize(clip=True),
                            linewidths=0.3)
                pcr.set_clim(vmin_r, vmax_r)
                ax_best_rho[xslice].add_collection(pcr)
                #v = [ vd[0,1:], vd[2,1:], vu[1,1:], vu[2,1:]]-np.array([vc_y, 0])
                #v = [ vd[0,1:], vd[2,1:], vu[0,1:], vu[2,1:]]-np.array([vc_y, 0]) #ok with the topo
                #v = [ vd[2,1:], vd[0,1:], vu[0,1:], vu[2,1:]]-np.array([vc_y, 0])
                pcs = PolyCollection( [v] ,  
                                        cmap=cmap_strength,
                                        #alpha=0.3,
                                        facecolors=color_vox_strength[l], #np.repeat(color_vox,axis=0),
                                        edgecolors=np.clip(color_vox_strength[l] - 0.5, 0, 1),  # brighter 
                                        norm=lognorm_s,
                            linewidths=0.3)
                pcs.set_clim(vmin_s, vmax_s)
                ax_best_strength[xslice].add_collection(pcs)

            print(f"cubes --- {(time.time() - start_time):.0f}  s ---") 
        
            
            for axb in [ax_best_rho[xslice], ax_best_strength[xslice]]:
                

                axb.xaxis.set_major_locator(MultipleLocator(1e2))
                axb.yaxis.set_major_locator(MultipleLocator(5e1))
                #axb.scatter(ltel_coord[:,0]-vc_x, ltel_coord[:,1]-vc_y, c='magenta', s=30,marker='o',)
                #axb.annotate(f"\nz = {zslice:.0f} m a.s.l", 
                #             xy=(0.5, .5), xycoords='axes fraction', xytext=(.05, .85), fontsize=14) #xytext=(.25, .85) #<- centered
                anchored_text = AnchoredText(f"X = {xslice-vc_x:.0f} m ", loc="upper left", frameon=True, prop=dict(fontsize=fontsize))
                axb.add_artist(anchored_text)
                ipan = AnchoredText(f"{l_ipan[ix_x]}", loc="upper right", frameon=True, prop=dict(fontsize=fontsize))
                axb.add_artist(ipan)
                #axb.annotate("$\\phi_{d}$"+f"={misfit_d[ix_rho0,i,j]:.2e}", xy=(0.5, .5), xycoords='axes fraction', xytext=(0.15, .85), fontsize=fontsize-2)
                axb.set_xlim(yrange)
                axb.set_ylim([zmin, zmax])
                axb.set_aspect('auto') #'equal'
                
                #North arrow
                x, y, arrow_length = 0.96, 0.15, 0.1
                axb.annotate('N', xy=(x, y), xycoords='axes fraction', xytext=(x-arrow_length, y),
                    arrowprops=dict(facecolor='black', width=5, headwidth=15, alpha=0.4),
                    ha="center", va="center",)
            

            
                
        for ax in [ax_best_rho, ax_best_strength]:
            ####Add color bar for density scale
            ax1 = ax[lxslice[-1]] 
            ax2 = ax[lxslice[1]]     
            ax[lxslice[-2]].set_xlabel("Northing Y [m]")
            ax[lxslice[-1]].set_xlabel("Northing Y [m]")
            ax[lxslice[0]].set_ylabel("Altitude Z [m]")
            ax[lxslice[-2]].set_ylabel("Altitude Z [m]")
            if ax == ax_best_rho : 
                cax = fig_best_rho.add_axes([ax1.get_position().x1+0.02,ax1.get_position().y0,0.02,ax2.get_position().y0])
                cb = fig_best_rho.colorbar(pcr,  cax=cax)
                locator = MultipleLocator(2e-1)
                cb.ax.yaxis.set_major_locator(locator)
                cb.set_label(label="density $\\~{\\rho}$ [g.cm$^{-3}$]", size=fontsize)#, labelpad=1.)
            if ax == ax_best_strength : 
                cax = fig_best_strength.add_axes([ax1.get_position().x1+0.02,ax1.get_position().y0,0.02,ax2.get_position().y0])
                cb = fig_best_strength.colorbar(ScalarMappable(norm=lognorm_s, cmap=cmap_strength), cax=cax)    
                cb.set_label(label="unixial compressive strength [MPa]", size=fontsize)#, labelpad=1.)
            cb.ax.tick_params(which="both",labelsize=fontsize)

            
            
        fout = out_dir / f"best_misfit_data_rho_post_rho0{rho0}_res{res}m_yz"
        fig_best_rho.savefig(f"{str(fout)}.png", transparent=True)
        print(f"save {fout}.png")
        fout = out_dir / f"best_misfit_strength_rho0{rho0}_res{res}m_yz"
        fig_best_strength.savefig(f"{str(fout)}.png", transparent=True)    
        print(f"save {fout}.png")
        plt.close()
        
            
        print(f"End --- {(time.time() - start_time):.0f}  s ---") 
        
        







