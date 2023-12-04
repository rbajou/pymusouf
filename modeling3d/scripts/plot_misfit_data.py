#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as cm
from matplotlib.cm import ScalarMappable
from mpl_toolkits.mplot3d.art3d import PolyCollection, Poly3DCollection, Line3DCollection
from matplotlib.ticker import MultipleLocator,EngFormatter, ScalarFormatter
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.gridspec import GridSpec
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.colors import Normalize
from matplotlib.offsetbox import AnchoredText
import sys
import scipy.io as sio
from pathlib import Path
from datetime import datetime, date, timezone
import logging
import pandas as pd
import mat73 #read v7.3 mat files
import palettable

#personal modules
from configuration import str2telescope, Telescope, dict_tel
from analysis import AcqVars, Observable
from plot_density_3d import VoxeledDome
from inversion import Inversion, KernelMatrix


if __name__=="__main__":
    data_dir = Path.home()/"data"
    run = "synthetic"
    res= int(sys.argv[1])
    tag_inv = "smoothing"
    out_dir = data_dir / "inversion" / run / tag_inv / sys.argv[2]
    
    ####
    print("load matrix dome...")
    ml_dir = Path.home()  / "MatLab" / "MuonCompute_Volumes_mnt" 
    fGmatrix_dome = ml_dir /  "matrices" / f"cubesMatrix_{res}.mat"
    fVolcanoCenter = ml_dir / "matrices" / "volcanoCenter.mat"
    mat_dome = mat73.loadmat(str(fGmatrix_dome))['cubesMatrix'] #shape=(N_cubes, N_par_cubes=31) 
    vol_center = sio.loadmat(str(fVolcanoCenter))['volcanoCenter'][0]
    vc_x = vol_center[0]
    vc_y = vol_center[1]
    dtc = np.sqrt( (mat_dome[:,25]-vc_x)**2 + (mat_dome[:,26]-vc_y)**2 ) 
    sv_center = (dtc <= 375) 
    
    dome = VoxeledDome(resolution=res, matrix=mat_dome)
    mask_dome = np.loadtxt(out_dir/f"mask_dome_res{res}m.txt")
    mask_dome = mask_dome == 1
    dome.cubes = dome.cubes[mask_dome] #select voxels of interest
    dome.barycenter = dome.barycenter[mask_dome]
    dome.xyz_up = dome.xyz_up[mask_dome]
    dome.xyz_down = dome.xyz_down[mask_dome]
    ilos_vox = dome.cubes
    #dome.get_distance_matrix()
   
   
   
    ltel_n = ["SNJ"] #default 
    if len(sys.argv) >= 3 :
        ltel_n = [sys.argv[i] for i in range(3, len(sys.argv))]
        print(f"ltel_n = {ltel_n}")
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
            print(mask_cubes.shape)
            union_mask_tel[i] = mask_cubes
        union_mask_tel = np.sum(union_mask_tel, axis=0)
        mask_cubes = union_mask_tel != 0 #convert to bool type
        
    mask_cubes = mask_cubes != 0 #convert to bool type
    mask_cyl = mask_cubes != 0
    print(f"mask_cubes = , shape = {mask_cubes.shape}")
    print(f"sv_center = {sv_center}, shape={sv_center.shape}")
    mask_cubes = mask_cubes & sv_center
    print(f"mask_cubes &  sv_center = {mask_cubes}, shape={mask_cubes[mask_cubes==1].shape}")
    print(f"dome total nvoxels : {len(mask_cubes)}")
    
    
    mat_rho0 = np.loadtxt(out_dir/ "rho0.txt",delimiter="\t")
    if mat_rho0.shape == ():  mat_rho0 = np.array([mat_rho0.mean()])
    mat_sigma = np.loadtxt(out_dir/ "sigma_prior.txt",delimiter="\t")
    if mat_sigma.shape == ():  mat_sigma = np.array([[mat_sigma.mean()]])
    mat_length = np.loadtxt(out_dir/ "correlation_length.txt", delimiter="\t")
    if mat_length.shape == ():  mat_length = np.array([[mat_length.mean()]])
    
    
    Gmat = np.loadtxt(out_dir/ f"Gmat_all_res{res}m.txt", delimiter="\t")
    
    vec_data_syn_all = np.loadtxt(out_dir/ f"data_syn_all_res{res}m.txt")
    vec_unc_syn_all = np.loadtxt(out_dir/ f"unc_syn_all_res{res}m.txt")
  
    fin = out_dir /  f"rho_post_res{res}m.npy"         
    with open(str(fin), 'rb') as f:
        rho_post = np.load(f)
    fin = out_dir /  f"rho_true_res{res}m.npy"         
    with open(str(fin), 'rb') as f:
        mat_rho_syn = np.load(f)
    
    rho_vox_true = np.load(out_dir /f"rho_true_res{res}m.npy")
    #unc_vox_true = np.load(out_dir /f"unc_true_res{res}m.npy")
    rho_min, rho_max, n = .1, 2.0, 100
    range_val = np.linspace(rho_min, rho_max, n)
    norm = cm.Normalize(vmin=rho_min, vmax=rho_max)(range_val)
    batlow=palettable.scientific.sequential.Batlow_20.mpl_colormap
    color_scale =  batlow(norm)
    
    
    
     
    jet = 'jet'#
    Gmatrix_dome = ml_dir /  "matrices" / f"cubesMatrix_{res}.mat"
    conf="3p1"
        
    fig_mis =  plt.figure(figsize=(12,8))
    ax_mis = fig_mis.subplot_mosaic(
        [[ "SB", "SNJ"], ["BR", "OM"]], #[lzslice],#
        sharex=False, sharey=False
    )
    fig_dat =  plt.figure(figsize=(12,8))
    ax_dat = fig_dat.subplot_mosaic(
        [["SB", "SNJ"], ["BR", "OM"]], #[lzslice],#
        sharex=False, sharey=False
    )
    
    ###select model for a given set of parameters
    sig_sel, lc_sel = 0.3, 160
    #ij_m = np.unravel_index(np.argmin(phi_m, axis=None), phi_m.shape) #np.argmin(abs(misfit_d[ix_rho0]-1), axis=1)
    #ij = (np.unravel_index(np.argmin(abs(mat_sigma-sig_sel)), mat_sigma.shape)[0], np.unravel_index(np.argmin(abs(mat_length-lc_sel)[1]), mat_length.shape)[1])
    try : 
        ij = (np.unravel_index(np.argmin(abs(mat_sigma-sig_sel)), mat_sigma.shape)[0], np.unravel_index(np.argmin(abs(mat_length-lc_sel)[1]), mat_length.shape)[1])
    except : 
        ij = np.unravel_index(np.argmin(abs(mat_sigma-sig_sel)), mat_sigma.shape)
    ix_rho, ix_sigma, ix_length = 0, ij[0], ij[1]
    rho0 = mat_rho0[ix_rho]
    md_min, md_max = 0.1, 2.5
    
    ###for inset axis
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
    ###
    
    for t in  ltel_n:
        ####detector data post
        tel = str2telescope(t)
        name = tel.name
        if t=="SNJ": name = "NJ"
        param_dir = Path.home() / 'muon_code_v2_0' / 'AcquisitionParams'
        acq_dir = param_dir / tel.name / "acqVars" / f"az{tel.azimuth}ze{tel.zenith}"
        print("acq_dir", acq_dir)
        acqVar = AcqVars(telescope=tel,acq_dir=acq_dir)
        topo = acqVar.topography[conf]
        mat_dome = mat73.loadmat(str(fGmatrix_dome))['cubesMatrix'] #shape=(N_cubes, N_par_cubes=31)
        fGmatrix_tel = ml_dir / "CubesMuonDependances" /  f"Resol_{res}" / name / f"CubesMuonDependances_{name}_{res}m.mat"
        try : 
            Gmat_tel = sio.loadmat(str(fGmatrix_tel))[f'CubesMuonDependances_{name}']  #shape=(N_los, N_cubes)
        except : 
            Gmat_tel = mat73.loadmat(str(fGmatrix_tel))[f'CubesMuonDependances_{name}']       
        print("ok")
        #exit()
        try : 
            vec_data_syn_tel = np.loadtxt(out_dir/ f"data_syn_{name}_res{res}m.txt")
            vec_unc_syn_tel = np.loadtxt(out_dir/ f"unc_syn_{name}_res{res}m.txt")
        except : 
            vec_data_syn_tel = np.loadtxt(out_dir/ f"data_syn_{name}_3p1_res{res}m.txt")
            vec_unc_syn_tel = np.loadtxt(out_dir/ f"unc_syn_{name}_3p1_res{res}m.txt")
        
        try :  
            fout = out_dir / f"mask_{name}_res{res}m.txt"
            mask_los = np.loadtxt(fout, delimiter="\t")
        except :     
            fout = out_dir / f"mask_{name}_3p1_res{res}m.txt"
            mask_los = np.loadtxt(fout, delimiter="\t")
        
        print(f"Gmat_tel = {Gmat_tel.shape}")
        print(f"mask_los = {len(mask_los)}")
        print(f"mask_cubes = {len(mask_cubes)}")
        
        Gmat_tel = Gmat_tel[:, mask_cubes==True] #select overlapped region voxels
        Gmat_tel = Gmat_tel[mask_los==True]
        #Gmat_tel[np.isnan(Gmat_tel)] = 0
        #Gmat_tel[Gmat_tel!=0] = 1
        nvox_per_los = np.count_nonzero(Gmat_tel, axis=1)
        nz = nvox_per_los != 0 
        Gmat_tel[nz]  = np.divide(Gmat_tel[nz].T, nvox_per_los[nz]).T       
        rhop = rho_post[ix_rho, ix_sigma, ix_length]
        print(Gmat_tel[nz].shape, len(rhop))
        
        
        
        data_post = Gmat_tel[nz] @ rhop  #/ nvox_per_los
       
        nlos, nvox = Gmat_tel[nz].shape
        
        
        
        X, Y = acqVar.az_tomo[conf], acqVar.ze_tomo[conf]    
        z = np.ones(len(mask_los))*np.nan    
        print(len(vec_data_syn_tel), len(data_post), len(vec_unc_syn_tel))
        arr_phi = (vec_data_syn_tel - data_post)  / vec_unc_syn_tel
        z[mask_los==True] =  arr_phi
        phi_d  = 1/nlos * np.sum(arr_phi**2)
        phi_d_std = np.nanstd(arr_phi)
        
        
        Z = z.reshape(X.shape)

        ######misfit 
        im_mis = ax_mis[t].pcolor(X,Y,Z, vmin=md_min, vmax=md_max, cmap=jet)
        ax_mis[t].set_ylim(40, 90)
        ax_mis[t].set_xlabel("$\\varphi$ [deg]")
        ax_mis[t].set_ylabel("$\\theta$ [deg]")
        ax_mis[t].invert_yaxis()
        ax_mis[t].grid(True, which='both',linestyle='-', linewidth="0.3 ", color='grey')
        ax_mis[t].plot(topo[0,:],topo[1,:], linewidth=1., color='black')
        ax_mis[t].set_title(f"{t} : "+"($\\varphi_t$, $\\theta_t$) = "+f"({tel.azimuth:.1f}, {tel.zenith:.1f})°")
        s = "$\\langle\\phi_d\\rangle$"+f" = {phi_d:.2f} $\\pm$ {phi_d_std:.2f} "
        anchored_text = AnchoredText(s, loc="upper left", frameon=True, prop=dict(fontsize=12))
        ax_mis[t].add_artist(anchored_text)
        
        ######data 
        z = np.ones(len(mask_los))*np.nan    
        z[mask_los==True] =  data_post
        Z = z.reshape(X.shape)
        im_dat = ax_dat[t].pcolor(X,Y,Z, vmin=rho_min, vmax=rho_max, cmap=batlow)
        rho_mean = np.nanmean(Z)
        rho_std = np.nanstd(Z)
        ax_dat[t].set_ylim(40, 90)
        ax_dat[t].set_xlabel("$\\varphi$ [deg]")
        ax_dat[t].set_ylabel("$\\theta$ [deg]")
        ax_dat[t].invert_yaxis()
        ax_dat[t].plot(topo[0,:],topo[1,:], linewidth=1., color='black')
        ax_dat[t].grid(True, which='both',linestyle='-', linewidth="0.3 ", color='grey')
        title=f"{t} : "+"($\\varphi_t$, $\\theta_t$) = "+f"({tel.azimuth:.1f}, {tel.zenith:.1f})°"
        ax_dat[t].set_title(title)
        s = "$\\langle\\overline{\\rho}\\rangle$"+f" = {rho_mean:.2f} $\\pm$ {rho_std:.2f} "+" g.cm$^{-3}$"
        anchored_text = AnchoredText(s, loc="upper left", frameon=True, prop=dict(fontsize=12))
        ax_dat[t].add_artist(anchored_text)
        
        
        
        ####inset axis
        axin1 = inset_axes(ax_mis[t],
                        width="22.5%", # width = 30% of parent_bbox
                        height=.75, # height : 1 inch
                        loc='upper right')
        axin2= inset_axes(ax_dat[t],
                        width="22.5%", # width = 30% of parent_bbox
                        height=.75, # height : 1 inch
                        loc='upper right')
        
        
        
        
        zslice=1300#m
        mask_slice =  (zslice - res <= dome.barycenter[:,-1]) & (dome.barycenter[:,-1] <= zslice )
        ilos_vox_s  = ilos_vox[mask_slice]
        nvox_slice = ilos_vox_s.shape[0]
        ilos_bar = dome.barycenter[mask_slice]
        ilos_xyzup = dome.xyz_up[mask_slice]
        ilos_xyzdown = dome.xyz_down[mask_slice]
        #ax_mis[t].annotate(s, xy=(0.5, .5), xycoords='axes fraction', xytext=(0.15, .85), fontsize=fontsize-2)                
        rho_vox_true_slice = rho_vox_true[mask_slice]
        arg_col_true =  [np.argmin(abs(range_val-v))for v in rho_vox_true_slice]
        color_vox_true = color_scale[arg_col_true]
        
        
        verts = ilos_xyzdown
        
        arg_col =  [np.argmin(abs(range_val-v))for v in rhop]    
        color_vox = color_scale[arg_col]
        for i, v in enumerate(verts):
                
            pc1 = PolyCollection([v[:,:-1]],  
                                    cmap=batlow,
                                    #alpha=0.3,
                                    facecolors=color_vox_true[i], #np.repeat(color_vox,axis=0),
                                    edgecolors=np.clip(color_vox_true[i] - 0.5, 0, 1),  # brighter 
                                    norm=Normalize(clip=True),
                                    linewidths=0.3)
            pc2 = PolyCollection([v[:,:-1]],  
                                    cmap=batlow,
                                    #alpha=0.3,
                                    facecolors=color_vox_true[i], #np.repeat(color_vox,axis=0),
                                    edgecolors=np.clip(color_vox_true[i] - 0.5, 0, 1),  # brighter 
                                    norm=Normalize(clip=True),
                                    linewidths=0.3)
            pc1.set_clim(1., 2.0)
            pc2.set_clim(1., 2.0)
            axin1.add_collection(pc1)
            axin2.add_collection(pc2)
        
        for ax in [axin1, axin2]:
            ax.set_xticklabels([])   
            ax.set_yticklabels([])
            ax.scatter(tel.utm[0], tel.utm[1], c='magenta', s=30,marker='.',)
            ax.set_xticklabels([])   
            ax.set_yticklabels([])
            ax.scatter(tel.utm[0], tel.utm[1], c='magenta', s=30,marker='.',)
            
            
            ax.set_xlim(xrange)
            ax.set_ylim(yrange)
            ax.tick_params(
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
        
        
        ####
        
    ####Add color bar for misfit scale
    ax1 = ax_mis["OM"]
    ax2 = ax_mis["SB"]
    cax = fig_mis.add_axes([ax1.get_position().x1+0.02,ax1.get_position().y0,0.02,ax2.get_position().y0])
    cb = fig_mis.colorbar(im_mis,  cax=cax)
    locator = MultipleLocator(5e-1)
    cb.ax.yaxis.set_major_locator(locator)
    cb.set_label(label="$\\phi_d$", size=14)#, labelpad=1.)
    cb.ax.tick_params(which="both",labelsize=14)
    fout = out_dir / f"misfit_all_det_rho0{rho0}_res{res}m"
    fig_mis.subplots_adjust(left=0.05, bottom=0.1, right=0.94, top=0.95, wspace=0.3, hspace=0.4)
    fig_mis.savefig(f"{str(fout)}.png")    
    print(f"save {fout}.png")
    plt.close()  
    
    
    ####Add color bar for misfit scale
    ax1 = ax_dat["OM"]
    ax2 = ax_dat["SB"]
    cax = fig_dat.add_axes([ax1.get_position().x1+0.02,ax1.get_position().y0,0.02,ax2.get_position().y0])
    cb = fig_dat.colorbar(im_dat,  cax=cax)
    locator = MultipleLocator(2e-1)
    cb.ax.yaxis.set_major_locator(locator)
    cb.set_label(label="$\\overline{\\rho}$ [g.cm$^{-3}$]", size=14)#, labelpad=1.)
    cb.ax.tick_params(which="both",labelsize=14)
    fout = out_dir / f"data_post_all_det_rho0{rho0}_res{res}m"
    fig_dat.subplots_adjust(left=0.05, bottom=0.1, right=0.94, top=0.95, wspace=0.3, hspace=0.4)
    fig_dat.savefig(f"{str(fout)}.png")    
    print(f"save {fout}.png")
    plt.close()         
