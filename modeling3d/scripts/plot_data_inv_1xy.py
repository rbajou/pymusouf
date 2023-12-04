#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as cm
import matplotlib.colors as colors
from mpl_toolkits.mplot3d.art3d import PolyCollection, Poly3DCollection, Line3DCollection
from matplotlib.ticker import MultipleLocator,EngFormatter, ScalarFormatter
from matplotlib.offsetbox import AnchoredText
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.gridspec import GridSpec
from matplotlib.cm import ScalarMappable
import sys
import scipy.io as sio
from pathlib import Path
import time
from datetime import datetime, date, timezone
import logging
import warnings
import pandas as pd
import mat73 #read v7.3 mat files
import shapefile
import utm 
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


####
params = {'legend.fontsize': 'xx-large',
        'axes.labelsize': 'xx-large',
        'axes.titlesize':'xx-large',
        'xtick.labelsize':'xx-large',
        'ytick.labelsize':'xx-large',
        'axes.labelpad':10}
plt.rcParams.update(params)
####


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
    ###coordinates
    ltel_n = ["SB", "SNJ", "BR", "OM"]
    #ltel_n = ["SNJ",]#, "OM"]
    str_tel =  "_".join(ltel_n)
    ltel_coord = np.array([ str2telescope(tel).utm[:-1] for tel in ltel_n])
    ltel_color = np.array([ str2telescope(tel).color for tel in ltel_n])

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
    
    mask_cubes = mask_cubes != 0 #convert to bool type
    mask_cyl = mask_cubes != 0
    #print(f"mask_cubes = , shape = {mask_cubes.shape}")
    #print(f"sv_center = {sv_center}, shape={sv_center.shape}")
    mask_cubes = mask_cubes & sv_center
    #print(f"mask_cubes &  sv_center = {mask_cubes}, shape={mask_cubes[mask_cubes==1].shape}")
    mask_cyl = mask_cyl & sv_cyl


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

    Gmat = np.loadtxt(out_dir/ f"Gmat_all_res{res}m.txt", delimiter="\t")#[mask_dome]
    data_real = np.loadtxt(out_dir/ f"data_all_res{res}m.txt",delimiter="\t")#[mask_dome]
    unc_real = np.loadtxt(out_dir/ f"unc_all_res{res}m.txt", delimiter="\t")#[mask_dome]

    
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
    fig.savefig(f"{fout}.png", transparent=True)    
    print(f"save {fout}.png")
    '''

    dx = 800#700
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
    with open(str(fin), 'rb') as f:
        arr_rho_post = np.load(f)
        
    fin = out_dir /  f"std_dev_res{res}m.npy"         
    with open(str(fin), 'rb') as f:
        arr_std = np.load(f)
    
    ####Plot density model
    cmap_rho = palettable.scientific.sequential.Batlow_20.mpl_colormap #'jet' #palettable.scientific.diverging.Roma_20
    rho_min, rho_max, n = .8, 2.7, 100
    range_val_rho = np.linspace(rho_min, rho_max, n)
    norm_r = cm.Normalize(vmin=rho_min, vmax=rho_max)(range_val_rho)
    color_scale_rho =  cmap_rho(norm_r)
    vmin_r, vmax_r = rho_min, rho_max

    std_min, std_max, n = .02, .2, 100
    range_val_std = np.linspace(std_min, std_max, n)
    norm_std = cm.Normalize(vmin=std_min, vmax=std_max)(range_val_std)
    color_scale_std =  cmap_rho(norm_std)
    vmin_std, vmax_std = std_min, std_max
    
    
    ####strength model scale
    cmap_strength = palettable.scientific.diverging.Roma_20.mpl_colormap
    strength_min, strength_max = 1e-1, frho_to_strength(rho_max)
    range_val_s = np.logspace(np.log10(strength_min), np.log10(strength_max), n)
    norm_s = cm.Normalize(vmin=strength_min, vmax=strength_max)
    lognorm_s = cm.LogNorm(vmin=strength_min, vmax=strength_max)
    vmin_s, vmax_s = strength_min, strength_max
    color_scale_strength = cmap_strength(norm_s(range_val_s))
    color_scale_strength = cmap_strength(lognorm_s(range_val_s))

    
    
    
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

    lzslice= [1.35e3] #1.4e3, 1.372e3, 1.324e3, 1.3e3, 1.292e3]#1.35e3
    l_ipan = ["a", "b", "c", "d"]
    
     ####Draw contours and craters: 
    fcontour = Path.home()/"QGis_Elements/dome/contours_100m.shp"
    shps = shapefile.Reader(str(fcontour)).shapes()
    TAR = np.array([6.429728210000000e5, 1.774237168000000e6])#, 1.5e3])
    CSS = np.array([6.430440754723654e5, 1.774125703493994e6])#, 1.5e3])
    BLK = np.array([643217.61, 1774144.31]) #1350
    G56 = np.array([643094.02, 1774205.92]) #1418
    FNO = np.array([642958.52, 1774500.03])
    NAP = np.array([642994.49, 1774232.5])
    ##isocurves altitudes
    #isocurves altitudes
    d_l = { 0:{"z":1200, "xy": np.array([642800-100, 1773880]), "r": -20},
            1:{"z":1300, "xy": np.array([642590, 1774343]), "r": 70},  
            2:{"z":1300, "xy": np.array([643365+50, 1773912+70]), "r": 30},  
            3:{"z":1400, "xy": np.array([643136-20, 1774437-30]), "r": -40}, 
           }
    ####fumeroles pos
    ffumerole = Path.home()/"QGis_Elements/dome/fumerole.shp"
    shps2 = shapefile.Reader(str(ffumerole)).shapes()
    ####ERT anomalies
    xyz_anom1_ert = np.loadtxt(Path.home() / "data/ERT/anomaly/Conductor_A1_xyz.txt", skiprows=1)
    xyz_anom2_ert = np.loadtxt(Path.home() / "data/ERT/anomaly/Conductor_A2_xyz.txt", skiprows=1)
 
    for zslice in lzslice : 
        for ix_rho0 in range(nrho): #
            rho0 = mat_rho0[ix_rho0]
            
            fig_best_rho = plt.figure(figsize=(12,9))#, constrained_layout=True)
        
            fig_best_rho.subplots_adjust(left=0.1, bottom=0.1, right=0.85, top=0.95, wspace=0.20, hspace=0.20)
            ax_best_rho = fig_best_rho.subplot_mosaic(
                    [[zslice]],
                    sharex=True, sharey=True
                )
            fig_best_strength = plt.figure(figsize=(12,9))#, constrained_layout=True)
            ax_best_strength = fig_best_strength.subplot_mosaic(
                    [[zslice]], 
                    sharex=True, sharey=True
                )
            fig_best_strength.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.95, wspace=0.20, hspace=0.20)
            fig_std = plt.figure(figsize=(12,9))#, constrained_layout=True)
            ax_std = fig_std.subplot_mosaic(
                    [[zslice]], 
                    sharex=True, sharey=True
                )
            fig_std.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.95, wspace=0.20, hspace=0.20)
            
            #plt.gcf().text(0.4, 0.93, f"$\\rho_{0}$ = {rho0}"+" g.cm$^{-3}$,"+f"\nz = {zslice} m a.s.l", fontsize=fontsize+4)

            

            ####Define slice mask
            mask_slice =  (zslice - res <= dome.barycenter[:,-1]) & (dome.barycenter[:,-1] <= zslice )
            ilos_vox_s  = ilos_vox[mask_slice]
            nvox_slice = ilos_vox_s.shape[0]
            ilos_bar = dome.barycenter[mask_slice]
            ilos_xyzup = dome.xyz_up[mask_slice]
            ilos_xyzdown = dome.xyz_down[mask_slice]
            #test if voxels are all nan
            if len(arr_rho_post[0,0,0][mask_slice]) == 0: continue

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
            fontsize = 10
            plt.gcf().text(0.4, 0.93, f"$\\rho_{0}$ = {rho0}"+" g.cm$^{-3}$,"+f"\nz = {zslice} m a.s.l", fontsize=fontsize+4)
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
                    std_dev = arr_std[ix_rho0,i,j][mask_slice]
                    misfit_d[ix_rho0,i,j] = 1/Nd *  np.sum( (data_real - data_post)**2 / unc_real**2) 
                    #print('rho_vox_post = ', rho_vox_post, len(rho_vox_post), len(rho_vox_post[rho_vox_post!=0]))
                    print(f'rho_vox_post = {np.nanmean(rho_vox_post):.3f} +/- {np.nanstd(rho_vox_post):.3f} g/cm^3')
                    print(f'std_dev = {np.nanmean(std_dev):.3f} +/- {np.nanstd(std_dev):.3f} g/cm^3')
                    if len(rho_vox_post) == 0 : continue
                    
                    arg_col =  [np.argmin(abs(range_val_rho-v))for v in rho_vox_post]    
                    color_vox = color_scale_rho[arg_col]
                    x,y,z = ilos_bar.T
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
                    '''
                    for l, v in enumerate(verts):
                        # if l > 100: exit()
                        # print(f"verts_down_{l}, verts_up_{l}= {v[:,1:]}, {ilos_xyzup[l,1:]}",)
                        # print(f"verts_{l}=",v[:,:-1])
                        pc = PolyCollection([v[:,:-1]],  
                                                cmap=cmap_rho,
                                                #alpha=0.3,
                                                facecolors=color_vox[l], #np.repeat(color_vox,axis=0),
                                                edgecolors=np.clip(color_vox[l] - 0.5, 0, 1),  # brighter 
                                                norm=Normalize(clip=True),
                                                linewidths=0.3)
                        pc.set_clim(vmin_r, vmax_r)
                        ax.add_collection(pc)
                    ax.scatter(ltel_coord[:,0], ltel_coord[:,1], c=ltel_color, s=30,marker='s',)
                    s = "$\\phi_{d}$"+f"={misfit_d[ix_rho0,i,j]:.2e}"
                    ax.annotate(s, xy=(0.5, .5), xycoords='axes fraction', xytext=(0.1, .75), fontsize=fontsize-2)
                    #anchored_text = AnchoredText(s, loc="upper left", frameon=False, prop=dict(fontsize=fontsize-2))
                    ax.set_xlim(xrange)
                    ax.set_ylim(yrange)
                    ax.set_aspect('auto') #'equal'
                    ax.grid(True, which='both',linestyle='-', linewidth="0.3 ", color='grey')
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
            #     ax1 = plt.subplot(gs[nrow-4, ncol-1])
            # except : 
            #     ax1 = plt.subplot(gs[0, 0])
            # cax = fig.add_axes([ax1.get_position().x1+0.03,ax1.get_position().y0,0.02,ax1.get_position().y0])
            # cb = fig.colorbar(pc,  cax=cax)
            # #ticks = np.linspace(0.2, 2.0, 10)
            # #cb.ax.set_yticklabels([f"{i:.1f}" for i in ticks])
            # locator = MultipleLocator(4e-1)
            # cb.ax.yaxis.set_major_locator(locator)
            # cb.set_label(label="mean density $\\overline{\\rho}$ [g.cm$^{-3}$]", size=fontsize+2, labelpad=1.)
            # cb.ax.tick_params(which="both",labelsize=fontsize)


            # fout = out_dir / f"rho_post_rho0{rho0}_z{int(zslice)}m_res{res}m"
            # fig.savefig(f"{str(fout)}.png")   
            # print(f"save {fout}.png")
            # print("End : ", time.strftime("%H:%M:%S", time.localtime()))#start time
            
                
            #####plot best misfit data inversion
            fontsize = 20
            ix= ij_d    
            sigma, length= mat_sigma[ix], mat_length[ix]
            #apply mask for given slice
            rho_vox_post = arr_rho_post[ix_rho0, ij_d[0], ij_d[1]][mask_slice]
            std_dev_post = arr_std[ix_rho0, ij_d[0], ij_d[1]][mask_slice]
            strength_vox_post = frho_to_strength(rho_vox_post)
            arg_col_rho =  [np.argmin(abs(range_val_rho-v))for v in rho_vox_post]   
            arg_col_strength =  [np.argmin(abs(range_val_s-v))for v in strength_vox_post]  
            arg_col_std =  [np.argmin(abs(range_val_std-v))for v in std_dev_post]      
            color_vox_rho = color_scale_rho[arg_col_rho]
            color_vox_strength = color_scale_strength[arg_col_strength]
            color_vox_std = color_scale_std[arg_col_std]
            for l, v in enumerate(verts):
                pcr = PolyCollection( [v[:,:-1]]-np.array([vc_x, vc_y]),  
                                cmap=cmap_rho,
                                #alpha=0.3,
                                facecolors=color_vox_rho[l], #np.repeat(color_vox,axis=0),
                                edgecolors=np.clip(color_vox_rho[l] - 0.5, 0, 1),  # brighter 
                                norm=colors.Normalize(clip=True),
                    linewidths=0.3)
                pcr.set_clim(vmin_r, vmax_r)
                ax_best_rho[zslice].add_collection(pcr)
                pcs = PolyCollection( [v[:,:-1]]-np.array([vc_x, vc_y]),  
                                        cmap=cmap_strength,
                                        #alpha=0.3,
                                        facecolors=color_vox_strength[l], #np.repeat(color_vox,axis=0),
                                        edgecolors=np.clip(color_vox_strength[l] - 0.5, 0, 1),  # brighter 
                                        norm=lognorm_s,
                            linewidths=0.3)
                pcs.set_clim(vmin_s, vmax_s)
                ax_best_strength[zslice].add_collection(pcs)
                pcstd = PolyCollection( [v[:,:-1]]-np.array([vc_x, vc_y]),  
                                        cmap=cmap_rho,
                                        #alpha=0.3,
                                        facecolors=color_vox_std[l], #np.repeat(color_vox,axis=0),
                                        edgecolors=np.clip(color_vox_std[l] - 0.5, 0, 1),  # brighter 
                                        norm=colors.Normalize(clip=True),
                            linewidths=0.3)
                pcstd.set_clim(vmin_std, vmax_std)
                ax_std[zslice].add_collection(pcstd)
                
                
            
            for axb in [ax_best_rho[zslice], ax_best_strength[zslice], ax_std[zslice]]:
                

                locator = MultipleLocator(5e2)
                axb.xaxis.set_major_locator(locator)
                axb.yaxis.set_major_locator(locator)
                axb.scatter(ltel_coord[:,0]-vc_x, ltel_coord[:,1]-vc_y, c=ltel_color, s=50,marker='s',edgecolor='black')
                #axb.scatter(ltel_coord[:,0]-vc_x, ltel_coord[:,1]-vc_y, c='magenta', s=30,marker='o',)
                #axb.annotate(f"\nz = {zslice:.0f} m a.s.l", 
                #             xy=(0.5, .5), xycoords='axes fraction', xytext=(.05, .85), fontsize=14) #xytext=(.25, .85) #<- centered
                anchored_text = AnchoredText(f"z = {zslice:.0f} m a.s.l", loc="upper left", frameon=True, prop=dict(fontsize=fontsize))
                axb.add_artist(anchored_text)
                #ipan = AnchoredText(f"{l_ipan[0]}", loc="upper right", frameon=True, prop=dict(fontsize=fontsize))
                #axb.add_artist(ipan)
                #axb.annotate("$\\phi_{d}$"+f"={misfit_d[ix_rho0,i,j]:.2e}", xy=(0.5, .5), xycoords='axes fraction', xytext=(0.15, .85), fontsize=fontsize-2)
                axb.set_xlim([-dx, +dx])
                axb.set_ylim([-dy, +dy])
                axb.set_aspect('auto') #'equal'
                axb.grid(True, which='both',linestyle='dotted', linewidth="0.3 ", color='grey')


        
        for ax in [ax_best_rho, ax_best_strength, ax_std]:
            ax1 = ax[zslice] 
            ax1.set_xlabel("X [m]", fontsize=fontsize)
            ax1.set_ylabel("Y [m]", fontsize=fontsize)
            ax1.tick_params(labelsize=fontsize-2)
            
            
            x,y,z = xyz_anom1_ert.T
            ax1.fill(x-vc_x,y-vc_y, color="orange", alpha=0.4, label="A1", edgecolor="orange")
            ax1.annotate("A1", xy =( 642650-vc_x,1773970-vc_y), xycoords='data', annotation_clip=True, fontsize=20, alpha=1.,fontweight='normal')  
            x,y,z = xyz_anom2_ert.T
            ax1.fill(x-vc_x,y-vc_y, color="red", alpha=0.4, label="A2", edgecolor="red")
            ax1.annotate("A2", xy =( 642850-vc_x,1773700-vc_y), xycoords='data', annotation_clip=True, fontsize=20, alpha=1.,fontweight='normal')  

            #####fumeroless
            for s, shape in enumerate(shps2):
                points = shape.points
                long, lat = points[0]
                x, y = utm.from_latlon(lat,long)[0],  utm.from_latlon(lat,long)[1]
                ax1.scatter(x-vc_x, y-vc_y,  s=35, marker="^", color="purple")
                
                
            for i, shape in enumerate(shps):
                points = shape.points
                lx, ly = np.zeros(len(points)),np.zeros(len(points))
                for j,pt in enumerate(points):  lx[j], ly[j] = pt
                ax1.plot(lx-vc_x, ly-vc_y, linewidth=1., marker=".",  markersize=0., color="black")   
                kwargs=dict( s=40, marker="D", color="deepskyblue", edgecolor="black")
                ax1.scatter(TAR[0]-vc_x, TAR[1]-vc_y,**kwargs) #"cyan"
                ax1.scatter(CSS[0]-vc_x, CSS[1]-vc_y,**kwargs) #"cyan"
                ax1.scatter(FNO[0]-vc_x, FNO[1]-vc_y,**kwargs) #"cyan"
                ax1.scatter(BLK[0]-vc_x, BLK[1]-vc_y,**kwargs) #"cyan"
                ax1.scatter(G56[0]-vc_x, G56[1]-vc_y,**kwargs) #"cyan"
                #ax1.scatter(NAP[0]-vc_x, NAP[1]-vc_y, s=30, marker="D", color="deepskyblue") #"cyan"
                for k, v in d_l.items():     
                    ax1.annotate(f"{v['z']}", 
                                xy =(v["xy"][0]-vc_x, v["xy"][1]-vc_y), 
                                xycoords='data', 
                                annotation_clip=True, 
                                fontsize=8, 
                                rotation=v["r"],
                                fontweight='ultralight')  
                


            ####Add color bar for density scale
            divider = make_axes_locatable(ax1)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            if ax == ax_best_rho : 
                #cax = fig_best_rho.add_axes([ax1.get_position().x1+0.02,ax1.get_position().y0,0.02,ax1.get_position().y0])
                cb = fig_best_rho.colorbar(pcr,  cax=cax)
                locator = MultipleLocator(2e-1)
                cb.ax.yaxis.set_major_locator(locator)
                cb.set_label(label="density $\\~{\\rho}$ [g.cm$^{-3}$]")#, size=fontsize)#, labelpad=1.)
            if ax == ax_best_strength : 
                #cax = fig_best_strength.add_axes([ax1.get_position().x1+0.02,ax1.get_position().y0,0.02,ax1.get_position().y0])
                cb = fig_best_strength.colorbar(ScalarMappable(norm=lognorm_s, cmap=cmap_strength), cax=cax)    
                cb.set_label(label="unixial compressive strength [MPa]")#, size=fontsize)#, labelpad=1.)

            if ax == ax_std : 
                #cax = fig_best_rho.add_axes([ax1.get_position().x1+0.02,ax1.get_position().y0,0.02,ax1.get_position().y0])
                cb = fig_std.colorbar(pcstd,  cax=cax)
                locator = MultipleLocator(4e-2)
                cb.ax.yaxis.set_major_locator(locator)
                cb.set_label(label="standard deviation $\\~{\\sigma}$ [g.cm$^{-3}$]")#, size=fontsize)#, labelpad=1.)
            
            cb.ax.tick_params(which="both",labelsize=fontsize)
            ##North arrow
            x, y, arrow_length = 0.96, 0.15, 0.05
            ax1.annotate('N', xy=(x, y), xytext=(x, y-arrow_length), color="black",
                arrowprops=dict(facecolor='black', width=5, headwidth=15, alpha=0.4),
                ha='center', va='center', fontsize=fontsize,
                xycoords=ax1.transAxes)
        
        fout = out_dir / f"best_misfit_data_rho_post_rho0{rho0}_res{res}m_z{zslice}m_xy"
        fig_best_rho.savefig(f"{str(fout)}.png", dpi=300, transparent=True)    
        print(f"save {fout}.png")
        
        
        # fig_best_strength.text(.3, .9, s=f"$\\rho_{0}$ = {rho0}"+" g.cm$^{-3}$, "
        #             +f"$\\sigma$ = {sigma}"+" g.cm$^{-3}$, " 
        #             +f"$l$ = {length}"+" m\n" , fontsize=14)

        fout = out_dir / f"best_misfit_strength_rho0{rho0}_res{res}m_z{zslice}m_xy"
        fig_best_strength.savefig(f"{str(fout)}.png", dpi=300, transparent=True )    
        print(f"save {fout}.png")
        
        fout = out_dir / f"std_dev_rho0{rho0}_res{res}m_z{zslice}m_xy"
        fig_std.savefig(f"{str(fout)}.png", dpi=300, transparent=True )    
        print(f"save {fout}.png")
        
        plt.close()

        #print("End : ", time.strftime("%H:%M:%S", time.localtime()))#start time
        print(f"End --- {(time.time() - start_time):.3f}  s ---") 



            

            









    '''
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    postp, rho0p, sigp, lp = arr_rho_post[-1,-1,-1], mat_rho0[-1], mat_sigma[-1], mat_length[-1]
    colors = color_scale[[np.argmin(abs(range_val_rho-v))for v in postp]]
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
