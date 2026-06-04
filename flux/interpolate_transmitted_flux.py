#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from matplotlib.colors import LogNorm, Normalize
import matplotlib.pyplot as plt
import numpy as np
from scipy import interpolate
import scipy.io as sio
from scipy.integrate import quad, quad_vec
from tqdm import tqdm

#Package modules
from acceptance import geometrical_acceptance
from config import STRUCT_DIR
from flux import FluxModel
from material import Rock
from stoppingpower import StoppingPower
from survey import CURRENT_SURVEY
from utils.tools import print_file_datetime

params = {'legend.fontsize': 'medium',
          'legend.title_fontsize' : 'xx-large',
         'axes.labelsize': 'xx-large',
         'axes.titlesize': 'xx-large',
         'xtick.labelsize':'xx-large',
         'ytick.labelsize':'xx-large',
         'axes.labelpad':1,
         'mathtext.fontset': 'stix',
         'font.family': 'STIXGeneral',
         'axes.grid': True,
         'grid.alpha':0.3,
         'figure.figsize': (8,8),
          'savefig.bbox': "tight",   
        'savefig.dpi':200    }
plt.rcParams.update(params)

if __name__=="__main__":

    survey_name = CURRENT_SURVEY.name   
    dir_survey = STRUCT_DIR / survey_name
    dir_flux = dir_survey / "flux"
    dir_tel = dir_survey / "telescope"

    dict_tel = CURRENT_SURVEY.telescopes
    tel = dict_tel["SNJ"]
    conf = tel.configurations["3p1"]
    tel.compute_angular_coordinates()
    u_edges, v_edges = conf.u_edges, conf.v_edges
    u,v = (u_edges[:-1] + u_edges[1:]) / 2, (v_edges[:-1] + v_edges[1:]) / 2
    nu, nv = len(u_edges)-1, len(v_edges)-1

    theta_os_tel = tel.zenith_os_matrix[conf.name]

    fluxop_file = dir_flux / 'IntegralFluxVsOpAndZaStructure_Corsika.mat'
    fluxop_grid = sio.loadmat(str(fluxop_file))['IntegralFluxVsOpAndZaStructure_rock_MuonsEKin_onlyModel'] 
    print_file_datetime(fluxop_file)
    fluxos_file = dir_flux / "openSkyFluxStructure.mat"
    fluxos_struct = sio.loadmat(str(fluxos_file))["openSkyFluxStructure"][0][0] 
    print(len(fluxos_struct), fluxos_struct[0].shape, fluxos_struct[1].shape, fluxos_struct[2].shape)
    # fig, axs = plt.subplots(ncols=3)
    # axs[0].plot(np.rad2deg(fluxos_struct[0].ravel()), fluxos_struct[1].ravel())
    # axs[1].plot(np.rad2deg(fluxos_struct[0].ravel()), fluxos_struct[2].ravel()) 
    # plt.show()
    # plt.close()

    k = "openSkyFluxStructure"
    theta_os_grid = fluxos_struct[0].ravel() * np.pi/180
    flux_os_grid = fluxos_struct[1].ravel() #
    uncflux_os_grid = fluxos_struct[2].ravel() #
    func = interpolate.interp1d(theta_os_grid, flux_os_grid)  
    flux_os_tel= func(theta_os_tel)
    flux_os_tel= np.interp(theta_os_tel, theta_os_grid, flux_os_grid)
    print("openSkyFluxStructure:", np.nanmin(flux_os_tel), np.nanmax(flux_os_tel))

    fluxos_file = dir_tel /tel.name / "flux"/ f"ExpectedOpenSkyFlux.mat"
    flux_os_grid = sio.loadmat(str(fluxos_file))[f"ExpectedFlux_calib_{conf.name[:-1]}"]
    print("ExpectedOpenSkyFlux:", np.nanmin(flux_os_grid),np.nanmax(flux_os_grid))

    model="guan"
    fout_npy = dir_flux/ f"flux_opensky_{model}_{tel.name}_{conf.name}.npy"
    # for model in  ["guan"]:
    #     flux_os_model = FluxModel().compute_opensky_flux(theta_os_tel.ravel(), model=model)
    #     np.save(fout_npy, flux_os_model)
    #     print(f"flux_{model}:", np.nanmin(flux_os_model),np.nanmax(flux_os_model))
    # flux_os_model = FluxModel().compute_opensky_flux(theta_os_tel.ravel(), model=model)
    flux_os_model = np.load(fout_npy)
    fig, axs = plt.subplots(ncols=2)
    norm=LogNorm(np.nanmin(flux_os_tel),np.nanmax(flux_os_tel))
    im0 = axs[0].pcolormesh(u, v, flux_os_grid.reshape(nu,nv), norm=norm)
    im1 = axs[1].pcolormesh(u, v, flux_os_model.reshape(nu,nv), norm=norm)
    fout_png = f"flux_opensky_{tel.name}_{conf.name}.png"
    fig.savefig(fout_png)
    print(f"Saved {fout_png}")
    # print(flux_os_tel)    
    # print(flux_os_grid)
    # fluxos_file = dir_flux / "ExpectedOpenSkyFlux.mat"
    # fluxos_grid = sio.loadmat(str(fluxop_file))['ExpectedOpenSkyFlux'] 

    # exit()
    opacity_log10 = fluxop_grid[0][0][1] #opacity shape: (bins_ene, bins_op) = (100, 600)
    opacity_grid = np.exp( np.log(10) * opacity_log10 )
    theta_grid = fluxop_grid[0][0][0] #zenith angle
    flux_grid = fluxop_grid[0][0][2] #Corsika flux
    flux_grid_low = fluxop_grid[0][0][3]
    flux_grid_up = fluxop_grid[0][0][4]
    # print(opacity_grid.shape, theta_grid.shape, flux_grid.shape)
    # opacity = np.sort(np.unique(opacity_grid)) #hg/cm^2 

    dir_voxel = dir_survey / "voxel" 
    # file_voxelray = dir_voxel / f"voxel_ray_matrix_{tel.name}_{conf.name}_vox8m.npz"
    file_ray_length =  dir_voxel / f"ray_length_{tel.name}_{conf.name}_vox4m.npy"
    raylength = np.load(file_ray_length).reshape(conf.shape_uv)
    
    delta_z = conf.length_z*1e-1 #mm > cm
    theta_tel = tel.zenith_matrix[conf.name]

    mask = (raylength > 1) & (theta_tel <= np.pi/2)
    opacity_tel = np.zeros_like(raylength) 
    opacity_tel[mask] = 2.65 * (raylength[mask])

    X,Y, values = theta_grid.ravel(), opacity_grid.ravel(), flux_grid.ravel()
    points = np.vstack((X.ravel(), Y.ravel())).T
    xi = np.vstack((theta_tel.ravel(), opacity_tel.ravel())).T

    flux_tel = interpolate.griddata(points, values, xi=xi, method='linear').reshape(nu, nv) # 1 / [cm^2.sr.s]
    flux_tel[~mask] = np.nan
    fig, ax = plt.subplots()
    im = ax.pcolormesh(u, v, flux_tel, cmap="RdBu", norm=LogNorm(np.nanmin(flux_tel),np.nanmax(flux_tel)))
    cb = fig.colorbar(im)
    cb.set_label("Integral Flux [cm$^2$ sr s]$^{-1}$")
    fout_png= f"{fluxop_file.stem}_{tel.name}_{conf.name}.png"
    fig.savefig(fout_png)
    print(f"Saved {fout_png}")
    