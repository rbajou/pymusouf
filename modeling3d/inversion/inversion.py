#!/usr/bin/env python3
# -*- coding: utf-8 -*-



from dataclasses import dataclass, field
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as cm
from matplotlib.colors import LogNorm
from mpl_toolkits.mplot3d.art3d import PolyCollection, Poly3DCollection, Line3DCollection
from matplotlib.colors import Normalize
from mpl_toolkits.axes_grid1 import make_axes_locatable
import scipy.io as sio
from pathlib import Path
import time
import mat73 #read v7.3 mat files

#personal modules
from config import MAIN_PATH
from modeling3d.voxel.voxel import DirectProblem, Voxel
from raypath import RayPathSoufriere
from telescope import str2telescope, Telescope
from survey import CURRENT_SURVEY


@dataclass
class SyntheticData:
   
    voxray_matrix: np.ndarray #shape (nray,nvox)
    rho: np.ndarray # (nvox,)
    unc: np.ndarray # (nvox,)
   
    def __post_init__(self):
       
        Gmat = self.voxray_matrix # (nray,nvox)
        self.obs  = Gmat @ self.rho # (nray,) 
        self.unc = Gmat @ self.unc # (nray,)
        mat_nvox_nz = np.count_nonzero(Gmat, axis=1) # (nray,)
        nz = mat_nvox_nz != 0 
        self.mask = nz
        #self.obs[nz] = self.obs[nz]/mat_nvox_nz[nz]
        #self.unc[nz] = self.unc[nz]/mat_nvox_nz[nz]
        self.gaus = np.zeros(self.obs.shape)
        self.gaus[nz] = np.random.normal(loc=self.obs[nz], scale=self.unc[nz])

    def plot_map_2d(self, fig, ax, x:np.ndarray, y:np.ndarray,  **kwargs):
       
        z = self.gaus.reshape(x.shape)
        z[ z==0 ] = np.nan
        im = ax.pcolor(x,y,z,  shading='auto', **kwargs)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        cbar = fig.colorbar(im, cax=cax, orientation="vertical")
        cbar.ax.tick_params(labelsize=12)
        cbar.set_label(label="mean density [g/cm$^3$]", size=12)
        ax.set_ylim(np.nanmin(y), 90)
        ax.invert_yaxis()
        # ax.grid(True, which='both',linestyle='-', linewidth="0.3 ", color='grey')
        # ax.set_xlabel(f"azimuth $\\varphi$ [deg]")
        # ax.set_ylabel(f"zenith $\\theta$ [deg]")


class Inversion:


    def __init__(self, voxel:Voxel, data:np.ndarray, unc:np.ndarray, voxray_matrix:np.ndarray, C_d:np.ndarray=None):

        self.voxel = voxel
        self.voxel_matrix =  self.voxel.vox_matrix
        self.data = data # self.voxray_matrix.T @ self.rho
        self.unc = unc # self.voxray_matrix.T @ self.unc 
        self.voxray_matrix = voxray_matrix
       
        self.C_d = np.diag(np.multiply(self.unc, self.unc)) # data covariance matrix
        if C_d is not None : self.C_d = C_d
        self.C_d_inv = np.linalg.inv(self.C_d)

        nvox = voxray_matrix.shape[1]
        self.C_smooth = np.zeros(shape=(nvox,nvox) )
        self.rho_post = np.zeros(shape=nvox)
        self.nvox = nvox
        
    def scaling_mrc(self, coord_tel):
        self.mat_scaling = np.ones(self.nvox)
        for i in range(self.nvox): 
            xyz_b = self.voxel.barycenter[i]
            d_tel = [np.linalg.norm(xyz_t - xyz_b) for xyz_t in coord_tel]
            rmin = d_tel[np.argmin(d_tel)]
            self.mat_scaling[i] = 1/rmin**1.5

    def get_model_post(self, rho0:np.ndarray):#, err_prior, d, l):
        '''
        rho0 : initial guess (shape=(nvox,))
        '''
        G = self.voxray_matrix
        self.C_rho_post = G.T @ self.C_d_inv @ G
        #print(f"self.C_smooth  = {self.C_smooth}")
        self.C_rho_post += np.linalg.inv(self.C_smooth)
        #print(f"self.C_rho_post  = {self.C_rho_post}, shape = {self.C_rho_post.shape}")
        #print(f"self.C_rho_post[nan] = {self.C_rho_post[np.isnan(self.C_rho_post)]}, shape = {self.C_rho_post[np.isnan(self.C_rho_post)].shape}")
        self.C_rho_post_inv =  np.linalg.inv(self.C_rho_post)
        self.rho_post = rho0 + self.C_rho_post_inv @ G.T @ self.C_d_inv @ ( self.data - G @ rho0) 
    

    def smoothing(self, err_prior:float, distance:np.ndarray, length:float,  damping:np.ndarray = None):
        """
        err_prior: a priori error on density [g/cm^3]
        distance : distance between voxels [m]
        length : correlation length [m]
        """
        nvox = self.voxray_matrix.shape[1]
        print(f"nvox = {nvox}")
        
        for i in range(nvox):
            for j in range(nvox):
                self.C_smooth[i,j] = err_prior**2 * np.exp(-distance[i,j]/length)
                if i == j : 
                    #diag damping
                    if damping is not None: self.C_smooth[i,j]  *= damping[i] 
                
    
    def gaus_smoothing(self, err_prior, distance, length, damping:np.ndarray = None):
        """
        err_prior: a priori error on density [g/cm^3]
        distance : distance between voxels [m]
        length : correlation length [m]
        """
        nvox = self.voxray_matrix.shape[1]
        #print(f"nvox = {nvox}")
        for i in range(nvox):
            for j in range(nvox):
                self.C_smooth[i,j] = err_prior**2 * np.exp(-(distance[i,j]/length)**2)
                if i == j : 
                    #diag damping
                    if damping is not None: self.C_smooth[i,j]  *= damping[i] 
 
    def get_misfit(self, rho0, err_prior):
        """
        misfit a.k.a cost function, objective function, least-squares function
        chi2 criterion to evaluate the goodness-of-fit
        """
        G = self.voxray_matrix
        self.misfit = ( self.data - G @ self.rho_post).T @ self.C_d_inv @ (self.data - G @ self.rho_post)
        self.misfit += err_prior * (self.rho_post-rho0).T @  self.C_rho_post_inv @ (self.rho_post-rho0)
        
        
if __name__ == "__main__":
    
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
        voxel.barycenters =  voxel.vox_matrix[:,19:22]

    else : 
        print(f"generateMesh() start")
        voxel.generateMesh()
        vox_matrix = voxel.vox_matrix
        np.save(fout_vox_struct, vox_matrix)

    print(f"generateMesh() end --- {time.time() - t0:.1f} s")

    print(vox_matrix.shape)
    print(voxel.barycenters.shape)

    '''
    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")
    voxel.getVoxels()
    color_vox = np.array([[0.98039216, 0.8,  0.98039216, 1.        ]])
    kwargs_mesh = dict(facecolor=color_vox)
    voxel.plot3Dmesh(ax=ax, vox_xyz=voxel.vox_xyz, **kwargs_mesh)#color_vox=color_vox)
    kwargs_topo = dict ( alpha=0.2, color='lightgrey', edgecolor='grey' )
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
   # mask = np.isnan(thickness).flatten()
   # tel.plot_ray_paths(ax=ax, front_panel=front, rear_panel=rear, mask=mask, rmax=1500,  color='grey', linewidth=0.3 )#
    '''

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

    vox_val = np.ones(shape=G.shape[1])*2.0   # (nray,nvox)
    vox_unc = vox_val*0.1
    ray_val  = G  @ vox_val # (nray,) 
    unc = G  @ vox_unc # (nray,)
    mat_nvox_nz = np.count_nonzero(G, axis=1) # (nray,)
    mat_nvox_nnan = ~np.all(np.isnan(G), axis=1) # (nray,)
    # print(np.all(np.isnan(G), axis=1))
    mask =  (mat_nvox_nz != 0 ) & mat_nvox_nnan
  
    #data_obs[mask] = data_obs[mask] / mat_nvox_nz[mask]
    #data_unc[mask] = data_unc[mask] / mat_nvox_nz[mask]
    data_gaus = np.zeros(ray_val.shape)
    print(f'{np.any(np.isnan(ray_val[mask]))}, {np.any(np.isnan(unc[mask]))}')
   
    data_gaus[mask] = np.random.normal(loc=ray_val[mask], scale=unc[mask])

    inversion = Inversion(voxel = voxel, 
                          data = data_gaus[mask], 
                          unc = unc[mask], 
                          voxray_matrix = G[mask], )

    voxel.getVoxelDistances()
    mat_voxdist = voxel.vox_distances
   
    ###test 
    l_rho0 = [1.8]#, 2.0]#[1.8, 2.0]
    l_sig = [3e-1, ] #np.linspace(0.1, 0.8, 8)#[1e-1, 2e-1, 3e-1, 4e-1]#[2e-1, 3e-1] #[1e-1, 2e-1, 3e-1, 4e-1, 5e-1, 6e-1] # 7e-1, 8e-1, 9e-1, 1]
    l_lc = [200, ] #np.linspace(50, 400, 8)#[1e2, 1.1e-1, 1.2e2, 1.3e-1, 1.4e2, 1.5e2, 1.6e2, 1.7e2, 1.8e2] #1e1, 2e1, 4e1, 6e1, 8e1, 
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
                print(f"smoothing: (rho0, err_prior, correlation length) = ({rho0:.1f} g/cm^3, {sig:.3e} g/cm^3, {lc:.3e} m) --- {(time.time() - start_time):.3f}  s ---") 
                vec_rho0 = rho0*np.ones(nvox)
                vec_sig  = sig*np.ones(nvox)
                inversion.get_model_post(rho0=vec_rho0)
                mat_rho_post[i,j,k] = inversion.rho_post

                print(f"rho_post = {mat_rho_post[i,j,k]} , shape={mat_rho_post[i,j,k].shape}")