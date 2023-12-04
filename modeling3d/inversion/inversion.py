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
from telescope import str2telescope, Telescope
from reco.reco import AcqVars, Observable

from plot_density_3d import VoxeledDome


@dataclass
class KernelMatrix:
    tel : Telescope
    resolution: float 
    matrix : np.ndarray
    def __str__(self): 
        return f"G matrix of spatial resolution {self.resolution}"
    def __post_init__(self):
        nvoxels = self.matrix.shape[0]
        object.__setattr__(self, 'nvoxels',  nvoxels )

@dataclass
class DataSynth:
    G: KernelMatrix #(nlos,nvox)
    rho: np.ndarray #(nvox,)
    unc: np.ndarray #(nvox,)
    def __post_init__(self):
        Gmat = self.G.matrix
        self.obs  = Gmat  @ self.rho #nlos 
        self.unc = Gmat  @ self.unc #nlos
        mat_nvox_nz = np.count_nonzero(Gmat, axis=1) #nlos
        nz = mat_nvox_nz != 0 
        self.mask = nz
        #self.obs[nz] = self.obs[nz]/mat_nvox_nz[nz]
        #self.unc[nz] = self.unc[nz]/mat_nvox_nz[nz]
        self.gaus = np.zeros(self.obs.shape)
        self.gaus[nz] = np.random.normal(loc=self.obs[nz], scale=self.unc[nz])

    def plot_data(self, fig, ax, x:np.ndarray, y:np.ndarray,  **kwargs):
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
        ax.grid(True, which='both',linestyle='-', linewidth="0.3 ", color='grey')
        ax.set_xlabel(f"azimuth $\\varphi$ [deg]")
        ax.set_ylabel(f"zenith $\\theta$ [deg]")


class Inversion:
    def __init__(self, voxel_dome:VoxeledDome, d_obs:np.ndarray, unc_obs:np.ndarray, G:np.ndarray, C_d:np.ndarray=None):
        self.dome = voxel_dome
        self.voxels =  self.dome.cubes
        self.d_obs = d_obs#(self.G.T @ self.rho)
        self.unc_obs = unc_obs#(self.G.T @ self.unc)
        self.G = G
        self.C_d = np.diag(self.unc_obs*self.unc_obs)#C_d
        self.C_d_inv = np.linalg.inv(self.C_d)
        nvox = G.shape[1]
        self.C_smooth = np.zeros(shape=(nvox,nvox) )
        self.rho_post = np.zeros(shape=nvox)
        self.nvox = nvox
        
    def scaling_mrc(self, coord_tel):
        self.mat_scaling = np.ones(self.nvox)
        for i in range(self.nvox): 
            xyz_b = self.dome.barycenter[i]
            d_tel = [np.linalg.norm(xyz_t - xyz_b) for xyz_t in coord_tel]
            rmin = d_tel[np.argmin(d_tel)]
            self.mat_scaling[i] = 1/rmin**1.5

    def get_model_expectation(self, rho0, err_prior, d, l):
        '''
        rho0 : initial guess (shape=(Nvox,))
        '''
        self.C_rho_post = self.G.T @ self.C_d_inv @  self.G
        #print(f"self.C_smooth  = {self.C_smooth}")
        self.C_rho_post += np.linalg.inv(self.C_smooth)
        #print(f"self.C_rho_post  = {self.C_rho_post}, shape = {self.C_rho_post.shape}")
        #print(f"self.C_rho_post[nan] = {self.C_rho_post[np.isnan(self.C_rho_post)]}, shape = {self.C_rho_post[np.isnan(self.C_rho_post)].shape}")
        self.C_rho_post_inv =  np.linalg.inv(self.C_rho_post)
        self.rho_post = rho0 + self.C_rho_post_inv @ self.G.T @ self.C_d_inv @ ( self.d_obs - self.G @ rho0) 
    
    def smoothing(self, err_prior, d, l,  damping:np.ndarray = None):
        """
        err_prior: a priori error on density [g/cm^3]
        d : distance between voxels [m]
        l : correlation length [m]
        """
        nvox = self.G.shape[1]
        #print(f"nvox = {nvox}")
        for i in range(nvox):
            for j in range(nvox):
                self.C_smooth[i,j] = err_prior**2 * np.exp(-d[i,j]/l)
                if i == j : 
                    #diag damping
                    if damping is not None: self.C_smooth[i,j]  *= damping[i] 
                
    
    def gaus_smoothing(self, err_prior, d, l, damping:np.ndarray = None):
        """
        err_prior: a priori error on density [g/cm^3]
        d : distance between voxels [m]
        l : correlation length [m]
        """
        nvox = self.G.shape[1]
        #print(f"nvox = {nvox}")
        for i in range(nvox):
            for j in range(nvox):
                self.C_smooth[i,j] = err_prior**2 * np.exp(-(d[i,j]/l)**2)
                if i == j : 
                    #diag damping
                    if damping is not None: self.C_smooth[i,j]  *= damping[i] 
 
    def get_misfit(self, rho0, err_prior):
        """
        misfit a.k.a cost function, objective function, least-squares function
        chi2 criterion to evaluate the goodness-of-fit
        """
        self.misfit = ( self.d_obs - self.G @ self.rho_post).T @ self.C_d_inv @ (self.d_obs - self.G @ self.rho_post)
        self.misfit += err_prior * (self.rho_post-rho0).T @  self.C_rho_post_inv @ (self.rho_post-rho0)
        
@dataclass
class AnaSet:
    tel: Telescope
    data_dir: Path
    tag_run: str
    tag_filter: str 
    tag_rho: str
    def __post_init__(self):
        self.dir = self.data_dir/self.tel.name/self.tag_run/"ana"/self.tag_filter
    
        
if __name__ == "__main__":
    
    
    
    ltel = ["SB", "SNJ", "BR", "OM"]
    
    
    data_dir = Path.home() / "data"
    str_flux = "corsika"
    day = "10" 
    str_date = f"{day}022023" #%d%m%y
    str_filter = f"{str_date}/filter_multiplicity"
    str_rho = f"fscale_1/{str_flux}/nearest"
    
    as_snj= AnaSet(tel=str2telescope("SNJ"), data_dir=data_dir, tag_run = "Tomo2", tag_filter = str_filter, tag_rho=str_rho)
    as_br = AnaSet(tel=str2telescope("BR"), data_dir=data_dir, tag_run = "3dat/tomo/2017_2018_2019", tag_filter = str_filter, tag_rho=str_rho)#Tomo6/2018_2019
    as_sb = AnaSet(tel=str2telescope("SB"), data_dir=data_dir, tag_run = "3dat/tomo", tag_filter = str_filter, tag_rho=str_rho)
    as_om = AnaSet(tel=str2telescope("OM"), data_dir=data_dir, tag_run = "3dat/tomo", tag_filter = str_filter, tag_rho=str_rho)
    dict_anasets = {
        "SB":  as_sb,
         #"SNJ":  as_snj, 
        #"BR":  as_br ,
        #"OM":  as_om  
        }


    conf='3p1'
    res= 64 #m
    ml_dir = Path.home()  / "MatLab" / "MuonCompute_Volumes_mnt" 
    fGmatrix_dome = ml_dir /  "matrices" / f"cubesMatrix_{res}.mat"
    fVolcanoCenter = ml_dir / "matrices" / "volcanoCenter.mat"
    ##overlap (SB & SNJ & BR & OM)
    fmask_overlap = ml_dir / "CubesMuonDependances" /  f"Resol_{res}" / "overlap" / f"mask_cubes_overlap_{res}m.mat"
    print("loading overlap region mask...")
    mask_cubes = sio.loadmat(str(fmask_overlap))['mask'].T[0]
    mask_cubes = mask_cubes != 0 #convert to bool type
    mat_dome = mat73.loadmat(str(fGmatrix_dome))['cubesMatrix'] #shape=(N_cubes, N_par_cubes=31)
 
    param_dir = Path.home() / 'muon_code_v2_0' / 'AcquisitionParams'
    for i, (name, ans) in enumerate(dict_anasets.items()):
        acq_dir = param_dir / name / "acqVars" / f"az{ans.tel.azimuth}ze{ans.tel.zenith}"
        #print(str(acq_dir), acq_dir.exists())
        acqVar = AcqVars(telescope=ans.tel, 
                        acq_dir=acq_dir,
                        mat_files=None,
                        tomo=True)
        thickness = acqVar.thickness[conf].flatten()
        
        nlos = len(thickness)
        
        sv_los = (~np.isnan(thickness)) & (acqVar.ze_tomo[conf].flatten() < 85)        
            
        rho_dir =ans.dir / "density"/ ans.tag_rho 
        
        #print(f"rho_dir = {rho_dir}")
        
        #frho = rho_dir /f"mean_density_3D_res{res}m_overlap_region_{conf}.txt" #G*d
        #func = rho_dir / f"unc_mean_density_3D_res{res}m_overlap_region_{conf}.txt" 
        rho_flat = np.loadtxt(str(rho_dir/ f"mean_density_{conf}.txt"), delimiter="\t").flatten()
        unc_flat = np.loadtxt(str(rho_dir/ f"unc_mean_density_{conf}.txt"), delimiter="\t").flatten()
        #print("unc_flat = ",  unc_flat)

        ###selection lines-of-sight        
        rho_flat[~sv_los] = np.nan
        unc_flat[~sv_los] = np.nan

        mask_los = ~np.isnan(rho_flat)
        rho_flat = rho_flat[mask_los]
        unc_flat = unc_flat[mask_los]
        #vox, mean_rho_vox, unc_rho_vox = np.loadtxt(frho).T  #G*d
        
        #mean_rho = Observable(f"mean_rho_{name}", value={conf:rho_flat}, error={conf:unc_flat})
        
        if name =="SNJ" : name="NJ"
        fGmatrix_tel = ml_dir / "CubesMuonDependances" /  f"Resol_{res}" / name / f"CubesMuonDependances_{name}_{res}m.mat"
        try : 
            Gmat_tel = sio.loadmat(str(fGmatrix_tel))[f'CubesMuonDependances_{name}']  #shape=(N_los, N_cubes)
        except : 
            Gmat_tel = mat73.loadmat(str(fGmatrix_tel))[f'CubesMuonDependances_{name}']
        Gmat_tel = Gmat_tel[:,mask_cubes] #select overlapped region voxels

        #Gmat_tel[~sv_los,:] = 0
        #print(Gmat_tel.shape)
        Gmat_tel = Gmat_tel[mask_los,:]
        print(f"max(Gmat_tel) = {np.nanmax(Gmat_tel)}")

        if i==0 :
            #mean_mean_rho = rho_flat[np.newaxis,:]
            mat_rho_flat = rho_flat
            mat_unc_flat = unc_flat
            Gmat = Gmat_tel
        else : 
            #mean_mean_rho = np.vstack((mean_mean_rho, mean_rho_vox))
            mat_rho_flat = np.concatenate(( mat_rho_flat, rho_flat ))  
            mat_unc_flat = np.concatenate(( mat_unc_flat, unc_flat))
            Gmat = np.vstack((Gmat, Gmat_tel))
        
    
        # print(mean_mean_rho.shape)
        # print(mean_rho_vox.shape)
        # print(mean_rho_flat.shape)
        # print(unc_flat.shape)
        #print(f"Gmat_tel ={Gmat_tel.shape}")
        #print(f"Gmat ={Gmat.shape}")

    #mean_mean_rho = np.nanmean(mean_mean_rho, axis=0)

    ###careful to nan values : https://stackoverflow.com/questions/71265228/np-linalg-inv-leads-to-array-full-of-np-nan
    Gmat[np.isnan(Gmat)] = 0
    Gmat[Gmat!=0] = 1
    print(Gmat.shape)
    t = np.count_nonzero(Gmat, axis=0)
    print(f"np.count_nonzero(Gmat_tel==1, axis=0) =  {t}, shape = {t.shape}")
    exit()
    
    C_d = np.diag(mat_unc_flat*mat_unc_flat)

    
    #print(f"C_d^-1= {np.linalg.inv(C_d)}")
    #exit()
   
    dome = VoxeledDome(resolution=res, matrix=mat_dome)
    dome.cubes = dome.cubes[mask_cubes==True] #select voxels of interest
    dome.barycenter = dome.barycenter[mask_cubes==True] 
    
    #print()
    dome.get_distance_matrix()

    start_time = time.time()
    print("Start inversion: ", time.strftime("%H:%M:%S", time.localtime()))#start time
    
    print(f"mat_rho_flat = {mat_rho_flat}")


    inv= Inversion(voxel_dome=dome, rho=mat_rho_flat, unc=mat_unc_flat, G=Gmat, C_d=C_d) 
    
    exit()

    
    #l_prior = np.logspace(np.log10(1e-3),np.log10(1e-1), 10) #
    #l_clength = np.logspace(np.log10(1e0),np.log10(1e2), 10) #correlation length

    #l_prior, l_clength = [], []
    

    is_data = True
    if is_data: run =  "_".join(ltel)
    else:  run = "synthetic"
    
    for sig in l_prior:
        for l in l_clength: 
    
            out_fold = Path(f"smoothing") / f"sig{sig}" / f"l{l}"

            out_dir = data_dir / "inversion" / run / str_date / out_fold
            out_dir.mkdir(parents=True, exist_ok=True)
            
            err_prior, d, l = 1e-1, dome.d, 100 #g/cm^3, m, m
            inv.smoothing(err_prior, d, l)
            print(f"smoothing --- {(time.time() - start_time):.3f}  s ---") 
            #print(inv.C_smooth)
            #print(np.linalg.inv(inv.C_smooth))
            nvox = Gmat.shape[1]
            rho_prior = np.ones(shape=nvox) * 1.8 #bulk density
            inv.get_model_expectation(rho0=rho_prior,err_prior=err_prior, d=d, l=l)
            print(f"compute model expectation --- {(time.time() - start_time):.3f}  s ---") 
            
            out_dir.mkdir(parents=True, exist_ok=True)
            timestr = time.strftime("%d%m%Y_%M%H")
            spar = f"parameters smoothing:\nerr_prior, correlation length = {err_prior} g/cm^3, {l} m"
            sout = f"{'_'.join(ltel)}\n{timestr}\n{str_rho}\n{spar}"
            fout = out_dir / "rho_post.txt"
            with open(str(out_dir/f"{timestr}.log"), 'w') as f : 
                f.write(sout)
            np.savetxt(fout, inv.rho_post, fmt="%.3e", delimiter="\t")
            print(f"rho_post = {inv.rho_post} g/cm^3")
            print(f"save {str(fout)}")
            
            


    # vmin, vmax = 0.5, 2.7#g/cm^3
    # val_cubes =  inv.rho_post
    # range_val = np.linspace(vmin, vmax, 50)
    # cmap = 'jet'
    # norm = cm.Normalize(vmin=vmin, vmax=vmax)(range_val)
    # color_scale =  plt.colormaps[cmap](norm)
    
    # arg_col =  [np.argmin(abs(range_val-v))for v in val_cubes]#, np.arange(0,n))#plt.colormaps[cmap](val_cubes)
    # colors = color_scale[arg_col]

    
    # fig = plt.figure()
    # ax = fig.add_subplot(111,projection='3d')
    # pc = Poly3DCollection(np.concatenate(dome.cubes),  
    #                             cmap=cmap,
    #                             alpha=0.3,
    #                             facecolors=np.repeat(colors,6,axis=0),
    #                             edgecolors=np.clip(colors - 0.5, 0, 1),  # brighter 
    #                             norm=Normalize(clip=True),
    #                             linewidths=0.3)
    
    
    # #print(f'vmin, vmax = {vmin}, {vmax}')
    # pc.set_clim(vmin, vmax)
    # #cb = plt.colorbar(pc, orientation='vertical', ax=ax, label="relative density $\\Delta\\overline{\\rho}$ [%]")#[g.cm$^{-3}$]")
    # cb = plt.colorbar(pc, orientation='vertical', ax=ax, label="mean density $\\overline{\\rho}$ [g.cm$^{-3}$]")
    # #ax.add_collection3d(lc_dome)
    # #ax.add_collection3d(pc_tel)
    # ax.add_collection3d(pc)
    # ax.view_init(90, 0)
    # ax.dist = 8    # define perspective (default=10)
    # # ax2 = fig.add_subplot(122, projection='3d')
    # # ax2.update_from(ax)
    # # ax2.view_init(30, azim)
    
    # fout = out_dir / f"rho_post_3D_res{res}m"
    # fig.savefig(f"{str(fout)}.png")
            #plt.show()
                
            