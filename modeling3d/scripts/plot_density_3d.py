#!/usr/bin/env python3
# -*- coding: utf-8 -*-



from dataclasses import dataclass, field
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as cm
from matplotlib.colors import LogNorm
from mpl_toolkits.mplot3d.art3d import PolyCollection, Poly3DCollection, Line3DCollection
from matplotlib.ticker import MultipleLocator,EngFormatter, ScalarFormatter, LogLocator
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.interpolate import griddata
import scipy.io as sio
from pathlib import Path
from datetime import datetime, date, timezone
import pandas as pd
import mat73 #read v7.3 mat files
#personal modules
from telescope import str2telescope, Telescope
from raypath import AcqVars

params = {'legend.fontsize': 'xx-large',
         'axes.labelsize': 'xx-large',
         'axes.titlesize':'xx-large',
         'xtick.labelsize':'xx-large',
         'ytick.labelsize':'xx-large',
         'axes.labelpad':10,
         'mathtext.fontset': 'stix',
         'font.family': 'STIXGeneral'
         
         }

plt.rcParams.update(params)

@dataclass
class VoxeledDome:
    resolution : float
    matrix : np.ndarray
    def __post_init__(self):
        nvoxels = len(self.matrix)
        object.__setattr__(self, 'nvoxels',  nvoxels )
        #### 
        x, y, z = self.matrix[:,1:5], self.matrix[:,9:13], self.matrix[:,17:25]
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
        self.xyz_down = np.array([x.T, y.T, self.matrix[:,17:21].T]).T #shape=(nvoxels,4,3)
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


params = {'legend.fontsize': 'x-large',
            'axes.labelsize': 'x-large',
            'axes.titlesize':'x-large',
            'xtick.labelsize':'x-large',
            'ytick.labelsize':'x-large',
            'axes.labelpad':15}
plt.rcParams.update(params)  
    
if __name__=="__main__":
    name = "SNJ" #sys.argv[1]
    tel = str2telescope(name)
    conf='3p1'
    param_dir = Path.home() / 'muon_code_v2_0' / 'AcquisitionParams'
    acq_dir = param_dir / name / "acqVars" / f"az{tel.azimuth}ze{tel.zenith}"
    acqVar = AcqVars(telescope=tel, 
                        acq_dir=acq_dir,
                        mat_files=None,
                        tomo=True)
    thickness = acqVar.thickness[conf].flatten()
    
    sv_los = (~np.isnan(thickness)) & (acqVar.ze_tomo[conf].flatten() < 85)
    #sv_los = (acqVar.ze_tomo[conf].flatten() <= 90)


    if tel.name == "SNJ" : tel.name='NJ'
    res = 64 #m
    print(f"res = {res} m")
    ml_dir = Path.home()  / "MatLab" / "MuonCompute_Volumes" 
    fGmatrix_dome = ml_dir /  "matrices" / f"cubesMatrix_{res}.mat"
    fVolcanoCenter = ml_dir / "matrices" / "volcanoCenter.mat"
    fGmatrix_tel = ml_dir / "CubesMuonDependances" /  f"Resol_{res}" / tel.name / f"CubesMuonDependances_{tel.name}_{res}m.mat"
    fmask_tel = ml_dir / "CubesMuonDependances" /  f"Resol_{res}" / tel.name / f"mask_cubes_{tel.name}_{res}m.mat"
    ##overlap (SB & SNJ & BR & OM)
    fmask_overlap = ml_dir / "CubesMuonDependances" /  f"Resol_{res}" / "overlap" / f"mask_cubes_overlap_{res}m.mat"

    
    mat_dome = mat73.loadmat(str(fGmatrix_dome))['cubesMatrix'] #shape=(N_cubes, N_par_cubes=31)
    vol_center = sio.loadmat(str(fVolcanoCenter))['volcanoCenter'][0]
    vc_x, vc_y = vol_center
    dtc = np.sqrt( (mat_dome[:,25]-vc_x)**2 + (mat_dome[:,26]-vc_y)**2 ) 
    sv_center = (dtc <= 375) 

    sv_cyl  = (dtc <= 100)
    print("loading telescope voxel matrix...")
    try : 
        mat_tel = sio.loadmat(str(fGmatrix_tel))[f'CubesMuonDependances_{tel.name}']  #shape=(N_los, N_cubes)
    except : 
        mat_tel = mat73.loadmat(str(fGmatrix_tel))[f'CubesMuonDependances_{tel.name}']
    
    is_tel = True 
    if is_tel :
        print("loading telescope mask...")
        mask_cubes = sio.loadmat(str(fmask_tel))['mask'].T[0]
    else : 
        print("loading overlap region mask...")
        mask_cubes = sio.loadmat(str(fmask_overlap))['mask'].T[0]

    
    mask_cubes = mask_cubes != 0 #convert to bool type
    
    print(f"mask_cubes = , shape = {mask_cubes.shape}")
    print(f"sv_center = {sv_center}, shape={sv_center.shape}")
    mask_cubes = mask_cubes & sv_center
    print(f"mask_cubes &  sv_center = {mask_cubes}, shape={mask_cubes[mask_cubes==1].shape}")
    

    
    dome = VoxeledDome(resolution=res, matrix=mat_dome)
    xrange = [np.min(dome.x), np.max(dome.x)]
    yrange = [np.min(dome.y), np.max(dome.y)]
    zrange = [np.min(dome.z), np.max(dome.z)]


    str_flux = "corsika_soufriere"
    str_date = "04042023" #%d%m%y
    str_filter = f"{str_date}/filter_multiplicity"
    if tel.name == "NJ" : 
        tel.name="SNJ"
        run = "Tomo2"
        tag_ana = str_filter #"filter_multiplicity/filter_tof"
        tag_rho = f"fscale_1/{str_flux}/nearest"
    
    elif tel.name == "BR": 
        run = "Tomo6/2018_2019"
        tag_ana = str_filter #"filter_multiplicity/filter_tof"
        tag_rho = f"fscale_1/{str_flux}/nearest"#"fscale_1/corsika/nearest"
    
    elif tel.name == "OM": 
        run = "3dat/tomo"
        tag_ana = str_filter #"filter_multiplicity/filter_tof"
        tag_rho = f"fscale_1/{str_flux}/nearest"#"fscale_1/corsika/nearest"    
    
    elif tel.name == "SB": 
        run = "3dat/tomo"
        tag_ana = str_filter
        tag_rho = f"fscale_1/{str_flux}/nearest"
    
    else: raise ValueError(f"Missing density matrix estimated for {tel.name}.")
   
    ana_dir = Path.home() /  "data" / tel.name /run /  "ana" / tag_ana
    rho_dir =  ana_dir / "density" / tag_rho

    rho_arr = np.loadtxt(str(rho_dir/ f"mean_density_{conf}.txt"), delimiter="\t")
    unc_arr = np.loadtxt(str(rho_dir/ f"unc_mean_density_sys_{conf}.txt"), delimiter="\t")
    rho_flat = rho_arr.flatten()
    unc_flat = unc_arr.flatten()
    rho0 = np.nanmean(rho_flat)
    drho_arr = (rho_arr-rho0) / rho0
    drho_flat = drho_arr.flatten()
    print(f"data: rho_min, rho_max = {np.nanmin(rho_flat):.2f},  {np.nanmax(rho_flat):.2f} g/cm^3")
    print(f"data: drho_min, drho_max = {np.nanmin(drho_flat)*100:.2f},  {np.nanmax(drho_flat)*100:.2f} %")
    

    mat_rho = np.ones(shape=mat_tel.shape)*np.nan
    print(f"rho0 = {rho0:.3f} g/cm^3")
    
    mat_tel = np.where((mat_tel > 0), 1, np.nan)
   
    
    nlos = mat_tel.shape[0]
    ilos_sel = np.arange(0,nlos)[sv_los] #range(336, 446)
    #print(f"index traversing line of sights = {ilos_sel}, shape= {len(ilos_sel)}")
    
    ix0 =  100
    #test
    if tel.name == "SNJ": ilos_sel = np.arange(436,446)# #range(ilos_sel[100], ilos_sel[100]+10) #SNJ : i=550 ~subwest  i=530 ~subeast
    else: ilos_sel = ilos_sel[ix0:ix0+10] 

   
    mat_rho = np.multiply(mat_tel.T, rho_flat).T
    mat_unc = np.multiply(mat_tel.T, unc_flat).T
    ###select a group of consecutive lines
    # mat_rho[0:np.min(ilos_sel), :]=np.nan#float("NaN")
    # mat_rho[(np.max(ilos_sel)+1):-1, :]=np.nan#float("NaN")
    mat_rho[~sv_los,:]= np.nan
    mat_rho[:, mask_cubes==False]=np.nan#float("NaN")
    mat_unc[~sv_los,:]= np.nan
    mat_unc[:, mask_cubes==False]=np.nan#float("NaN")

    #print(f"mat_rho = ; shape = {mat_rho.shape};  min, max = {np.nanmin(mat_rho):.3f}, {np.nanmax(mat_rho):.3f}; mean = {np.nanmean(mat_rho):.3f}; std = {np.nanstd(mat_rho):.3f}")
    # axis=0 -> mean on line of sights (out: shape=(Nvox,)); 
    # axis=1 -> mean on voxels (out: shape=(Nlos,))
    mat_rho_mean = np.ones(shape=mat_rho.shape[1])*np.nan #shape:(Nvox,)
    mat_unc_mean = np.ones(shape=mat_rho.shape[1])*np.nan #shape:(Nvox,)

    mask_mean = np.any(~np.isnan(mat_rho), axis=0)

    mat_rho_mean[mask_mean] = np.nanmean(mat_rho[:,mask_mean], axis=0)
    mat_unc_mean[mask_mean] = np.nanmean(mat_unc[:,mask_mean], axis=0)
    #print(f"mat_rho_mean = ; shape = {mat_rho_mean.shape};  min, max = {np.nanmin(mat_rho_mean):.3f}, {np.nanmax(mat_rho_mean):.3f}; mean = {np.nanmean(mat_rho_mean):.3f}; std = {np.nanstd(mat_rho_mean):.3f}")
    zslice = 1.3e3

    #exit()
   
    ndome_cubes = dome.cubes.shape[0]

    lc_dome = Line3DCollection(np.concatenate(dome.cubes), colors='red', edgecolor="k", alpha=0.25, linewidths=0.4, linestyles='dotted') 
    ml_dir = Path.home()  / "MatLab" / "MuonCompute_Volumes" 
    fVolcanoCenter = ml_dir / "matrices" / "volcanoCenter.mat"
    vol_center = sio.loadmat(str(fVolcanoCenter))['volcanoCenter'][0]
    dx = 600
    xrange = [vol_center[0]-dx, vol_center[0]+dx]
    yrange = [vol_center[1]-dx, vol_center[1]+dx]
    
    #xrange = [642128+dx, 643792-dx]
    #yrange = [1773448+dx, 1775112-dx]
    zrange = [1.0494e+03, 1.4658e+03 + 50]
    #zrange = [1.3e3-res/2, 1.3e3+res/2]
    fig = plt.figure(figsize=(12,8))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlim(xrange)
    ax.set_ylim(yrange)
    ax.set_zlim(zrange)
    ax.set_aspect('auto') #'equal'
    ax.grid()
    
    front, rear = tel.configurations[conf][0], tel.configurations[conf][-1]
    tel.plot_los(ax=ax, front_panel=front, rear_panel=rear, mask=~sv_los, rmax=800,  linewidth=0.3 )#
    

    #rho_min, rho_max, n = -1., 1., 50 #0.8, 2.0, 20
    #rho_min, rho_max, n = .5, 2.5, 50
    import palettable
    cmap_rho = palettable.scientific.sequential.Batlow_20.mpl_colormap #'jet' #palettable.scientific.diverging.Roma_20
    rho_min, rho_max, n = .8, 2.7, 100
    range_val = np.linspace(rho_min, rho_max, n)
    norm_r = cm.Normalize(vmin=rho_min, vmax=rho_max)
    color_scale_rho =  cmap_rho(norm_r(range_val))
    vmin_r, vmax_r = rho_min, rho_max
    
    #cmap = 'jet'
    #range_val = np.linspace(rho_min, rho_max, n)

    ilos_cubes= dome.cubes[mask_mean]
    #ilos_anom = dome.cubes[mask_mean & mask_cyl]

    ####plot selected lines of sight
    # dir = np.flipud(tel.dir)
    # for i in ilos_sel : 
    #     ax.plot(dir[i,:, 0], dir[i,:, 1], dir[i,:, 2], 
    #                         c="black", linewidth=1)
  
    #val_cubes = mat_rho_mean[mask_mean]
    #######SYNTHETIC TEST
    #val_cubes = 2*np.ones(shape = mat_rho_mean[(mask_mean==True) & (mask_slice==True)].shape)
    val_cubes = 2*np.ones(shape = mat_rho_mean[(mask_mean==True)].shape)
    print(val_cubes.shape)
    #print(f'(mask_mean==True) & (mask_slice==True) = {mat_rho_mean[(mask_mean==True) & (mask_slice==True)].shape}')
    


    mask_cyl = mask_cubes != 0
    mask_cyl = mask_cyl & sv_cyl
    print(f"mask_mean = {mask_mean.shape},\nmask_cubes = {mask_cubes.shape},\nmask_mean = {mask_mean.shape}" )
    print(f"mask_cyl = {mask_cyl.shape}")
    #mask_cyl = mask_cubes[(mask_cyl==True) & (mask_mean==True) & (mask_slice==True)]
    mask_cyl = mask_cyl[mask_mean]
    val_cubes[mask_cyl] = 1.0

    
    ####Apply slice mask
    '''
    mask_slice =  (zslice - res/2 < dome.barycenter[:,-1]) & (dome.barycenter[:,-1] <  zslice + res/2)
    mask_slice = mask_slice[mask_mean]
    val_cubes = val_cubes[mask_slice]
    print(f"ilos_cubes = {ilos_cubes.shape}")
    ilos_cubes = ilos_cubes[mask_slice]
    '''
    ####
    
    ######
    #####COLOR SCALE -> COLOR VOXELS
   # norm = cm.Normalize(vmin=rho_min, vmax=rho_max)(range_val)
   # color_scale =  plt.colormaps[cmap_rho](norm)
    arg_col =  [np.argmin(abs(range_val-v))for v in val_cubes]#, np.arange(0,n))#plt.colormaps[cmap](val_cubes)
    color_vox = color_scale_rho[arg_col]
    #print(f"ilos_cubes = {ilos_cubes.shape}")
   # ilos_cubes = ilos_cubes[mask_slice]
   # print(f"ilos_cubes[mask_slice] = {ilos_cubes[mask_slice].shape}")
   # color_vox = color_vox[mask_slice]
    
    ########
    # pc_tel = Poly3DCollection(np.concatenate(tel_cubes),  
    #                             cmap=cmap,
    #                             alpha=0.3,
    #                             facecolors=np.repeat(color_vox,6,axis=0),
    #                             edgecolors=np.clip(color_vox - 0.5, 0, 1),  # brighter 
    #                             norm=Normalize(clip=True),
    #                             linewidths=0.3)
    
    pc_sel = Poly3DCollection(np.concatenate(ilos_cubes),  
                                cmap=cmap_rho,
                                alpha=0.3,
                                facecolors=np.repeat(color_vox,6,axis=0),
                                edgecolors=np.clip(color_vox - 0.5, 0, 1),  # brighter 
                                norm=Normalize(clip=True),
                                linewidths=0.3)
    
    vmin, vmax = rho_min, rho_max#np.nanmin(val_cubes), np.nanmax(val_cubes)
    #print(f'vmin, vmax = {vmin}, {vmax}')
    pc_sel.set_clim(vmin, vmax)
    #cb = plt.colorbar(pc_sel, orientation='vertical', ax=ax, label="relative density $\\Delta\\overline{\\rho}$ [%]")#[g.cm$^{-3}$]")
    cax = fig.add_axes([ax.get_position().x1+0.12,ax.get_position().y0,0.03, ax.get_position().y1])
    #cb = plt.colorbar(pc_sel, orientation='vertical', ax=ax, label="mean density $\\overline{\\rho}$ [g.cm$^{-3}$]",cax=cax, alpha=1.)
    cb = plt.colorbar(ScalarMappable(norm=norm_r, cmap=cmap_rho), orientation='vertical', ax=ax, label="density $\\rho_{true}$ [g.cm$^{-3}$]",cax=cax, alpha=1.)
    #ax.add_collection3d(lc_dome)
    #ax.add_collection3d(pc_tel)
    cb.ax.yaxis.set_major_locator(MultipleLocator(2e-1))
    ax.add_collection3d(pc_sel)
    #ax.view_init(90, -90)
    ax.view_init(30, -60)
    ax.dist = 8    # define perspective (default=10)
    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    #ax.set_zlabel("Z [m]")
    # ax2 = fig.add_subplot(122, projection='3d')
    # ax2.update_from(ax)
    # ax2.view_init(30, azim)
    fig.subplots_adjust(left=0.05, bottom=0.1, right=0.87, top=0.95, wspace=0.15, hspace=0.15)

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
    fout = rho_dir / "grid_dem_tmp.txt"
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

    ltel_n = ["SB", "SNJ", "BR", "OM"]
    #ltel_n = ["SNJ",]#, "OM"]
    str_tel =  "_".join(ltel_n)
    ltel_coord = np.array([ str2telescope(tel).utm for tel in ltel_n])
    ltel_color = np.array([ str2telescope(tel).color for tel in ltel_n])
    ax.scatter(ltel_coord[:,0], ltel_coord[:,1], ltel_coord[:,-1], c=ltel_color, s=30,marker='s',)



    if is_tel : 
        fout = rho_dir / f"mean_density_3D_res{res}m_{conf}"
    else : 
        fout = rho_dir / f"mean_density_3D_res{res}m_overlap_region_{conf}"
    fig.savefig(f"{str(fout)}.png", transparent=True)
    #plt.show()
        
    ix_cubes= np.arange(0,len(mat_rho_mean))
    mat_rho_out = np.vstack((ix_cubes, mat_rho_mean, mat_unc_mean)).T
    np.savetxt(f"{str(fout)}.txt", mat_rho_out, delimiter="\t", fmt=["%.0f", "%.3f", "%.3f"], header="ix_voxel\trho_mean[g/cm^3]\tunc[g/cm^3]")
    print(f"save {str(fout)}.png")
    print(f"save {str(fout)}.txt")


    

