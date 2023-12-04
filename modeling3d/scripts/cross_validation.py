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

from sklearn.model_selection import train_test_split
from sklearn import datasets
from sklearn import svm

#personal modules
from telescope import str2telescope, Telescope, dict_tel
from raypath import AcqVars
from plot_density_3d import VoxeledDome
from modeling3d.modeling3d import Inversion, KernelMatrix, DataSynth

  

if __name__=="__main__":

    conf='3p1'
    param_dir = Path.home() / 'muon_code_v2_0' / 'AcquisitionParams'
    
    res= sys.argv[1] # in [16,32,64] m
    
    tag_run=None
    if len(sys.argv) > 2:  tag_run = sys.argv[2]



    data_dir = Path.home()/"data" 
    realdata_dir = data_dir / "multi_tel"/"SB_SNJ_BR_OM"/"ana"/"04042023"/"filter_multiplicity"/"tomo"
    strdate = time.strftime("%d%m%Y_%H%M")
    label_run = strdate
    #if tag_run is not None: label_run = f"{strdate}_{tag_run}"
    run = "synthetic"
    tag_inv = "smoothing"
    out_dir = data_dir / "inversion" / run / tag_inv / "cross_validation" / f"res{res}m" / label_run 
    #out_dir = data_dir / "inversion" / run / tag_inv / "N" / "runs" / f"r{res}" / label_run 
    out_dir.mkdir(parents=True, exist_ok=True)

    
    data_dir = data_dir / ""
    
    
    
    ml_dir = Path.home()  / "MatLab" / "MuonCompute_Volumes_mnt" 
    fGmatrix_dome = ml_dir /  "matrices" / f"cubesMatrix_{res}.mat"
    fVolcanoCenter = ml_dir / "matrices" / "volcanoCenter.mat"
    
    mat_dome = mat73.loadmat(str(fGmatrix_dome))['cubesMatrix'] #shape=(N_cubes, N_par_cubes=31)
    vol_center = sio.loadmat(str(fVolcanoCenter))['volcanoCenter'][0]
    vc_x = vol_center[0]
    vc_y = vol_center[1]
    dtc = np.sqrt( (mat_dome[:,25]-vc_x)**2 + (mat_dome[:,26]-vc_y)**2 ) 
    sv_center = (dtc <= 375) 
    
    ltel_n = ["SNJ","SB","BR","OM"]#,"SB","SNJ"]#,"OM"]
    ltel = [dict_tel[n] for n in ltel_n]
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
            print(mask_cubes.shape)
            union_mask_tel[i] = mask_cubes
        union_mask_tel = np.sum(union_mask_tel, axis=0)
        mask_cubes = union_mask_tel != 0 #convert to bool type
        
    #exit()
    #print(f"mask_cubes = , shape = {mask_cubes.shape}")
    
    
    mask_cubes = mask_cubes != 0 #convert to bool type
    mask_cyl = mask_cubes != 0
    print(f"mask_cubes = , shape = {mask_cubes.shape}")
    print(f"sv_center = {sv_center}, shape={sv_center.shape}")
    mask_cubes = mask_cubes & sv_center
    print(f"mask_cubes &  sv_center = {mask_cubes}, shape={mask_cubes[mask_cubes==1].shape}")
    mask_cyl = mask_cyl & sv_cyl
    print(f"dome total nvoxels : {len(mask_cubes)}")
    print(f"dome dtc nvoxels : {len(mask_cyl[mask_cubes])}")
    print(f"bulk dtc nvoxels : {len(mask_cyl[mask_cubes][mask_cyl[mask_cubes]==False])}")
    print(f"anomaly nvoxels : {len(mask_cyl[mask_cubes][mask_cyl[mask_cubes]==True])}")

    ##############
    dome = VoxeledDome(resolution=res, matrix=mat_dome)
    nvox = len(mask_cubes[mask_cubes==True])
    mask_dome = (mask_cubes==True)
    dome.cubes = dome.cubes[mask_dome] #select voxels of interest
    dome.barycenter = dome.barycenter[mask_dome]
    dome.xyz_up = dome.xyz_up[mask_dome]
    dome.get_distance_matrix()
    coord_tel = [tel.utm for tel in ltel]
    
    fout_data_syn = out_dir / f"data_syn_all_res{res}m.txt"
    fout_unc_syn = out_dir / f"unc_syn_all_res{res}m.txt"
    fout_Gmat = out_dir / f"Gmat_all_res{res}m.txt"
    
    #if not all([fout_data_syn.exists(), fout_unc_syn.exists(), fout_Gmat.exists()]):

    print("Generate synthetic data") 
    fig =  plt.figure(figsize=(12,8))
    ax_dict = fig.subplot_mosaic(
        [["SB", "SNJ"], ["BR", "OM"]], #[lzslice],#
        sharex=False, sharey=False
    )
    fig.subplots_adjust(wspace=0.35, hspace=0.35)

    cmap = palettable.scientific.sequential.Batlow_20.mpl_colormap
        
    

###GENERATE DATA with GAUSSIAN noise
    for i, tel in enumerate(ltel):
        sconfig = list(tel.configurations.keys())
        for conf in sconfig :
            if conf=="3p2": continue
            print(f"{tel.name}_{conf}") 

            unc_sys_tel = np.loadtxt(realdata_dir / f"unc_density_sys_tot_{tel.name}_{conf}.txt", delimiter="\t")
            unc_stat_tel = np.loadtxt(realdata_dir / f"unc_density_stat_{tel.name}_{conf}.txt", delimiter="\t")
            unc_tot_tel = unc_sys_tel + unc_stat_tel
            
            acq_dir = param_dir / tel.name / "acqVars" / f"az{tel.azimuth}ze{tel.zenith}"
            acqVar = AcqVars(telescope=tel, acq_dir=acq_dir)
            thickness = acqVar.thickness[conf].flatten()
            terrain_path = Path.home() / "data" / tel.name /"terrain"
            ##positions computed with 'pos_border_crater.py' script
            crater = {  "TAR" : np.loadtxt(terrain_path/f"TAR_on_border.txt", delimiter="\t") , 
                        "SC" : np.loadtxt(terrain_path/f"CS_on_border.txt", delimiter="\t"),
                        "BLK" : np.loadtxt(terrain_path/f"BLK_on_border.txt", delimiter="\t"),
                        "G56" : np.loadtxt(terrain_path/f"G56_on_border.txt", delimiter="\t"),
                        "FNO" : np.loadtxt(terrain_path/f"FNO_on_border.txt", delimiter="\t")
                }
            
            sv_los = (~np.isnan(thickness)) & (acqVar.ze_tomo[conf].flatten() < 90)
            front, rear = tel.configurations[conf][0], tel.configurations[conf][-1]
            mat_los = tel.get_los(front, rear)
            nlos = mat_los.shape[0]*mat_los.shape[1]#number of lines of sight
            print("loading telescope voxel matrix...")
            tel_name = tel.name
            if tel.name == "SNJ" : tel_name='NJ'
            fGmatrix_tel = ml_dir / "CubesMuonDependances" /  f"Resol_{res}" / tel_name / f"CubesMuonDependances_{tel_name}_{res}m.mat"
            try : 
                Gmat_tel = sio.loadmat(str(fGmatrix_tel))[f'CubesMuonDependances_{tel_name}']  #shape=(N_los, N_cubes)
            except : 
                Gmat_tel = mat73.loadmat(str(fGmatrix_tel))[f'CubesMuonDependances_{tel_name}']        
            Gmat_tel = Gmat_tel[:,mask_cubes==True] #select overlapped region voxels
            Gmat_tel[np.isnan(Gmat_tel)] = 0
            Gmat_tel[Gmat_tel!=0] = 1
            nvox_per_los = np.count_nonzero(Gmat_tel, axis=1)
            nz = nvox_per_los != 0 
            Gmat_tel[nz]  = np.divide(Gmat_tel[nz].T, nvox_per_los[nz]).T
            nlos, nvox = Gmat_tel.shape
            
            ####DEFINE TRUE DENSITY VALUES
            rho_flat = np.ones(nvox) * 2
            rho_flat[mask_cyl[mask_cubes]] = 1
            real_unc = unc_tot_tel.flatten()
            err_perc = 0.1
            unc_flat = err_perc*rho_flat #np.ones(rho_flat.shape)*0.05 #uncertainty vector 

            #####
            print(f"nlos, nvox={Gmat_tel.shape}")
            
            
            if i==0 : 
                np.save(out_dir / f"rho_true_res{res}m.npy", rho_flat)
                np.save(out_dir / f"unc_true_res{res}m.npy", unc_flat)
            
            #print(f"rho_flat = {rho_flat.shape}")
            
            #####MASK
            theta_horizon =  83 #deg
            mask_tel = ~(np.isnan(unc_sys_tel)) & ~(np.isnan(unc_stat_tel))   & (acqVar.ze_tomo[conf] < theta_horizon)
            mask_tot = (nz)  & mask_tel.flatten()
            ####GENERATE GAUSSIAN SYNTHETIC DATA (and apply mask)
            #####            
            d_true  = Gmat_tel  @ rho_flat #nlos 
            d_syn = np.zeros(d_true.shape)
            d_unc = unc_tot_tel.flatten()
            d_unc[np.isnan(d_unc)]= 0
            d_syn[mask_tot] = np.random.normal(loc=d_true[mask_tot], scale=d_unc[mask_tot])
            
            
            ####PLOT DATA
            ax = ax_dict[tel.name]
            ax.set_title(f"Synthetic data {tel.name} ({tel.azimuth:.1f}, {tel.elevation:.1f})°")
            x, y = acqVar.az_tomo[conf], acqVar.ze_tomo[conf]
            z = d_syn.reshape(x.shape)
            z[ z==0 ] = np.nan
            im = ax.pcolor(x,y,z,  shading='auto', cmap=cmap,vmin=0.8, vmax=2.5)
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
            topo = acqVar.topography[conf]
            ax.plot(topo[0,:],topo[1,:], linewidth=3, color='black')
            marker, dz, size = MarkerStyle("*"),-0.5, 17
            for key, value in crater.items():    
                if key =="SC": mark_col = "magenta"
                elif key =="TAR": mark_col="white"
                elif key =="G56": mark_col="lightgrey"
                elif key =="BLK": mark_col="grey"
                else: mark_col="black"
                az_cr, ze_cr =  value[0], value[1]
                #ax.plot(az_cr,ze_cr+dz, marker=marker, color=mark_col, markersize=size, markeredgewidth=1., markeredgecolor="black")
                #ax.annotate(key, ((az_cr+0.5, ze_cr+dz-1)), fontsize=14)
            ####
            
            if i==0 :
                data_syn = d_syn[mask_tot]
                unc_syn = d_unc[mask_tot]
                Gmat = Gmat_tel[mask_tot,:]
            else : 
                data_syn = np.concatenate(( data_syn,d_syn[mask_tot] ))  
                unc_syn = np.concatenate(( unc_syn, d_unc[mask_tot]))
                Gmat = np.vstack((Gmat, Gmat_tel[mask_tot,:]))
                # data_syn = np.concatenate(( data_syn, d_syn.gaus ))  
                # unc_syn = np.concatenate(( unc_syn, d_syn.unc))
                # Gmat = np.vstack((Gmat, Gmat_tel))
            
            if tel_name=="SNJ" or tel_name =="NJ" : tel_name = f"{tel_name}_{conf}"
            
            fout = out_dir / f"data_syn_{tel_name}_res{res}m.txt"
            np.savetxt(fout, d_syn[nz], delimiter="\t", fmt="%.3e")
            print(f"save {fout}")
            fout = out_dir / f"unc_syn_{tel_name}_res{res}m.txt"
            np.savetxt(fout, d_unc[nz], delimiter="\t", fmt="%.3e")
            print(f"save {fout}")
            fout = out_dir / f"Gmat_{tel_name}_res{res}m.txt"
            np.savetxt(fout, Gmat_tel[nz,:], delimiter="\t", fmt="%.3e")
            print(f"save {fout}")
    
    fout = out_dir / f"fig_data_syn_res{res}m.png"
    plt.savefig(str(fout))#, bbox_inches='tight')#,pad_inches=1)
    print(f"save {str(fout)}")
    
    fout = out_dir / f"data_syn_all_res{res}m.txt"
    np.savetxt(fout, data_syn, delimiter="\t", fmt="%.3e")
    print(f"save {fout}")
    fout = out_dir / f"unc_syn_all_res{res}m.txt"
    np.savetxt(fout, unc_syn, delimiter="\t", fmt="%.3e")
    print(f"save {fout}")
    fout = out_dir / f"Gmat_all_res{res}m"
    np.savetxt(f"{fout}.txt", Gmat, delimiter="\t", fmt="%.3e")
    print(f"save {fout}.txt")
    
            
    '''
    else : 
        print(f"load {str(fout_data_syn)}, {str(fout_unc_syn)}, {str(fout_Gmat)}")
        data_syn = np.loadtxt(fout_data_syn, delimiter="\t")
        unc_syn = np.loadtxt(fout_unc_syn, delimiter="\t")
        Gmat = np.loadtxt(fout_Gmat, delimiter="\t")
    '''     
        
    # fout = out_dir / f"Gmat_all_res{res}m"
    # np.savetxt(f"{fout}.txt", Gmat, delimiter="\t", fmt="%.3e")
    # np.save(f"{fout}.npy", Gmat)
    # print(f"save {fout}")
      

    fout = out_dir / f"rho_true_res{res}m.npy"
    rho_true = np.load(fout)
    
    # Gunit = np.where(Gmat!=0, 1, 0)
    # Xtmp = np.multiply(Gunit,rho_true)
    # mat_nvox_nz = np.count_nonzero(Xtmp, axis=1) #nlos 
    # nvox_per_los = np.count_nonzero(Gmat, axis=1)
    # maskz = (Xtmp != 0)
    # print(f"Xtmp = {Xtmp.shape}")
    # print(f"mat_nvox_nz = {mat_nvox_nz.shape}")
    # print(f"maskz = {maskz.shape}")
    
    # X = []
    # for i, m in enumerate(maskz):
    #     #print(m.shape)
    #     x = Xtmp[i,:][m]
    #     X.append(x)
    '''
    print(f"len(X) = {len(X)}")
    print(f"len(X[0]) = {len(X[0])}")
    print(mat_nvox_nz[0])
    print(nvox_per_los[0])
    print(f"len(X[-1]) = {len(X[-1])}")
    print(mat_nvox_nz[-1])
    print(nvox_per_los[-1])
    '''
    
    test_size = 0.25 # relative size of testing sub data set in [0,1]
    
    
    l_rho0 = [1.8]#[1.8, 2.0]
    l_sig = np.linspace(0.05, 0.8, 16)#[0.1]#[1e-1, 2e-1, 3e-1, 4e-1]#[2e-1, 3e-1] #[1e-1, 2e-1, 3e-1, 4e-1, 5e-1, 6e-1] # 7e-1, 8e-1, 9e-1, 1]
    l_lc = np.linspace(50, 800, 16)#[100]#[1.2e2, 1.4e2, 1.6e2, 1.8e2]#[1e2, 1.1e-1, 1.2e2, 1.3e-1, 1.4e2, 1.5e2, 1.6e2, 1.7e2, 1.8e2] #1e1, 2e1, 4e1, 6e1, 8e1, 
    ndat, nvox = Gmat.shape
    

    
    mat_prior =  np.tile(l_sig, (len(l_lc), 1)).T
    mat_lc =  np.tile(l_lc, (len(l_sig),1))

    mat_rho_post = np.ones(shape=(len(l_rho0), len(l_sig), len(l_lc), nvox) )
    mat_std_dev = np.ones(shape=(len(l_rho0), len(l_sig), len(l_lc), nvox) )
    mat_misfit = np.ones(shape=(len(l_rho0), len(l_sig), len(l_lc)) )
    N = len(data_syn)

    arr_MSE = np.ones(shape=(len(l_rho0), len(l_sig), len(l_lc)))

    kfold = int(1/test_size) #kfold
    print(f"kfold = {kfold}")
    
    #d = dome.d #distance inter-voxels 
    random_state = np.random.randint(0, 100, size=kfold)

    ##inversion
    start_time = time.time()
    print("Start inversion: ", time.strftime("%H:%M:%S", time.localtime()))#start time  
    
    for i, rho0 in enumerate(l_rho0):
        for j, sig in enumerate(l_sig):
            for k, lc in enumerate(l_lc):  
                s = 0    
                #arr_rho_post_temp = 
                for kf in range(kfold):
                    G_train, G_test, dat_train, dat_test = train_test_split(Gmat, data_syn, test_size=test_size,random_state=random_state[kf]) # #test_size=0.25,
                    _, _, unc_train, unc_test = train_test_split(Gmat, unc_syn, test_size=test_size,random_state=random_state[kf])
                    #print("np.all(G_train==G_train_): ",np.all(G_train==G_train_))
                    inv = Inversion(voxel_dome=dome, d_obs=dat_train, unc_obs=unc_train, G=G_train)
                    # print(f"X_train = {G_train[:10]}, {G_train.shape}")
                    # print(f"X_test = {G_test[:10]}, {G_test.shape}")
                    # print(f"y_train = {dat_train[:10]}, {len(dat_train)}")
                    # print(f"y_test = {dat_test[:10]}, {len(dat_test)}")
                    # print(f"y_test/y_tot = {len(G_test)/(len(G_train)+len(G_test))}")
                    inv.smoothing(err_prior=sig, d=dome.d, l=lc, damping=None)
                    #inv.gaus_smoothing(err_prior=sig, d=dome.d, l=lc,damping=None)
                    print(f"smoothing: (rho0, err_prior, correlation length) = ({rho0:.1f} g/cm^3, {sig:.3e} g/cm^3, {lc:.3e} m) --- {(time.time() - start_time):.3f}  s ---") 
                    ndat_train, nvox_train = G_train.shape
                    ndat_test, nvox_test = G_test.shape
                    vec_rho0 = rho0*np.ones(nvox_train)
                    print(f"vec_rho0 = {vec_rho0}")
                    vec_sig  = sig*np.ones(nvox_train)
                    print(f"vec_sig = {vec_sig}")
                    inv.get_model_expectation(rho0=vec_rho0,err_prior=sig, d=dome.d, l=lc)
                    print(f"inv.rho_post = {inv.rho_post}")
                    dat_test_post = G_test @ inv.rho_post
                    print(f"dat_test_post = {dat_test_post}")
                    s += np.linalg.norm(dat_test - dat_test_post)**2
                    #mat_rho_post[i,j,k] = inv.rho_post
                
                MSE = 1/(kfold*ndat_test)  * s
                arr_MSE[i,j,k] = MSE
                print(f"MSE({rho0:.1f} g/cm^3, {sig:.3e} g/cm^3, {lc:.3e} m) = {MSE:.3f} --- {(time.time() - start_time):.3f}  s ---") 

    print(f"MSE = {arr_MSE}")
    
    
    #np.save(out_dir /  f"rho_post_res{res}m.npy", mat_rho_post, allow_pickle=True, fix_imports=True)
    #np.save(out_dir /  f"std_dev_res{res}m.npy", mat_std_dev, allow_pickle=True, fix_imports=True)
    np.savetxt(out_dir/ "rho0.txt",l_rho0, delimiter="\t", fmt="%.3e")
    np.savetxt(out_dir/ "sigma_prior.txt",mat_prior, delimiter="\t", fmt="%.3e")
    np.savetxt(out_dir/ "correlation_length.txt",mat_lc, delimiter="\t", fmt="%.3e")
    np.save(out_dir/ "MSE.npy",arr_MSE)
    #print(f"save {out_dir /  f'rho_post_res{res}m.npy'}")
    #print(f"save {out_dir /  f'std_dev_res{res}m.npy'}")
    print(f"save {out_dir/ 'sigma_prior.txt'}")
    print(f"save {out_dir/ 'correlation_length.txt'}")
    print(f"save {out_dir/ 'MSE.npy'}")
        