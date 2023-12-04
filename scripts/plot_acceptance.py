import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable
import os
from pathlib import Path
import sys
import inspect
from dataclasses import dataclass, field
import glob
import scipy.io as sio
import pickle
import warnings
warnings.filterwarnings("ignore")
import time
from scipy.integrate import quad, dblquad, nquad
from scipy.interpolate import interp2d

####my modules
from telescope import dict_tel
from tomo.acceptance import GeometricalAcceptance



if __name__ == "__main__":
    
    main_dir = Path("/Users/raphael/")
    code_dir = main_dir/"muon_code_v2_0" 
    param_dir = code_dir/"AcquisitionParams"
    tel ="SNJ"
    calib_label= "CALIB2"
    #data_dir = main_dir/"data"/tel/"ana"
    #file_acceptance_th = code_dir/"Acceptance"/"A_theo_16x16_120cm.txt"
    #file_data_acceptance = data_dir/calib_label/"acceptance"/f"acceptance_3p1_{calib_label}.txt"
    #data_acceptance = np.loadtxt(file_data_acceptance)

    # file_IntFlux = main_dir/"cosmic_flux/flux_vs_opacity/rock/2.65/gaisser/flux.txt"
    # IntFlux = np.loadtxt(file_IntFlux)
    # file_ze = param_dir/"SNJ"/"acqVars_3p.mat"
    # acqVars_3p_mat = sio.loadmat(file_ze)
    # data_ze = acqVars_3p_mat['zenithAngleMatrix']*180/np.pi
    # data_az = np.fliplr(acqVars_3p_mat['azimutAngleMatrix']*180/np.pi)
    # file_thickness =  param_dir/"SNJ"/"App_thickness_3panels.txt"
    # data_thickness= np.loadtxt(file_thickness) ###in m 
    # IntegralFluxVsOpAndZaStructure_Corsika = sio.loadmat(os.path.join(param_dir,'common','IntegralFluxVsOpAndZaStructure_Corsika.mat')) 
    # simu_ze = IntegralFluxVsOpAndZaStructure_Corsika['IntegralFluxVsOpAndZaStructure_rock_MuonsEKin_onlyModel'][0][0][0] #zenith angle
    # simu_log10opacity = IntegralFluxVsOpAndZaStructure_Corsika['IntegralFluxVsOpAndZaStructure_rock_MuonsEKin_onlyModel'][0][0][1] #opacity
    # simu_op = np.exp(np.log(10) * simu_log10opacity)  ####mwe s
    
    
    
    ###########
    geom = GeometricalAcceptance(telescope=dict_tel["SNJ"], configuration="3p1")
    acc, acc_w = np.zeros(shape=(31,31)), np.zeros(shape=(31,31))
    dx, dy = np.arange(-15, 16),np.arange(-15, 16)
    DX, DY = np.meshgrid(dx,dy)
    x = DX * np.arctan(geom.width/geom.length)*180/np.pi
    y = DY * np.arctan(geom.width/geom.length)*180/np.pi




    for i in range(0, 31):
        for j in range(0, 31):
            acc[i,j] = geom.acceptance_axis_approx(geom.dxdy[i,j])
            x1, x2, y1, y2 = geom.mat_xy[i,j,:] * geom.width *1e-1 #mm -> cm
            #print(x1, x2, y1, y2)
            acc_w [i,j] = geom.acceptance_willis( x1, x2, y1, y2 )
    print(f"acc_jourde in [{np.min(acc)}; {np.max(acc)}] cm2.sr")
    print(f"acc_willis in [{np.min(acc_w)}; {np.max(acc_w)}] cm2.sr")



    exit()
    fig, ax = plt.subplots(1, figsize= (16,9))
    ax1 = fig.add_subplot(projection='3d') 
    ax1.view_init(elev=15., azim=45)      
    im1 = ax1.plot_surface(
        x,
        y,
        acc_w,
        cmap="jet", #cm.coolwarm
        linewidth=0,
        antialiased=False,
        alpha=1,
        vmin=0, vmax=np.max(acc)
    )

    #3D plot 
    ax1.set_zlim(0, np.max(acc) )
    #ax1.get_zaxis().set_visible(False)
    cbar = plt.colorbar(im1,  shrink=0.5, orientation="vertical")
    cbar.ax.tick_params(labelsize=12)
    plt.show()
    exit()
    ############
   
    ##run_duration
    dT= 145*24*3600 #s
    ###Expected opacity for rock
    rho0 = 2.65 #g/cm^3
    #points = np.vstack((simu_ze.flatten(), simu_op.flatten())).T
    #values = IntFlux.flatten()
    start_time = time.time()
    print(start_time)
    fout = main_dir/"Desktop" /f'IntFlux_interp_func_SNJ_3p1.pkl'
    if not fout.exists():
        f_IntFlux = interp2d(simu_ze, simu_op, IntFlux, kind='cubic')
        print(f"Interpolation --- {(time.time() - start_time):.3f}  s ---")  
        print(f"Save function in {fout}")
        with open(fout, 'wb') as out:
            pickle.dump(f_IntFlux, out, pickle.HIGHEST_PROTOCOL)
    else:     
        with open(fout, 'rb') as inp:
            f_IntFlux = pickle.load(inp)
            
    
    expected_opacity = (data_thickness*1e2 * rho0) *1e-2 #g/cm2 -> hg/cm2=mwe
   
    fm = FluxModel()
    flux_exp, nmu_exp = fm.number_expected_muons(dt=dT, 
                                    acceptance=data_acceptance, 
                                    rho=rho0, 
                                    thickness=data_thickness,
                                    azimuth=data_az,
                                    zenith=data_ze,
                                    func_flux=f_IntFlux
    )

    
    
    file_Nexp= main_dir/"Desktop"/"nmu_exp.txt"
   
            
    print(f"Calcul Nexp --- {(time.time() - start_time):.3f}  s ---")          
    print(nmu_exp)
    np.savetxt(file_Nexp, nmu_exp, fmt="%.3e")
   
    var={"Rock Thickness [m]":data_thickness, "Flux [cm$^{-2}$.s$^{-1}$.sr$^{-1}$]":flux_exp, "Number of muons":nmu_exp}
    cmaps = ["jet", "jet_r", "viridis"]
    ncols= len(var)
    fig, ax =plt.subplots(nrows=1, ncols=ncols)
    for i, ((k, v), cm) in enumerate(zip(var.items(), cmaps)):
        v = np.fliplr(v)
        im=ax[i].imshow(v, cmap=cm, norm=LogNorm(vmin=np.nanmin(v[v!=0]), vmax=np.nanmax(v)))
        #im = ax[i].pcolor(data_az, data_ze, v, cmap=cm, norm=LogNorm(vmin=np.nanmin(v[v!=0]), vmax=np.nanmax(v)))
        divider = make_axes_locatable(ax[i])
        cax = divider.append_axes("right", size="5%", pad=0.05)
        cbar = fig.colorbar(im, cax=cax, orientation="vertical")
        cbar.ax.tick_params(labelsize=6)
        #cbar.set_label(label=k, size=10)
        ax[i].set_title(k)
    fig.tight_layout()
    plt.show()

    
    
    