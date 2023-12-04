
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines  #use for legend settings
import matplotlib.patches as mpatch  #use for legend settings
import os
import time 
import argparse
from pathlib import Path

from forwardsolver.stoppingpower import StoppingPower
from material import Rock, Water


if __name__ == "__main__":
    home_dir = Path(os.environ["HOME"])
    work_dir = Path(__file__).parents[1]
    
    t_start = time.time()
    print("Start: ", time.strftime("%H:%M:%S", time.localtime()))#start time

    parser=argparse.ArgumentParser(
    description='''Plot muon stopping power (-dE/dx) as a function of kinetic energy for standard rock and water''', epilog="""All is well that ends well.""")
    parser.add_argument('--out_dir', '-o', default=str(work_dir / "out"), help='Path to processing output', type=str)
    args=parser.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    T = np.logspace(np.log10(1e3),np.log10(1e9),100) ####MeV
    mu = 105.65839 #MeV 
    E_MeV = T+mu 
    E_GeV = E_MeV*1e-3
    
    fig, ax = plt.subplots()
    
    handles=[]
    
    for medium in [Rock, Water]:
        print(medium.name)
        l_rho = [medium.rho]
        l_col = ["blue"]
        if medium.name=="rock": 
            l_rho = [1.0, 1.8, 2.65]
            l_col = ["lightgreen", "yellowgreen", "darkgreen"]
        f_dedx_table = work_dir/"files"/f"dEdx_{medium.name}_pdg.csv"
        ene_pdg, dEdx_ion_pdg = np.loadtxt(f_dedx_table, skiprows=2, usecols=(0,2), unpack=True)
        I = medium.I
        A, Z, Z_A = medium.A, medium.Z, medium.Z_A 
        sp = StoppingPower(I=I)
        for rho, col in zip(l_rho, l_col) :      
            out_dir_tmp = out_dir / medium.name / str(rho)
            ftotout = out_dir_tmp / "dEdx_tot.csv"
            faout = out_dir_tmp / "dEdx_ion.csv"
            fbout = out_dir_tmp / "b_rad.csv"
            cut_rad = (ene_pdg  > 20*mu)
            
            if ftotout.exists():
                _, dEdx_ion, dEdx_rad = np.loadtxt(ftotout, unpack=True, skiprows=1) 
                dEdx_tot = dEdx_rad+dEdx_ion
                
            else :
                dEdx_ion = sp.bethe_bloch(ene_pdg, Z_A=Z_A, rho=rho, brem_corr=True) #in MeV/(g/cm^2)
                arr_ion = np.vstack((ene_pdg, dEdx_ion)).T
                np.savetxt(faout, arr_ion, fmt="%.5e", delimiter="      ", header="T(MeV)      dEdx")
                print(f"save {faout}")
                dEdx_tot = np.copy(dEdx_ion)
                gamma = ene_pdg/mu 
                beta = np.sqrt(1-1/gamma**2)
                ene_pdg_GeV = ene_pdg*1e-3
                sub_E_GeV =  ene_pdg_GeV[cut_rad][::5]
                b_pair = [sp.b_pair(e, A,Z) for e in sub_E_GeV] #in cm^2/g
                b_pair_interp = np.interp(ene_pdg_GeV[cut_rad], sub_E_GeV, b_pair)  #in cm^2/g
                b_brems = [sp.b_brems(e, A,Z) for e in ene_pdg_GeV[cut_rad]] #in cm^2/g
                b_phnuc = [sp.b_nuc(e, A,Z) for e in ene_pdg_GeV[cut_rad]] #in cm^2/g
                b_tot =   np.sum(np.vstack((b_pair_interp, b_brems, b_phnuc)).T,axis=1)
                dEdx_rad = np.zeros(len(dEdx_tot))
                arr_rad = np.vstack((ene_pdg[cut_rad], b_pair_interp, b_brems, b_phnuc, b_tot)).T
                np.savetxt(fbout, arr_rad,  fmt="%.5e", delimiter="      ", header="E(MeV)      b_pair      b_brem     b_nucl     b_total(cm^2/g)")
                print(f"save {fbout}")
                dEdx_rad[cut_rad] = b_tot*ene_pdg[cut_rad]
                dEdx_tot[cut_rad] += b_tot*ene_pdg[cut_rad]
                arr_tot = np.vstack((ene_pdg, dEdx_ion, dEdx_rad)).T
                np.savetxt(ftotout, arr_tot, fmt="%.5e", delimiter="      ", header="E(MeV)      dEdx_ion(MeV.cm^2/g)      dEdx_rad(MeV.cm^2/g)")
                print(f"save {ftotout}")
            
            x = ene_pdg*1e-3 #GeV    #gamma*beta ###gamma*beta = p/M
            #ax.plot(x, dEdx_ion_wo_delta, color='green', linestyle="dotted", alpha=0.5, label="ionization wo delta")
            #ax.plot(x, dEdx_bethe, color='magenta', alpha=0.5, label="Bethe-Bloch")
            #ax.plot(x[cut_rad], b_pair_interp*E_MeV[cut_rad], color="red", label="pair", linestyle="dashed")
            #ax.plot(x[cut_rad], b_brems*E_MeV[cut_rad], color="blue", label="Bremsstrahlung", linestyle="dashed")
            ax.plot(x, dEdx_ion, color=col, linestyle="dashdot", label="ionization")
            ax.plot(x, dEdx_rad, color=col, label="radiative", linestyle="dashed")
            ax.plot(x, dEdx_tot, color=col, label="total", linestyle="-")
            hdl = mpatch.Patch(color=col, label=f"{medium.name} ({rho} "+"g.cm$^{-3}$)")
            handles.append(hdl)
        #idx = np.argwhere(np.diff(np.sign(dEdx_ion[cut_rad] - b_tot*ene_pdg[cut_rad]))).flatten()
        #print(f"idx={idx}")
        #print(f"Ecut={x[cut_rad][idx]} GeV")
        #ax.axvline(x[cut_rad][idx], color='grey', linestyle="dashed")
    #plt.text(x[cut_rad][idx],0,'E_{c}',rotation=0)
    
    dedx_ion_min = np.nanmin(dEdx_ion)
    dedx_ion_mean = np.nanmean(dEdx_ion)
    emin = ene_pdg[np.argmin(dEdx_ion)]*1e-3
    print(f"min(dEdx_ion)_{medium.name} ({emin:.3f} GeV)  = {dedx_ion_min} MeV/(g/cm2)")
    print(f"mean(dEdx_ion)_{medium.name} = {dedx_ion_mean:.3f} MeV/(g/cm2)")
    
    ax.tick_params(which="both", bottom=True, top=True, left=True, right=True)
    ax.tick_params(which="both", labelbottom=True, labeltop=False, labelleft=True, labelright=False)

    ax.grid(True, which='both',linestyle='dotted', linewidth="0.5", color='grey')
    handle_ion = mlines.Line2D([], [], color='black',  linestyle='dashdot',
                            markersize=10, label="ionization")
    handle_rad = mlines.Line2D([], [], color='black',  linestyle='dashed',
                            markersize=10, label="radiative")
    handle_tot = mlines.Line2D([], [], color='black',  linestyle='-',
                            markersize=10, label="total")
    handles.extend([handle_ion, handle_rad, handle_tot])
    
    ax.legend(handles=handles, loc="best")
    ax.axis([np.min(x), 1e5, 1, 1e2])
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Kinetic Energy [GeV]")
    ax.set_ylabel("$\\langle -dE/dx \\rangle$ [MeV.cm$^{2}$.g$^{-1}$]")
    


    fout = f"dedx_full"
    plt.savefig(f"{out_dir}/{fout}.png", dpi=300)
    print(f"save {out_dir}/{fout}.png")
    print(f'End: dt = {time.time()-t_start:.2f} s')
