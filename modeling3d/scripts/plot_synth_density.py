#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from dataclasses import dataclass, field
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.markers import MarkerStyle
from matplotlib.ticker import MultipleLocator,EngFormatter
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.offsetbox import AnchoredText
import sys
import os
from pathlib import Path
import inspect
from datetime import datetime, date, timezone
import time
import json
import warnings
import pandas as pd
warnings.filterwarnings("ignore")
#personal modules
filename = inspect.getframeinfo(inspect.currentframe()).filename
script_path = os.path.dirname(os.path.abspath(filename))
wd_path = os.path.abspath(os.path.join(script_path, os.pardir))#working directory path
sys.path.append(script_path)
from telescope import str2telescope, Telescope
#from processing import InputType
from raypath import AcqVars
#from acceptance import Acceptance
#from tomo import ImageFeature, Topography, Tomo 
#from tools.tools import pretty, fill_empty_pixels
#from filter import FilterCharge
import palettable

params = {'legend.fontsize': 'x-large',
            'axes.labelsize': 'x-large',
            'axes.titlesize':'x-large',
            'xtick.labelsize':'x-large',
            'ytick.labelsize':'x-large',
            'axes.labelpad':10}
plt.rcParams.update(params)  


if __name__ == "__main__":
    fig = plt.figure(figsize=(16,12))
        
    ax_dict = fig.subplot_mosaic(
        [[ "SB", "SNJ"], ["BR", "OM"]],
        sharex=False, sharey=True
    ) 

    
    
    xlabel = "azimuth $\\varphi$ [deg]"
    ylabel = "zenith $\\theta$ [deg]"
    
    
    
    
    data_dir = Path.home()/"data"
    param_dir = Path.home() / "muon_code_v2_0" / "AcquisitionParams"
    run = "synthetic" #"real"
    res=sys.argv[1]
    tag_inv = "smoothing"
    datestr = sys.argv[2]
    ltel_name = ["SB", "SNJ", "BR", "OM"]
    str_tel = "SB_SNJ_BR_OM"
    out_dir = data_dir / "inversion" / run / tag_inv / datestr #"23022023_1410"

    timestr = time.strftime("%d%m%Y-%H%M%S")

    
    fout = out_dir / f"synth_data_res{str(res)}m.png"
    
    vmin, vmax = 0.8, 2.7 #g/cm^3
    zlabel = "mean density $\\overline{\\rho}$ [g.cm$^{-3}$]"
    
    batlow = palettable.scientific.sequential.Batlow_20.mpl_colormap
    
    for i, (tel_name, ipan) in enumerate(zip(ltel_name, ["a", "c", "b", "d"])):
        tel = str2telescope(tel_name)
        acq_dir = Path(param_dir) / tel_name / "acqVars" / f"az{tel.azimuth}ze{tel.zenith}"
        
        if tel.name=="SNJ": 
            conf="3p1"
            tel_name = f"NJ_{conf}"
        else :conf="3p1"
        acqVar = AcqVars(telescope=tel, 
                            acq_dir=acq_dir,
                            mat_files=None,
                            tomo=True)
    
        terrain_path = Path.home() / "data" / tel.name /"terrain"
        ##positions computed with 'pos_border_crater.py' script
        crater = { "TAR": np.loadtxt(terrain_path/f"TAR_on_border.txt", delimiter="\t") , 
                "CS": np.loadtxt(terrain_path/f"CS_on_border.txt", delimiter="\t")
            }
        crater["BLK"] = np.loadtxt(terrain_path/f"BLK_on_border.txt", delimiter="\t")
        #crater["G56"] = np.loadtxt(terrain_path/f"G56_on_border.txt", delimiter="\t")
        crater["FNO"] = np.loadtxt(terrain_path/f"FNO_on_border.txt", delimiter="\t")

        ax = ax_dict[tel.name]
        ax.set_title(f"{tel.name} ({tel.site})")#({tel.azimuth:.1f},{tel.elevation:.1f})°")
        ax.set_title(f"{tel.name} ({tel.site})")

        zval = np.loadtxt(str(out_dir/f"data_syn_{tel_name}_res{str(res)}m.txt"))

        X,Y = acqVar.az_tomo[conf], acqVar.ze_tomo[conf]
        Z = np.zeros(shape=(len(X)**2))
        print(len(X))
        
        mask = np.loadtxt(str(out_dir/f"mask_{tel_name}_res{str(res)}m.txt"))
        mask = (mask == 1)
        print(mask.shape, mask[mask==True].shape, zval.shape)
        Z[mask] = zval
        Z[Z==0] = np.nan
        Z= Z.reshape(X.shape)
        ax.tick_params(axis='both', which='major')#, labelsize=10)
        ax.set_ylim([np.nanmin(Y), 90])#deg
        ax.invert_yaxis()
        ax.set_aspect('auto') #'equal'
        ax.set_xlabel("$\\varphi$ [deg]")
        ax.set_ylabel("$\\theta$ [deg]")
        
        kwargs = dict(cmap=batlow, vmin=vmin, vmax=vmax)
        threshold=4. #g/cm3
        aberrant = (zval >= threshold) 
        zval[aberrant] = np.nan
        
        print(X.shape,Y.shape,Z.shape)
        
        im = ax.pcolor(X,Y,Z,  shading='auto', **kwargs)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        cbar = fig.colorbar(im, cax=cax, orientation="vertical")
        #cbar.ax.tick_params(labelsize=14)
        cbar.set_label(label=zlabel, size=16)
        if zlabel.startswith("mean density"): 
            locator = MultipleLocator(2e-1)
            cbar.ax.yaxis.set_major_locator(locator)
       
       
        rho_mean, rho_std = np.nanmean(zval), np.nanstd(zval)
        s = "$\\langle\\overline{\\rho}\\rangle$"+f" = {rho_mean:.1f} $\\pm$ {rho_std:.1f}"+" g.cm$^{-3}$"
        anchored_text = AnchoredText(s, loc="upper left", frameon=True, prop=dict(fontsize=12))
        ax.add_artist(anchored_text)
        ax.grid(True, which='both',linestyle='dotted', linewidth="0.3 ", color='grey')
        rho0 = np.nanmean(zval)
        
        
        topo = acqVar.topography[conf]
        ax.plot(topo[0,:],topo[1,:], linewidth=3, color='black')
        

        ipanel = AnchoredText(ipan, loc="upper right", frameon=True, prop=dict(fontsize=12))
        
    plt.subplots_adjust(left=0.05, bottom=0.1, right=0.94, top=0.95, wspace=0.3, hspace=0.4)
    fig.savefig(str(fout), transparent=True)#, bbox_inches='tight')#,pad_inches=1)
    print(f"save {str(fout)}")
    

