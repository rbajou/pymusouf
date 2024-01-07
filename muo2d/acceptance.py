#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 08 2021
@author: Raphaël Bajou
"""
# Librairies
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.axes3d import Axes3D
from typing import Union
from pathlib import Path
from matplotlib.gridspec import GridSpec
from mpl_toolkits.axes_grid1 import make_axes_locatable
import pickle
#package module(s)
from reco import HitMap, RecoData, RansacData
from telescope import Telescope, DICT_TEL



params = {'legend.fontsize': 'x-large',
         'axes.labelsize': 'x-large',
         'axes.titlesize':'x-large',
         'xtick.labelsize':'x-large',
         'ytick.labelsize':'x-large',
         'axes.labelpad':10,
         'mathtext.fontset': 'stix',
         'font.family': 'STIXGeneral',
        
         }
plt.rcParams.update(params)




if __name__=="__main__":


    tel = DICT_TEL['SNJ']
    data_path = Path.home()/f"data/{tel.name}/CALIB2/reco/merge"
    reco_file = str( data_path / "reco.csv.gz")
    reco_data = RecoData(reco_file, tel)
    ransac_file = str(data_path / "inlier.csv.gz")
    ransac_data = RansacData(ransac_file, tel)
    df = ransac_data.df
    

    hm_cal = HitMap(tel, reco_data.df)

    from config import MAIN_PATH

    flux_model_tel_path = MAIN_PATH / "files" / "telescopes" /  tel.name /  "flux"
    file_int_flux_sky = flux_model_tel_path / 'integrated_flux_opensky.pkl'
    import pickle
    with open(str(file_int_flux_sky), 'rb') as f: 
        int_flux_opensky = pickle.load(f)

    acc = Acceptance(telescope=tel,
                        hitmap=hm_cal,
                                flux=int_flux_opensky)
    acc.compute()
    #acc.plot_fig_2d()

    fig = plt.figure(figsize=(12, 7))
    ax = fig.add_subplot(projection='3d')
    x, y = np.arange(-15, 16), np.arange(-15, 16)
    Z = acc.estimate['3p1']
    kwargs = {'cmap':'jet'}
    acc.plot_fig_3d(ax=ax, x=x, y=y, Z=Z, **kwargs)
    plt.show()
