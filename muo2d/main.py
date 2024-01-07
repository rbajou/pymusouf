#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#%%
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import scipy.io as sio
from pathlib import Path
import argparse
import time
from datetime import datetime, timezone
import logging
import glob
import yaml
#import pickle
import json
import pandas as pd
import copy
import palettable
import pickle

import sys
sys.path.append(str(Path(__file__).parents[1]))  #needed if interactive mode
#package module(s)
#from acceptance import Acceptance
from config import MAIN_PATH, SURVEY_DIR
from forwardsolver import FluxModel 
from filter import *
from muo2d import Acceptance, TransmittedFluxModel, Muo2D
from raypath import RayPathSoufriere
from reco import RansacData, RecoData, EvtRate, HitMap, Charge
from telescope import str2telescope
from survey import CURRENT_SURVEY

#%%
start_time = time.time()
print("Start: ", time.strftime("%H:%M:%S", time.localtime()))#start time

###Load default script arguments stored in .yaml file
parser=argparse.ArgumentParser(description='''Estimate acceptance from calib data; flux, opacity and mean density from tomo data.''', epilog="""All is well that ends well.""")
parser.add_argument('--telescope', '-tel',  required=True, help='Input telescope key (e.g "SNJ"). It provides the associated configuration(s).',  type=str2telescope) #required=True,
args=parser.parse_args()
tel = args.telescope
sconfig  = list(tel.configurations.keys())
nc = len(sconfig)

survey = CURRENT_SURVEY

home = Path.home()
main_path = Path(__file__).parents[1]
out_path = main_path / "out"
out_path.mkdir(parents=True, exist_ok=True)
t_start = time.perf_counter()

run_survey = survey.runs[tel.name]
run_path = run_survey.run_path

run_tomo = run_survey.run_tomo
run_calib = run_survey.run_calib

ftrack_tomo, finlier_tomo = [], []
for run in run_tomo:
    data_path = run_path['tomo'][run.name] 
    recopath_tomo = run.check_recopath(data_path)
    ftrack_tomo.append(glob.glob(str(recopath_tomo / f"*track*") )[0] )
    finlier_tomo.append(glob.glob(str(recopath_tomo / f"*inlier*") )[0] )


ftrack_cal, finlier_cal = [], []
for run in run_calib:
    data_path = run_path['calib'][run.name] 
    recopath_cal = run.check_recopath(data_path)
    ftrack_cal.append(glob.glob(str(recopath_cal / f"*track*") )[0] )
    finlier_cal.append(glob.glob(str(recopath_cal / f"*inlier*") )[0] )

kwargs_dat = { "index_col": 0, "delimiter": "\t", "nrows": None}
reco_data_tomo = RecoData(file=ftrack_tomo, telescope=tel)
ransac_data_tomo = RansacData(file=finlier_tomo, telescope=tel, kwargs=kwargs_dat)
reco_data_cal = RecoData(file=ftrack_cal, telescope=tel)
ransac_data_cal = RansacData(file=finlier_cal, telescope=tel, kwargs=kwargs_dat)


print(f"Load dataframe(s) -- {(time.time() - start_time):.1f} s")   
#%%
##Reindexing tomo dataframe (because of potential evtID doublons)
df = ransac_data_tomo.df
old_ix = df.index.to_numpy()
repeat = np.diff(np.where(np.concatenate(([old_ix[0]], old_ix[:-1] != old_ix[1:], [True])))[0]) #repeat consecutive index
df2 = reco_data_tomo.df 
df2.reset_index(inplace=True) 
new_ix = np.repeat(df2.index, repeat) #repeat new index for ransac dataframe
df.index = pd.Index(new_ix)
print(f"Reindexing -- {time.time()-start_time:.3f} s")

hm_tomo = HitMap(tel, reco_data_tomo.df)
#print(f"Before: nevts = {len(hm_tomo.df_DXDY['3p1'])}")
hm_cal = HitMap(tel, reco_data_cal.df)

##Filter(s)
# print("Filter(s)")
# fim_tomo = FilterInlierMultipliciy(tel, ransac_data_tomo.df)
# fim_tomo.apply_cut_front_rear(hm_tomo, 2, 2)    


# fim_cal = FilterInlierMultipliciy(tel, ransac_data_cal.df)
# fim_cal.apply_cut_front_rear(hm_cal, 2, 2)    

# ftof_tomo = FilterToF(tel, ransac_data_tomo.df)
# ftof_tomo.get_dict_filter(dtof=10)
# hm_tomo.fill_dxdy(dict_filter=ftof_tomo.dict_filter)
# print(f"After filter: hm = {len(hm_tomo.df_DXDY['3p1'])}")

# fig, ax = plt.subplots(figsize=(12,7))
# kwargs={'label':'tof'}
# ftof_tomo.plot_tof(ax, **kwargs)
# plt.show()

# ftof_cal = FilterToF(tel, ransac_data_cal.df)
# ftof_cal.get_dict_filter(dtof=5) #dtof in ns
# hm_cal.fill_dxdy(dict_filter=ftof_cal.dict_filter)
# print(f"After filter: nevts = {len(hm_cal.df_DXDY['3p1'])}")

##When several filters: 
# dict_list = [filter_time.dict_filter, filter_tof.dict_filter]
# new_dict_filter = intersect_multiple_filters(dict_list)


#%%
#Flux model
survey_path = CURRENT_SURVEY.path
print(f"survey_path= {survey_path}")
flux_model_path = survey_path  / "flux" 
cors_path =  flux_model_path / "corsika" / "muons" / "032023"
print(f"cors_path= {cors_path}")

if not cors_path.exists() : raise ValueError("Check path corsika flux.")
file_diff_flux_model = cors_path / "diff_flux.pkl"
with open(str(file_diff_flux_model), 'rb') as f: 
    diff_flux = pickle.load(f)
dflux_mean, dflux_std, energy_bins, theta_bins = diff_flux['mean'], diff_flux['std'], diff_flux['energy_bins'], diff_flux['theta_bins']*np.pi/180
fm = FluxModel(altitude=tel.altitude, 
               corsika_flux=dflux_mean, 
               corsika_std=dflux_std, 
               energy_bins=energy_bins, 
               theta_bins=theta_bins)



##Acceptance
flux_model_tel_path = survey_path / "telescope" /   tel.name /  "flux"
file_int_flux_sky = flux_model_tel_path / 'integral_flux_opensky.pkl'
with open(str(file_int_flux_sky), 'rb') as f: 
    int_flux_opensky = pickle.load(f)

acc = Acceptance(telescope=tel,
                     hitmap=hm_cal,
                             flux=int_flux_opensky)
acc.compute()

##plot acceptance
# fig = plt.figure(figsize=(12, 7))
# # acc.plot_fig_2d()
# ax = fig.add_subplot(projection='3d')
# conf = '3p1'
# front, rear = tel.configurations[key][0], tel.configurations[key][-1]
# ray_matrix = tel.get_ray_matrix(front, rear)
# X, Y = ray_matrix[:,:,0], ray_matrix[:,:,1]
# Z = acc.estimate[key]
# kwargs = {'cmap':'jet'}
# acc.plot_fig_3d(ax=ax, grid_x=X, grid_y=Y, grid_z=Z, **kwargs)
# plt.show()
# figkey = out_path / f"acceptance_{key}.png"
# plt.savefig(figkey)
# print("Save figure {figkey}")
file = out_path / "acceptance.pkl"
file.parent.mkdir(parents=True, exist_ok=True)
acc.save(file)
print(f"Acceptance -- {(time.time() - start_time):.3f}  s") 

filter_time = FilterTimePeriod(telescope=tel, df=reco_data_tomo.df)

tlim = [ datetime(2016, 9, 27, hour=00,minute=00,second=00), 
                datetime(2017, 2, 7, hour=16,minute=00,second=00)   ]


filter_time.get_dict_filter(tlim=tlim)

fig, ax = plt.subplots(figsize=(16,8))
ftraw = recopath_tomo / "traw.csv.gz"
if ftraw.exists():
    dftraw = pd.read_csv(ftraw, index_col=0, delimiter="\t")
    evtrateTomo_raw = EvtRate(df=dftraw)
    evtrateTomo_raw(ax, width=3600, label="raw", alpha=0.3)
df = reco_data_tomo.df
evtrateTomo = EvtRate(df=df)
evtrateTomo(ax, width=3600, label= "reco") #width = size time bin width in seconds 
df_f = df.loc[filter_time.idx]
evtrateTomo_f = EvtRate(df=df_f)
evtrateTomo_f(ax, width=3600, label="filter")
ax.legend(loc='best')
figfile = out_path/ 'evtrate.png'
plt.savefig(figfile)
print(f"Save {figfile}")
plt.close()



old_cors_path = flux_model_path /  'corsika' / 'muons' / 'former' 
file_int_flux = old_cors_path / 'int_flux_opacity_zenith_grid.pkl'
with open(str(file_int_flux), 'rb') as f:
    print(f"Load {file_int_flux}")
    dict_int_flux = pickle.load(f,)

model = TransmittedFluxModel(
    zenith = dict_int_flux['zenith'],
    flux = dict_int_flux['flux'], 
    opacity = dict_int_flux['opacity'], 
    error = dict_int_flux['sigma'], 
)

files_path = main_path / 'files'

raypath = RayPathSoufriere[tel.name]

thickness = {key: ray['thickness'] for key, ray in raypath.raypath.items() }
muo2d = Muo2D(telescope=tel, 
              hitmap=hm_tomo, 
              acceptance=acc, 
              evtrate=evtrateTomo,
              model=model,
              thickness=thickness)
fout_flux = out_path / "flux.pkl"
fout_op = out_path / "opacity.pkl"
fout_rho = out_path / "mean_density.pkl"
muo2d.flux.save(fout_flux)
muo2d.opacity.save(fout_op)
muo2d.mean_density.save(fout_rho)

###plot transmitted flux, opacity, mean density maps

for key, _ in tel.configurations.items():
   
    X, Y = tel.azimuthMatrix[key]*180/np.pi, tel.zenithMatrix[key]*180/np.pi
    
    Z = muo2d.flux.estimate[key]
    Z[Z==0] = np.nan
    fig, ax = plt.subplots(figsize=(12,7))
    vmin, vmax = np.nanmin(Z), np.nanmax(Z)
    kwargs_flux = dict(cmap='viridis',  shading='auto', norm=LogNorm(vmin=vmin, vmax=vmax), label = 'Transmitted Flux [cm$^{-2}$.s$^{-1}$.sr$^{-1}$]')
    muo2d.plot_map_2d(fig, ax, grid_x=X, grid_y=Y, grid_z=Z, **kwargs_flux)
    topo = raypath.raypath[key]['profile_topo']
    ax.plot(topo[:,0], topo[:,1], linewidth=3, color='black')
    figfile = out_path / f"flux_{key}.png"
    plt.savefig(figfile)
    print(f"Save figure {figfile}")
    plt.close()

    Z = muo2d.opacity.estimate[key]
    Z[Z==0] = np.nan
    fig, ax = plt.subplots(figsize=(12,7))
    vmin, vmax = np.nanmin(Z), np.nanmax(Z)
    kwargs_op = dict(cmap='jet',  shading='auto', norm=LogNorm(vmin=vmin, vmax=vmax), label = 'Opacity $\\varrho$ [mwe]')
    muo2d.plot_map_2d(fig, ax, grid_x=X, grid_y=Y, grid_z=Z, **kwargs_op)
    figfile = out_path / f"opacity_{key}.png"
    plt.savefig(figfile)
    print(f"Save figure {figfile}")
    plt.close()

    Z = muo2d.mean_density.estimate[key]
    Z[Z==0] = np.nan
    cmap = palettable.scientific.sequential.Batlow_20.mpl_colormap
    fig, ax = plt.subplots(figsize=(12,7))
    vmin, vmax = np.nanmin(Z), 3#np.nanmax(Z)
    kwargs_rho = dict(cmap=cmap,  shading='auto', vmin=vmin, vmax=vmax, label = 'Mean Density $\\overline{\\rho}$ [g.cm$^{-3}$]')
    muo2d.plot_map_2d(fig, ax, grid_x=X, grid_y=Y, grid_z=Z, **kwargs_rho)
    figfile = out_path / f"mean_density_{key}.png"
    plt.savefig(figfile)
    print(f"Save figure {figfile}")
    plt.close()

print(f"End -- {(time.time() - start_time):.1f} s")






