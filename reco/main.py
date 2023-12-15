#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
import time
import glob
import pandas as pd
import warnings
warnings.filterwarnings("ignore")
#package module(s)
from telescope import str2telescope
from tracking import InputType
from reco import RecoData, RansacData
from eventrate import EvtRate
from hitmap import HitMap


start_time = time.time()
print("Start: ", time.strftime("%H:%M:%S", time.localtime()))#start time
t_start = time.perf_counter()
home_path = Path.home()
wd_dir = Path(__file__).parents[1]

parser=argparse.ArgumentParser(
description='''Check track reconstruction processing output (hitmaps, event rate, charge distributions)''', epilog="""All is well that ends well.""")
parser.add_argument('--telescope', '-tel', default="SNJ", help='Input telescope name (e.g "COP"). It provides the associated configuration.',  type=str2telescope)
parser.add_argument('--in_dir', '-i', help="/path/to/reco/files", type=str, required=True)
parser.add_argument('--out_dir', '-o', default=f"{str(wd_dir)}/out/", help="/path/to/out/dir/", type=str)
args=parser.parse_args()

tel = args.telescope
sconfig = list(tel.configurations.keys())
nc = len(sconfig)
in_dir =  Path(args.in_dir )
out_dir = Path(args.out_dir )
out_dir.mkdir(parents=True, exist_ok=True)

###Get reco and inlier files
try:
    freco = glob.glob(str(in_dir / f'*reco*'))[0]
    finlier = glob.glob(str(in_dir / f'*inlier*'))[0]

except AttributeError:
    exit(f"Files missing in {in_dir}.")

recoFile = RecoData(file=freco, telescope=tel)

start_time = time.time()

print(f"Load dataframe -- {(time.time() - start_time):.1f}  s")   

#####Hitmap
hmDir = out_dir/ "hitmap"
hmDir.mkdir(parents=True,exist_ok=True)
hmFiles= {c : hmDir/f"hitmap_{c}.txt" for c in sconfig}
hmDXDY  = {}
print("Hit Map(s)")
hm = HitMap(telescope=tel, df=recoFile.df)
##loop on DXDY histograms (one per tel config) and save it as .txt files
for c, h in  hm.hDXDY.items():  np.savetxt(hmFiles[c], h, delimiter='\t', fmt='%.5e')

npan = len(tel.panels)
hm.plot_xy_map(transpose=True) #hits per panel
fout = hmDir / f'xy.png'
plt.savefig(fout) 
print(f"\tSave {fout}")
plt.close()

hm.plot_dxdy_map() #hits per telescope config (3-panel : 1config, 4-panel : 3configs)
fout = hmDir/f'dxdy.png'
plt.savefig(fout)
print(f"\tSave {fout}")
plt.close()


# #####Event rate
print("Event Rate")
erDir= out_dir/ "evtrate"
erDir.mkdir(parents=True, exist_ok=True)
fout = erDir / "event_rate"
fig, ax = plt.subplots(figsize=(16,9))
er = EvtRate(df=recoFile.df)
label = "all"
er(ax, width=3600, label=label) #width = size time bin width in seconds 
ax.legend(loc='best')
plt.savefig(str(fout))
print(f"\tSave {fout}"+".png")
plt.close()
print(f"\tRun duration = {er.run_duration:1.3e} s = {er.run_duration/(3600):1.3e} h = {er.run_duration/(24*3600):1.3e} days")
kwargs={}
er.to_csv(str(fout)+".csv", **kwargs)
print(f"\tSave {fout}"+".csv")
#######

#####Charge distributions per panel (in ADC)
####COMMENT all this below if you are not interested
print("Charge Panel Distributions")
qDir = out_dir / "charge"
qDir.mkdir(parents=True,exist_ok=True)
###golden events (GOLD) = those with exaclty 1XY/panel = closest to real muons
###we use them to compute charge calibration constante per panel to convert from ADC unit to MIP fraction (i.e muon fraction)
(qDir/"gold").mkdir(parents=True, exist_ok=True)
kwargs_dat = { "index_col": 0,"delimiter": '\t'}#, "nrows": 10000} #optional arguments to parse to pandas dataframe use in RecoData()
ransacData = RansacData(file=finlier, telescope=tel, kwargs=kwargs_dat)
from charge import Charge
mask_gold = ransacData.df_inlier['gold'] == 1 
qGold = Charge(df=ransacData.df_inlier[mask_gold], telescope=tel, input_type=InputType.DATA)
fig, axs = plt.subplots(figsize=(16,8), ncols=len(tel.panels), nrows=1, sharey=True)
kwargs_gold = {'color': 'orange', 'label':'inlier', 'alpha':0.5}
for i, (ax, panel) in enumerate(zip(axs, tel.panels)):
    qGold.plot_charge_panel(ax=ax, panel=panel, **kwargs_gold)
    qGold.fit_charge_distrib(panel=panel)
    qGold.plot_fit_panel(ax, panel)
    ax.set_xlabel('charge [ADC]')
    ax.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
    ax.tick_params(axis='both')
    ax.legend(loc='upper right', title=f"{panel.position.loc} ID{panel.ID}")
axs[0].set_ylabel('entries')
fig.tight_layout()

ofile_fit_gold = qDir / "gold" / f"fit_charge.csv"
qGold.df_fit.to_csv( ofile_fit_gold, sep='\t')
print(f"\tSave {ofile_fit_gold}")
fout = qDir / "gold" / "distrib_charge.png"
plt.savefig(str(fout))
print(f"\tSave {fout}")
ofile_perc = qDir / "gold" / f"percentiles_charge.csv"
qGold.df_perc.to_csv( ofile_perc, sep='\t')
print(f"\tSave {ofile_perc}")

qInlier = Charge(df=ransacData.df_inlier, telescope=tel, input_type=InputType.DATA)
qOutlier = Charge(df=ransacData.df_outlier, telescope=tel, input_type=InputType.DATA)
kwargs_in = {'color': 'green', 'label':'inlier', 'alpha':0.5}
kwargs_out = {'color': 'red', 'label':'outlier', 'alpha':0.5}
df_cal = pd.read_csv(ofile_fit_gold, delimiter="\t", index_col=[0], header=[0, 1], skipinitialspace=True) #read multi cols 
dict_fcal = { pan.ID: df_cal.loc[pan.ID]['MPV']['value'] for pan in tel.panels}
fig, axs = plt.subplots(figsize=(16,8), ncols=len(tel.panels), nrows=1, sharey=True)
for i, (ax, panel) in enumerate(zip(axs, tel.panels)): 
    fcal = dict_fcal[panel.ID]
    qInlier.plot_charge_panel(ax=ax, panel=panel, fcal=fcal, **kwargs_in)
    qOutlier.plot_charge_panel(ax=ax, panel=panel, fcal=fcal, **kwargs_out)
    ax.set_xlabel('charge [MIP fraction]')
    ax.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
    ax.tick_params(axis='both')
    ax.legend(loc='upper right', title=f"{panel.position.loc} ID{panel.ID}")
axs[0].set_ylabel('entries')
fig.tight_layout()

fout = qDir / "distrib_charge.png"
plt.savefig(str(fout))
print(f"\tSave {fout}")
ofile_perc = qDir / f"percentiles_charge.csv"
qInlier.df_perc.to_csv( ofile_perc, sep='\t')
print(f"\tSave {ofile_perc}")

print(f"End -- {(time.time() - start_time):.1f}  s")