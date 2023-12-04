#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path
import argparse
import time
import glob
import pandas as pd
import warnings
warnings.filterwarnings("ignore")
#personal modules
from telescope import str2telescope
from tracking import InputType
from reco import EvtRate, EventType, RecoData, AnaBase, AnaHitMap, AnaCharge, PlotHitMap


start_time = time.time()
print("Start: ", time.strftime("%H:%M:%S", time.localtime()))#start time
t_start = time.perf_counter()
home_path = Path.home()
wd_dir = Path(__file__).parents[1]

parser=argparse.ArgumentParser(
description='''Check track reconstruction processing output (hitmaps, event rate, charge distributions)''', epilog="""All is well that ends well.""")
parser.add_argument('--telescope', '-tel', default="COP", help='Input telescope name (e.g "COP"). It provides the associated configuration.',  type=str2telescope)
parser.add_argument('--in_dir', '-i', help="/path/to/output/tracking/", type=str, required=True)
parser.add_argument('--out_dir', '-o', default=f"{str(wd_dir)}/out/", help="/path/to/output/directory/", type=str)
args=parser.parse_args()

tel = args.telescope
sconfig = list(tel.configurations.keys())
nc = len(sconfig)
inDir =  Path(args.in_dir )
outDir = Path(args.out_dir )
outDir.mkdir(parents=True, exist_ok=True)

###Get reco and inlier files
try:
    freco = glob.glob(str(inDir / f'*reco*'))[0]
    finlier = glob.glob(str(inDir / f'*inlier*'))[0]

except AttributeError:
    exit(f"Files missing in {inDir}.")

print("ANALYSIS...")
start_time = time.time()
input_type = InputType.DATA
finfo  = Path(args.out_dir) / f"info.json"
run_info = {}
recoFile = RecoData(file=freco, 
                            telescope=tel,  
                            input_type=input_type)

print(f"load files --- {(time.time() - start_time):.3f}  s ---")   


####if you want to set a time period
# tlim = ( int(datetime(2019, 3, 1, hour=00,minute=00,second=00).replace(tzinfo=timezone.utc).timestamp()) , 
#         int(datetime(2019, 5, 1, hour=23,minute=59,second=59).replace(tzinfo=timezone.utc).timestamp())   )

ab = AnaBase(recofile=recoFile, 
                    label="", 
                    tlim=None)


#####Hitmap
hmDir = outDir/ "hitmap"
hmDir.mkdir(parents=True,exist_ok=True)
hmFiles= {c : hmDir/f"hitmap_{c}.txt" for c in sconfig}
hmDXDY  = {}
print("Hit maps (DXDY)")
hm = AnaHitMap(anabase=ab, input_type=input_type, panels=tel.panels)
##loop on DXDY histograms (one per tel config) and save it as .txt files
for c, h in  hm.hDXDY.items():  np.savetxt(hmFiles[c], h, delimiter='\t', fmt='%.5e')
pl = PlotHitMap(hitmaps=[hm], outdir=str(hmDir))
pl.XY_map(transpose=True) #hits per panel
pl.DXDY_map() #hits per telescope config (3-panel : 1config, 4-panel : 3configs)
print("save in:")
os.system(f"ls {hmDir}/*")
########

#####Event rate
fout = outDir / "event_rate.png"
fig, ax = plt.subplots(figsize=(16,9))
evtrateCal = EvtRate(df=ab.df)
label = "all"
evtrateCal(ax, width=3600, label=label) #width = size time bin width in seconds 
ax.legend(loc='best')
plt.savefig(str(fout))
plt.close()
print(f"save {str(fout)}")
#######


#####Charge distributions per panel (in ADC)
####COMMENT all this below if you are not interested
chargeDir = outDir / "charge"
chargeDir.mkdir(parents=True,exist_ok=True)
###golden events (GOLD) = those with exaclty 1XY/panel = closest to real muons
###we use them to compute charge calibration constante per panel to convert from ADC unit to MIP fraction (i.e muon fraction)
(chargeDir/"gold").mkdir(parents=True, exist_ok=True)
kwargs = { "index_col": 0,"delimiter": '\t'}#, "nrows": 10000} #optional arguments to parse to pandas dataframe use in RecoData()
inlierData = RecoData(file=finlier, telescope=tel, input_type=input_type, kwargs=kwargs, is_inlier=True)
outlierData = RecoData(file=finlier, telescope=tel, input_type=input_type, kwargs=kwargs, is_inlier=False)
acGold= AnaCharge(inlier_file=inlierData,
                                outlier_file=outlierData,
                                outdir=str(chargeDir/"gold"), 
                                evttype=EventType.GOLD,
                                label=label,
                                )
do_fit = True
dict_charge_gold = {('Inlier', 'green', do_fit) :acGold.ADC_XY_inlier, ('Outlier', 'red', False):acGold.ADC_XY_outlier} 
qcal_file = chargeDir / "gold" / "fit_dQ_Inlier.csv" 
if len(acGold.ADC_XY_inlier) != 0 and not qcal_file.exists():    
    print("Fit gold charge distributions")
    acGold.plot_charge_panels(charge=dict_charge_gold, 
                                is_scaling=False, 
                                input_type=input_type.name)  
fcal = { pan.ID: 1 for pan in tel.panels}
unc_fcal = { pan.ID: 0 for pan in tel.panels} 
if qcal_file.exists(): 
    ###Load gold calibration constantes for conversion ADC->MIP
    df_cal = pd.read_csv(str(qcal_file), delimiter="\t", index_col=[0], header=[0, 1], skipinitialspace=True) #read multi cols 
    fcal = { pan.ID: df_cal.loc[pan.ID]['MPV']['value'] for pan in tel.panels}
    unc_fcal = { pan.ID: df_cal.loc[pan.ID]['MPV']['error'] for pan in tel.panels}

###rest of events (MAIN)
(chargeDir/"main").mkdir(parents=True, exist_ok=True)
acMain= AnaCharge(inlier_file=inlierData,
                    outlier_file=outlierData,
                    outdir=str(chargeDir/"main"), 
                    evttype=EventType.MAIN,
                    label=label,
                    )
dict_charge_main = {('Inlier', 'green', False) :acMain.ADC_XY_inlier, ('Outlier', 'red', False):acMain.ADC_XY_outlier} 
acMain.plot_charge_panels(charge=dict_charge_main, 
                                is_scaling=False, 
                                input_type=input_type.name,
                                fcal=fcal,
                                unc_fcal=unc_fcal)  
print("save in:")
os.system(f"ls {chargeDir}/*")
###########

print(f"end --- {(time.time() - start_time):.3f}  s ---")  

