#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#%%
import numpy as np
import matplotlib.pyplot as plt
import os
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

#personal modules
from telescope import str2telescope, dict_tel
from tracking import InputType
from muo2d import Muography
try : 
    from acceptance import Acceptance
except : 
    from muo2d import Acceptance
from reco import InlierData, RecoData, Cut, EventType, AnaBase, AnaCharge, AnaHitMap, PlotHitMap, EvtRate
from utils.tools import pretty
from forwardsolver import FluxModel 
from raypath import AcqVars

###Load default script arguments stored in .yaml file
def_args={}
#parser=argparse.ArgumentParser(description='''Plot event rate, hit, flux, opacity and density maps''', epilog="""All is well that ends well.""")
#parser.add_argument('--telescope', '-tel',  required=True, help='Input telescope name (e.g "SNJ"). It provides the associated configuration.',  type=str2telescope) #required=True,
#args=parser.parse_args()
tel = dict_tel['OM']#args.telescope
sconfig  = list(tel.configurations.keys())
nc = len(sconfig)
####default arguments/paths are written in a yaml config file associated to telescope
with open( str(Path(__file__).parents[1] / 'files' / 'telescopes' / tel.name /'run.yaml') ) as fyaml:
    try:
        # The FullLoader parameter handles the conversion from YAML
        # scalar values to Python the dictionary format
        def_args = yaml.load(fyaml, Loader=yaml.SafeLoader)
        #def_args = yaml.safe_load(fyaml)
    except yaml.YAMLError as exc:
        print(exc)
    
pretty(def_args)

start_time = time.time()
print("Start: ", time.strftime("%H:%M:%S", time.localtime()))#start time
t_start = time.perf_counter()
home = Path.home()
main_path = Path(__file__).parents[1]
out_path = main_path / 'out'
tomoRun = def_args["reco_tomo"]['run']
print(tomoRun)
print(isinstance(tomoRun, list))
ext="csv.gz"

if isinstance(tomoRun, list): 
    input_tomo= []
    freco_tomo, finlier_tomo = [], []
    for run in tomoRun:
        input_tomo = home / run
        freco_tomo.append(glob.glob(str(input_tomo / f'*reco.{ext}') )[0] )
        finlier_tomo.append(glob.glob(str(input_tomo / f'*inlier.{ext}') )[0] )
        #df = pd.concat([pd.read_csv(f, delimiter="\t", index_col=0) for f in chunck_files])
else: 
    input_tomo = home / def_args["reco_tomo"]['run']
    freco_tomo = glob.glob(str(input_tomo / f'*reco.{ext}') )[0] 
    finlier_tomo = glob.glob(str(input_tomo / f'*inlier.{ext}') )[0] 

input_calib = home / def_args["reco_calib"]
eff_dir = home /  def_args["eff_dir"]

input_type = InputType.DATA

print(f"input_tomo: {input_tomo}")

freco_cal = glob.glob(str(input_calib / f'*reco.{ext}') )[0] 
finlier_cal = glob.glob(str(input_calib / f'*inlier.{ext}') )[0] 


label_tomo = def_args["label_tomo"]
label_calib = def_args["label_calib"]

kwargs = { "index_col": 0,"delimiter": '\t', "nrows": None}

iD_tomo = RecoData(file=finlier_tomo, telescope=tel, input_type=input_type, kwargs=kwargs, is_all=True) #inlier pts
oD_tomo = copy.deepcopy(iD_tomo) #outlier 
oD_tomo.df = iD_tomo.df[iD_tomo.df['inlier']==0]
rD_tomo = RecoData(file=freco_tomo, telescope=tel, index=list(set(iD_tomo.df.index)), input_type=input_type, kwargs=kwargs) #tracks

iD_cal = RecoData(file=finlier_cal, telescope=tel, input_type=input_type, kwargs=kwargs, is_all=True)
oD_cal = copy.deepcopy(iD_cal)
oD_cal.df = iD_cal.df[iD_cal.df['inlier']==0]
rD_cal = RecoData(file=freco_cal, telescope=tel, input_type=input_type, kwargs=kwargs)

####reindexing tomo dataset to avoid duplicated evtIDs:
old_ix_tomo = iD_tomo.df.index.to_numpy()
#array with consecutive index number of repetitions, repeat should have the length of recodata.df 
repeat_tomo = np.diff(np.where(np.concatenate(([old_ix_tomo[0]], old_ix_tomo[:-1] != old_ix_tomo[1:], [True])))[0])
rD_tomo.df = rD_tomo.df.reset_index()
new_ix_tomo = np.repeat(rD_tomo.df.index, repeat_tomo)
iD_tomo.df.index = pd.Index(new_ix_tomo)
#oD_tomo.df.index = pd.Index(new_ix_tomo)
print(f"reindexing -- {time.time()-start_time:.3f} s -- ")
######

# terrain_path = main_path / "telescope" / "files" / tel.name /"terrain"
# crater = { 
#         "TAR": np.loadtxt(terrain_path/f"TAR_on_border.txt", delimiter="\t") , 
#         "CS": np.loadtxt(terrain_path/f"CS_on_border.txt", delimiter="\t"),
#         "BLK": np.loadtxt(terrain_path/f"BLK_on_border.txt", delimiter="\t"),
#         "G56": np.loadtxt(terrain_path/f"G56_on_border.txt", delimiter="\t"),
#         "FNO": np.loadtxt(terrain_path/f"FNO_on_border.txt", delimiter="\t")
#         }


#############
chi2_min, chi2_max = 0., 8.
cut_chi2 = Cut(column="rchi2", vmin=chi2_min, vmax=chi2_max, label=f"{chi2_min} <"+"$\\chi^{2}$/ndf"+f"< {chi2_max}")
#cut_tof = Cut(column="time-of-flight", vmin=0, vmax=10, label=f"tof>0")
cut= cut_chi2


is_filter_chi2 = True
is_filter_tof = False
is_filter_charge = False
is_filter_multiplicity = False
    

print(f"is_filter_chi2 : {is_filter_chi2}")

############ DEFINE PATHS



tomoDir =  out_path / def_args["out_dir_tomo"] ##ana out path
calibDir = out_path / def_args["out_dir_calib"]

date_str = time.strftime('%d%m%Y')
date_str = "04042023"
tomoDir =  tomoDir / date_str  ##ana out path
calibDir = calibDir / date_str



####
scint_eff = 1.#0.9**2
eff = { conf : np.ones(tel.los[conf].shape[:-1])*scint_eff for conf in sconfig }
unc_eff = { conf : np.ones(tel.los[conf].shape[:-1])*scint_eff for conf in sconfig }

print("before")
print("iD_tomo.df = ", iD_tomo.df.head)
    
ix_gold_tomo = iD_tomo.df[iD_tomo.df['gold'] == 1].index
ix_gold_cal = iD_cal.df[iD_cal.df['gold'] == 1].index
print(f"ngold_tomo = {len(ix_gold_tomo)}")
print(f"ngold_cal = {len(ix_gold_cal)}")

if is_filter_chi2 : 
    tomoDir = tomoDir / "cut_chi2" / f"{cut.vmin}_{cut.vmax}"
    calibDir = calibDir / "cut_chi2" / f"{cut.vmin}_{cut.vmax}"
    tomoDir.mkdir(parents=True, exist_ok=True)
    calibDir.mkdir(parents=True, exist_ok=True)
if is_filter_charge :
    tomoDir = tomoDir / "filter_charge"
    calibDir = calibDir / "filter_charge"
if is_filter_multiplicity : 
    tomoDir = tomoDir / "filter_multiplicity"
    calibDir = calibDir / "filter_multiplicity"
    eff_dir = eff_dir / "multiplicity"
    if eff_dir.exists():pass
    else : raise Exception(f"{str(eff_dir)} does not exist.")
    ###define cut
    is_inlier = False
    cut_m_front, cut_m_rear = 1, 2 # n XY hit(s) per panel 
    if tel.name == "BR" or tel.name == "OM": 
        cut_m_front, cut_m_rear = 2, 3 
    ###Get hit multiplicty per evt on each panel...
    print("Multiplicity")
    
    df_tmp_tomo = iD_tomo.df.groupby([iD_tomo.df.index, "Z", "inlier"]).count()#six
    df_tmp_cal = iD_cal.df.groupby([iD_cal.df.index, "Z", "inlier"]).count()#six
    Z = [p.position.z for p in tel.panels]
    lidx_tomo = [ (idx, z, i)  for idx in rD_tomo.df.index.unique()  for z in Z for i in [0,1] ] 
    lidx_cal = [ (idx, z, i)  for idx in rD_cal.df.index.unique()  for z in Z  for i in [0,1] ] 
    print("Before reindexing")
    mix_tomo = pd.MultiIndex.from_tuples(lidx_tomo, names=['ix', 'Z', 'inlier']) #six
    mix_cal = pd.MultiIndex.from_tuples(lidx_cal, names=['ix', 'Z', 'inlier']) #six
    
    ####will fill missing panel with 0 
    df_tmp_tomo = df_tmp_tomo.reindex(mix_tomo, fill_value=0) #multiindex '(evtID, z)'
    df_tmp_cal = df_tmp_cal.reindex(mix_cal, fill_value=0) #multiindex '(evtID, z)'
    print(f"After multi reindexing --- {(time.time() - start_time):.3f}  s --- ")
    
    str_rand_col= 'timestamp_ns' ###get count in column (could be any column name)
    df_multi_tomo = df_tmp_tomo[[str_rand_col]] 
    df_multi_cal = df_tmp_cal[[str_rand_col]] 
    df_multi_tomo.columns = df_multi_tomo.columns.str.replace(str_rand_col, 'm') #rename column 'm' = 'multiplicity'
    df_multi_cal.columns = df_multi_cal.columns.str.replace(str_rand_col, 'm') #rename column 'm' = 'multiplicity'
    dict_df_multi_tomo=  {pan.position.loc : df_multi_tomo.xs([pan.position.z, int(is_inlier)], level=[1,2], drop_level=[1,2])['m'] for pan in tel.panels }
    dict_df_multi_cal=  {pan.position.loc : df_multi_cal.xs([pan.position.z, int(is_inlier)], level=[1,2], drop_level=[1,2])['m'] for pan in tel.panels }


    #dead_time = 0.8 #portion of deadtime
    
    '''
    for conf in sconfig : 
        if conf[0] == '3' : sconf = '3p1'
        else : sconf = '4p'
        eff[conf]  = np.loadtxt(glob.glob(str(eff_dir/ f'eff*{sconf}*'))[0])*scint_eff#*dead_time
        #print(eff[conf].shape)
        var =  np.loadtxt(glob.glob(str(eff_dir/ f'var*{sconf}*'))[0])*scint_eff#*dead_time
        unc_eff[conf] =  np.sqrt(var) 
        #print(unc_eff[conf].shape)
    '''
    fout = tomoDir/"cut_m.txt"
    tomoDir.mkdir(parents=True, exist_ok=True)
    with open(str(fout), "w") as f:
        f.write(f"multi_front, multi_rear = {cut_m_front}, {cut_m_rear} ")
    fout = calibDir/"cut_m.txt"
    calibDir.mkdir(parents=True, exist_ok=True)
    with open(str(fout), "w") as f:
        f.write(f"multi_front, multi_rear =  {cut_m_front}, {cut_m_rear} ns")


if is_filter_tof: 
    print("filter_tof")
#cut_s = (df_tof_s_cal['timestamp_s'] > 0)
    tomoDir = tomoDir / "filter_tof"
    calibDir = calibDir / "filter_tof"
    tomoDir.mkdir(parents=True, exist_ok=True)
    calibDir.mkdir(parents=True, exist_ok=True)


#df_tof_s_tomo = iD_tomo.df.groupby(["evtID"])[['Z','timestamp_s']].max() - iD_tomo.df.groupby(["evtID"])[['Z','timestamp_s']].min()
#df_tof_ns_tomo = iD_tomo.df.groupby(["evtID"])[['Z','timestamp_ns']].max() - iD_tomo.df.groupby(["evtID"])[['Z','timestamp_ns']].min()
# df_tof_s_tomo = iD_tomo.df.groupby([iD_tomo.df.index])[['Z','timestamp_s']].max() - iD_tomo.df.groupby([iD_tomo.df.index])[['Z','timestamp_s']].min()
# df_tof_ns_tomo = iD_tomo.df.groupby([iD_tomo.df.index])[['Z','timestamp_ns']].max() - iD_tomo.df.groupby([iD_tomo.df.index])[['Z','timestamp_ns']].min()
iD_tomo.df.rename_axis('evtID', inplace=True)
#print("after rename_axis : iD_tomo.df.head =",iD_tomo.df.head)
iD_tomo.df.reset_index(inplace=True)
#print("after rest_index : iD_tomo.df.head =",iD_tomo.df.head)
iD_cal.df.rename_axis('evtID', inplace=True)
#print("after rename_axis : iD_cal.df.head =",iD_cal.df.head)
iD_cal.df.reset_index(inplace=True)
#print("after rest_index : iD_cal.df.head =",iD_cal.df.head)
idx_max = iD_tomo.df.groupby(['evtID'])['Z'].transform('max') == iD_tomo.df["Z"]
idx_min = iD_tomo.df.groupby(['evtID'])['Z'].transform('min') == iD_tomo.df["Z"]
df_tof_s_tomo = (iD_tomo.df[idx_max].groupby(['evtID'])['timestamp_s'].max() - iD_tomo.df[idx_min].groupby(['evtID'])['timestamp_s'].max())*10
df_tof_ns_tomo = (iD_tomo.df[idx_max].groupby(['evtID'])['timestamp_ns'].max() - iD_tomo.df[idx_min].groupby(['evtID'])['timestamp_ns'].max())*10
iD_cal.df.reset_index(inplace=True)
idx_max = iD_cal.df.groupby(['evtID'])['Z'].transform('max') == iD_cal.df["Z"]
idx_min = iD_cal.df.groupby(['evtID'])['Z'].transform('min') == iD_cal.df["Z"]
df_tof_s_cal = (iD_cal.df[idx_max].groupby(['evtID'])['timestamp_s'].max() - iD_cal.df[idx_min].groupby(['evtID'])['timestamp_s'].max())*10
df_tof_ns_cal = (iD_cal.df[idx_max].groupby(['evtID'])['timestamp_ns'].max() - iD_cal.df[idx_min].groupby(['evtID'])['timestamp_ns'].max())*10


#df_tof_s_cal = iD_cal.df.groupby(["evtID"])[['Z','timestamp_s']].max() - iD_cal.df.groupby(["evtID"])[['Z','timestamp_s']].min()
#df_tof_ns_cal = iD_cal.df.groupby(["evtID"])[['Z','timestamp_ns']].max() - iD_cal.df.groupby(["evtID"])[['Z','timestamp_ns']].min()
#df_tof_s_cal = iD_cal.df.groupby([iD_cal.df.index])[['Z','timestamp_s']].max() - iD_cal.df.groupby([iD_cal.df.index])[['Z','timestamp_s']].min()
#df_tof_ns_cal = iD_cal.df.groupby([iD_cal.df.index])[['Z','timestamp_ns']].max() - iD_cal.df.groupby([iD_cal.df.index])[['Z','timestamp_ns']].min()



dtof = 20 #in ns

fig, ax = plt.subplots()
bins_tof, range_tof = 20, [-100,100]
density=False
em,bin,_= ax.hist(df_tof_ns_tomo, bins=bins_tof, range=range_tof, label="all", density=density)
binc=(bin[:-1]+ bin[1:])/2
tof_peak = binc[np.argmax(em)]
binw=abs(bin[:-1]- bin[1:])
hg = df_tof_ns_tomo[df_tof_ns_tomo.index.isin(ix_gold_tomo)]
eg,_,_=ax.hist(hg, bins=bins_tof, range=range_tof, color="orange", label="gold", density=density)
ax.set_xlabel('time-of-flight [ns]')
ax.set_ylabel('density entries')
ax.set_yscale('log')
ax.vlines([tof_peak-dtof, tof_peak+dtof], ymin=0, ymax=1, transform=ax.get_xaxis_transform(), color='red', linestyle="dashed", alpha=0.6)
ax.legend()
fout = tomoDir/"tof"
plt.savefig(str(fout)+".png", transparent=True)
print(f"save {fout}")
header ="dtbin_c\tdtbin_w\tent_main\tent_gold\t"
mat_out = np.vstack((binc,binw, em,eg)).T
np.savetxt(f"{str(fout)}.txt", mat_out, delimiter="\t", header=header, fmt="%.0f")
plt.close()
cut_s = (df_tof_s_tomo> 0)
cut_ns_tomo = (df_tof_ns_tomo < tof_peak-dtof) | (tof_peak+dtof < df_tof_ns_tomo ) 
fout = tomoDir/"cut_tof.txt"
header ="tof_min\ttof_peak\ttof_max\t[ns]"
mat_out = np.vstack((tof_peak-dtof,tof_peak,tof_peak+dtof)).T
np.savetxt(fout, mat_out, delimiter="\t", header=header, fmt="%.1f")

fig, ax = plt.subplots()
em,bin,_ =  ax.hist(df_tof_ns_cal, bins=bins_tof, range=range_tof, label="all",density=density)
binc=(bin[:-1]+ bin[1:])/2
tof_peak = binc[np.argmax(em)]
hg = df_tof_ns_cal[df_tof_ns_cal.index.isin(ix_gold_cal)]
eg,_,_= ax.hist(hg, bins=bins_tof, range=range_tof, color="orange", label="gold", density=density)
ax.set_xlabel('time-of-flight [ns]')
ax.set_ylabel('entries')
ax.set_yscale('log')
ax.vlines([tof_peak-dtof, tof_peak+dtof], ymin=0, ymax=1, transform=ax.get_xaxis_transform(), color='red', linestyle="dashed", alpha=0.6)
ax.legend()
fout = calibDir/"tof"
plt.savefig(str(fout)+".png", transparent=True)
print(f"save {fout}")
header ="dtbin_c\tdtbin_w\tent_main\tent_gold\t"
mat_out = np.vstack((binc,binw,em,eg )).T
np.savetxt(f"{str(fout)}.txt", mat_out, delimiter="\t", header=header, fmt="%.0f")
plt.close()

cut_s = (df_tof_s_cal> 0)
cut_ns_cal =  (df_tof_ns_cal < tof_peak-dtof) | (tof_peak+dtof < df_tof_ns_cal ) 

fout = calibDir/"cut_tof.txt"
header ="tof_min\ttof_peak\ttof_max\t[ns]"
mat_out = np.vstack((tof_peak-dtof,tof_peak,tof_peak+dtof)).T
np.savetxt(fout, mat_out, delimiter="\t", header=header, fmt="%.1f")


tomoDir.mkdir(parents=True, exist_ok=True)
calibDir.mkdir(parents=True, exist_ok=True)


print("End cell #1")
#%%

#old_stdout = sys.stdout
#log_file = open(str(tomoDir/"message.log"),"w")
#sys.stdout = log_file

print(f"###########\ntomoDir={tomoDir}\n###########\ncalibDir={calibDir}")

timestr = time.strftime("%d%m%Y")
logging.basicConfig(filename=str(tomoDir/f'run_{label_tomo}_{timestr}.log'), level=logging.INFO)#, filemode='w')
timestr = time.strftime("%d%m%Y-%H%M%S")
logging.info(timestr)
logging.info(def_args)

#print("ANALYSIS...")
start_time = time.time()
#print("Start: ", time.strftime("%H:%M:%S", time.localtime()))#start time
flux_model_path = main_path / 'files'  / "flux" 
flux_model_tel_path = main_path / 'files' / 'telescopes' /   tel.name /  'flux'
####OPEN-SKY flux for ACCEPTANCE computation either CORSIKA or model
model = 'corsika'#'corsika'#'corsika_soufriere'
lmodel_avail = ['guan', 'gaisser', 'modgaisser', 'corsika_soufriere']
os_flux = {conf: np.zeros((mat.shape[0], mat.shape[1]) ) for conf,mat  in tel.los.items()}
Corsika_OpenSkyFlux = sio.loadmat(str(flux_model_tel_path  / 'ExpectedOpenSkyFlux.mat'))
for conf in sconfig: 
    ###loop on telescope configurations 
    if model == 'corsika' or model not in lmodel_avail: 
        flux = Corsika_OpenSkyFlux[f'ExpectedFlux_calib_{conf[0]}p']#open-sky flux  
    else : 
        flux_file = flux_model_tel_path / f'ExpectedOpenSkyFlux_{conf[0]}p_{model}.txt'
        flux = np.loadtxt(flux_file, delimiter="\t")
    os_flux[conf]  = flux

fm = FluxModel(altitude=tel.utm[-1])
    
if model == "corsika_soufriere":
    cors_path =  flux_model_path / "corsika" / "soufriere" / "muons" / "032023"
    if not cors_path.exists() : raise ValueError("Check path corsika flux.")
    mat_newfile_corsika = cors_path / "032023" / "muonFlux_laSoufriere.mat"
    struct_corsika = sio.loadmat(str(mat_newfile_corsika))
    energy_bins = np.logspace(-0.9500,5.9500, 70)
    theta_bins = np.linspace(2.5,87.5,18)
    corsika_flux_mean = struct_corsika['muonFlux']['diffFlux_mean'][0][0]
    corsika_flux_std = struct_corsika['muonFlux']['diffFlux_std'][0][0]
    fm = FluxModel(altitude=1140., corsika_flux=corsika_flux_mean, corsika_std=corsika_flux_std, energy_bins=energy_bins, theta_bins=theta_bins*np.pi/180)

        
print(f"Open-sky flux used for experimental acceptance computation from '{model}'")
#####
#####Pixel detector angular coordinates matrix contained 'acqVars.mat' files
acqVarDir = main_path / 'files' / 'telescopes' / tel.name / "acqvars" / f"az{tel.azimuth}ze{tel.zenith}"

acqVars = AcqVars(telescope=tel, 
                    dir=acqVarDir,
                    tomo=True)
az_calib  = acqVars.az_os
AZ_CALIB = acqVars.AZ_OS_MESH
az_tomo   = acqVars.az_tomo
ze_calib   = acqVars.ze_os
ZE_CALIB = acqVars.ZE_OS_MESH
ze_tomo   = acqVars.ze_tomo

print(f"azimuth{sconfig[0]}: phi in [{np.min(az_tomo[sconfig[0]]):.2f}, {np.max(az_tomo[sconfig[0]]):.2f}]\nzenith{sconfig[0]}: theta in [{np.min(ze_tomo[sconfig[0]]):.2f}, {np.max(ze_tomo[sconfig[0]]):.2f}]°")
if len(sconfig)>1 : print(f"azimuth{sconfig[-1]}: phi in [{np.min(az_tomo[sconfig[-1]]):.2f}°, {np.max(az_tomo[sconfig[-1]]):.2f}]\nzenith{sconfig[-1]}: theta in [{np.min(ze_tomo[sconfig[-1]]):.2f}, {np.max(ze_tomo[sconfig[-1]]):.2f}]°")
print(f"load acqVars --- {(time.time() - start_time):.3f}  s ---")              
########
print("ACCEPTANCE...")
accept_path = main_path / 'files' / "telescopes" / tel.name / 'acceptance' 
laccth =[]
if len(tel.panels) == 3 : 
    acc_th_120 = np.loadtxt(str(accept_path /  "A_theo_32x32_120cm.txt")) #integrated acceptance
    laccth.append(acc_th_120)
elif len(tel.panels) == 4 :
    acc_th_120 = np.loadtxt(str(accept_path / "A_theo_16x16_120cm.txt"))
    acc_th_180 = np.loadtxt(str(accept_path / "A_theo_16x16_180cm.txt"))
    laccth.extend([acc_th_120, acc_th_120,acc_th_180])
else: raise ValueError('Unknown telescope configuration')
acc_th = { sconfig[i]: a for i,a in enumerate(laccth) }

abCal = AnaBase(recofile=rD_cal, label=label_calib, tlim = None )

####GOF calib
sigma = 50#mm
dfCal = abCal.df
res = dfCal['quadsumres']/sigma**2
ndf = dfCal['npts']-2
gof = res/ndf
dfCal['rchi2'] = gof
if tel.name =="BR" : 
    #####
    tlim_cal1 = ( int(datetime(2017, 4, 2, hour=00,minute=00,second=00).replace(tzinfo=timezone.utc).timestamp()) , 
    int(datetime(2017, 4, 2, hour=16,minute=00,second=00).replace(tzinfo=timezone.utc).timestamp())   )
    tlim_cal2 = ( int(datetime(2017, 4, 3, hour=00,minute=00,second=00).replace(tzinfo=timezone.utc).timestamp()) , 
    int(datetime(2017, 4, 3, hour=18,minute=00,second=00).replace(tzinfo=timezone.utc).timestamp())   )
    t =  abCal.df['timestamp_s']
    cut_time_br = (( tlim_cal1[0] < t ) &  ( t < tlim_cal1[1] )) | (( tlim_cal2[0] < t ) &  ( t < tlim_cal2[1] )  )
    abCal.df = abCal.df[cut_time_br]
    
if is_filter_chi2 :
    abCal.df = cut(dfCal)
#####

hmCal = AnaHitMap(anabase=abCal, input_type=input_type, panels=tel.panels, dict_filter=None) 
#print(f"hmCal.idx['3p1'] = {hmCal.idx['3p1']}")

####CHARGE GOLD EVENTS
fn_out = "evtID_filter.csv.gz"
fout_filter_tomo = tomoDir / fn_out
fout_filter_cal = calibDir / fn_out    
levtID_filter_cal, levtID_filter_tomo = [], []

#if len(dfCal[dfCal["gold"]==1]) != 0: 
#####Gold charge calib
q_dir = calibDir / "charge"
(q_dir/"gold").mkdir(parents=True, exist_ok=True)
anaChargeGold_cal = AnaCharge(inlier_file=iD_cal,
                        outlier_file=oD_cal,
                        evttype=EventType.GOLD,
                        outdir=str(q_dir/"gold"), 
                        label=label_calib,
                    ) 

qcal_file = q_dir / "gold" / "fit_dQ_Inlier.csv" 
do_fit = True
#if qcal_file.exists() : do_fit = False
###compute calibration C_charge/mip constante
dict_charge_gold = {('Inlier', 'green', do_fit) :anaChargeGold_cal.ADC_XY_inlier, ('Outlier', 'red', False):anaChargeGold_cal.ADC_XY_outlier}
#plot charge distribution per panel
if len(dict_charge_gold[('Inlier', 'green', do_fit)] ) != 0 :    anaChargeGold_cal.plot_charge_panels(charge=dict_charge_gold, is_scaling=False, input_type=input_type.name)  
#################


hmDXDY = hmCal.hDXDY
#####
dict_filter = {conf : [] for conf in sconfig}
'''
if is_filter_charge : 
    print("Filter charge calib")
    path_calib_mip = q_dir / "gold"
    df_percentiles = pd.read_csv(str(path_calib_mip / "percentiles_dQ_Inlier.csv"), index_col=0, delimiter="\t")
    perc = 5
    dict_cuts = { pan.ID : df_percentiles.loc[perc][f"panel_{pan.ID}"] for pan in tel.panels }
    print("Apply filter")
    dq_gold_cal = {pan.ID : dq for pan, dq in zip(tel.panels, anaChargeGold_cal.ADC_XY_inlier)}
    ###filter calib data
    filter_charge_cal = FilterCharge(telescope=tel, inlier_data=iD_cal, outlier_data=oD_cal, dict_cuts=dict_cuts)
    levtID_filter_cal = filter_charge_cal.filter()
    with gzip.open(str(fout_filter_cal), "w" ) as f : 
        f.write("\n".join([str(i) for i in levtID_filter_cal]).encode('utf-8'))
        print(f"save {str(fout_filter_cal)}")
    hmCal_filter = AnaHitMap(anabase=abCal, input_type=input_type, panels=tel.panels, dict_filter=dict_filter) 
    hmDXDY = hmCal_filter.hDXDY
'''
if is_filter_multiplicity: 
    print("Filter multiplicity calib")
    for conf, panels in tel.configurations.items():
        front = panels[0]
        rear  = panels[-1]
        xpos = front.position.loc
        ypos = rear.position.loc
        idx_conf = hmCal.idx[conf]
        df_front = dict_df_multi_cal[xpos].loc[idx_conf]  #index 'evtID'
        df_rear  = dict_df_multi_cal[ypos].loc[idx_conf]  #index 'evtID'
        idx_front  = df_front[df_front < cut_m_front].index
        idx_rear   = df_rear [df_rear < cut_m_rear].index #
        idx_front_rear = list(set(idx_rear).intersection(set(idx_front)))
        dict_filter[conf].extend(idx_front_rear)
        dict_filter[conf] = list(set(dict_filter[conf]))
    
    hmCal_filter = AnaHitMap(anabase=abCal, input_type=input_type, panels=tel.panels, dict_filter=dict_filter) 
    hmDXDY = hmCal_filter.hDXDY

if is_filter_tof: 
    print("Filter tof calib")
    idx = df_tof_ns_cal[~cut_ns_cal].index.get_level_values(0)
    for conf, panels in tel.configurations.items():
        dict_filter[conf].extend(idx)
        dict_filter[conf] = list(set(dict_filter[conf]))
        
    hmCal_filter = AnaHitMap(anabase=abCal, input_type=input_type, panels=tel.panels, dict_filter=dict_filter) 
    hmDXDY = hmCal_filter.hDXDY
    
##remove duplicated indexes
dict_filter = {k: list(set(v)) for k, v in dict_filter.items()}


hm_dir_cal = calibDir / "hitmap"
hm_dir_cal.mkdir(parents=True, exist_ok=True)
hmFiles_cal = {c : hm_dir_cal/f"hitmap_{c}.txt" for c in sconfig}
for c, hm in hmDXDY.items(): np.savetxt(hmFiles_cal[c], hm, delimiter='\t', fmt='%.5e')

print(f"apply filter calib --- {(time.time() - start_time):.3f}  s ---") 

print("End cell #2")
#%%
####Plot evt rate w/o and with filter
rateDir = calibDir / "event_rate"
rateDir.mkdir(parents=True, exist_ok=True)
width = 3600 #s
ftraw = input_calib / "traw.csv.gz"
fig, ax = plt.subplots(figsize=(16,9))
if ftraw.exists():
    print("Raw event rate")
    dftraw = pd.read_csv(ftraw, index_col=0, delimiter="\t")
    print("dftraw = ",dftraw.head)
    try: 
        traw = dftraw["timestamp_s"]
    except: 
        traw = list(dftraw.index)
    ntimebins =  int((np.nanmax(traw) - np.nanmin(traw)) / width)
    (nevt_raw, dtbin, patches) = ax.hist(traw, bins=ntimebins, edgecolor='None', alpha=0.5, label=f"raw\nnevts={len(traw):1.3e}")
    ftraw_out = rateDir / "event_rate_raw"
    dtbinc =  (dtbin[1:] + dtbin[:-1])/2
    header ="tbin_center\tnevts\t(tbin_width=3600s)"
    mat_traw  = np.vstack((dtbinc,nevt_raw)).T
    np.savetxt(f"{str(ftraw_out)}.txt", mat_traw, delimiter="\t", header=header, fmt="%.0f")
    print(f"save {str(ftraw_out)}.txt")

fout = rateDir / "event_rate.png"
evtrateCal = EvtRate(df=abCal.df)
label = "all"
evtrateCal(ax, width=3600, label=label)
ax.legend(loc='best')
plt.savefig(
    str(fout), 
    transparent=True
    
)
plt.close()
print(f"save {str(fout)}")
print("End cell #3")

#%%

finfo  = calibDir / f"info.json"
run_info = {}
nevts = len(abCal.df)
run_info['Nevts'] = nevts
run_info['run_start'] = str(evtrateCal.start) #datetime
run_info['run_end'] = str(evtrateCal.end) #datetime
run_info['run_duration'] = float(f"{evtrateCal.run_duration:.1f}") #s
run_info['mean_evtrate'] = float(f"{evtrateCal.mean/3600:.3f}") #evt.s^-1
run_info['std_evtrate'] = float(f"{evtrateCal.std/3600:.3f}") #evt.s^-1
outstr=json.dumps(run_info)
with open(finfo, 'w') as f: 
    f.write(outstr)


###Event rate per telescope config
for conf in sconfig:
    fig, ax = plt.subplots(figsize=(16,9))
    ###select index corresponding to configuration 'conf'
    idx_conf = hmCal.idx[conf] #multi-index '(evtID, timestamp_s)'
    print(f"hmCal.idx[conf] = {idx_conf}")
    df = abCal.df.loc[idx_conf]
    print(f"abCal.df.index = {abCal.df.index}")
    er = EvtRate(df)
    width = 3600 #s
    er(ax, width=width, label=f"{conf}: all")
    dtbinc =  (er.dtbin[1:] + er.dtbin[:-1])/2
    mat_rate  = np.vstack((dtbinc,er.nevt)).T
    if len(dict_filter[conf]) != 0: 
        df_filter = df[df.index.isin(dict_filter[conf])]
        er_filter = EvtRate(df_filter)
        er_filter(ax, width=width, label=f"{conf}: filter")
        mat_rate = np.vstack((mat_rate.T, er_filter.nevt)).T
    frate_out = rateDir / f"event_rate_{conf}"
    header ="tbin_center\tnevts\tnevts_filter\t(tbin_width=3600s)"
    np.savetxt(f"{str(frate_out)}.txt", mat_rate, delimiter="\t", header=header, fmt="%.0f")
    ax.legend(loc='best')
    plt.savefig(
        f"{str(frate_out)}.png", transparent=True
    )
    plt.close()
    print(f"save {str(frate_out)}.txt")
    print(f"save {str(frate_out)}.png")
    
    if is_filter_multiplicity:
        finfo  = calibDir / f"info_{conf}.json"
        run_info = {}
        nevts, nevts_filter = len(df), len(df_filter)
        run_info['conf'] = conf
        run_info['Nevts_tot'] = nevts
        run_info['Nevts_filter'] = nevts_filter
        run_info['run_start'] = str(er.start) #datetime
        run_info['run_end'] = str(er.end) #datetime
        run_info['run_duration'] = float(f"{er.run_duration:.1f}") #s
        run_info['mean_evtrate'] = float(f"{er.mean/width:.3f}") #evt.s^-1
        run_info['std_evtrate'] = float(f"{er.std/width:.3f}") #evt.s^-1
        run_info['mean_evtrate_filter'] = float(f"{er_filter.mean/width:.3f}") #evt.s^-1
        run_info['std_evtrate_filter'] = float(f"{er_filter.std/width:.3f}") #evt.s^-1
        outstr=json.dumps(run_info)
        with open(finfo, 'w') as f: 
            f.write(outstr)

fCalib = { pan.ID: 1 for pan in tel.panels}
unc_fCalib = { pan.ID: 0 for pan in tel.panels}
"""
if len(rD_tomo.df[rD_tomo.df["gold"]==1]) != 0: 
    q_dir_tomo = tomoDir / "charge"  
    (q_dir_tomo/"gold").mkdir(parents=True, exist_ok=True)
    anaChargeGold_tomo = AnaCharge(inlier_file=iD_tomo,
                                    outlier_file=oD_tomo,
                                    evttype=EventType.GOLD,
                                    outdir=str(q_dir_tomo/"gold"), 
                                    label=label_tomo,
                                    ) 
    dict_charge = {('Inlier', 'orange',False):anaChargeGold_tomo.ADC_XY_inlier, ('Outlier', 'red', False):anaChargeGold_tomo.ADC_XY_outlier}
    if qcal_file.exists(): 
        ###Load calibration constante
        dfCalib = pd.read_csv(str(qcal_file), delimiter="\t", index_col=[0], header=[0, 1], skipinitialspace=True) #read multi cols 
        fCalib = { pan.ID: dfCalib.loc[pan.ID]['MPV']['value'] for pan in tel.panels}
        unc_fCalib = { pan.ID: dfCalib.loc[pan.ID]['MPV']['error'] for pan in tel.panels}
    anaChargeGold_tomo.plot_charge_panels(charge=dict_charge, fcal=fCalib, unc_fcal=unc_fCalib, input_type=input_type.name )
    if is_filter_charge : 
        print("FILTER charge TOMO")
        '''
        if fout_filter_tomo.exists() and fout_filter_cal.exists(): 
            print("Load filtered evt ids")
            ftomo= pd.read_csv(str(fout_filter_tomo), compression="gzip",  dtype=int) 
            levtID_filter_tomo = ftomo.values.T.flatten()
        else :
        '''
        print("Apply filter")
        dq_gold_tomo = {pan.ID : dq for pan, dq in zip(tel.panels,anaChargeGold_tomo.ADC_XY_inlier)}
        ###filter calib data
        filter_charge_tomo = FilterCharge(telescope=tel, inlier_data=iD_tomo, outlier_data=oD_tomo, dict_cuts=dict_cuts)
        levtID_filter_tomo = filter_charge_tomo.filter()
        print(f"levtID_filter_tomo = {len(levtID_filter_tomo)}")
        with gzip.open(str(fout_filter_tomo), "w" ) as f : 
            f.write("\n".join([str(i) for i in levtID_filter_tomo]).encode('utf-8'))
            print(f"save {str(fout_filter_tomo)}")
        
        df_filter = rD_tomo.df.loc[levtID_filter_tomo]
"""
#########
#########
acc_dir = calibDir / "acceptance" / model
accFiles= {c : acc_dir/f"acceptance_{c}.txt" for c in sconfig}
uaccFiles = {c : acc_dir/ f"unc_acc_{c}.txt" for c in sconfig} 
print(list(accFiles.values()))


print("Compute acceptance..")
acc_dir.mkdir(parents=True, exist_ok=True)
pl = PlotHitMap(hitmaps=[hmCal], outdir=str(hm_dir_cal))
pl.XY_map()
pl.DXDY_map()
hmCal.hDXDY = {conf: h/eff[conf] for conf, h in hmCal.hDXDY.items()}
A = Acceptance(hitmap=hmCal,
            evttype=EventType.MAIN,
                            outdir=acc_dir, 
                            opensky_flux=os_flux,
                            theoric=acc_th)
acceptance = {conf: v for conf,v in A.acceptance.items()}
unc_acc = {conf: v for conf,v in A.unc.items()}
for (conf, aexp), (_, ath), (_,AZ), (_,ZE) in zip(acceptance.items(), acc_th.items(), AZ_CALIB.items(), ZE_CALIB.items()): 
    A.plot_acceptance_3D(acc_exp=aexp, acc_th=ath, AZ=AZ, ZE=ZE, label=conf)
    A.plot_ratio_acc(acc_exp=aexp, acc_th=ath, az=az_calib[conf], ze=ze_calib[conf], label=conf)

print(f"acceptance --- {(time.time() - start_time):.3f}  s ---") 
print(f"end CALIB")


########




#ts_max = int(datetime(2019, 8, 31, hour=23,minute=59,second=59).replace(tzinfo=timezone.utc).timestamp())
#tlim = (0, ts_max) 

abTomo = AnaBase(recofile=rD_tomo,
                label=label_tomo,
                tlim=None) 


if tel.name =="SB" : 
    #####
    tlim1 = ( int(datetime(2016, 9, 27, hour=00,minute=00,second=00).replace(tzinfo=timezone.utc).timestamp()) , 
    int(datetime(2017, 2, 7, hour=16,minute=00,second=00).replace(tzinfo=timezone.utc).timestamp())   )
    # tlim_cal2 = ( int(datetime(2017, 2, 1, hour=00,minute=00,second=00).replace(tzinfo=timezone.utc).timestamp()) , 
    # int(datetime(2017, 4, 3, hour=18,minute=00,second=00).replace(tzinfo=timezone.utc).timestamp())   )
    t =  abTomo.df['timestamp_s']
    cut_time_sb = (( tlim1[0] < t ) &  ( t < tlim1[1] ))# | (( tlim_cal2[0] < t ) &  ( t < tlim_cal2[1] )  )
    abTomo.df = abTomo.df[cut_time_sb]

if 'quadsumres' in list(abTomo.df.columns) :
    df = abTomo.df
    res = df['quadsumres']/sigma**2
    ndf = df['npts']-2
    gof = res/ndf
    df['rchi2'] = gof    
    #cut_gof = (0.<gof) & (gof < chi2_max) 
    #####GoF
    # fig = plt.figure(constrained_layout=True)
    # ax = fig.subplot_mosaic([["main", "gold"]])#, sharey=True)
    # GoF(ax=ax["main"], df=df, is_gold=True, column='rchi2', bins=100, range=[0,10])
    # gold = df[df["gold"] == 1]
    # GoF(ax=ax["gold"], df=gold, column='rchi2', bins=100, range=[0,10], color="orange")
    # ax["gold"].set_ylabel('')
    # plt.savefig(str(tomoDir /f"gof.png"))
    # plt.close()
    # print("ngold_tomo=", len(df[df["gold"]==1]))


if is_filter_chi2 :
    abTomo.df = cut_chi2(df)

######TOMO DATASET ANALYSIS
topo = acqVars.topography 

thickness = acqVars.thickness
mask = { c:( np.isnan(thickness[c]) ) for c in sconfig }

###Hitmaps tomo

hmTomo = AnaHitMap(anabase=abTomo, 
                    input_type=input_type, 
                    panels=tel.panels, 
                    dict_filter=None)
hmDXDY = hmTomo.hDXDY





'''
GoF_inlier_outlier(abTomo.df, str(tomoDir /f"gof_inlier_outlier_all.png"))
evt_3p = abTomo.df[abTomo.df["nimp"] == 3]
evt_4p = abTomo.df[abTomo.df["nimp"] == 4]
GoF_inlier_outlier(evt_3p, str(tomoDir /f"gof_inlier_outlier_3p.png"))
if len(evt_4p) != 0:  GoF_inlier_outlier(evt_4p, str(tomoDir /f"gof_inlier_outlier_4p.png"))
'''


####PREPARE TOMOGRAPHY DATASET
##filters
dict_filter = {conf : [] for conf in sconfig}
'''
if is_filter_charge: 
    dict_charge_filter = {}
    for conf, _ in tel.configurations.items():    
        dict_charge_filter[conf] = filter_charge_tomo.filter(conf)
    dict_filter = dict_charge_filter
else : dict_charge_filter=None
'''
###....


####APPLY filter

if is_filter_multiplicity: 
    print("Filter multiplicity tomo")
    for conf, panels in tel.configurations.items():
        front = panels[0]
        rear  = panels[-1]
        xpos = front.position.loc
        ypos = rear.position.loc
        idx_conf = hmTomo.idx[conf]
        df_front = dict_df_multi_tomo[xpos].loc[idx_conf]  #index 'evtID'
        df_rear  = dict_df_multi_tomo[ypos].loc[idx_conf]  #index 'evtID'
        idx_front  = df_front[df_front < cut_m_front].index
        idx_rear   = df_rear [df_rear < cut_m_rear].index #
        idx_front_rear = list(set(idx_rear).intersection(set(idx_front)))
        dict_filter[conf].extend(idx_front_rear)
        dict_filter[conf] = list(set(dict_filter[conf]))

    hmTomo_filter = AnaHitMap(anabase=abTomo, input_type=input_type, panels=tel.panels, dict_filter=dict_filter) 
    hmDXDY = hmTomo_filter.hDXDY
    
if is_filter_tof: 
    print("Filter tof tomo")
    idx = df_tof_ns_tomo[~cut_ns_tomo].index.get_level_values(0)
    for conf, panels in tel.configurations.items():
        dict_filter[conf].extend(idx)
        dict_filter[conf] = list(set(dict_filter[conf]))
        
    hmTomo_filter = AnaHitMap(anabase=abTomo, input_type=input_type, panels=tel.panels, dict_filter=dict_filter) 
    hmDXDY = hmTomo_filter.hDXDY


##remove duplicated indexes
dict_filter = {k: list(set(v)) for k, v in dict_filter.items()}

    
hm_dir = tomoDir / "hitmap"
hm_dir.mkdir(parents=True, exist_ok=True)
hmFiles= {c : hm_dir/f"hitmap_{c}.txt" for c in sconfig}
hm_dir.mkdir(parents=True,exist_ok=True)
for c, hm in hmDXDY.items(): np.savetxt(hmFiles[c], hm, delimiter='\t', fmt='%.5e')
pl = PlotHitMap(hitmaps=[hmTomo], outdir=str(hm_dir))
flipud = False
if tel.name =="OM":  flipud = True  
pl.DXDY_map(flipud=flipud)

print(f"hit maps tomo --- {(time.time() - start_time):.3f}  s ---") 


print(f"apply filter tomo --- {(time.time() - start_time):.3f}  s ---") 



print("Event rate...")
rateDir = tomoDir/ "event_rate" 
rateDir.mkdir(parents=True, exist_ok=True)
width = 3600 #s
dftraw = pd.DataFrame()
if isinstance(input_tomo, list):
    ldf = []
    for f in input_tomo:
        ftraw = f / "traw.csv.gz"
        if ftraw.exists(): ldf.append(pd.read_csv(f, delimiter="\t", index_col=0))
    dftraw = pd.concat(ldf)
else : 
    ftraw = input_tomo / "traw.csv.gz"
    if ftraw.exists():
        print("Raw event rate")
        dftraw = pd.read_csv(ftraw, index_col=0, delimiter="\t")
    
if len(dftraw)!=0:
    print("dftraw = ",dftraw.head)
    try: 
        traw = dftraw["timestamp_s"]
    except: 
        traw = list(dftraw.index)
    ntimebins =  int((np.nanmax(traw) - np.nanmin(traw)) / width)
    (nevt_raw, dtbin, patches) = ax.hist(traw, bins=ntimebins, edgecolor='None', alpha=0.5, label=f"raw\nnevts={len(traw):1.3e}")
    ftraw_out = rateDir / "event_rate_raw"
    dtbinc =  (dtbin[1:] + dtbin[:-1])/2
    header ="tbin_center\tnevts\t(tbin_width=3600s)"
    mat_traw  = np.vstack((dtbinc,nevt_raw)).T
    np.savetxt(f"{str(ftraw_out)}.txt", mat_traw, delimiter="\t", header=header, fmt="%.0f")
    print(f"save {str(ftraw_out)}.txt")
finfo  = tomoDir / f"info.json"
run_info = {}
evtrateTomo=EvtRate(abTomo.df)
fig, ax = plt.subplots(figsize=(16,9))
evtrateTomo(ax, width=width, label="all")

ax.legend(loc='best')
plt.savefig(
    str(rateDir / f"event_rate.png"), 
    transparent=True
)
plt.close()
run_info['Nevts_tot'] = str(evtrateTomo.nevt_tot)
run_info['run_start'] = str(evtrateTomo.start) #s
run_info['run_end'] = str(evtrateTomo.end) #s
run_info['run_duration'] = float(f"{evtrateTomo.run_duration:.1f}") #s
run_info['mean_evtrate'] = float(f"{evtrateTomo.mean/3600:.3f}") #evt.s^-1
run_info['std_evtrate'] = float(f"{evtrateTomo.std/3600:.3f}") #evt.s^-1
outstr=json.dumps(run_info)

with open(finfo, 'w') as f: 
    f.write(outstr)



###Event rate per telescope config
for conf in sconfig:
    fig, ax = plt.subplots(figsize=(16,9))
    ###select index corresponding to configuration 'conf'
    idx_conf = hmTomo.idx[conf] #multi-index '(evtID, timestamp_s)'
    df = abTomo.df.loc[idx_conf]
    er = EvtRate(df)
    width = 3600 #s
    er(ax, width=width, label=f"{conf}: all")
    frate_out = rateDir / f"event_rate_{conf}"
    dtbinc =  (er.dtbin[1:] + er.dtbin[:-1])/2
    header ="tbin_center\tnevts\tnevts_filter\t(tbin_width=3600s)"
    mat_rate  = np.vstack((dtbinc,er.nevt)).T
    if len(dict_filter[conf]) != 0: 
        #df_filter = df.loc[dict_filter[conf]]
        df_filter = df[df.index.isin(dict_filter[conf])]
        er_filter = EvtRate(df_filter)
        er_filter(ax, width=width, label=f"{conf}: filter")
        mat_rate = np.vstack((mat_rate.T, er_filter.nevt)).T
    np.savetxt(f"{str(frate_out)}.txt", mat_rate, delimiter="\t", header=header, fmt="%.0f")
    ax.legend(loc='best')
    plt.savefig(
        f"{str(frate_out)}.png", transparent=True
    )
    plt.close()
    print(f"save {str(frate_out)}.txt")
    print(f"save {str(frate_out)}.png")
    if is_filter_multiplicity:
        finfo  = tomoDir / f"info_{conf}.json"
        run_info_conf = {}
        nevts, nevts_filter = len(df), len(df_filter)
        run_info_conf['conf'] = conf
        run_info_conf['Nevts_tot'] = nevts
        run_info_conf['Nevts_filter'] = nevts_filter
        run_info_conf['run_start'] = str(er.start) #datetime
        run_info_conf['run_end'] = str(er.end) #datetime
        run_info_conf['run_duration'] = float(f"{er.run_duration:.1f}") #s
        run_info_conf['mean_evtrate'] = float(f"{er.mean/width:.3f}") #evt.s^-1
        run_info_conf['std_evtrate'] = float(f"{er.std/width:.3f}") #evt.s^-1
        run_info_conf['mean_evtrate_filter'] = float(f"{er_filter.mean/width:.3f}") #evt.s^-1
        run_info_conf['std_evtrate_filter'] = float(f"{er_filter.std/width:.3f}") #evt.s^-1
        outstr=json.dumps(run_info_conf)
        with open(finfo, 'w') as f: 
            f.write(outstr)




print("TOMO --- \n")



####test with geometric acceptance 
#acceptance = acc_th

if tel.name =="OM": 
    for k,v in hmDXDY.items(): hmDXDY[k] = np.flipud(v)

tomo = Muography(
            telescope = tel,
            hitmap = hmDXDY, 
            label = label_tomo,
            outdir = tomoDir,
            acceptance = acceptance,
            mask = mask,
            topography=topo,
            info = run_info
        )

####load reconstruction efficiency
print("Compute transmitted flux [cm^-2.sr^-1.s^-1]")
flux_data_path = tomoDir / "flux"
flux_data_path.mkdir(parents=True, exist_ok=True)
fluxFiles= {c : flux_data_path/f"flux_{c}.txt" for c in sconfig}
#fluxFiles = {c : glob.glob(str(flux_data_path / f"flux*{c}")) for c in sconfig}
#ufluxFiles = {c : f for c, f in zip(sconfig,glob.glob( str(hm_dir/f"unc_flux*{c}*") ))} #glob.glob(str(flux_data_path / "unc_flux*"))
ufluxFiles= {c : flux_data_path/f"unc_flux_{c}.txt" for c in sconfig}


'''
if all(f.exists() for _,f in fluxFiles.items()) :
    flux_tomo = {c : np.loadtxt(f) for c,f in fluxFiles.items()}
    uflux_tomo = {c : np.loadtxt(f) for c,f in ufluxFiles.items()}
else : 
'''



tomo.compute_flux( unc_acc=unc_acc, efficiency=eff, unc_eff=unc_eff)
flux_tomo = tomo.flux
uflux_tomo = tomo.unc_flux

flux_range = (1e-7, 3e-2) 
#flux_range = (1e-8, 1e-1) 
tomo.plot_flux(flux=flux_tomo, range=flux_range, az=az_tomo, ze=ze_tomo, topography=None, colorbar=True, sigma=(0.5, 0.5))
####
eff_os = {conf:np.ones(shape=(acqVars.az_tomo[conf].shape[1], 4)) for conf in sconfig}

#if model == "guan":
print("Correction with open-sky portion in tomo run")

arr_ze =  np.linspace(0, 90.5, 100)
flux_model = np.ones(len(arr_ze))

'''
flux_tomo_corr = {conf:np.zeros(shape=flux_tomo[conf].shape) for conf in sconfig}
uflux_tomo_corr = {conf:np.zeros(shape=flux_tomo[conf].shape) for conf in sconfig}
for z,ze in enumerate(arr_ze): 
    flux_model[z]  = fm.ComputeOpenSkyFlux(ze*np.pi/180, emin=0.2, model=model)
xmodel, guan = arr_ze, flux_model
###efficiency computed from open-sky portion of tomo set, to correct model bias
for conf in sconfig: 
    #print(conf)
    z, a = np.copy(acqVars.ze_tomo[conf]), np.copy(acqVars.az_tomo[conf])
    sel_sky = np.isnan(acqVars.thickness[conf]) & (z < 65)
    #az_c = np.median((a[:,0] + a[:,-1])/2)   
    #laz_c = a[0,:-1] + a[0,1:]/2
    mat_eff = np.ones(shape=(acqVars.az_tomo[conf].shape[1], 5))
    for ix, phi in enumerate(a[0,:]): 
        #ix_az = np.argmin(abs(a[0,:]-az_c))
        s = sel_sky[:,ix]
        f, u = copy.deepcopy(flux_tomo[conf][:,ix]), copy.deepcopy(uflux_tomo[conf][:,ix])
        z_sky = z[:,ix]
        z_sky[~s] = np.nan
        f[~s], u[~s]= np.nan, np.nan
        #print("z_sky = ",z_sky.shape)
        fmod  = np.array([ flux_model[np.argmin(abs(z-arr_ze))] for z in z_sky])
        #print(f"fmod_{conf} = {fmod}, {fmod.shape}")
        #print(f"f_{conf} = {f}, {f.shape}")
        #print(f"z_sky_{conf} = {z_sky}, {z_sky.shape}")
        err = f / fmod
        #print(err, err.shape)
        mean_eff = np.nanmean(err)
        std_eff = np.nanstd(err)
        min_eff = np.nanmin(err)
        max_eff = np.nanmax(err)
        eff_os[conf][ix,:] = np.array([mean_eff, std_eff,min_eff, max_eff]) 
        mat_eff[ix,:] = np.vstack((phi, mean_eff,std_eff, min_eff,max_eff)).T    
        flux_tomo_corr[conf][:,ix] = flux_tomo[conf][:,ix] / mean_eff
        uflux_tomo_corr[conf][:,ix] = np.sqrt((uflux_tomo[conf][:,ix]/mean_eff)**2 + (std_eff/mean_eff**2 * f)**2 )
        
        if ix == 10:
            fig, axt =plt.subplots(figsize=(12,9))
            x = z_sky
            xc = (x[:-1] + x[1:]) /2
            xw = abs(x[:-1]-x[1:])
            xerr = xw/2
            ydat, yunc = f, u
            ydatc, yerr = (ydat[:-1] + ydat[1:]) /2, (yunc[:-1] + yunc[1:]) /2
            axt.errorbar(xc, ydatc, xerr=xerr, yerr=yerr, fmt='o', linestyle='none',label=f"{tel.name} data", capsize=5, color=tel.color, fillstyle='none')
            ymod  = np.array([ flux_model[np.argmin(abs(x-arr_ze))] for x in xc]) 
            err = ydatc / ymod
            mean_eff = np.nanmean(err)
            axt.errorbar(xc, ydatc/mean_eff, xerr=xerr, yerr=yerr, fmt='o', linestyle='none',label=f"{tel.name} corrected", capsize=5, color=tel.color, fillstyle='full')
            std_eff = np.nanstd(err)
            min_eff = np.nanmin(err)
            max_eff = np.nanmax(err)
            axt.set_xlim(40, 70)
            ymin = np.nanmin(ydatc)
            axt.set_ylim(ymin, 1e-2)
            unc_eff = std_eff
            yerr = np.sqrt((yerr/mean_eff)**2 + (std_eff/mean_eff**2 * ydatc)**2 )
            axt.set_title(f"az={phi:.3f}°")
            axt.legend(loc="best")
            axt.set_yscale("log")
            axt.plot(arr_ze, flux_model, color="black")
            fout = tomoDir / f"open_sky_flux_tomo_vs_model_{phi:.0f}.png"
            fig.savefig(str(fout))
            print(f"save {str(fout)}")
        
    
    fout_eff = tomoDir/f"eff_open_sky_{conf}.txt"
    np.savetxt(f"{fout_eff}", mat_eff, delimiter="\t", header="az\tmean\tstd\tmin\tmax", fmt="%.3f")
    print(f"save {fout_eff}")
tomo.plot_flux(flux=flux_tomo_corr, range=flux_range, az=az_tomo, ze=ze_tomo, topography=topo, colorbar=True, label="corrected", sigma=(0.5, 0.5))       
#exit()

fluxFiles= {c : flux_data_path/f"flux_{c}_corr.txt" for c in sconfig}
ufluxFiles= {c : flux_data_path/f"unc_flux_{c}_corr.txt" for c in sconfig}
for c in sconfig:
    np.savetxt(fluxFiles[c], flux_tomo_corr[c] , fmt='%.5e', delimiter='\t')
    np.savetxt(ufluxFiles[c], uflux_tomo_corr[c] , fmt='%.5e', delimiter='\t')
'''


# tomo.plot_quantities_vs_elevation( q1=flux_tomo, unc_q1=uflux_tomo, label1="mean flux $\\langle\\phi\\rangle$ [cm$^{-2}$.sr$^{-1}$.s$^{-1}$]", 
#                                   q2=None, unc_q2=None, label2=None,
#                                   ze=ze_tomo, mask=mask, outdir=str(Path(tomoDir)/"flux"))



# outdir =  flux_data_path / "interp"
# outdir.mkdir(parents=True, exist_ok=True)
# flux_tomo_interp= fill_empty_pixels(flux_tomo, az_tomo, ze_tomo, outdir=outdir, filename="flux", mask=tomo.mask)
# uflux_tomo_interp = fill_empty_pixels(tomo.unc_flux, az_tomo, ze_tomo, outdir=outdir, filename="unc_flux",mask=tomo.mask)
# tomo.plot_flux(flux=flux_tomo_interp, range=flux_range, az=az_tomo, ze=ze_tomo, topography=topo, colorbar=True, sigma=(0.5, 0.5),outdir=outdir)
#    exit()

medium, rho  = "rock", 2.65 
IntegralFluxVsOpAndZaStructure_Corsika = sio.loadmat( str(flux_model_path /  'corsika' / 'soufriere' / 'muons' / 'former' / 'IntegralFluxVsOpAndZaStructure_Corsika.mat')) #dictionary out of 100x600 matrix
simu_ze = IntegralFluxVsOpAndZaStructure_Corsika['IntegralFluxVsOpAndZaStructure_rock_MuonsEKin_onlyModel'][0][0][0] #zenith angle
simu_logopacity = IntegralFluxVsOpAndZaStructure_Corsika['IntegralFluxVsOpAndZaStructure_rock_MuonsEKin_onlyModel'][0][0][1] #opacity
simu_opacity = np.exp( np.log(10) * simu_logopacity )
if model == 'corsika' or model not in lmodel_avail: 
    simu_flux = IntegralFluxVsOpAndZaStructure_Corsika['IntegralFluxVsOpAndZaStructure_rock_MuonsEKin_onlyModel'][0][0][2] #Corsika flux
    simu_flux_low = IntegralFluxVsOpAndZaStructure_Corsika['IntegralFluxVsOpAndZaStructure_rock_MuonsEKin_onlyModel'][0][0][3]
    simu_flux_up = IntegralFluxVsOpAndZaStructure_Corsika['IntegralFluxVsOpAndZaStructure_rock_MuonsEKin_onlyModel'][0][0][4]
#####TEST with model flux instead of former CORSIKA
else : 
    flux_file = Path.out_path() / "cosmic_flux" / "flux_vs_opacity" / medium / str(rho) / model / "flux.txt"
    flux_low_file = Path.out_path() / "cosmic_flux" / "flux_vs_opacity" / medium / str(rho) / model / "flux_low.txt"
    flux_up_file = Path.out_path() / "cosmic_flux" / "flux_vs_opacity" / medium / str(rho) / model / "flux_up.txt"
    simu_ze = np.arange(0,90.5,0.5) #shape=(181,)
    simu_opacity = simu_opacity[0,:] #shape=(600,)
    simu_opacity, simu_ze = np.meshgrid(simu_opacity, simu_ze) 
    simu_flux = np.loadtxt(flux_file, delimiter=" ")
    simu_flux_low = np.loadtxt(flux_low_file, delimiter=" ")
    simu_flux_up = np.loadtxt(flux_up_file, delimiter=" ")

sig_model = (simu_flux_up.flatten() - simu_flux_low.flatten())/2
    
print(f"Interpolation with {model}")




#simu_flux *= fscale 
nz = (flux_tomo['3p1'] != 0)
print(f"(flux_min, flux_max)_data = ({np.nanmin(flux_tomo['3p1'][nz]):.3e}, {np.nanmax(flux_tomo['3p1']):.3e})  1/(cm^2.sr.s)")
print(f"(flux_min, flux_max)_{model} = ({np.nanmin(simu_flux):.3e}, {np.nanmax(simu_flux):.3e})  1/(cm^2.sr.s)")

tomo.plot_thickness(az=az_tomo,ze=ze_tomo, app_thick=thickness)

print("Interpolation Opacity [m.w.e]")

interp = "nearest" #nearest, linear or cubic 
op_dir = tomoDir / 'opacity' / model / interp
op_dir.mkdir(parents=True, exist_ok=True)
de_dir = tomoDir / 'density' / model / interp
de_dir.mkdir(parents=True, exist_ok=True)
'''
if all(f.exists() for _,f in opaFiles.items()) :
    opacity = { c:np.loadtxt(f) for c, f in opaFiles.items()}  
    unc_opacity = { c:np.loadtxt(f) for c, f in uopaFiles.items()}  
    density = { c:np.loadtxt(f) for c, f in denFiles.items()} 
    unc_density = { c:np.loadtxt(f) for c, f in udenFiles.items()} 
else : 
'''
tomo.interpolate_opacity(
                        range_ze=simu_ze,
                        range_flux=simu_flux,
                        sig_model = sig_model,
                        range_op=simu_opacity,
                        tomo_ze= ze_tomo, 
                        tomo_flux=flux_tomo,
                        app_thick=thickness,
                        method=interp
                        )

print(f"interpolate opacity --- {(time.time() - start_time):.3f}  s ---") 
opacity = tomo.opacity
density = tomo.density
opaFiles= {c : op_dir/f"opacity_{c}.txt" for c in sconfig}
uopaStatFiles = {c : op_dir/f"unc_opacity_stat_{c}.txt" for c in sconfig}
uopaSysFiles = {c : op_dir/f"unc_opacity_sys_{c}.txt" for c in sconfig}
uopaTotFiles = {c : op_dir/f"unc_opacity_tot_{c}.txt" for c in sconfig}
denFiles = {c : de_dir/f"mean_density_{c}.txt" for c in sconfig}
udenStatFiles = {c : de_dir/f"unc_mean_density_stat_{c}.txt" for c in sconfig}
udenSysFiles = {c : de_dir/f"unc_mean_density_sys_{c}.txt" for c in sconfig}
udenTotFiles = {c : de_dir/f"unc_mean_density_tot_{c}.txt" for c in sconfig}
for c in sconfig:
    np.savetxt(opaFiles[c], tomo.opacity[c] , fmt='%.5e', delimiter='\t')
    np.savetxt(uopaStatFiles[c], tomo.unc_opacity_stat[c] , fmt='%.5e', delimiter='\t')
    np.savetxt(uopaSysFiles[c], tomo.unc_opacity_sys[c] , fmt='%.5e', delimiter='\t')
    np.savetxt(uopaTotFiles[c], tomo.unc_opacity_tot[c] , fmt='%.5e', delimiter='\t')
    np.savetxt(denFiles[c], tomo.density[c] , fmt='%.5e', delimiter='\t')
    np.savetxt(udenStatFiles[c], tomo.unc_density_stat[c] , fmt='%.5e', delimiter='\t')
    np.savetxt(udenSysFiles[c], tomo.unc_density_sys[c] , fmt='%.5e', delimiter='\t')
    np.savetxt(udenTotFiles[c], tomo.unc_density_tot[c] , fmt='%.5e', delimiter='\t')



dump_mean_op=json.dumps({k: f"{np.mean(d[~np.isnan(d)]):.5f}" for k, d in opacity.items()})
dump_range_op = json.dumps({k: f"({np.min(d[~np.isnan(d)]):.5f}, {np.max(d[~np.isnan(d)]):.5f})" for k, d in opacity.items() })
with open(str(op_dir/"average_opacity.json"), 'w') as f: f.write(dump_mean_op)
with open(str(op_dir/"range_opacity.json"), 'w') as f: f.write(dump_range_op)

dump_mean_density=json.dumps({k: f"{np.nanmean(d):.5f}" for k, d in density.items()})
dump_median_density=json.dumps({k: f"{np.nanmedian(d):.5f}" for k, d in density.items()})
dump_std_density=json.dumps({k: f"{np.nanstd(d):.5f}" for k, d in density.items()})
dump_range_density = json.dumps({k: f"({np.nanmin(d):.5f}, {np.nanmax(d):.5f})" for k, d in density.items() })
with open(str(de_dir/"average_mean_density.json"), 'w') as f: f.write(dump_mean_density)
with open(str(de_dir/"median_mean_density.json"), 'w') as f: f.write(dump_median_density)
with open(str(de_dir/"std_mean_density.json"), 'w') as f: f.write(dump_std_density)
with open(str(de_dir/"range_mean_density.json"), 'w') as f: f.write(dump_range_density)

print(f"mean density --- {(time.time() - start_time):.3f}  s ---")




'''
####apply correction
tomo.interpolate_opacity(
                        range_ze=simu_ze,
                        range_flux=simu_flux,
                        sig_model = sig_model,
                        range_op=simu_opacity,
                        tomo_ze= ze_tomo, 
                        tomo_flux=flux_tomo_corr,
                        app_thick=thickness,
                        method=interp
                        )

opacity_corr=tomo.opacity
unc_opacity_corr = tomo.unc_opacity_tot
density_corr=tomo.density
unc_density_corr=tomo.unc_density_tot
opaFiles= {c : op_dir/f"opacity_{c}_corr.txt" for c in sconfig}
uopaStatFiles = {c : op_dir/f"unc_opacity_stat_{c}_corr.txt" for c in sconfig}
uopaSysFiles = {c : op_dir/f"unc_opacity_sys_{c}_corr.txt" for c in sconfig}
uopaTotFiles = {c : op_dir/f"unc_opacity_tot_{c}_corr.txt" for c in sconfig}
denFiles = {c : de_dir/f"mean_density_{c}_corr.txt" for c in sconfig}
udenStatFiles = {c : de_dir/f"unc_mean_density_stat_{c}_corr.txt" for c in sconfig}
udenSysFiles = {c : de_dir/f"unc_mean_density_sys_{c}_corr.txt" for c in sconfig}
udenTotFiles = {c : de_dir/f"unc_mean_density_tot_{c}_corr.txt" for c in sconfig}
for c in sconfig:
    np.savetxt(opaFiles[c], tomo.opacity[c] , fmt='%.5e', delimiter='\t')
    np.savetxt(uopaStatFiles[c], tomo.unc_opacity_stat[c] , fmt='%.5e', delimiter='\t')
    np.savetxt(uopaSysFiles[c], tomo.unc_opacity_sys[c] , fmt='%.5e', delimiter='\t')
    np.savetxt(uopaTotFiles[c], tomo.unc_opacity_tot[c] , fmt='%.5e', delimiter='\t')
    np.savetxt(denFiles[c], tomo.density[c] , fmt='%.5e', delimiter='\t')
    np.savetxt(udenStatFiles[c], tomo.unc_density_stat[c] , fmt='%.5e', delimiter='\t')
    np.savetxt(udenSysFiles[c], tomo.unc_density_sys[c] , fmt='%.5e', delimiter='\t')
    np.savetxt(udenTotFiles[c], tomo.unc_density_tot[c] , fmt='%.5e', delimiter='\t')

print(f"interpolate with correction --- {(time.time() - start_time):.3f}  s ---")     
'''


opa_range=(1e1, 3e3)#10e1/dthick) #MWE
sigma=(1.,1.)
tomo.plot_mean_density(quantity="opacity $\\varrho$ [m.w.e]",  
                    val=opacity, 
                    range=opa_range, 
                    az=az_tomo,
                    ze=ze_tomo, 
                    sigma=sigma, 
                    outdir=str(op_dir), 
                    label=label_tomo,
                    topography=topo,
                    lognorm=True)
'''
tomo.plot_mean_density(quantity="opacity $\\varrho$ [m.w.e]",  
                    val=opacity_corr, 
                    range=opa_range, 
                    az=az_tomo,
                    ze=ze_tomo, 
                    sigma=sigma, 
                    outdir=str(op_dir), 
                    label=label_tomo+'_corr',
                    topography=topo,
                    lognorm=True)
'''
print("DENSITYYYYYYY")
rho_min, rho_max =np.nanmin(np.array([np.nanmin(d[~np.isnan(d)]) for k, d in density.items()])), np.nanmax(np.array([np.nanmax(d[~np.isnan(d)]) for k, d in density.items()]))
rho_mean = np.nanmean(np.array([np.nanmean(d[~np.isnan(d)]) for k, d in density.items()]))
rho_median = np.array([np.nanmedian(d[~np.isnan(d)]) for k, d in density.items()])
rho_std = np.array([np.nanstd(d[~np.isnan(d)]) for k, d in density.items()])
print(f"(rho_min, rho_max)_data = ({rho_min:.3f}, {rho_max:.3f})  g/cm^3")
print(f"rho_median = {rho_median}  g/cm^3")
print(f"rho_std = {rho_std}  g/cm^3")
print(f"<rho>_data = {rho_mean:.3f} g/cm^3")
#rho_range=(0.8,2.0)
rho_range=(0.8,2.7)
cmap = palettable.scientific.sequential.Batlow_20.mpl_colormap
tomo.plot_mean_density(quantity="mean density $\\overline{\\rho}$ [g.cm$^{-3}$]",  
                        val=density, 
                        range=rho_range, 
                        az=az_tomo,
                        ze=ze_tomo,
                        sigma=sigma,
                        outdir=str(de_dir), 
                        label=label_tomo,   
                        topography=topo, 
                        mask=mask, 
                        cmap=cmap, 
                        threshold=3.5, 
                        crater=None
                        )
'''
####apply correction
rho_min, rho_max =np.nanmin(np.array([np.nanmin(d[~np.isnan(d)]) for k, d in density_corr.items()])), np.nanmax(np.array([np.nanmax(d[~np.isnan(d)]) for k, d in density_corr.items()]))
rho_mean = np.nanmean(np.array([np.nanmean(d[~np.isnan(d)]) for k, d in density_corr.items()]))
rho_median = np.array([np.nanmedian(d[~np.isnan(d)]) for k, d in density_corr.items()])
rho_std = np.array([np.nanstd(d[~np.isnan(d)]) for k, d in density_corr.items()])
print(f"(rho_min, rho_max)_data_corr = ({rho_min:.3f}, {rho_max:.3f})  g/cm^3")
print(f"rho_median_corr = {rho_median}  g/cm^3")
print(f"rho_std_corr = {rho_std}  g/cm^3")
print(f"<rho>_data_corr = {rho_mean:.3f} g/cm^3")
#rho_range=(0.8,2.0)
rho_range=(0.8,2.7)
cmap = palettable.scientific.sequential.Batlow_20.mpl_colormap
tomo.plot_mean_density(quantity="mean density $\\overline{\\rho}$ [g.cm$^{-3}$]",  
                        val=density_corr, 
                        range=rho_range, 
                        az=az_tomo,
                        ze=ze_tomo,
                        sigma=sigma,
                        outdir=str(de_dir), 
                        label=label_tomo+'_corr',  
                        topography=topo, 
                        mask=mask, 
                        cmap=cmap, 
                        threshold=3.5, 
                        crater=crater
                        )
'''


rho_0 = rho_mean
drho_range = [rho_mean-2*np.mean(rho_std), rho_mean+2*np.mean(rho_std)]

out_dens = (de_dir/f"rel_var/{rho_0:.2f}")
out_dens.mkdir(parents=True, exist_ok=True)
rel_density = {c: (rho-rho_0)/ rho_0 for c, rho in density.items()}
rho_min, rho_max =np.nanmin(np.array([np.nanmin(d[~np.isnan(d)]) for k, d in rel_density.items()])), np.nanmax(np.array([np.nanmax(d[~np.isnan(d)]) for k, d in rel_density.items()]))
rho_mean = np.nanmean(np.array([np.nanmean(d[~np.isnan(d)]) for k, d in rel_density.items()]))
rho_std = np.nanmean(np.array([np.nanstd(d[~np.isnan(d)]) for k, d in rel_density.items()]))
print(f"D(rho_min, rho_max)_data = ({rho_min:.3f}, {rho_max:.3f})")
print(f"<Drho_mean>_data = {rho_mean:.3f}")
drho_sc = 2*rho_std
#drho_range = [rho_mean-drho_sc, rho_mean+drho_sc]
#drho_range = [-0.7, 0.7]
drho_range = [-1., 1.]
tomo.plot_mean_density(quantity="($\\rho$ - $\\rho_0$) / $\\rho_0$",  val=rel_density, range=drho_range, 
                        az=az_tomo,
                        ze=ze_tomo, 
                        sigma=sigma, 
                        outdir=str(out_dens), 
                        label=label_tomo,   
                        topography=topo, 
                        mask=mask, 
                        cmap="jet", 
                        threshold=None,
                        crater=None,
                        ) 

#    tomo.plot_mean_density(quantity="mean density $\\overline{\\rho}$ [g.cm$^-3$]",  val=density_corr,outdir=str(de_dir/ 'corr'), label='w_corr_fact', range=rho_range_corr, az=az_tomo,ze=ze_tomo, sigma=sigma, topography=topo, mask=mask)
# rho_0 = {c :np.mean(v) for c,v in density.items()} #2.1 #g.cm^-3
# relvar = {c : (v-rho_0[c])/rho_0[c] for c,v in density.items()}
# drho_range=(-3,3)

#os.system(f"open {tomoDir}")

t_sec = round(time.time() - start_time)
(t_min, t_sec) = divmod(t_sec,60)
(t_hour,t_min) = divmod(t_min,60)  
print('runtime analysis: {}hour:{}min:{}sec'.format(t_hour,t_min,t_sec))
print(f"end --- {(time.time() - start_time):.3f}  s ---")