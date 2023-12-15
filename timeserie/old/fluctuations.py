#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from tkinter.ttk import LabeledScale
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.colors import LogNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.gridspec import GridSpec
import matplotlib.lines as mlines  #use for legend settings
import sys
import os
import scipy.io as sio
import scipy.ndimage
from scipy.interpolate import griddata
from scipy import interpolate
from pathlib import Path
import inspect
import argparse
import logging
import glob
import pandas as pd
import time
from datetime import datetime, timezone
import re
import yaml
import pickle
import warnings
warnings.filterwarnings("ignore")
#personal modules
filename = inspect.getframeinfo(inspect.currentframe()).filename
script_path = os.path.dirname(os.path.abspath(filename))
wd_path = os.path.abspath(os.path.join(script_path, os.pardir))#working directory path
#print(wd_path)
print(script_path)
sys.path.append(script_path)
from configuration import dict_tel, str2telescope, Telescope
from processing import InputType, Data #Event, Impact, ReconstructedParticle, Intersection
import analysis as ana
from acceptance import Acceptance
from tomo import ImageFeature, Tomo, Topography
from tools.tools import var_wt, pretty
from sklearn.preprocessing import scale # Data scaling
from sklearn import decomposition #PCA
#from PyAstronomy import pyasl


def time_sort_df(df:pd.DataFrame, outfile:str):
    df = df.sort_values(by=["timestamp_s",'timestamp_ns'])
    time_sort = df['timestamp_s'] + df['timestamp_ns']*10**(-8)
    dtime = np.diff(time_sort) 
    run_duration = np.sum(dtime[dtime < 3600])  # in second
    print( f"run_duration={(run_duration)/(24*3600):.3f}days")
    df["date"] = pd.to_datetime(df["timestamp_s"], unit='s')
    df['gap'] = np.concatenate(([0],np.diff(time_sort))) #df.date.diff().dt.seconds
    df['duration'] = pd.to_timedelta( df['date'] - df['date'].iloc[0] ).astype('timedelta64[s]')
    dtgap = 3600 #s     
    print(df.iloc[-1])
    print( f"run_duration={(df['duration'].iloc[-1])/(24*3600):.3f}days")
    dt_sum_gaps = np.sum( df['gap'][df['gap'] > dtgap ] )
    print(f"dt_sum_gaps={dt_sum_gaps/(24*3600):.3f}days")
    print(f"run_duration={(df.iloc[-1]['duration']  - dt_sum_gaps)/(24*3600):.3f}days" )
    #print(f"run_duration={np.sum())/(24*3600):.3f}days" )    
    for i, gap_id in enumerate(df['duration'][df['gap'] > dtgap ].index): 
        id_above_gap = df.loc[gap_id:].index
        df.loc[id_above_gap, 'duration'] = df.loc[id_above_gap, 'duration'] - df.loc[gap_id]['gap']
    print( f"run_duration={df.iloc[-1]['duration']/(24*3600):.3f}days")
    del df["gap"]
    df['duration'] = np.ndarray.astype(df['duration'].values, dtype=int)
    df = df.set_index(df["date"])
    del df["date"]
    #print(df.head)
    df.to_csv(outfile, compression='gzip', sep='\t')
    


class Fluctuations:
    def __init__(self, telescope:Telescope, dT:str, df:pd.DataFrame, outdir:str,  tlim:tuple=None):
        self.tel = telescope
        ####tlim 
        if tlim is not None:
            cut = (tlim[0] <= df['timestamp_s']) & (df['timestamp_s'] <= tlim[1])
            df = df[cut]
        #####DateTimeIndex fmt
        df.index = pd.to_datetime(df.index)
        
        
        g = df.groupby(pd.Grouper(level='date',freq=dT))
    
        self.bin_width_dt = g.tail(1)['duration'].values - g.head(1)['duration'].values
        
        self.bin_edges_date = [(dmin,dmax) for dmin, dmax in zip(g.head(1).index.values, g.tail(1).index.values)]

        self.bin_edges_timestamp = [(tmin,tmax) for tmin, tmax in zip(g.head(1)['timestamp_s'].values, g.tail(1)['timestamp_s'].values)]

        self.nbins = len(self.bin_width_dt)
        ########
        sconfig  = list(self.tel.configurations.keys())
        nlos = self.tel.los[sconfig[0]].shape
        wscint = self.tel.panels[0].matrix.scintillator.width
        DXmin, DXmax, DYmin, DYmax = np.min(self.tel.los[sconfig[0]][:,:,0]),np.max(self.tel.los[sconfig[0]][:,:,0]),np.min(self.tel.los[sconfig[0]][:,:,1]),np.max(self.tel.los[sconfig[0]][:,:,1])  
        self.rangeDXDY = [ (DXmin*wscint, DXmax*wscint), (DYmin*wscint, DYmax*wscint) ]
        nc = len(sconfig)
        ###Flux
        self.arr_DXDY = np.zeros(shape=(nc, self.nbins, nlos[0], nlos[1]))
        self.arr_flux = np.zeros(shape=(nc, self.nbins, nlos[0], nlos[1]))
        self.arr_err_flux = np.zeros(shape=(nc, self.nbins, nlos[0], nlos[1]))
        ###Opacity
        self.arr_op = np.zeros(shape=(nc, self.nbins, nlos[0], nlos[1]))
        self.arr_err_op = np.zeros(shape=(nc, self.nbins, nlos[0], nlos[1]))
        self.arr_rho = np.zeros(shape=(nc, self.nbins, nlos[0], nlos[1]))
        self.arr_err_rho = np.zeros(shape=(nc, self.nbins, nlos[0], nlos[1]))
        self.sconfig=sconfig
        self.df = df
        self.nlos = nlos
        
        
        ####Attempt to speed up with removing the DateTime index
        df = df.reset_index()
        
        print(df.head())

    
    
    def compute_flux_timeseries(self, acceptance, err_acc,  outdir, label):               
        for c, conf in enumerate(self.sconfig):
            acc, u_acc  = acceptance[conf], err_acc[conf]
            for t, (tmin, tmax) in enumerate(self.bin_edges_timestamp):
                #print(f"t_range = ( {tmin} -> {tmax} )s")
                t_range = self.df[(tmin<self.df['timestamp_s']) & (self.df['timestamp_s'] < tmax)].index #pd.t_range(start=tmin, end=tmax)  #returns: DateTimeIndex    # 
                #print(f"trange: {time.time()}")
                DX, DY = self.df.loc[t_range, f'DX_{conf}'].values,  self.df.loc[t_range, f'DY_{conf}'].values 
                #print(f"DX, DY: {time.time()}")
                hDXDY = np.histogram2d(DX, DY, bins=(self.nlos[0], self.nlos[1]), range=self.rangeDXDY)
                Ntomo, u_Ntomo = hDXDY[0], np.sqrt(hDXDY[0])
                #eff, u_eff  = efficiency[conf], err_eff[conf]
                self.arr_DXDY[c,t,:,:]= Ntomo
                #print(f"dT_s={dT}")
                self.arr_flux[c,t,:,:]=  Ntomo/(acc*self.bin_width_dt[t])
                self.arr_err_flux[c,t,:,:]= self.arr_flux[c,t]*np.sqrt( (u_Ntomo/Ntomo)**2 + (u_acc/acc)**2)# + (u_eff/eff)**2 )
                   
        # with open( str( outDir/ 'flux.pkl' ), 'wb') as fFlux, open( str( outDir/ 'err_flux.pkl' ), 'wb') as fErr:
        #     pickle.dump(arr_hFlux, fFlux, pickle.HIGHEST_PROTOCOL)
        #     pickle.dump(arr_err_flux, fErr, pickle.HIGHEST_PROTOCOL)

    def interpolate_opacity(self,outdir:str, label:str, tomo_ze:dict, simu_ze:np.ndarray, op:np.ndarray, tomo_flux:dict, simu_flux:np.ndarray, app_thick:dict, method:str='linear'):

        opacity, density={},{}
    
        points = np.zeros(shape=(60000, 2))
        points[:, 0] = simu_ze.flatten() #zenith 1D array
        points[:, 1] = simu_flux.flatten()
        values = np.exp( np.log(10) * op.flatten() ) #op=exp(log10 varrho)
        outDir = Path(outdir)/"opacity" 
        for (conf,ze), (_,flux), (_,thick) in zip(tomo_ze.items(), tomo_flux.items(), app_thick.items()):
            grid_x, grid_y = ze, flux
            #grid_x.reshape(31,31)
            grid_x.reshape(self.tel.los[conf].shape[0], self.tel.los[conf].shape[1])
            grid_op = griddata(points, values, (grid_x, grid_y), method=method)
            thick[np.isnan(thick)] = 0
            grid_op[(thick==0)] = np.nan
            Path(os.path.join(outdir, 'opacity')).mkdir(parents=True, exist_ok=True)
            np.savetxt(os.path.join(outdir, 'opacity', f'opacity_{label}_{conf}.txt'), grid_op , fmt='%.5e', delimiter='\t')
            opacity[conf] = grid_op
            density[conf] = grid_op/thick
       
        return opacity, density
    
    def compute_opacity_timeseries(self, outdir, label, tomo_ze:dict, simu_ze:np.ndarray, op:np.ndarray, simu_flux:np.ndarray,app_thick:dict, method:str='linear',mask=None):    


        for t, (tmin, tmax) in enumerate(self.bin_edges_date):
            tomo_flux = {conf: self.arr_flux[c,t] for c,conf in enumerate(self.sconfig)}
            tminfmt = datetime.fromtimestamp(tmin).strftime("%d%m%y")
            dict_op, dict_rho = self.interpolate_opacity(
                                                    outdir=outdir, 
                                                    label=label+f"_{tminfmt}_",
                                                    tomo_ze=tomo_ze,
                                                    simu_ze=simu_ze,
                                                    op=op,
                                                    tomo_flux=tomo_flux,
                                                    simu_flux = simu_flux,
                                                    app_thick = app_thick
                                                    )
            
            for c, conf in enumerate(self.sconfig):
                #t_range = df[(dtbin[t]<df['timestamp_s']) & (df['timestamp_s'] < dtbin[t+1])].index 
                self.arr_op[c,t,:,:], self.arr_rho[c,t,:,:]  =  dict_op[conf], dict_rho[conf]
                
    # with open( str( outDir/ 'opacity.pkl' ), 'wb') as fout:
    #     pickle.dump(arr_hOp, fout, pickle.HIGHEST_PROTOCOL)
    # with open( str( outDir/ 'density.pkl' ), 'wb') as fout:
    #     pickle.dump(arr_hRho, fout, pickle.HIGHEST_PROTOCOL)
    

    def compute_mean_quantity(self, quantity:str, array:np.ndarray,err_array:np.ndarray, mask:dict, outdir:str, label:str):
        nc = len(sconfig)
        df_flux = pd.DataFrame(columns = ['tstart', 'tend']  )
        df_flux['tstart'] = np.array([self.bin_edges_timestamp[i][0] for i in range(self.nbins)])
        df_flux['tend'] = np.array([self.bin_edges_timestamp[i][1] for i in range(self.nbins)])
        arr_mean = np.zeros(shape=(nc, self.nbins))
        arr_stdev= np.zeros(shape=(nc, self.nbins))
        arr_mean_err = np.zeros(shape=(nc, self.nbins))
        for c, conf in enumerate(sconfig):
            for t in range(self.nbins):
                if mask is not None: array[c,t,:,:][mask[conf]] = np.nan
                arr_stdev[c,t] = np.std(array[c,t,:,:][~np.isnan(array[c,t,:,:])])
                arr_mean[c,t]= np.mean(array[c,t,:,:][~np.isnan(array[c,t,:,:])])
                thresh_all = arr_mean[c,t]*1e-6
                sel_all = (~np.isnan(err_array[c,t, :,:]) & ~np.isnan(array[c,t,:,:]))
                x_all, sig_all =  array[c,t,:, :][sel_all], err_array[c,t, :, :][sel_all]
                res_all = var_wt(x=x_all, sigma_i=sig_all, thresh=thresh_all)
                arr_mean[c,t] = res_all['Mean']
                arr_mean_err[c,t] = np.sqrt(res_all['Variance'])
            df_flux[f'mean_{quantity}_{conf}'] = arr_mean[c,:]
            df_flux[f'stdev_{quantity}_{conf}'] = arr_stdev[c,:]
            df_flux[f'mean_err_{quantity}_{conf}'] = arr_mean_err[c,:]
        outDir = Path(outdir)
        df_flux.to_csv(str(outDir/f"df_{quantity}_{label}.csv"), sep='\t') 
        return df_flux
        
    def mean_var_vs_time(self, quantity:str, df:pd.DataFrame, label:str, outdir:str, logscale:bool=False):
        fig, axs = plt.subplots(nrows=len(self.sconfig), ncols=1, figsize=(16, 9))
        tc = ((df['tstart']+df['tend'])/2).values
        #tc_fmt = [ datetime.fromtimestamp(t) for t in tc ]
        tcinter = np.linspace(np.min(tc), np.max(tc), 100 )
        tcinter_fmt = [ datetime.fromtimestamp(x) for x in tcinter ]
        for c, (conf, ax) in enumerate(zip(self.sconfig, axs.ravel())):
            #fig.suptitle(f"$\\Delta$T={dT/(24*3600):.1f} days" , fontsize=18, y=0.95)
            ax.set_title(f"{conf}")
            myFmt = mdates.DateFormatter('%d-%m-%Y')
            ax.set_xticklabels(ax.get_xticks(), rotation = 45)
            ax.xaxis.set_major_formatter(myFmt) 
            mean_flux  = df[f'mean_{quantity}_{conf}']
            stdev_flux  = df[f'stdev_{quantity}_{conf}']
            err_mean_flux  = df[f'mean_err_{quantity}_{conf}']
            rel_flux = (mean_flux-mean_flux[0])/ mean_flux
            #err_flux = 2*rel_flux*np.sqrt((arr_stdev_flux[c,:]/df_flux[f'mean_flux_{conf}'])**2+(arr_stdev_flux[c,0]/df_flux.iloc[0][f'mean_flux_{conf}'])**2)
            err_var_flux = err_mean_flux/mean_flux #2*np.sqrt((stdev_flux/mean_flux)**2+(stdev_flux[0]/mean_flux[0])**2)
            #ax.plot(tc_fmt, rel_flux, color='red')
            _, caps, _ = ax.errorbar(x=tc, y=rel_flux, yerr=err_var_flux, xerr=None, fmt='o', color='red', linestyle=None, capsize=15)
            for cap in caps:
                cap.set_markeredgewidth(1)
            # ax.fill_between(tc_fmt, rel_flux-err_var_flux, rel_flux+err_var_flux, alpha=0.4, edgecolor='red', facecolor='tomato',
            #     linewidth=1, linestyle='-', antialiased=True)
            #ax.plot(tc_fmt, rel_flux, color='red')
            tck = interpolate.splrep(tc, rel_flux )
            res = interpolate.splev(tcinter, tck)
            
            ax.plot(tcinter_fmt, res, color='red')
            
            ax.set_xlabel("time [days]", fontsize=10)
            if logscale : ax.set_yscale("log")
            ax.grid(True, which="both", ls="-")
            ax.tick_params(axis='both', labelsize=10)
            #ax.set_ylabel("Mean Flux $\\langle\\phi\\rangle$ [cm$^{-2}$.s$^{-1}$.sr$^{-1}$]",  fontsize=8)
            ax.set_ylabel(f"Relative {quantity} variation",  fontsize=10)#+ "($\\langle\\phi\\rangle$-$\\langle\\phi\\rangle_{t0}$)/$\\langle\\phi\\rangle_{t0}$ ",  fontsize=10)
            # leg = mlines.Line2D([], [], color='blue',  linestyle='-',
            #                     markersize=10, label=label)
            #ax.legend(handles=[leg], fontsize=12, loc='upper left')
            
        fig.tight_layout()
        fout = os.path.join(outdir,f"mean_{quantity}_vs_time_{label}.png")
        print(f"save {fout}")
        plt.savefig(fout)

    def plot_maps(self, quantity:str,array:np.ndarray,nplots:tuple, vrange:tuple, sigma:tuple, az:dict, ze:dict, outdir:str, label:str, topography:dict=None, lognorm:bool=False, cmap='jet_r', mask:dict=None):
        vmin, vmax= vrange
        dt_center = [ (t0+t1) /2  for t0, t1 in self.bin_edges_timestamp  ] 
        for c, conf in enumerate(self.sconfig):
            fig, axs = plt.subplots(nrows=nplots[0], ncols=nplots[1], figsize=(16, 9),sharex=True, sharey=True)
            fig.suptitle("", fontsize=18, y=0.95)
            a,z= az[conf], ze[conf]
            az_min , az_max, ze_min , ze_max = np.min(a), np.max(a), np.min(z), np.max(z)
            for t, ax in enumerate(axs.ravel()):
                if t >= self.nbins: break
                if mask is not None: array[c,t,:,:][mask[conf]]=np.nan
                f = np.flipud(array[c,t,:,:])
                f = np.fliplr(f)
                
                if lognorm:
                    im = ax.imshow( f , cmap=cmap, norm=LogNorm(vmin=vmin, vmax=vmax),  extent=[az_min , az_max,  ze_min , ze_max], aspect='auto')
                else:
                    im = ax.imshow( f , cmap=cmap, vmin=vmin, vmax=vmax, extent=[az_min , az_max,  ze_min , ze_max], aspect='auto') 
                #im0 = ax0.pcolor(a,z, f, cmap='jet_r',  shading='auto', norm=LogNorm(vmin=vmin, vmax=vmax)) 
            # zgaus = scipy.ndimage.filters.gaussian_filter(z, sigma, mode='constant')
                # Diplay filtered array
            # zgaus[zgaus<=vmin]=np.nan
                ax.grid(False)
                ax.set(ylim=[50, 90]) #deg
                if conf=='4p': 
                    ax.set(ylim=[55, 85]) #deg
                
                ax.set(adjustable='box', aspect='equal')
                # if lognorm:
                #     im = ax.pcolor(a,z, f, cmap=cmap, norm=LogNorm(vmin=vmin, vmax=vmax))
                # else:
                #     im = ax.pcolor(a,z, f, cmap=cmap,  vmin=vmin, vmax=vmax)
            
                if  topography is not None: 
                    ax.plot(topography[conf][:,0], topography[conf][:,1], linewidth=3, color='black')
                ax.invert_yaxis()
                #forceAspect(ax,aspect=1)
                t_ = datetime.fromtimestamp(dt_center[t])
                ax.set_title(t_.strftime("%d/%m/%y"))
            cax = fig.add_axes([0.90, 0.15, 0.03, 0.7]) #[0.15, 0.15, 0.5, 0.05]) #[left, bottom, length/width, height]
            cbar = fig.colorbar(im, cax=cax, orientation="vertical")
            #cbar = fig.colorbar(c, ax=ax, shrink=0.75, format='%.0e', orientation="horizontal")
            cbar.ax.tick_params(labelsize=8)
            #cbar.set_label(label=u'Flux (cm$^{-2}$.s$^{-1}$.sr$^{-1}$)', size=12)
            fig.subplots_adjust(right=0.85)
            t_min, t_max = np.min(dt_center), np.max(dt_center)
            date_min, date_max = datetime.fromtimestamp(t_min), datetime.fromtimestamp(t_max)
            date_min_fmt, date_max_fmt = date_min.strftime("%d%m%y"), date_max.strftime("%d%m%y")
            fout = os.path.join(outdir, f"{quantity}_{date_min_fmt}_{date_max_fmt}_{conf}_{label}.png")
            print(f"save {fout}")
            plt.savefig(fout)
        


    
    
    
if __name__=="__main__":
    
    
    
    start_time = time.time()
    print("Start: ", time.strftime("%H:%M:%S", time.localtime()))#start time
    ###Load default script arguments stored in .yaml file
    def_args={}
    with open(os.path.join(script_path, 'config_files','configSNJ.yaml') ) as fyaml:
        try:
            # The FullLoader parameter handles the conversion from YAML
            # scalar values to Python the dictionary format
            def_args = yaml.load(fyaml, Loader=yaml.FullLoader)
        except yaml.YAMLError as exc:
            print(exc)
    
    pretty(def_args)   
    
    t_start = time.perf_counter()
    home_path = os.environ["HOME"]
    parser=argparse.ArgumentParser(
    description='''For a given muon telescope configuration, this script allows to format .root simu output file, process the format data and analyze it.''', epilog="""All is well that ends well.""")
    parser.add_argument('--telescope', '-tel', default=def_args["telescope"], help='Input telescope name (e.g "tel_SNJ"). It provides the associated configuration.',  type=str2telescope)
    parser.add_argument('--input_tomo', '-tom', default=def_args["reco_tomo"],  help='Input reco tomo file', type=str)
    parser.add_argument('--input_calib', '-cal', default=def_args["reco_calib"],  help='Input reco calib file', type=str)
    parser.add_argument('--label_tomo', '-lt', default=def_args["label_tomo"], help='Label of the dataset', type=str)
    parser.add_argument('--label_calib', '-lc', default=def_args["label_calib"], help='Label of the dataset', type=str)
    parser.add_argument('--ana_dir', '-o', default=def_args["out_dir_tomo"], help='Path to processing output', type=str) 
    args=parser.parse_args()
    tel = args.telescope
    recofileTomo = glob.glob(os.path.join(args.input_tomo, "*reco*"))[0]
    recofileCal = glob.glob(os.path.join(args.input_calib, "*reco*"))[0]
    tlim = ( int(datetime(2019, 2, 1, hour=00,minute=00,second=00).replace(tzinfo=timezone.utc).timestamp()) , int(datetime(2019, 8, 31, hour=23,minute=59,second=59).replace(tzinfo=timezone.utc).timestamp())   )
    #tlim = ( int(datetime(2019, 2, 1, hour=00,minute=00,second=00).replace(tzinfo=timezone.utc).timestamp()) , int(datetime(2019, 3, 15, hour=23,minute=59,second=59).replace(tzinfo=timezone.utc).timestamp())   )
    t_min, t_max = tlim
 
    evttype = ana.EventType.MAIN
    fn_dxdy = os.path.join(args.ana_dir, f"out_NDXDY_{args.label_tomo}.csv.gz")
  
    
  
    
    # print(f"read df_dxdy --- {(time.time() - start_time):.3f}  s ---")  
    ###load existing dataframe
    fn_sort = Path(args.ana_dir) / f"out_NDXDY_{args.label_tomo}_sort.csv.gz"
    kwargs = { "index_col": 0,"delimiter": '\t', "nrows": 1e5}
    #if not fn_sort.exists():    
    #    if not os.path.exists(fn_dxdy): 
            
    anaBaseTomo = ana.AnaBase(recofile=ana.RecoData(file=recofileTomo, 
                                                            telescope=tel,  
                                                            input_type=InputType.DATA, 
                                                            kwargs=kwargs), 
                        label=args.label_tomo) 
    #print(anaBaseTomo.df.head)
    hmTomo = ana.AnaHitMap(anabase=anaBaseTomo, input_type=InputType.DATA, panels=tel.panels)
    df = hmTomo.df_DXDY
            #df.to_csv(fn_dxdy, sep='\t')
    
        # df = pd.read_csv(fn_dxdy, delimiter="\t", compression="gzip", index_col=0) 
    time_sort_df(df=df,outfile=fn_sort)
        # print(f"sort df --- {(time.time() - start_time):.3f}  s ---")  
    
    df = pd.read_csv(fn_sort, compression='gzip', delimiter="\t", index_col=0)
    print(f"read df_sort --- {(time.time() - start_time):.3f}  s ---")  
    #print(df.head, df.tail )


    #####
    dT = "1D" #days
    F = Fluctuations( telescope=tel,
                                tlim=tlim,
                                dT=dT,
                                df=df,
                                outdir=args.ana_dir,
    )
    
    
    
    print(f"set tomo --- {(time.time() - start_time):.3f}  s ---")              
    
    sconfig = list(tel.configurations.keys())
    param_dir = os.path.join(script_path,'', 'AcquisitionParams')
    Corsika_OpenSkyFlux = sio.loadmat(os.path.join(param_dir, f'{tel.name}', 'ExpectedOpenSkyFlux.mat'))
    OSFlux_calib_3p = Corsika_OpenSkyFlux['ExpectedFlux_calib_3p']#open flux
    OSFlux_calib_4p = Corsika_OpenSkyFlux['ExpectedFlux_calib_4p']
    os_flux = { sconfig[0]: OSFlux_calib_3p,  sconfig[1]:OSFlux_calib_3p,  sconfig[2]:OSFlux_calib_4p } 
    accDir = os.path.join(args.ana_dir, 'acceptance')
    accFiles = glob.glob(os.path.join(accDir, f"acceptance*.txt"))
    uaccFiles = glob.glob(os.path.join(accDir, f"unc_acc*.txt"))
    if len(accFiles)==0:
        anaBaseCal = ana.AnaBase(recofile=ana.RecoData(file=recofileCal, 
                                                                telescope=tel,  
                                                                input_type=InputType.DATA), 
                    label=args.label_calib)
        hmCal = ana.AnaHitMap(anabase=anaBaseCal, 
                                input_type=InputType.DATA, 
                                panels=tel.panels
                                )
        A = Acceptance(hitmap=hmCal,
                                    outdir=args.ana_dir, 
                                    evttype=evttype, 
                                    opensky_flux=os_flux,
                                    theoric=None)
        acceptance = {k: v for k,v in A.acceptance.items()}
        err_acc = {k: v for k,v in A.unc.items()}
        
    else : 
        acceptance = {conf: np.loadtxt(f) for conf, f in zip(sconfig, accFiles)}
        err_acc = {conf: np.loadtxt(f) for conf, f in zip(sconfig, uaccFiles)}
    
    
   
    print(f"set acceptance --- {(time.time() - start_time):.3f} s ---")              
    #exit()

    
    date_min, date_max = datetime.fromtimestamp(t_min), datetime.fromtimestamp(t_max)
    date_min_fmt, date_max_fmt = date_min.strftime("%d%m%y"), date_max.strftime("%d%m%y")

    outDir = Path(args.ana_dir) / "flux_variations" / f"{date_min_fmt}_{date_max_fmt}_dT{dT}"
    outDir.mkdir(parents=True, exist_ok=True)
    print(f"outDir  = {outDir}")
    F.compute_flux_timeseries(acceptance=acceptance, err_acc=err_acc,  outdir=outDir, label=args.label_tomo)
    

    
    IntegralFluxVsOpAndZaStructure_Corsika = sio.loadmat(os.path.join(param_dir, 'common', 'IntegralFluxVsOpAndZaStructure_Corsika.mat')) #dictionary out of 100x600 matrix
    simu_ze = IntegralFluxVsOpAndZaStructure_Corsika['IntegralFluxVsOpAndZaStructure_rock_MuonsEKin_onlyModel'][0][0][0] #zenith angle
    simu_opacity = IntegralFluxVsOpAndZaStructure_Corsika['IntegralFluxVsOpAndZaStructure_rock_MuonsEKin_onlyModel'][0][0][1] #opacity
    simu_flux = IntegralFluxVsOpAndZaStructure_Corsika['IntegralFluxVsOpAndZaStructure_rock_MuonsEKin_onlyModel'][0][0][2] #Corsika flux
    
    acqVar = ana.AcqVars(telescope=tel, tomo=True)
        # with open( str( fAcqVars ), 'wb') as fout:
        #     pickle.dump(acqVar, fout, pickle.HIGHEST_PROTOCOL)
              
    print(f"load acqVar --- {(time.time() - start_time):.3f} s ---")              
    az_tomo   = acqVar.az_tomo
    ze_tomo   = acqVar.ze_tomo
    topo = acqVar.topography
    thickness = acqVar.thickness

    sky_mask = { c:( np.isnan(thickness[c]) ) for c in sconfig }

    cs_mask = { c: (~((ze_tomo[c]<80) & (25<az_tomo[c]) & ~sky_mask[c])  ) for c in sconfig }
     
    sigma = (0.5, 0.5)
    nplots=(2,3)
    flux_min, flux_max = (3e-6, 1e-2)
    flux_dir = os.path.join(str(outDir), 'flux')
    Path(flux_dir).mkdir(parents=True,exist_ok=True)
    #F.plot_maps(quantity="flux",mask=cs_mask, 
    #                       label=args.label_tomo+"_cs_mask", 
    #                        array=F.arr_flux,nplots=nplots, 
    #                        vrange=(flux_min, flux_max), 
    #                         sigma=sigma, 
    #                           az=az_tomo, 
    #                           ze=ze_tomo, 
    #                           outdir=outDir,  
    #                           topography=topo,    
    #                           lognorm=True, cmap='jet_r')

    
    #exit()
    
  
    
    print(f"compute mean flux --- {(time.time() - start_time):.3f} s ---"  )   
    ####Plots
    df_flux = F.compute_mean_quantity( quantity="flux", array=F.arr_flux, err_array=F.arr_err_flux, mask=None, outdir=flux_dir, label=args.label_tomo) 
    print("df_flux = ", df_flux)
    
    #F.mean_var_vs_time(quantity='flux',df=df_flux, label=args.label_tomo, outdir=flux_dir)
    #F.plot_maps(quantity="flux",array=F.arr_flux,nplots=nplots, vrange=(flux_min, flux_max), sigma=sigma, az=az_tomo, ze=ze_tomo, outdir=outDir, label=args.label_tomo, topography=None, lognorm=True, cmap='jet_r')
    
    ''''
    Test PCA
    '''
    print( F.arr_flux.shape)
    #X = F.arr_flux[0,:,:,:]
    #pca = decomposition.PCA(n_components=3)
    #pca.fit(X)

    '''Hamming window'''
    #y = df_flux['mean_flux_3p1']
    #dTwindow =  len(y)/#days
    #sm2 = pyasl.smooth(y, dTwindow, 'hamming') #https://pyastronomy.readthedocs.io/en/latest/pyaslDoc/aslDoc/smooth.html
    #print(f"y = {y}")
    #print(f"sm2 = {sm2}")
    
    
    
    exit()
    F.compute_opacity_timeseries(
                        label=args.label_tomo,
                        outdir=str(outDir),
                        tomo_ze= ze_tomo, 
                        simu_ze=simu_ze,
                        op=simu_opacity,
                        simu_flux=simu_flux,
                        app_thick=thickness,
                        method='linear')
    
    op_range=(3e1, 3e3)#10e1/dthick) #MWE
    op_dir = os.path.join(str(outDir), "opacity" )
    Path(op_dir).mkdir(parents=True,exist_ok=True)
    F.plot_maps(quantity="opacity", mask=None,label=args.label_tomo, array=F.arr_op,nplots=nplots, vrange=op_range, sigma=sigma, az=az_tomo, ze=ze_tomo, outdir=op_dir,topography=topo, lognorm=True, cmap='jet')
    F.plot_maps(quantity="opacity", mask=cs_mask, label=args.label_tomo+"_cs_mask", array=F.arr_op,nplots=nplots, vrange=op_range, sigma=sigma, az=az_tomo, ze=ze_tomo, outdir=op_dir,topography=topo, lognorm=True, cmap='jet')

    df_opacity = F.compute_mean_quantity( quantity="opacity", mask=cs_mask, label=args.label_tomo+"_cs_mask", array=F.arr_op, err_array=F.arr_err_op, outdir=op_dir) 
    F.mean_var_vs_time(quantity='opacity',df=df_opacity, label=args.label_tomo+"_cs_mask", outdir=op_dir)
    
    
    rho_min, rho_max=0.1, 2.0
    de_dir = os.path.join(str(outDir), "density" )
    Path(de_dir).mkdir(parents=True,exist_ok=True)
    F.plot_maps(quantity="density", label=args.label_tomo, array=F.arr_rho,nplots=nplots, vrange=(rho_min, rho_max), sigma=sigma, az=az_tomo, ze=ze_tomo, outdir=de_dir,topography=None, lognorm=False, cmap='jet')
    
  
  
  
  
     #######
    # fn_sort = Path(args.ana_dir) / f"out_NDXDY_{args.label_tomo}_sort.csv.gz"
    # df_sort = pd.read_csv(fn_sort, compression='gzip', delimiter="\t", index_col=0)
    # df_sort.index = pd.to_datetime(df_sort.index)
    # df_sort = df_sort.set_index(df_sort["date"], inplace=True)
    # del df_sort["date"]
    # print(df_sort.head)
    # print(df_sort.index.to_list()[:10])
    # print(type(df_sort.index))
    # df_sort.to_csv(fn_sort, compression='gzip',sep='\t')
    
    
    # g = df_sort.groupby(pd.Grouper(level='date',freq='50D'))
 
    # dt = g.tail(1)['duration'].values - g.head(1)['duration'].values
    
    # ddates = [(dmin,dmax) for dmin, dmax in zip(g.head(1).index.values, g.tail(1).index.values)]
    # dtimestamps = [(tmin,tmax) for tmin, tmax in zip(g.head(1)['timestamp_s'].values, g.tail(1)['timestamp_s'].values)]
    
    
    
    # index = g.groups.values()
    
    # print(index)
    
    # print(g.head(1)['duration'])
    # print(g.tail(1)['duration'])
    
    # print(dt)
    # print(ddates)
    # print(dtimestamps)
    
    #exit()
  #######
  
  
  
  
'''
class Test:
    def __init__(self, telescope:Telescope, tlim:tuple, dT:float, df:pd.DataFrame, outdir:str ):
        self.tel = telescope
        self.tmin, self.tmax = tlim
        indexes = df['timestamp_s'][(self.tmin <= df['timestamp_s']) & (df['timestamp_s'] <= self.tmax)].index
        df = df.loc[indexes]
        
        run_duration = df['duration'].iloc[-1]
        self.nbins = int(np.ceil(run_duration/dT))
        #df['duration'].hist(bins=self.nbins)
        #plt.show()
        (_, self.dtbin) = np.histogram(df['timestamp_s'], bins=self.nbins) 
        print(self.nbins, self.dtbin )
        #df["date"] = pd.to_datetime(df["timestamp_s"], unit='s')
       
        
        
        #print(df[df['timestamp_s']==self.dtbin[0]])
        #print(df[df['timestamp_s']==self.dtbin[1]])
        
        # dt_width = np.array([ df[df['timestamp_s']==self.dtbin[i+1], 'duration'] - df[df['timestamp_s']==self.dtbin[i], 'duration']
        #                      for i in range(self.nbins)])
        # print(dt_width)
        # print( np.array([self.dtbin[i+1] - self.dtbin[i] for i in range(self.nbins)]) ) 
        
        #s = df[df['gap']==df['gap'].max()].index
        # print(df.loc[s])
        # print(type(s))
        # for i in s : print(i)
        # print(df.index.get_loc(s[0]))
        # print()
        # print(df.iloc[df.index.get_loc(s[0]) - 1])
        # print()
        # print(df.iloc[df.index.get_loc(s[0])])
        # print()
        # print(df.iloc[df.index.get_loc(s[0]) + 1])
        #print(df.get_loc(s))
        
        #print(df.iloc[s.iloc[-1]])
       # print(df.iloc[loc_gap-1])
      #  print(df.iloc[loc_gap+1])
       # print(df.iloc[loc_gap+2])
#        print(df.loc[loc_gap]['date'], df.loc[loc_gap-1]['date'], df.loc[loc_gap-2]['date'])
       # print(df['duration'].head)
        
        #report = pd.pandas_profiling.ProfileReport(data)report
        
        #time_sort = np.sort(df['timestamp_s'][(self.tmin <= df['timestamp_s']) & (df['timestamp_s'] <= self.tmax)].values)
        #dtime = np.diff(time_sort) 
        #print(np.max(dtime/(24*3600)))
        # run_duration = np.sum(dtime[dtime < 3600])
        # t_start = int(np.min(time_sort))
        # t_end = int(np.max(time_sort))
        
        # self.nbins = int(np.ceil(run_duration/dT))
        # (_, self.dtbin) = np.histogram(time_sort, bins=self.nbins) 
        # print(self.nbins, len(self.dtbin) )
        ####time slices
        #t_range = [ df[(dtbin[i]<df['timestamp_s']) & (df['timestamp_s'] < dtbin[i+1])].index for i in range(len(dtbin)-1) ]
       
        
        
        #df["gap"] = df.groupby(["duration"]).diff()  < 10
        
        #df = df.sort_values(by="date")
        #print(df.head)
        #
        #print(df[df['duration']>3600])
        #print(df.head)
        #exit()
        # print(df.iloc[0]['date'], df.iloc[0]['date'])
        # print((df.iloc[-1]['timestamp_s']- df.iloc[0]['timestamp_s'])/(24*3600))
        # print(df.head)
        #exit()        #dict_time_sort = {'time':time_sort}
        #df_tmp = pd.DataFrame(time_sort, columns=['time'])#, index=df.index.values)
        # df_tmp.reindex_like(df) 
        
        # print(df_tmp.head)
        # df_tmp['time']= pd.to_datetime(df_tmp['time'], unit='s')
        # print()
        # print(df_tmp.head)
        
        # # 
        
        # print(run_duration)
        # nbins = 
        # print(nbins)
        
        # self.nbins = np.linspace(0, run_duration, nbins)
        # print(self.nbins)
        # print(len(self.nbins))
        # #print(self.nbins)
        # labels=[str(i) for i in range(nbins-1)]
        # print(len(labels))
        # df_tmp['Time Bin'] = pd.cut(df_tmp.time.dt.hour, self.nbins, labels=labels)#, right=False)#,  labels=[str(i) for i in range(self.nbins-1)])
        # print(df_tmp.time.dt.second)
        
        # print(df_tmp.head)


  
  
  '''