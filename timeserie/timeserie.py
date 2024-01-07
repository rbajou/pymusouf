#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Raphaël Bajou
"""
from dataclasses import dataclass, field
from typing import Union
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.markers import MarkerStyle
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from pathlib import Path
from matplotlib.gridspec import GridSpec
from matplotlib.colors import LogNorm, Normalize
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.ticker import LogLocator, MultipleLocator,EngFormatter, ScalarFormatter
from matplotlib.offsetbox import AnchoredText
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import pickle
import pandas as pd
import warnings
warnings.filterwarnings("ignore")
import time
import glob 
import yaml
from datetime import datetime
#package module(s)
from reco import HitMap, EvtRate, RecoData
from telescope import Telescope, dict_tel
from muo2d import Acceptance, Flux, Opacity, MeanDensity
from raypath import RayPathSoufriere
from utils.tools import var_wt

params = {'legend.fontsize': 'x-large',
         'axes.labelsize': 'xx-large',
         'axes.titlesize':'xx-large',
         'xtick.labelsize':'x-large',
         'ytick.labelsize':'x-large',
         'axes.labelpad':10,
         'mathtext.fontset': 'stix',
         'font.family': 'STIXGeneral',
         }
plt.rcParams.update(params)



def time_sort_df(df:pd.DataFrame, dtgap=3600, tres =1e-8)-> pd.DataFrame:
    '''
    Returns dataframe with timedelta column obtained after substraction of all time gaps between consecutive events superior to 'dtgap' in sec.
    '''

    df.sort_values(by=["timestamp_s","timestamp_ns"], inplace=True)
    time_sort = df["timestamp_s"] + df["timestamp_ns"]*tres
    #freq = np.diff(time_sort) 
    #run_duration = np.sum(freq[freq < dtgap])  # in second
    #print( f"run_duration={(run_duration)/(24*3600):.3f}days")

    df["date"] = pd.to_datetime(df["timestamp_s"], unit='s')
    df["gap"] = np.concatenate(([0],np.diff(time_sort))) #df.date.diff().dt.seconds
    df["timedelta"] = pd.to_timedelta( df["date"] - df["date"].iloc[0], unit='s' ).astype('timedelta64[s]')

    #dt_sum_gaps = np.sum( df["gap"][df["gap"] > dtgap ] )
    #print(f"dt_sum_gaps={dt_sum_gaps/(24*3600):.3f}days")
    #print(f"run_duration={(df.iloc[-1]['timedelta'].total_seconds()  - dt_sum_gaps)/(24*3600):.3f}days" )
   
    idx_gap =  df["timedelta"][df["gap"] > dtgap ].index

    #print(f"idx_gap = {idx_gap}")

    for i, ix in enumerate(idx_gap): 
        id_above_gap = df.loc[ix:].index
        res = df.loc[id_above_gap, "timedelta"] - df.loc[ix]["gap"].astype('timedelta64[s]')
        #print(f"i, res = {i}, {res}")
        df.loc[id_above_gap, "timedelta"] = res
   
    #print( f"run_duration={df.iloc[-1]['timedelta'].total_seconds()/(24*3600):.3f}days")
    del df["gap"]
    df["timedelta"] = np.ndarray.astype(df["timedelta"].values, dtype=int)
    #df.set_index(df["date"], inplace=True)
    #del df["date"]
    #print(df.head)
    return df 


class TimeSerie:

    def __init__(self, telescope:Telescope,  df:pd.DataFrame, key:str='timedelta', freq="10D"):
        self.tel = telescope
        self.df = df.copy()
        #g = self.df.groupby(pd.Grouper(level='date',freq=freq)) #if 'date' is the index
        g = self.df.groupby(pd.Grouper(key=key,freq=freq))

        head, tail = g.head(1), g.tail(1) #of each time bin
        self.bin_edges_date = np.array([[dmin,dmax] for dmin, dmax in zip(head['date'].values, tail['date'].values)])
        #print(f'bin_edges_date = {self.bin_edges_date} \n')
        self.bin_edges_timestamp = np.array([[tmin,tmax] for tmin, tmax in zip(head['timestamp_s'].values, tail['timestamp_s'].values)])
        #print(f'bin_edges_timestamp = {self.bin_edges_timestamp} \n')
        self.bin_width_dt = (tail['timedelta'].values - head['timedelta'].values).astype('timedelta64[s]')
        self.bin_width_dt = np.ndarray.astype(self.bin_width_dt, dtype=int)
        # print(f'bin_width_dt = {self.bin_width_dt} \n')

        self.ts = self.df['timestamp_s']
        data = {'bin_low' : self.bin_edges_timestamp[:,0],
                'bin_up' : self.bin_edges_timestamp[:,1],
                'bin_width' : self.bin_width_dt}
        for name, conf in tel.configurations.items(): 
            data[f'mean_{name}']= np.ones(len(self.bin_width_dt))*np.nan
            data[f'std_{name}'] =  np.ones(len(self.bin_width_dt))*np.nan
            data[f'unc_{name}'] =  np.ones(len(self.bin_width_dt))*np.nan
          
        rayMatrix = { name :  self.tel.get_ray_matrix(front_panel=conf.panels[0], rear_panel=conf.panels[-1]) for name,conf in self.tel.configurations.items()}
        shape = {name : (m.shape[0], m.shape[1]) for name, m in rayMatrix.items()}
        n = len(self.bin_width_dt)
        self.estimate_serie = {}
        self.estimate_serie['estimate'] = { name : {ti : np.ones(shp)*np.nan for ti in range(n)} for name, shp in shape.items()}
        self.estimate_serie['unc'] = { name : {ti : np.ones(shp)*np.nan for ti in range(n)} for name, shp in shape.items()}
        self.df_serie = pd.DataFrame(data=data)
        

    def save(self, file:Union[Path, str]): 
        ''' 
        Save 'estimate_serie' in binary pickle file format.
        '''
        if isinstance(file, Path): file=str(file)
        with open(file, 'wb') as f:
            pickle.dump(self.estimate_serie, f, pickle.HIGHEST_PROTOCOL)
            print(f"Save pickle file {file}")


    def compute(self, observable:Union[Flux,Opacity,MeanDensity], scale_factor:int=1e4,**args):
        
        shape = (self.bin_edges_timestamp.shape[0], len(self.tel.configurations.keys()))
        arr_mean, arr_std, arr_unc = np.ones(shape)*np.nan,  np.ones(shape)*np.nan, np.ones(shape)*np.nan
        
        for c, name in enumerate(self.tel.configurations.keys()): 

            for ti, (tmin, tmax) in enumerate(self.bin_edges_timestamp):
                if c == 0: 
                    t_window = (tmin < self.ts) & (self.ts < tmax)
                    df_window = self.df[t_window]
                    df_window.reset_index(inplace=True)
                    if 'hitmap' in args.keys(): args['hitmap'] = HitMap(self.tel, df_window)
                    if 'time' in args.keys():  args['time'] = tmax - tmin
                    observable.compute(**args)
                    
                est = observable.estimate[name]
                self.estimate_serie['estimate'][name][ti] = est
                unc = np.sqrt(observable.unc_stat[name]**2 + observable.unc_sys[name]**2)
                self.estimate_serie['unc'][name][ti] = unc
                mean, std = np.nanmean(est), np.nanstd(est)
                sel = (~np.isnan(mean) & ~np.isnan(unc))
                thresh = mean*1e-6
                var = var_wt(x=est[sel], sigma_i=unc[sel], thresh=thresh)['variance']
                arr_line = np.array([mean, std, np.sqrt(var)]) * scale_factor
                arr_mean[ti, c], arr_std[ti, c], arr_unc[ti, c] = arr_line.T
            
            self.df_serie[f'mean_{name}'], self.df_serie[f'std_{name}'], self.df_serie[f'unc_{name}'] = arr_mean, arr_std, arr_unc

    def plot_timeserie(self, ax:Axes, x:np.ndarray, y:np.ndarray, yerr:np.ndarray, **kwargs):
        ax.plot(x, y, color=kwargs['color'])
        # _, caps, _ = ax.errorbar(x=x, y=y, yerr=yerr, xerr=None, fmt='o', color='red', linestyle=None, capsize=15)
        # for cap in caps:
        #     cap.set_markeredgewidth(1)
        ax.fill_between(x, y-yerr,  y+yerr, **kwargs)


    def plot_maps(self, fig:Figure, grid_x:np.ndarray, grid_y:np.ndarray, grid_z:dict, fmt:str = '%d/%m/%Y', **kwargs) : 
        
        nbin = len(self.bin_width_dt)
        ncol = 5
        nmax = np.round(len(self.bin_edges_date)+ncol, -1)
        nrow = int(nmax/ncol)
        if (ncol*nrow)  >= (nbin+ncol) : nrow-=1

        gs = GridSpec(nrow, ncol,
                wspace=0.0, hspace=0.0, 
                top=1-0.1/(nrow+1), bottom=0.3/(nrow+1), 
                left=0.3/(ncol+1), right=0.95-0.1/(ncol+1))
        ti = 0
        for i in range(nrow):
            for j in range(ncol): 
                ax = plt.subplot(gs[i,j])
                if ti in grid_z.keys(): 
                    Z = grid_z[ti]
                    im =  ax.pcolor(grid_x, grid_y, Z, **kwargs)
                    date_start, date_end = datetime.fromtimestamp(self.bin_edges_timestamp[ti][0]).strftime(fmt), datetime.fromtimestamp(self.bin_edges_timestamp[ti][1]).strftime(fmt)
                    text = f"{date_start} -> {date_end}"
                    anchored_text = AnchoredText(text, loc="lower center", frameon=False, prop=dict(fontsize='large'))
                    ax.add_artist(anchored_text)
                    #ax.invert_yaxis()
                else: 
                    Z = np.ones(grid_x.shape)*np.nan
                    ax.pcolor(grid_x, grid_y, Z, **kwargs)
                ax.invert_yaxis()
                if i != nrow - 1 :  
                    ax.set_xticklabels([]) 
                if j != 0 : 
                    ax.set_yticklabels([]) 
                ti += 1
        ax1 = fig.get_axes()[-1]
        ax2 = fig.get_axes()[-(1+(nrow-1)*ncol)]
        cax = fig.add_axes([ax1.get_position().x1+0.01,ax1.get_position().y0, 0.01, ax2.get_position().y0])
        cb = fig.colorbar(im,  cax=cax)
        locator = LogLocator()
        cb.ax.yaxis.set_major_locator(locator)
        if 'label' in kwargs.keys(): cb.set_label(label=kwargs['label'], labelpad=1.)
        plt.gcf().text(0.5, 0.008, f"Azimuth $\\varphi$ [deg]", fontsize='xx-large')
        plt.gcf().text(0.008, 0.5, f"Zenith $\\theta$ [deg]",  fontsize='xx-large', rotation='vertical')



if __name__ == "__main__":

    tel = dict_tel['BR']
    ####default arguments/paths are written in a yaml config file associated to telescope
    main_path = Path(__file__).parents[1]#MAIN_PATH# #
    print(f"main_path: {main_path}")
    with open( str(main_path / "files" / "telescopes" / tel.name /"run.yaml") ) as fyaml:
        try: def_args = yaml.load(fyaml, Loader=yaml.SafeLoader)
        except yaml.YAMLError as exc: print(exc)

    out_path = main_path / "out" / tel.name
    out_path.mkdir(parents=True, exist_ok=True)
    t_start = time.perf_counter()
    home = Path.home()
    tomoRun = def_args["reco_tomo"]["run"]

    if isinstance(tomoRun, list): 
        input_tomo= []
        freco_tomo, finlier_tomo = [], []
        for run in tomoRun:
            input_tomo = home / run
            freco_tomo.append(glob.glob(str(input_tomo / f"*reco*") )[0] )
            finlier_tomo.append(glob.glob(str(input_tomo / f"*inlier*") )[0] )
    else: 
        input_tomo = home / def_args["reco_tomo"]["run"]
        freco_tomo = glob.glob(str(input_tomo / f"*reco*") )[0] 
        finlier_tomo = glob.glob(str(input_tomo / f"*inlier*") )[0] 
    start_time = time.time()

    # print(plt.rcParams.keys())
    file_sort = out_path / "reco_sort.csv.gz"
    if file_sort.exists():
        print(f"Load file {file_sort}")
        index_col_date = -2 #penultimate column
        #df_sort = pd.read_csv(file_sort,  compression='gzip', delimiter='\t', index_col=index_col_date, parse_dates=[index_col_date]) #DateTimeIndex fmt
        df_sort = pd.read_csv(file_sort,  compression='gzip', delimiter='\t', parse_dates=[index_col_date])
    else : 
        kwargs_dat = { "index_col": 0, "delimiter": "\t", "nrows": None}
        reco_data_tomo = RecoData(file=freco_tomo, telescope=tel, kwargs=kwargs_dat)
        df_ini = reco_data_tomo.df.copy()
        df_ini.reset_index(inplace=True) 
        print(f"time_sort_df(..) -- {time.time() - start_time:.1f} s")
        df_sort = time_sort_df(df_ini)
        df_sort.to_csv(file_sort, compression='gzip', sep='\t')
        print(f"Save file {file_sort} -- {time.time() - start_time:.1f} s")
    
    df_sort['timedelta'] = pd.to_timedelta(df_sort['timedelta'], unit='s') 

    freq = '10D' 

    ts = TimeSerie(tel, df_sort, key='timedelta', freq=freq)

    input_calib = home / def_args["reco_calib"]
    freco_cal = glob.glob(str(input_calib / f"*reco*") )[0] 
    reco_data_cal = RecoData(file=freco_cal, telescope=tel)
    hm_cal = HitMap(tel, reco_data_cal.df)
    
    flux_model_tel_path = main_path / "files" / "telescopes" /  tel.name /  "flux"
    file_int_flux_sky = flux_model_tel_path / 'integral_flux_opensky.pkl'
    with open(str(file_int_flux_sky), 'rb') as f: 
        int_flux_opensky = pickle.load(f)
    
    acceptance=Acceptance(tel, hm_cal, int_flux_opensky)
    acceptance.compute()
    args = {'hitmap' : None, 'acceptance' : acceptance, 'time' : None}
    
    raypath = RayPathSoufriere[tel.name]
    thickness = {name: rp['thickness'] for name, rp in raypath.raypath.items() }
    mask = {name : np.isnan(tk) for name,tk in thickness.items()}
    args['mask'] = mask
    
    ts.compute(Flux(), **args)
    ts.save(out_path/f'estimate_serie_f{freq}.pkl')

    for name, _ in tel.configurations.items():
        fig, ax  = plt.subplots(figsize=(12,7))
        x, y, yunc = (ts.df_serie['bin_low'] + ts.df_serie['bin_up'])[:-1] /2 , ts.df_serie[f'mean_{name}'][:-1], ts.df_serie[f'unc_{name}'][:-1]
        print(f"ts.df_serie={ts.df_serie.head}")
        #print(f'x, y, yerr = {x}, {y}, {yunc}')
        kwargs_ts= {'color': 'blue', 'linewidth':0.5, 'alpha': 0.5}
        y0 = y.iloc[0] 
        ynew = (y - y0) / y0 
        yunc = yunc / y0
        #print(ynew, yunc)

        ts.plot_timeserie(ax, x, ynew, yunc, **kwargs_ts)
        ax.set_ylabel('Relative flux variation $(\\langle I \\rangle - \\langle I_{0} \\rangle) / \\langle I_{0} \\rangle$')
        ax.set_xlabel('Time')
        datetime_ticks = [datetime.fromtimestamp(int(ts)).strftime('%d-%m-%y') for ts in ax.get_xticks()]
        ax.set_xticklabels(datetime_ticks, rotation=45, ha='right')
        fig.tight_layout()
        figfile = out_path / f"relflux_timeserie_{name}_f{freq}.png"
        fig.savefig(figfile)
        print(f"Save figure {figfile}")
        plt.close()
    
    
    for key, conf in tel.configurations.items():
        panels = conf.panels
        front, rear = panels[0],panels[-1]
        ray_matrix = tel.get_ray_matrix(front, rear)

        #grid_x, grid_y =  ray_matrix[:,:,0], ray_matrix[:,:,1]
        grid_x, grid_y = tel.azimuthMatrix[key], tel.zenithMatrix[key]
        grid_z = ts.estimate_serie['estimate'][key]
        nn = grid_z[0] != 0
        vmin, vmax = np.nanmin(grid_z[0][nn]), np.nanmax(grid_z[0])
        kwargs = {'cmap':'jet_r', 'norm' : LogNorm(vmin=vmin, vmax=vmax), 'label' : 'Transmitted Flux [cm$^{-2}$.s$^{-1}$.sr$^{-1}$]' }
        fig = plt.figure(figsize=(16,12))
        ts.plot_maps(fig, grid_x, grid_y, grid_z, **kwargs)
        figfile = out_path / f'estimate_serie_{key}_f{freq}.png'
        plt.savefig(figfile)
        print(f'Save figure {figfile}')
        plt.close()



    print(f"End -- {time.time() - start_time:.1f} s")


