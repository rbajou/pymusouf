#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from argparse import ArgumentError
from dataclasses import dataclass, field
from typing import List, Union, Dict
from enum import Enum, auto
import numpy as np
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
import matplotlib.pyplot as plt
plt.rc('font', size=12)
plt.rc('axes', labelsize=12)
plt.rc('xtick', labelsize=12)
plt.rc('ytick', labelsize=12)
plt.rc('legend', fontsize=12)
from matplotlib.gridspec import GridSpec
from matplotlib.offsetbox import AnchoredText
from matplotlib.colors import LogNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable
import inspect
import scipy.io as sio
from scipy.optimize import curve_fit
import pandas as pd
import pylandau
import os
from pathlib import Path
from datetime import datetime, date, timezone
filename = inspect.getframeinfo(inspect.currentframe()).filename
script_path = os.path.dirname(os.path.abspath(filename))
#personal modules
from telescope import Telescope, Panel
from tracking import InputType
from utils import tools, functions


@dataclass
class Observable:
    name : str
    value: Union[dict, np.ndarray]
    error: Union[dict, np.ndarray] = None

@dataclass
class InlierData:
    file: Union[str, List]
    input_type : InputType
    telescope : Telescope
    df : pd.DataFrame = field(init=False) 
    kwargs : Dict = field(default_factory=lambda : {"index_col":0, "delimiter":'\t'})
    index : List = field(default_factory=lambda: []) #index specific evts
    is_all : bool = field(default_factory=lambda: False)
    is_inlier : bool = field(default_factory=lambda: True)
    def __post_init__(self):
        if isinstance(self.file, list):
            self.df = pd.concat([pd.read_csv(f, **self.kwargs) for f in self.file])
        else : 
            if self.file.endswith(".csv") or self.file.endswith(".csv.gz"): 
                self.df= pd.read_csv(self.file, **self.kwargs) 
            else : raise ValueError("Input file should be a .csv file.") 
        
        if len(self.index) != 0 :      
            self.df = self.df[self.df.index.isin(self.index)]
        
        if not self.is_all : 
            if 'inlier' in list(self.df.columns):
                ####RANSAC tagging
                if self.is_inlier : 
                    self.df = self.df[self.df['inlier']==1]
                else : 
                    self.df = self.df[self.df['inlier']==0]

@dataclass
class RecoData: 
    file : str
    input_type : InputType
    telescope : Telescope
    df : pd.DataFrame = field(init=False) 
    kwargs : Dict = field(default_factory=lambda : {"index_col":0, "delimiter":'\t'})
    index : List = field(default_factory=lambda: []) #index specific evts
    is_all : bool = field(default_factory=lambda: False)
    is_inlier : bool = field(default_factory=lambda: True)
    def __post_init__(self):
        if isinstance(self.file, list):
            self.df = pd.concat([pd.read_csv(f, **self.kwargs) for f in self.file])
        else : 
            if self.file.endswith(".csv") or self.file.endswith(".csv.gz"): 
                self.df= pd.read_csv(self.file, **self.kwargs) 
            else : raise ValueError("Input file should be a .csv file.")  
        
        if len(self.index) != 0 :      
            self.df = self.df[self.df.index.isin(self.index)]

        if not self.is_all : 
            if 'inlier' in list(self.df.columns):
                ####RANSAC tagging
                if self.is_inlier : 
                    self.df = self.df[self.df['inlier']==1]

                else : 
                    self.df = self.df[self.df['inlier']==0]
                
class EventType(Enum):
    GOLD = auto()
    MAIN = auto()
    PRIMARY = auto()

class DataType(Enum):
    CALIB = auto()
    TOMO  = auto()

        
class Cut:
    def __init__(self, column:str, vmin:float=None, vmax:float=None, label:str=""):
        self.column = column            
        self.vmin = vmin
        self.vmax = vmax
        self.label = label
        self.evtID = None
    def __call__(self, df:pd.DataFrame):
        if self.column not in df.columns: raise ValueError(f"'{self.column}' not in '{df.columns}'")
        self.cut = None
        ix = df.index
        if self.vmin is not None and self.vmax is not None:
            self.cut = (self.vmin<df[self.column])  & (df[self.column]<self.vmax)
        elif self.vmin is not None:
            cut_vmin = (self.vmin<df[self.column])
            self.cut =  cut_vmin
        elif self.vmax is not None:
            self.cut (df[self.column]<self.vmax)
        else : self.cut = np.ones(shape=len(ix), dtype=bool)
        self.loss = len(self.cut[self.cut== False])/ len(ix)
        df_new = df[self.cut]
        self.evtID  = df_new.index
        return df_new
                
class AnaBase: 
    def __init__(self, recofile:RecoData, label:str, evtIDs:list=None, tlim:tuple=None, cuts:List=None):
        self.recofile = recofile
        self.df= self.recofile.df
        self.label  = label
        #if tlim is None: self.tlim = (0, int(datetime(2032, 12, 31, hour=23,minute=59,second=59).replace(tzinfo=timezone.utc).timestamp()))
        #else : self.tlim = tlim
        #self.df= self.df.loc[ ( (self.tlim[0]<self.df['timestamp_s']) & (self.df['timestamp_s']<self.tlim[1]) )]

        self.df_gold= self.df.loc[self.df['gold']==1.]
        
        if cuts is not None:
            for cut in cuts: self.df = cut(self.df)
            
        if evtIDs:
            df_tmp = self.df.loc[evtIDs, : ] #Reconstructed primaries
            self.df= df_tmp[~df_tmp.index.duplicated(keep='first')]
            self.evtIDs=evtIDs
        else: 
            self.evtIDs = list(self.df.index)


def GoF(ax, df, column, color:str="blue", is_gold:bool=False, *args, **kwargs):
    if column not in df.columns: raise ValueError(f"'{column}' not in '{df.columns}'")
    entries, edges = np.histogram(df[column], *args, **kwargs)
    norm = np.sum(entries)
    fmain = entries/norm
    centers    = 0.5*(edges[1:]+edges[:-1])
    widths     =   edges[1:]-edges[:-1] 
    handles=[]
    hdl = ax.bar(centers, fmain, widths, color=color)
    handles.append(hdl)
    if is_gold: 
        entries_gold, edges = np.histogram(df[df["gold"]==1][column],  *args, **kwargs)
        fgold = entries_gold/norm
        centers    = 0.5*(edges[1:]+edges[:-1])
        widths     =   edges[1:]-edges[:-1] 
        hdl = ax.bar(centers, fgold, widths, color="orange")
        handles.append(hdl)
    ax.legend(handles=handles)
    ax.tick_params(axis='both', which='both', bottom=True)
    ax.xaxis.set_major_locator(MultipleLocator(1))
    ax.xaxis.set_major_formatter(FormatStrFormatter('%d'))
    ax.xaxis.set_minor_locator(MultipleLocator(0.2))
    ax.set_xlabel('goodness-of-fit', fontsize=14)
    ax.set_ylabel('probability density', fontsize=14)         
    plt.close()
def GoF_inlier_outlier(df:pd.DataFrame, outfile:str):
    #####GoF
    ####Check if 
    l_subcol = ["ninl", "noutl", "rchi2"]
    if not all(col in list(df.columns) for col  in l_subcol):
        return f"Check if {l_subcol} in {list(df.columns)}"
    
    range_gof= [0,10]

    chi2_noutl = { n : df[df["noutl"] == n]["rchi2"] for n in np.arange(0,6)}
    chi2_noutl[f">={6}"] = df[df["noutl"] >= 6]["rchi2"]
    fig = plt.figure(constrained_layout=True)
    ax = fig.subplot_mosaic([["outliers", "inliers"]], sharey=True)
    nbins=50
    entries, x = np.histogram(df["rchi2"], bins=nbins, range=range_gof)
    xc = (x[1:] + x[:-1]) /2
    w = x[1:] - x[:-1]
    norm = np.sum(entries)
    #GoF(ax=ax["outliers"], df=df, is_gold=True, column='rchi2', bins=nbins, range=[0,10])
    ax["outliers"].bar(xc, entries/norm, w ,color="blue", alpha=0.5, fill=False, edgecolor="blue" )
    bottom = np.zeros(len(entries))
    for i, ((k,chi2), color) in enumerate(zip(chi2_noutl.items(), [ "black", "purple", "blue", "green", "yellow", "orange", "red"] )):
        e, x = np.histogram(chi2, bins=nbins, range=range_gof )
        e_norm =e/norm
        ax["outliers"].bar(xc, e/norm, w ,color=color, alpha=0.5, label=k, bottom=bottom) #bottom=e[-1]
        bottom += e_norm
    ax["outliers"].legend(fontsize=14, title="#outliers")
    ax["outliers"].set_xlabel("GoF")
    ax["outliers"].set_ylabel("probability density")
    chi2_ninl = { n : df[df["ninl"] == n]["rchi2"] for n in np.arange(3,6)}
    chi2_ninl[f">={6}"] = df[df["ninl"] >= 6]["rchi2"]
    bottom = np.zeros(len(entries))
    for i, ((k,chi2), color) in enumerate(zip(chi2_ninl.items(), [ "green", "yellow", "orange", "red"] )):
        e, x = np.histogram(chi2, bins=nbins, range=range_gof )
        e_norm =e/norm
        ax["inliers"].bar(xc, e/norm, w ,color=color, alpha=0.5, label=k, bottom=bottom) #bottom=e[-1]
        bottom += e_norm

    ax["inliers"].legend(fontsize=14, title="#inliers")
    ax["inliers"].set_xlabel("GoF")
    plt.savefig(outfile)
    plt.close()
    
class EvtRate:
    def __init__(self, df:pd.DataFrame, dt_gap:int=3600):
        self.df = df
        #self.timestamp_s = np.zeros(len(self.df))
        if 'timestamp_s' in self.df.columns : 
            pass
        else : 
            try : 
                #timestamp_s = self.df.index.get_level_values('timestamp_s')
                self.df = self.df.reset_index(level='timestamp_s')
            except : 
                raise KeyError("Dataframe has no 'timestamp_s' in index or column.")
            
        self.run_duration = 0
        time = self.df['timestamp_s'] + self.df['timestamp_ns']*10**(-8)
        self.nevts = len(time)
        time_sort = np.sort(time)
        dtime = np.diff(time_sort) 
        self.run_duration = np.sum(dtime[dtime < dt_gap])  # in second
        self.mean = 0
    def __call__(self, ax, width:float=3600, label:str="", tlim=None, t_off:float=0.):
        if tlim is None: tlim =  ( 0, 
        int(datetime(2032, 4, 2, hour=16,minute=00,second=00).replace(tzinfo=timezone.utc).timestamp())   )
        t_min, t_max = tlim
        if t_min > t_max: raise ArgumentError("t_min > t_max")
        time = self.df['timestamp_s'][(t_min <= self.df['timestamp_s']) & (self.df['timestamp_s'] <= t_max)].values
        
        #mask = ((t_start <= time) & (time <= t_end))
        t_start = int(np.min(time))
        t_end = int(np.max(time))
        date_start = datetime.fromtimestamp(t_start+t_off)
        self.start = date_start
        date_start = date(date_start.year, date_start.month, date_start.day )#str(datetime.fromtimestamp(data_tomo[:, 1][0]))
        date_end = datetime.fromtimestamp(t_end+t_off)
        self.end = date_end
        date_end = date(date_end.year, date_end.month, date_end.day )#str(datetime.fromtimestamp(data_tomo[:, 1][-1]))
        self.date_start, self.date_end=  date_start, date_end
        ntimebins = int(abs(t_end - t_start)/width) #hour
        print(f"run duration = {self.run_duration:1.3e}s = {self.run_duration/(3600):1.3e}h = {self.run_duration/(24*3600):1.3e}days")
        (self.nevt, self.dtbin, self.patches) = ax.hist(time, bins=ntimebins, edgecolor='None', alpha=0.5, label=f"{label}\nnevts={len(time):1.3e}")
        datetime_ticks = [datetime.fromtimestamp(int(ts)).strftime('%d/%m %H:%M') for ts in ax.get_xticks()]
        ax.set_xticklabels(datetime_ticks)
        self.nevt_tot = np.nansum(self.nevt)
        self.mean = np.nanmean(self.nevt)#np.sum(dtbin_centers*nevt)/np.sum(nevt)
        self.std = np.nanstd(self.nevt)#np.sum(nevt*(dtbin_centers-self.mean)**2)/(np.sum(nevt)-1)
        ax.set_ylabel('events', fontsize=23)
        ax.set_xlabel("time", fontsize=22)
        plt.figtext(.5,.95, f"Event time distribution from {str(date_start)} to {str(date_end)}", fontsize=14, ha='center')
        plt.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
       
        
        

class AnaHitMap: 
    def __init__(self, anabase:AnaBase, input_type:InputType, panels:List[Panel], dict_filter:dict=None):#,  binsXY:tuple, rangeXY:tuple, binsDXDY:tuple, rangeDXDY:tuple):
        self.tel = anabase.recofile.telescope
        self.panels = panels
        self.df= anabase.df
        self.input_type = input_type
        self.label = anabase.label
        self.dict_filter = dict_filter       
        self.sconfig = list(self.tel.configurations.keys()) #panels config in telescope (i.e 4-panel configuration = 2*3p + 1*4p)
        self.XY, self.DXDY, self.hDXDY = {}, {}, {}
        #brut (X,Y) hit maps
        self.binsXY = {pan.position.loc : (pan.matrix.nbarsX, pan.matrix.nbarsY) for pan in self.panels}
        self.rangeXY = { pan.position.loc : (  ( 0, int(pan.matrix.nbarsX) * float(pan.matrix.scintillator.width) ), (0, int(pan.matrix.nbarsY) * float(pan.matrix.scintillator.width) ) ) for pan in self.panels }
        
        #(DX,DY) maps : hits per telescope pixel (=line-of-sight) r_(DX,DY) 
        self.binsDXDY  = { conf :  (los.shape[0], los.shape[1]) for conf, los in self.tel.los.items()}
        w = self.tel.panels[0].matrix.scintillator.width
        self.rangeDXDY = { conf : ((np.min(los[:,:,0])*w, np.max(los[:,:,0])*w), (np.min(los[:,:,1])*w, np.max(los[:,:,1])*w) ) for conf,los in self.tel.los.items()}
        
        self.df=self.df.copy()
        cols = ['timestamp_s', 'timestamp_ns']
        cols.extend([f'{i}_{c}'  for c in self.sconfig for i in ['DX', 'DY']])
        self.df_DXDY = pd.DataFrame(index=self.df.index, columns=cols)
        #####N.B Sometimes tomography dataset contains events that share the same index evtID, so better reindex the dataframe with the row num
        #colnames= ['timestamp_ns', 'gold']#,'residuals']
        #self.df_DXDY = self.df[colnames]
        #self.mix  = pd.MultiIndex.from_arrays([self.df.index,self.df['timestamp_s']], names=['evtID', 'timestamp_s'])
        #self.df_DXDY = self.df_DXDY.set_index(self.mix)
        self.fill_dxdy()
    
        
    def fill_dxdy(self):
        df=self.df.copy()
        #df = df.set_index(self.mix)
        for pan in self.tel.panels : 
            key = pan.position.loc
            xpos, ypos = f"X_{key}", f"Y_{key}"
            ((Xmin, Xmax), (Ymin, Ymax)) = self.rangeXY[key]
            sel =  ( (Xmin< df[xpos]) & (df[xpos]<Xmax) &  (Ymin< df[ypos]) & (df[ypos]<Ymax) )
            self.XY[key] = [df[sel][xpos].values, df[sel][ypos].values]
            
        self.idx, DX, DY  =  {}, {}, {}
        for conf, panels in self.tel.configurations.items():
            front, rear = panels[0].position.loc, panels[-1].position.loc
            ((Xminf, Xmaxf), (Yminf, Ymaxf)) = self.rangeXY[front]
            ((Xminr, Xmaxr), (Yminr, Ymaxr)) = self.rangeXY[rear]
            xposf, yposf = f"X_{front}", f"Y_{front}"
            xposr, yposr = f"X_{rear}", f"Y_{rear}"
            sfront =  ( (Xminf< df[xposf]) & (df[xposf]<Xmaxf) &  (Yminf< df[yposf]) & (df[yposf]<Ymaxf) )
            srear = ( (Xminr < df[xposr]) & (df[xposr]<Xmaxr) &  (Yminr< df[yposr]) & (df[yposr]<Ymaxr) )
            sel = (sfront & srear)
            ###apply filter on evt ids
            if self.dict_filter is not None:                
                filter = self.dict_filter[conf]
                self.idx[conf]  = df[sel].loc[filter].index
                #self.idx[conf]  = df[df.index.isin(filter)][sel].index
            else : self.idx[conf]  = df[sel].index
            dftmp = df.loc[self.idx[conf]] 
            DX[conf], DY[conf] =  dftmp[xposf].values - dftmp[xposr].values, dftmp[yposf].values - dftmp[yposr].values

        self.hDXDY = { conf : np.histogram2d(DX[conf], DY[conf], bins=[bdx,bdy], range=[dxlim, dylim] )[0] for (conf,(bdx,bdy)), (_,(dxlim, dylim)) in zip(self.binsDXDY.items(), self.rangeDXDY.items())  }
        
class PlotHitMap:
    """Class to plot reconstructed trajectories hit maps."""
    def __init__(self, hitmaps:List[AnaHitMap], outdir:str) :
        self.hitmaps = hitmaps
        self.sconfig = self.hitmaps[0].sconfig
        self.panels  = self.hitmaps[0].panels
        self.outdir  = outdir
        
    def XY_map(self, invert_yaxis:bool=False, transpose:bool=False):
        """Plot hit map for reconstructed primaries and all primaries"""

        if len(self.hitmaps)==0:
            raise Exception("Fill all XY vectors first")
       
        fig = plt.figure(0, figsize=(9,16))
        gs = GridSpec( len(self.panels), len(self.hitmaps), left=0.05, right=0.95, wspace=0.2, hspace=0.5)
       
        labels = [ hm.label for hm in self.hitmaps]
        for l, (name, hm) in enumerate(zip(labels, self.hitmaps)) : #dict_XY.items()): 
            tools.create_subtitle(fig, gs[l, ::], f'{name}')
            for i, p in enumerate(self.panels):
                
                key = p.position.loc
                
                X, Y = hm.XY[key]
                ax = fig.add_subplot(gs[i,l], aspect='equal')
                #ax.get_yaxis().set_visible(False)
                ax.set_title(f"{p.position.loc}")
                if i == len(self.panels)-1: ax.set_title("Rear")
                ax.grid(False)
                counts, xedges, yedges, im1 = ax.hist2d( X, Y,cmap='viridis', bins=hm.binsXY[key], range=hm.rangeXY[key] ) #im1 = ax.imshow(hXY[i])
                if transpose: 
                    ax.hist2d(Y, X, cmap='viridis', bins=hm.binsXY[key], range=hm.rangeXY[key] ) #im1 = ax.imshow(hXY[i])
                if invert_yaxis: ax.invert_yaxis()
                if i == len(self.panels)-1 : ax.set_xlabel('Y')
                ax.set_ylabel('X')
                divider1 = make_axes_locatable(ax)
                cax1 = divider1.append_axes("right", size="5%", pad=0.05)
                plt.colorbar(im1, cax=cax1, format='%.0e')
                #cbar.ax.tick_params(labelsize=8)        

        gs.tight_layout(fig)
        plt.savefig(
            os.path.join(self.outdir,'', f'brut_hit_maps.png')
        ) 
        print(f"save {os.path.join(self.outdir,'', f'brut_hit_maps.png')}")
        plt.close()
        
    def DXDY_map(self, invert_xaxis:bool=True, invert_yaxis:bool=False, transpose:bool=False, fliplr:bool=False, flipud:bool=False):
        if len(self.hitmaps)==0:
            raise Exception("Fill all DXDY vectors first")
        labels = [ hm.label for hm in self.hitmaps]
        fig = plt.figure(1, figsize= (16,9))
        nconfigs = len(self.sconfig)
        gs = GridSpec(len(self.hitmaps), nconfigs , left=0.05, right=0.95, wspace=0.2, hspace=0.1)
        for l, (name, hm) in enumerate(zip(labels, self.hitmaps)):
            tools.create_subtitle(fig, gs[l, ::], f'{name}')
            for i, conf in enumerate(self.sconfig):
                ax1 = fig.add_subplot(gs[l,i], aspect='equal')
                #if c == 0 : 
                #    ax1.set_ylabel('$\\Delta$Y [mm]', fontsize=16)
                #else : ax1.get_yaxis().set_visible(False)
                if i == 0 : ax1.set_ylabel('$\\Delta$X [mm]')#, fontsize=16)
                ax1.set_xlabel('$\\Delta$Y [mm]')#, fontsize=16)
                DX_min, DX_max = hm.rangeDXDY[conf][0]
                DY_min, DY_max = hm.rangeDXDY[conf][1]
                h = hm.hDXDY[conf]
                if transpose: h = h.T
                if fliplr : h = np.fliplr(h)
                if flipud : h = np.flipud(h)
                h[h==0] = np.nan 
                im1 = ax1.imshow(h, cmap='viridis', norm=LogNorm(vmin=1, vmax=np.max(h[~np.isnan(h)])), extent=[DX_min, DX_max, DY_min, DY_max] )
                ax1.grid(False)
                #hist, xedges, yedges, im1 = ax1.hist2d( DY[c], DX[c], edgecolor='black', linewidth=0., bins=hm.binsDXDY[c], range=hm.rangeDXDY[c], weights=None, cmap='viridis', norm=LogNorm(vmin=1, vmax=np.max(hm.hDXDY[c]) ) ) #    
                if invert_xaxis:  ax1.invert_xaxis()
                if invert_yaxis:  ax1.invert_yaxis()
                divider1 = make_axes_locatable(ax1)
                cax1 = divider1.append_axes("right", size="5%", pad=0.05)
                plt.colorbar(im1, cax=cax1, extend='max')
        gs.tight_layout(fig)
        plt.savefig(
            os.path.join(self.outdir, f'NHits_dXdY_map.png')
        )
        print(f"save {os.path.join(self.outdir, f'NHits_dXdY_map.png')}")
        plt.close()
     
    def DXDY_map_az_ze(self, az=None, ze=None, invert_yaxis:bool=False, transpose:bool=False, fliplr:bool=False):
        ####NHits map in (Az,Ze) plane
        # ze_min, ze_max = np.min(ze), np.max(ze)
        # az_min, az_max = np.min(az), np.max(az)
        if len(self.hitmaps)==0:
            raise Exception("Fill DXDY histograms first")
        nconfigs= len(self.hitmaps[0].hDXDY)
        gs = GridSpec(len(self.hitmaps),nconfigs)

        #sns.set_style("whitegrid")
        fig = plt.figure(1, figsize= (16,9))
        for l,  hm in enumerate(self.hitmaps):
            name = hm.label
            tools.create_subtitle(fig, gs[l, ::], f'{name}')
            for c, conf in enumerate(self.sconfig):
                hist = hm.hDXDY[conf]
                ax = fig.add_subplot(gs[l,c], aspect='equal')
                h = hist[conf]
                if transpose: h = h.T
                if fliplr : h = np.fliplr(h)
                im1 = ax.imshow(h, cmap='viridis', norm=LogNorm())
                #ax.grid(color='b', linestyle='-', linewidth=0.25)
                ax.grid(False)
                locs = ax.get_xticks()[1:-1]  # Get the current locations and labels.
                #print(az[c])
                new_x = [str(int(az[conf][int(l)])) for l in locs]
                new_y = [str(int(ze[conf][int(l)])) for l in locs]
                ax.set_xlabel('$\\varphi$ [deg]', fontsize=16)
                ax.set_xticks(locs, new_x)
                ax.set_ylabel('$\\theta$ [deg]', fontsize=16)
                ax.set_yticks(locs, new_y)
                if invert_yaxis : ax.invert_yaxis()
                ax.set_title(conf, fontsize=16)
                divider = make_axes_locatable(ax)
                cax = divider.append_axes("right", size="5%", pad=0.05)
                fig.colorbar(im1, cax=cax, orientation='vertical')
        gs.tight_layout(fig)
        label = '_'.join([hm.label for hm in self.hitmaps ])
        plt.savefig(
            os.path.join(self.outdir, f"NHits_AzZe_map_{label}.png")
        )
        plt.close()

 
class AnaCharge:
    """
    Class to checkout charge distributions per panel
    'Gold' events : events that formed exactly 1 XY-hit per panel (~MIP events)
    """
    def __init__(self, inlier_file:RecoData,  outlier_file:RecoData, outdir:str, label:str, evttype:EventType=None):
        self.inlier_file  = inlier_file
        self.outlier_file = outlier_file
        self.df_inlier  = self.inlier_file.df
        self.df_outlier = self.outlier_file.df
        if  evttype is not None: 
            if evttype.name == "GOLD" : 
               self.df_inlier = self.df_inlier.loc[self.df_inlier['gold']==1.]
               self.df_outlier = self.df_outlier.loc[self.df_outlier['gold']==1.]
        self.outdir = outdir 
        self.evtNo_reco = list(self.df_inlier.index)
        self.label = label 
        self.panels = inlier_file.telescope.panels
        self.binsDX = 2*self.panels[0].matrix.nbarsX-1
        self.binsDY = 2*self.panels[0].matrix.nbarsY-1
        self.ADC_XY_inlier  = [] #sum(ADC_X+ADC_Y) 
        self.ADC_XY_outlier = []
        self.fill_charge_arrays()
        
    def fill_charge_arrays(self):
        Z = np.sort(list(set(self.df_inlier['Z'])))
        sumADC_XY_in = self.df_inlier['ADC_X'] + self.df_inlier['ADC_Y'] 
        self.df_inlier = self.df_inlier.assign(ADC_SUM=pd.Series(sumADC_XY_in).values)
        sumADC_XY_out = self.df_outlier['ADC_X'] + self.df_outlier['ADC_Y'] 
        self.df_outlier = self.df_outlier.assign(ADC_SUM=pd.Series(sumADC_XY_out).values)
        
        self.ADC_XY_inlier = [ np.array(self.df_inlier.loc[self.df_inlier['Z'] ==z]["ADC_SUM"].values) for z in Z]
        self.ADC_XY_outlier = [ np.array(self.df_outlier.loc[self.df_outlier['Z'] ==z]["ADC_SUM"].values) for z in Z]

        
    def langau(x, mpv, eta, sigma, amp): return pylandau.langau(x, mpv, eta, sigma, amp, scale_langau=True) 
    
    
    def fit_dQ(self, q:np.ndarray, nbins:int=100, is_scaling:bool=False, input_type:str="DATA"):    
        fscale= 1 
        if is_scaling:   fscale = 1e3 ###needed to fit with Landau function from pylandau
        q = q*fscale
        xmax_fig = np.mean(q) + 5*np.std(q)
        xmax_fit = xmax_fig
        entries, bins = np.histogram(q,  range=(0,  xmax_fit), bins =  nbins)
        widths = np.diff(bins)
        bin_centers = np.array([ (bins[i+1]+bins[i])/2 for i in range(len(bins)-1)])
        N = len(bin_centers)     
        mean = sum( bin_centers * entries) / sum(entries)#(n*max(nentries))#sum(np.multiply(bin_centers, nentries)) / n
        sigma = np.sqrt( sum( entries*(bin_centers - mean)**2  )  /  ((N-1)/N*sum(entries))  )
        rough_max = np.max( bin_centers[bin_centers>0][entries.argmax()] )#bin_centers[np.where(entries==max(entries))] )
        #fitrange =  ( ( rough_max*0.2 < bin_centers ) & (bin_centers< 3*rough_max ) )
        fitrange =  ( ( rough_max*0.2 < bin_centers ) & (bin_centers < xmax_fit ) )
        yerr = np.array([np.sqrt(n) for n in entries[fitrange] ]) 
        yerr[entries[fitrange]<1] = 1
        xfit = bin_centers[fitrange]
        yfit = entries[fitrange]
        bin_w = np.diff(bin_centers[fitrange] )
        mpv, eta, amp = int(rough_max), sigma, np.max(entries)
        print(mpv, eta, sigma, amp)
    
        if input_type == "MC" : 
            values, pcov = curve_fit(pylandau.landau, xfit, yfit,
                sigma=yerr,
                absolute_sigma=False,
                p0=(mpv, eta, amp)
                # bounds=( (0.1*mpv, 0.1*eta, 0.1*amp ),
                #         (3*mpv, 3*eta, 3*amp )
                #         )
                )
            errors = np.sqrt(np.diag(pcov))
            values[0], values[1] =  values[0]/fscale, values[1]/fscale
            errors[0], errors[1] =  errors[0]/fscale, errors[1]/fscale
        elif input_type == "DATA" : 
            values, errors, m = functions.fit_landau_migrad(
                                            xfit,
                                            yfit,
                                            p0=[mpv, eta, sigma, amp],#
                                            limit_mpv=(rough_max*0.8,rough_max*1.2), #(10., 100.)
                                            limit_eta=(0.3*eta,1.5*eta), #(0.8*eta,1.2*eta)
                                            limit_sigma=(0.3*sigma,1.5*sigma), #(0.8*sigma,1.2*sigma)
                                            limit_A=(0.8*amp,1.2*amp) #(0.8*amp,1.2*amp)
                                            ) 
            values[0], values[1], values[2] =  values[0]/fscale, values[1]/fscale, values[2]/fscale
            errors[0], errors[1], errors[2] =  errors[0]/fscale, errors[1]/fscale, errors[2]/fscale 
        else : raise ValueError('Unknown InpuType.')
     
        xfit, yfit = xfit/fscale, yfit/fscale

        xyrange = [[np.nanmin(xfit), np.nanmax(xfit)], [np.nanmin(yfit), np.nanmax(yfit)]]
        return values, errors, xyrange

    def plot_charge_panels(self, charge:dict, nbins:int=100,  fcal:dict=None, unc_fcal:dict=None, xlabel:str='dQ [ADC]', is_scaling :bool=False,input_type: InputType=InputType.DATA) : 

        fig, f_axs = plt.subplots(ncols=len(self.panels), nrows=1, figsize=(16,9), sharey=True)
        
        if fcal is None: fcal = {p.ID : 1 for p in self.panels}
        
        if input_type == 'DATA': lpar_fit = ['MPV', 'eta', 'sigma', 'A']##parameters landauxgaussian distribution
        elif input_type == 'MC': lpar_fit = ['MPV', 'eta', 'A'] ##parameters landau distribution
        else : raise ValueError()
        lmes = ['value', 'error']
        lpar = lpar_fit
        lpar.extend(['xmin', 'xmax', 'entries'])
        mix = pd.Index([pan.ID for pan in self.panels], name="panel_id")
        cols = pd.MultiIndex.from_tuples([(par, mes) for par in lpar for mes in lmes])
        df_par = pd.DataFrame(columns=cols,index=mix)
        
        
        for i, ((tag, color, do_fit), charge_panel) in enumerate(charge.items()):
            df_entries= pd.DataFrame(index=np.arange(0, nbins)) ###(entries, bin) / panel 
            df_percentiles = pd.DataFrame(index=np.arange(1, 101), columns=[f"panel_{panel.ID}" for panel in self.panels])
            for col, panel in enumerate(self.panels): 
                
                if len(charge_panel[col]) == 0 : continue
                
                q = charge_panel[col]/fcal[panel.ID]
                if fcal[panel.ID] != 1: xlabel="dE [MIP fraction]"
                ax = f_axs[col]
                xmax_fig = np.mean(q) + 5*np.std(q)
                ax.set_xlim(0, xmax_fig)
                ax.set_xlabel(xlabel) 
                if col == 0 : ax.set_ylabel("entries") 
                entries, bins = np.histogram(q,  range=(0,  xmax_fig), bins =  nbins)
                widths = np.diff(bins)
                
                ax.bar(bins[:-1], entries, widths,color='None', edgecolor=color, label=f"{tag}")
                
                
                if fcal is not None and unc_fcal is not None: 
                    ####if calibration constante C_ADC/MIPfraction is parsed (measured with 'golden' events)
                    if i == 0 : ax.axvspan(1-unc_fcal[panel.ID]/fcal[panel.ID], 1+unc_fcal[panel.ID]/fcal[panel.ID], color='orange', alpha=0.2,
                               label = "Gold evts peak")
                    ax.set_xlim(0, xmax_fig)
                

                ax.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
                ax.tick_params(axis='both')
                ax.legend(loc='upper right', title=f"{panel.position.loc} (ID={panel.ID})")

                bin_centers = np.array([ (bins[i+1]+bins[i])/2 for i in range(len(bins)-1)])    
                bin_w = np.diff(bin_centers)

                # df_entries.loc[fitrange,f'bin_{panel.ID}'] = xfit / fscale
                # df_entries.loc[fitrange,f'entry_{panel.ID}'] = yfit
                for per in range(1,101) : df_percentiles.loc[per][f"panel_{panel.ID}"] = np.around(np.percentile(q,per),3)

                if not do_fit : continue
                ## FIT plot and params              
                values, errors, xyrange = self.fit_dQ(q, nbins=nbins, is_scaling=is_scaling, input_type=input_type)
                [xmin, xmax], [ymin, ymax] = xyrange
                sym_err = [np.max(np.abs(e)) for e in errors]
         
                for par, val, err in zip(lpar_fit, values, sym_err):
                    df_par.loc[panel.ID][par] = [np.around(val,3),np.around(err,3)] 
         
                df_par.loc[panel.ID]['xmin'] = [np.around(xmin,3),np.around(bin_w[0],3)] 
                df_par.loc[panel.ID]['xmax'] = [np.around(xmax,3),np.around(bin_w[-1],3)]  
                df_par.loc[panel.ID]['entries'] = [int(np.sum(entries)), 0]
                print("df_par =", df_par)
                str_par_fit = ['MPV', '$\\eta$', '$\\sigma$', 'A']
                if input_type == 'MC':str_par_fit = ['MPV', '$\\eta$', 'A']

                label = ""#"$\\bf{"+ name +"}$\n"
                label += ' '.join('{}={:0.1f}$\\pm${:0.1f} ADC\n'.format(p, value, error) for p, value, error in zip(str_par_fit[:-1], values[:-1], sym_err [:-1]  )   )
                
                ax.legend(loc='best', title=f"Panel {panel.ID}")
                amp = df_par.loc[panel.ID]['A']['value']
                ax.set_ylim(0, 1.2*amp)
            
                xfit = np.linspace(xmin, xmax, 100)
                if input_type == "MC" : 
                    ax.plot(xfit, pylandau.landau(xfit, *values), '-', label=label+'{}={:0.1f}$\\pm${:0.1f}'.format(str_par_fit[-1], values[-1],sym_err[-1]), color=color )
                elif input_type == "DATA": 
                    ax.plot(xfit, pylandau.langau(xfit, *values), '-', label=label+'{}={:0.1f}$\\pm${:0.1f}'.format(str_par_fit[-1], values[-1],sym_err[-1]), color=color )
                else : 
                    raise ValueError('Unknown InpuType.')
                        
        
            #ofile_ent = os.path.join(self.outdir, f"entries_dQ_{tag}.csv")
            ofile_perc = os.path.join(self.outdir, f"percentiles_dQ_{tag}.csv")
            #df_entries.to_csv( ofile_ent, sep='\t')
            df_percentiles.to_csv( ofile_perc, sep='\t')
            if do_fit : 
                ofile_par = os.path.join(self.outdir, f'fit_dQ_{tag}.csv')
                #df_par = pd.DataFrame.from_dict(dict_par, columns=['value', 'error'],orient='index')
                df_par.to_csv( ofile_par, sep='\t')
            
        fig.tight_layout()
        plt.savefig(
            os.path.join(self.outdir, "",  f"charge_distributions.png")
        )
        plt.close()


    def scatter_plot_dQ(self, fig, gs, dQx:dict, dQy:dict, rangex:tuple=None, rangey:tuple=None, nbins:int=100) : 
        for i, (((tagx, colorx, do_fitx), valx), ((tagy, colory, do_fity),valy))  in enumerate( zip(dQx.items(), dQy.items() )) :

            ax = fig.add_subplot(gs[1, 0])
            ax_histx = fig.add_subplot(gs[0, 0], sharex=ax)
            #if fcal[panel.ID]!=1 : 
                #ax.set_xlabel('dQ [MIP fraction]', fontsize=fontsize) 
            atx = AnchoredText('dQ_front',
                        prop=dict(size=14), frameon=True,
                        loc='upper right',
                        )
            atx.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
            ax_histx.add_artist(atx) 
            ax_histy = fig.add_subplot(gs[1, 1], sharey=ax)
            aty = AnchoredText('dQ_rear',
                        prop=dict(size=14), frameon=True,
                        loc='upper right',
                        )
            aty.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
            ax_histy.add_artist(aty) 
            ax_histx.tick_params(axis="x", labelbottom=False)
            ax_histx.set_ylabel("entries", fontsize=10)
            ax_histx.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
            ax_histx.set_yscale('log')
            ax_histy.tick_params(axis="y", labelleft=False)
            ax_histy.set_xlabel("entries", fontsize=10)
            ax_histy.ticklabel_format(axis="x", style="sci", scilimits=(0,0))
            ax_histy.set_xscale('log')
        # the scatter plot:
        #data = np.vstack([x, y])
        #kde = gaussian_kde(data)
    
        
        #Xgrid, Ygrid = np.meshgrid(x, y)
        #Z = kde.evaluate(np.vstack([Xgrid.ravel(), Ygrid.ravel()]))

    # Plot the result as an image
        #ax.imshow(Z.reshape(Xgrid.shape),
            #    origin='lower', aspect='auto',
            #    extent=[0, max(x), 0, max(y)],
            #    cmap='Blues')
        #cb = plt.colorbar()
        #cb.set_label("density")
        
    
            
            if rangex is None and rangey is None : 
                xmax_2d = np.mean(valx) + 5*np.std(valx)
                ymax_2d = np.mean(valy) + 5*np.std(valy)
                rangex, rangey= (0,xmax_2d), (0,ymax_2d)
               
            #print("rangex, rangey= ",rangex, rangey)
            entries_x, bins_x = np.histogram(valx,  range=rangex, bins =  nbins)#ax_histx.hist(valx, bins=nbins, range =rangex, color ='lightgreen', alpha=1., label='X', edgecolor='none')
            widths_x = np.diff(bins_x)
            ax_histx.bar(bins_x[:-1], entries_x, widths_x,color='None', edgecolor='lightgreen', label=f"X")
            ax_histx.set_xlim(rangex)
            
            entries_y, bins_y = np.histogram(valy,  range=rangey, bins =  nbins)
            widths_y = np.diff(bins_y)
            ax_histy.barh(bins_y[:-1], entries_y, widths_y,color='None', edgecolor='lightgreen', label=f"Y")
            ax_histy.set_ylim(rangey)
            #entries_y, bins_y, _ = ax_histy.hist(valy, bins=nbins, range =rangey, orientation='horizontal', color = 'lightgreen', alpha=1., label='Y', edgecolor='none')
            #bins_center_x = np.array([ (bins_x[i+1]+bins_x[i])/2 for i in range(len(bins_x)-1)])
            #bins_center_y = np.array([ (bins_y[i+1]+bins_y[i])/2 for i in range(len(bins_y)-1)])
            
            h, xedges, yedges =  np.histogram2d(valx, valy, bins=nbins, range=[rangex, rangey])#, norm=mcolors.PowerNorm(gamma), cmap='jet') #mcolors.LogNorm(vmin=1, vmax=max_)
            # gamma = 0.3
            # h, xedges, yedges, im = ax.hist2d(valx, valy, bins=nbins, range=[rangex, rangey], norm=mcolors.PowerNorm(gamma), cmap='jet') #mcolors.LogNorm(vmin=1, vmax=max_)
            #im = ax.imshow(, interpolation='none', origin='lower', cmap = 'jet')
            Z = np.ma.masked_where(h < 1, h).T
            xc, yc = xedges, yedges#(xedges[1:] + xedges[:-1])/2, (yedges[1:] + yedges[:-1])/2
            X, Y = np.meshgrid(xc, yc)
            im = ax.pcolormesh(X,  Y, Z, shading='auto', cmap='jet')#,norm=LogNorm(vmin=vmin, vmax=vmax) )
        return ax, h, xedges, yedges



if __name__ == '__main__':
    pass