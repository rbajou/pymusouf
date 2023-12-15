#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from dataclasses import dataclass, field
from typing import List, Union
from enum import Enum, auto
import numpy as np
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
import matplotlib.pyplot as plt
import pandas as pd

#package module(s)
from telescope import Telescope, Panel
from tracking import InputType
from utils import tools, functions


@dataclass
class Observable:
    name : str
    value: Union[dict, np.ndarray]
    error: Union[dict, np.ndarray] = None

class RecoData: 
    
    def __init__(self, file:Union[str, List], telescope:Telescope, kwargs:dict={"index_col":0, "delimiter":'\t'}):
        self.file = file
        self.tel = telescope
        self.kwargs = kwargs
        self.df = None
        if isinstance(self.file, list):
            self.df = pd.concat([pd.read_csv(f, **self.kwargs) for f in self.file])
        else : 
            if self.file.endswith(".csv") or self.file.endswith(".csv.gz"): 
                self.df= pd.read_csv(self.file, **self.kwargs) 
            else : raise ValueError("Input file should be a .csv file.")  

       
class RansacData(RecoData): 
    
    def __init__(self, file:Union[str, List], telescope:Telescope, kwargs:dict={"index_col":0, "delimiter":'\t'}):
        RecoData.__init__(self, file, telescope, kwargs)
        try:
            ####RANSAC tagging
            self.df_inlier = self.df[self.df['inlier']==1]
            self.df_outlier = self.df[self.df['inlier']==0]
        except: 
            raise ValueError("No 'inlier' column was found in input dataframe.")
        


class EventType(Enum):
    GOLD = auto()
    MAIN = auto()
    PRIMARY = auto()

        
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
    


 


if __name__ == '__main__':
    pass