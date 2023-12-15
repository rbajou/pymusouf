#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Raphaël Bajou
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
import pandas as pd
from pathlib import Path
import time
import warnings
warnings.filterwarnings("ignore")
from typing import Union
from datetime import datetime, timezone
#package module(s)
from telescope import dict_tel, Telescope
from reco import RecoData, RansacData, HitMap


params = {'legend.fontsize': 'xx-large',
         'axes.labelsize': 'xx-large',
         'axes.titlesize':'xx-large',
         'xtick.labelsize':'xx-large',
         'ytick.labelsize':'xx-large',
         'axes.labelpad':10,
         'mathtext.fontset': 'stix',
         'font.family': 'STIXGeneral',
         'axes.grid' : True,
         'grid.linestyle' : 'dotted',
         }
plt.rcParams.update(params)


class Filter:
    

    def __init__(self, telescope:Telescope, df:pd.DataFrame):
        self.tel = telescope
        self.df = df.copy()
        self.dict_filter = {conf : [] for conf, _ in self.tel.configurations.items()}
        self.label = None


class FilterInlierMultipliciy(Filter):


    def __init__(self, telescope:Telescope, df:pd.DataFrame, is_inlier:bool=True, label:str="multiplicity"):
        #print("\tFilterInlierMultipliciy")
        #start_time = time.time()
        Filter.__init__(self, telescope, df)
        self.label = label
        self.get_df_multi(is_inlier)
        #print(f"End FilterInlierMultipliciy -- {(time.time() - start_time):.1f}  s")


    def get_df_multi(self, is_inlier):
        
        df_tmp = self.df.groupby([self.df.index, "Z", "inlier"]).count()#six
        Z = [p.position.z for p in self.tel.panels]
        lidx = [ (idx, z, i)  for idx in self.df.index.unique()  for z in Z for i in [0,1] ] 
        
        #print("Before reindexing")
        mix = pd.MultiIndex.from_tuples(lidx, names=['ix', 'Z', 'inlier']) #six
        start_time = time.time()
        ####will fill missing panel with 0 
        df_tmp = df_tmp.reindex(mix, fill_value=0) #multiindex '(evtID, z)'
        print(f"\tAfter multi reindexing -- {(time.time() - start_time):.3f}  s ")
        
        str_rand_col= 'timestamp_ns' ###get count in column (could be any column name)
        df_multi = df_tmp[[str_rand_col]] 
        df_multi.columns = df_multi.columns.str.replace(str_rand_col, 'm') #rename column 'm' = 'multiplicity'
        self.dict_df_multi=  {pan.ID : df_multi.xs((pan.position.z, int(is_inlier)), level=[1,2], drop_level=[1,2])['m'] for pan in self.tel.panels }
       

    def get_dict_filter(self, hitmap:HitMap, cut_front:int, cut_rear:int):

        for conf, panels in self.tel.configurations.items():
            
            front, rear = panels[0], panels[-1]
            idx_conf = hitmap.idx[conf]
            
            df_front = self.dict_df_multi[front.ID].loc[idx_conf]  #index 'evtID'
            df_rear  = self.dict_df_multi[rear.ID].loc[idx_conf]  #index 'evtID'
            
            idx_front  = df_front[df_front < cut_front].index
            idx_rear   = df_rear [df_rear < cut_rear].index #
            idx_front_rear = list(set(idx_rear).intersection(set(idx_front)))
            
            self.dict_filter[conf].extend(idx_front_rear)
            self.dict_filter[conf] = list(set(self.dict_filter[conf]))
        



class FilterToF(Filter):

    def __init__(self, telescope:Telescope, df:pd.DataFrame):
        #print("\tFilterTof")
        Filter.__init__(self, telescope, df)
        self.df.rename_axis('evtID', inplace=True)
        self.df.reset_index(inplace=True)
        idx_min = self.df.groupby(['evtID'])['Z'].transform('min') == self.df["Z"]
        idx_max = self.df.groupby(['evtID'])['Z'].transform('max') == self.df["Z"]
        #df_tof_s = (df[idx_max].groupby(['evtID'])['timestamp_s'].max() - df[idx_min].groupby(['evtID'])['timestamp_s'].max())*10
        self.df_tof_ns = (self.df[idx_max].groupby(['evtID'])['timestamp_ns'].max() - self.df[idx_min].groupby(['evtID'])['timestamp_ns'].max())
        self.df_tof_ns *= 10
        self.tof_peak = None


    def get_tof(self, nbins:int=20, range:list=[-100,100]):
        
        entries, bins = np.histogram(self.df_tof_ns,  range=range, bins = nbins)
        bins_w=abs(bins[:-1]- bins[1:])
        bins_c=(bins[:-1]+ bins[1:])/2
        self.tof_peak = bins_c[np.argmax(entries)]
        return entries, bins_c, bins_w


    def plot_tof(self, ax:Axes, nbins:int=20, range:list=[-100,100], dtof:float=None, **kwargs):
       
        y, x, dx = self.get_tof(nbins, range)
        ax.bar(x,y,dx, **kwargs)
        ax.set_xlabel('time-of-flight [ns]')
        ax.set_ylabel('entries')
        ax.set_yscale('log')
        ax.legend()
    

    def get_dict_filter(self, dtof:float=20.):

        if self.tof_peak is None: self.get_tof()
        mask = (self.tof_peak-dtof  < self.df_tof_ns ) & (self.df_tof_ns < self.tof_peak+dtof ) 
        self.idx = self.df_tof_ns[mask].index.get_level_values(0)
        for conf, _ in self.tel.configurations.items():
            self.dict_filter[conf].extend(self.idx)
            self.dict_filter[conf] = list(set(self.dict_filter[conf]))
       

class FilterTimePeriod(Filter):


    def __init__(self, telescope:Telescope, df:pd.DataFrame, colname:str='timestamp_s'):
    
        Filter.__init__(self, telescope, df)
        try: 
            self.ts = self.df[colname]
        except : 
            raise ValueError('Check the timestamp column name in the dataframe.')

    def get_dict_filter(self, tlim:Union[list, np.ndarray]):
        '''
        tlim (array-like): (tmin, tmax) datetime or timestamp format 
        '''
        tmin, tmax = tlim
        if all( isinstance(t, datetime) for t in tlim):
            tmin, tmax = tmin.replace(tzinfo=timezone.utc).timestamp(), tmax.replace(tzinfo=timezone.utc).timestamp()
        mask =  ( tmin < self.ts ) & ( self.ts < tmax )
        self.idx = self.df[mask].index#.get_level_values(0)
        for conf, _ in self.tel.configurations.items():
            self.dict_filter[conf].extend(self.idx)
            self.dict_filter[conf] = list(set(self.dict_filter[conf]))

# class MultipleFilter:

#     def __init__(self, List[Filter]):
#         pass

def intersect_multiple_filters(dict_list):
    
    new_dict_filter = {}
    d0 = dict_list[0]
    if not all(d.keys() == d0.keys() for d in dict_list):
        return "Check that all dicts in 'list' have identical keys."
        #print(f"d={d}")
        #new_ix = []
    for key, ix in d0.items():
        new_ix = ix
        for d in dict_list:
            new_ix = list(set(d[key]).intersection(set(new_ix)))  
        new_dict_filter[key] = new_ix
        
    return new_dict_filter


if __name__=='__main__':
    
    tel = dict_tel['SB']
    reco_file = str(Path.home() / "data/SB/3dat/tomo/reco/merge/reco.csv.gz")
    reco_data = RecoData(reco_file, tel)
    ransac_file = str(Path.home() / "data/SB/3dat/tomo/reco/merge/inlier.csv.gz")
    ransac_data = RansacData(ransac_file, tel)
    df = ransac_data.df

    # fim = FilterInlierMultipliciy(tel, df)
    # hm = HitMap(tel, reco_data.df)
    # fim.get_dict_filter(hm, 2, 2)

    # ftof = FilterToF(tel, df)
    # fig, ax = plt.subplots(figsize=(12,7))
    # kwargs={'label':'tof'}
    # ftof.plot_tof(ax, **kwargs)
    # plt.show()