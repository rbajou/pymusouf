#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt

params = {'legend.fontsize': 'xx-large',
         'axes.labelsize': 'xx-large',
         'axes.titlesize':'xx-large',
         'xtick.labelsize':'xx-large',
         'ytick.labelsize':'xx-large',
         'axes.labelpad':10,
         'mathtext.fontset': 'stix',
         'font.family': 'STIXGeneral',
         'axes.grid' : True,
         }
plt.rcParams.update(params)

from datetime import datetime, date, timezone
from argparse import ArgumentError
from pathlib import Path
from typing import Union


class EvtRate:


    def __init__(self, df:pd.DataFrame, dt_gap:int=3600):
        self.df = df.copy()
        self.run_duration = 0
        try: 
            self.ts = self.df['timestamp_s'] #+ self.df['timestamp_ns']*1e-8
        except : 
            raise ValueError('Check the timestamp column name in the dataframe.')
        self.nevts = len(self.ts)
        time_sort = np.sort(self.ts)
        dtime = np.diff(time_sort) 
        self.run_duration = np.sum(dtime[dtime < dt_gap])  # in second
        self.mean = 0

    def __call__(self, ax, width:float=3600, label:str="",  t_off:float=0., tlim=None, **kwargs):
        if tlim is None: tlim =  ( 0, int(datetime(2032, 4, 2, hour=16,minute=00,second=00).replace(tzinfo=timezone.utc).timestamp())   )
        t_min, t_max = tlim
        if t_min > t_max: raise ArgumentError("t_min > t_max")
        mask = (t_min <= self.ts) & (self.ts <= t_max)
        time = self.ts[mask].values
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
        (self.entries, self.tbin, self.patch) = ax.hist(time, bins=ntimebins, edgecolor='None', label=f"{label}\nnevts={len(time):1.3e}", **kwargs)
        datetime_ticks = [datetime.fromtimestamp(int(ts)).strftime('%d/%m %H:%M') for ts in ax.get_xticks()]
        ax.set_xticklabels(datetime_ticks)
        ax.set_ylabel("nevents")
        ax.set_xlabel("time")
        title =  f"Event time distribution from {str(date_start)} to {str(date_end)}"
        ax.set_title(title)
        #plt.figtext(.5,.95, title, ha='center')
        plt.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
    
    def to_csv(self, file:Union[str, Path], **kwargs):
        tbinc = (self.tbin[:-1] +  self.tbin[1:]) / 2
        self.df = pd.DataFrame(data={'nevt': self.entries, 'tbinc':tbinc}, **kwargs)
        self.df.to_csv(file)

if __name__ == "__main__":


    main_path = Path(__file__).parent[1]
    out_dir = main_path / 'out'

    reco_file = Path.home() / 'data' / 'SNJ' / 'CALIB2' / 'reco' / 'merge' / 'reco.csv.gz'
    # ab = AnaBase(recofile=reco_file, 
    #                 label="", 
    #                 tlim=None)
    # #####Event rate
    # fout = 'out' / "event_rate.png"
    # fig, ax = plt.subplots(figsize=(16,9))
    # evtrateCal = EvtRate(df=ab.df)
    # label = "all"
    # evtrateCal(ax, width=3600, label=label) #width = size time bin width in seconds 
    # ax.legend(loc='best')
    # plt.savefig(str(fout))
    # plt.close()
    # print(f"save {str(fout)}")
    # #######