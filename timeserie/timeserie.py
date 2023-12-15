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
from matplotlib.colors import LogNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable
import pickle
import pandas as pd
import warnings
warnings.filterwarnings("ignore")
import time
import glob 
import yaml
#package module(s)
from reco import HitMap, EvtRate, RecoData
from telescope import Telescope, dict_tel
from muo2d import Flux


def time_sort_df(df:pd.DataFrame, dtgap=3600, tres =1e-8):
   
    df = df.sort_values(by=["timestamp_s",'timestamp_ns'])
    time_sort = df['timestamp_s'] + df['timestamp_ns']*tres
    dtime = np.diff(time_sort) 
    run_duration = np.sum(dtime[dtime < 3600])  # in second

    print( f"run_duration={(run_duration)/(24*3600):.3f}days")

    df["date"] = pd.to_datetime(df["timestamp_s"], unit='s')
    df['gap'] = np.concatenate(([0],np.diff(time_sort))) #df.date.diff().dt.seconds
    df['timedelta'] = pd.to_timedelta( df['date'] - df['date'].iloc[0] ).astype('timedelta64[s]')

    print(df.iloc[-1])

    dt_sum_gaps = np.sum( df['gap'][df['gap'] > dtgap ] )
   
    print(f"dt_sum_gaps={dt_sum_gaps/(24*3600):.3f}days")
    print(f"run_duration={(df.iloc[-1]['timedelta'].total_seconds()  - dt_sum_gaps)/(24*3600):.3f}days" )
   
    idx_gap =  df['timedelta'][df['gap'] > dtgap ].index

    print(f"idx_gap = {idx_gap}")

    for i, ix in enumerate(idx_gap): 
        id_above_gap = df.loc[ix:].index
        res = df.loc[id_above_gap, 'timedelta'] - df.loc[ix]['gap'].astype('timedelta64[s]')
        print(f'i, res = {i}, {res}')
        df.loc[id_above_gap, 'timedelta'] = res
   
    print( f"run_duration={df.iloc[-1]['timedelta'].total_seconds()/(24*3600):.3f}days")
    del df["gap"]
    df['timedelta'] = np.ndarray.astype(df['timedelta'].values, dtype=int)
    df = df.set_index(df["date"])
    del df["date"]
    #print(df.head)
    return df 
    #df.to_csv(file, compression='gzip', sep='\t')


class TimeSerie:

    def __init__(self, telescope:Telescope,  df:pd.DataFrame):
        self.tel = telescope
        self.df = df.copy()
        pass


if __name__ == "__main__":
    

    from datetime import timedelta
    year = timedelta(days=365)
    ten_years = 10 * year
    
    month = timedelta(days=30)
    dt = year-month
    print(dt, type(dt))
    dts = dt.total_seconds()
    print(dts, type(dts))

    #exit()

    tel = dict_tel['SB']
    ####default arguments/paths are written in a yaml config file associated to telescope
    main_path = Path(__file__).parents[1]#MAIN_PATH# #
    with open( str(main_path / "files" / "telescopes" / tel.name /"run.yaml") ) as fyaml:
        try: def_args = yaml.load(fyaml, Loader=yaml.SafeLoader)
        except yaml.YAMLError as exc: print(exc)

    out_path = main_path / "out"
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
    kwargs_dat = { "index_col": 0, "delimiter": "\t", "nrows": None}
    reco_data_tomo = RecoData(file=freco_tomo, telescope=tel)
    df_ini = reco_data_tomo.df.copy()
    df_ini.reset_index(inplace=True) 
    print(f"time_sort_df(..) -- {time.time() - start_time:.1f} s")
    df = time_sort_df(df_ini)

    print(f"df.head = {df.head}")
    print(f"End -- {time.time() - start_time:.1f} s")