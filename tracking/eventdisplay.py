#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from typing import List, Union
from dataclasses import dataclass
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines  #use for legend settings
import os
from pathlib import Path
import time
import pandas as pd
import glob
import argparse
from random import sample, choice
#personal modules
from telescope import Telescope, str2telescope
from .tracking import InputType, Event
from reco import EventType, RecoData


class RawEvtDisplay:
    def __init__(self, telescope:Telescope, label:str="", outdir:str=os.environ['HOME'], max_nevt:int=1):
        self.telescope = telescope
        self.label = label
        self.outdir = outdir
        self.nevt = 0
        self.max = max_nevt
        self.handles =[]
        
    def plotCanvas(self, fig, ax): 
        self.fig, self.ax = fig, ax
        self.ax.set_facecolor('white')
        self.telescope.plot3D(self.ax, position=np.zeros(3))
        #self.str_evt= f"{self.label}"
        
    
    def addEvt(self, evt:Event, color:str=None, **kwargs):
        if color is None: color = "#"+''.join([choice('0123456789ABCDEF') for i in range(6)])
        if self.nevt > self.max : return 0
        xyz = evt.xyz
        sum_adc = evt.adc[:,0] + evt.adc[:,1]
        self.ax.view_init(elev=10., azim=-60)
        #self.str_evt += f"\nevtNo {evt.ID}"
        leg = mlines.Line2D([], [], color=color,  linestyle=None, marker='o', markersize=8, label=f"{evt.ID}", **kwargs)
        self.handles.append(leg)
        #self.ax.set_title( self.str_evt )
        hdl = self.ax.scatter(
                xyz[:, 0],
                xyz[:, 1],
                xyz[:, 2],
                c=color,
                s=10,
                marker='.',
                linewidths=1,
                **kwargs
            )
        if sum_adc is not None:
            arr_norm_sum_adc = sum_adc/np.max(sum_adc)
            vis_factor = 100
            hdl = self.ax.scatter(
                xyz[:, 0],
                xyz[:, 1],
                xyz[:, 2],
                s= arr_norm_sum_adc*vis_factor,
                c= color,
                marker='o',
                edgecolor=color,
                label= 'raw' , **kwargs
            )
            self.nevt +=1
        self.handles.append(hdl)
    
    def plot_legend(self, **kwargs): 
        self.ax.legend( handles=self.handles, **kwargs)
    
    def addPoints(self, ax, xyz, adc, is_trk:bool=False, handle=None, **kwargs ):
        if handle is not None : 
            self.handles.append(handle)
        vis_factor = 100
        handles = [] #legend            

        xyz = xyz[~np.all( (xyz[:, :-1] == 0.), axis=1)]
        if len(xyz) ==0 : return
        if is_trk: 
            xs, ys, zs = xyz[0,:].T #1st intersection pt
            xe, ye, ze = xyz[-1,:].T #last intersection pt
            ax.plot([xs, xe],
                [ys, ye],
                [zs, ze], 
                c="blue", linewidth=0.75)
            ax.scatter(
                xyz[:, 0],
                xyz[:, 1],
                xyz[:, 2],
                c='blue',
                s=10,
                marker='+',
                linewidths=1, 
                **kwargs
            )
        else: 
           
            if xyz.ndim == 1 : xyz = xyz[np.newaxis, :]
            adc = adc[:,:-1]
            arr_norm_sum_adc = np.sum(adc, axis=1)/np.max(adc)
            
            
            ax.scatter(
                xyz[:, 0],
                xyz[:, 1],
                xyz[:, 2],
                s= arr_norm_sum_adc*vis_factor,
                c= 'limegreen',
                marker='o',
                edgecolor='green',
                label= 'inlier' ,
                **kwargs
            )
            ax.scatter(
                xyz[:, 0],
                xyz[:, 1],
                xyz[:, 2],
                c='green',
                s=10,
                marker='.',
                linewidths=1,
                **kwargs
            )
                
    
    def save(self, outdir:str, filename:str):
        outdir = Path(outdir)
        outdir.mkdir(parents=True, exist_ok=True)
        plt.savefig(str(outdir/filename))
        plt.close()
        
class RecoEvtDisplay:
    def __init__(self, telescope:Telescope, recodir:str="", label:str="", outdir:str=os.environ['HOME'], input_type=InputType.DATA, kwargs:dict=None):
        self.telescope = telescope
        self.recodir = recodir
        self.label = label
        self.outdir = outdir
        try:
            f_reco = glob.glob(os.path.join(recodir,  '', '*reco*') )[0]
            f_inlier = glob.glob(os.path.join(recodir,  '', '*inlier*') )[0]
        except: raise ValueError
        reco_trk  = RecoData(file=f_reco, 
                            telescope=self.telescope, 
                            input_type=input_type, 
                            kwargs=kwargs)
        inlier_data = RecoData(file=f_inlier, 
                                telescope=self.telescope, 
                                input_type=input_type,
                                kwargs=kwargs,
                                is_all=True)
      
#        self.df = self.df[self.df.index.isin(self.index)]
        self.df_reco = reco_trk.df
        
        self.df_inlier = inlier_data.df[inlier_data.df["inlier"]==1]
        self.evtID_in = list(set(self.df_inlier.index))
        #print(self.df_inlier)
        sumADC_XY_in = self.df_inlier['ADC_X'] + self.df_inlier['ADC_Y'] 
        self.df_inlier = self.df_inlier.assign(sumADC_XY=pd.Series(sumADC_XY_in).values)
        
        self.df_outlier = inlier_data.df[inlier_data.df["inlier"]==0]
        #print("self.df_outlier", self.df_outlier)
        self.evtID_out = list(set(self.df_outlier.index))
        sumADC_XY_out = self.df_outlier['ADC_X'] + self.df_outlier['ADC_Y'] 
        self.df_outlier = self.df_outlier.assign(sumADC_XY=pd.Series(sumADC_XY_out).values)
        
        self.handles = [] #legend    
        
    def get_points(self, evtID:Union[List, int]):

        l_evtID = []
        if type(evtID)==list: l_evtID.extend(evtID)
        else : l_evtID.append(evtID)
        
        Z = np.sort(list(set(self.df_inlier['Z'])))
        self.xyz_reco = None
        if len(self.df_reco) > 0 : 
            self.xyz_reco = { i : np.array([ np.concatenate( (self.df_reco.loc[i][[f'X_{p.position.loc}', f'Y_{p.position.loc}']].to_numpy(), p.position.z) , axis=None )   for p in self.telescope.panels ]) for i in l_evtID}

        self.xyz_in = { i : self.df_inlier.loc[i][['X', 'Y', 'Z']].to_numpy() if i in self.evtID_in else np.zeros(3) for i in l_evtID}
        self.adc_in = { i : self.df_inlier.loc[i]['sumADC_XY'].tolist() if i in self.evtID_in else [0] for i in l_evtID }
        self.xyz_out = { i : self.df_outlier.loc[i][['X', 'Y', 'Z']].to_numpy() if i in self.evtID_out else np.zeros(3) for i in l_evtID}
        self.adc_out = { i : self.df_outlier.loc[i]['sumADC_XY'].tolist() if i in self.evtID_out else [0] for i in l_evtID}
        #print("self.adc_in", self.adc_in)
        #print("self.adc_out", self.adc_out)

    def plot3D(self, fig, ax, evtID:Union[List, int], isReco:bool=True, isInlier:bool=True, isOutlier:bool=True  ):
        self.telescope.plot3D(fig, ax, position=np.zeros(3))
        ax.set_facecolor('white')
        vis_factor = 100
        l_evtID = []
        handles = [] #legend            
        if type(evtID)==list: l_evtID.extend(evtID)
        else : l_evtID.append(evtID)
        for i in l_evtID : 
            if isReco==True:
                ####Track fit
                xyz_r = self.xyz_reco[i]
                xyz_r = xyz_r[~np.all( (xyz_r[:, :-1] == 0.), axis=1)]
                xs, ys, zs = xyz_r[0] #1st intersection pt
                xe, ye, ze = xyz_r[-1] #last intersection pt
                ax.plot([xs, xe],
                        [ys, ye],
                        [zs, ze], 
                        c="blue", linewidth=0.75)
                ax.scatter(
                    xyz_r[:, 0],
                    xyz_r[:, 1],
                    xyz_r[:, 2],
                    c='blue',
                    s=10,
                    marker='+',
                    linewidths=1
                )

            ###Inliers & Outliers
            xyz_i, xyz_o = self.xyz_in[i], self.xyz_out[i]
            if xyz_i.ndim == 1: xyz_i = xyz_i[np.newaxis, :]
            if xyz_o.ndim == 1: xyz_o = xyz_o[np.newaxis, :]
           
           
            adc_i, adc_o = list(), list()
            if type(self.adc_in[i]) == float :  adc_i.append(self.adc_in[i])
            else: adc_i = self.adc_in[i]
            if type(self.adc_out[i]) == float :  adc_o.append(self.adc_out[i])
            else : adc_o = self.adc_out[i]
            adc_all = adc_i+adc_o
            
            if isInlier==True and xyz_i.any() != False:
                
                arr_norm_sum_adc_in = adc_i/np.max(adc_all)
                
                ax.scatter(
                    xyz_i[:, 0],
                    xyz_i[:, 1],
                    xyz_i[:, 2],
                    s= arr_norm_sum_adc_in*vis_factor,
                    c= 'limegreen',
                    marker='o',
                    edgecolor='green',
                    label= 'inlier' 
                )
                ax.scatter(
                    xyz_i[:, 0],
                    xyz_i[:, 1],
                    xyz_i[:, 2],
                    c='green',
                    s=10,
                    marker='.',
                    linewidths=1
                )
                inlier_mark = mlines.Line2D([], [], color='limegreen',  marker='o', linestyle='None',
                            markersize=10, label='inlier hit')
                handles.append(inlier_mark)
            
            if isOutlier==True and xyz_o.any() != False :
                arr_norm_sum_adc_out = adc_o/np.max(adc_all)
                ax.scatter(
                    xyz_o[:, 0],
                    xyz_o[:, 1],
                    xyz_o[:, 2],
                    c='tomato',
                    s= arr_norm_sum_adc_out*vis_factor,
                    marker='o',
                    edgecolor='red',
                    label='outlier'
                )
            
                ax.scatter(
                    xyz_o[:, 0],
                    xyz_o[:, 1],
                    xyz_o[:, 2],
                    c='r',
                    s=10,
                    marker='.',
                    linewidths=1
                )
                outlier_mark = mlines.Line2D([], [], color='tomato',  marker='o', linestyle='None',
                            markersize=10, label='outlier hit')
                handles.append(outlier_mark)
    
        
           
            
            #for adc in np.round(np.linspace(round(min(adc),-1), round(max(adc), -1), ntouched_panels), -2):
            # for adc in adc[id_nearest]:
            #     ax.scatter( [], [], c= 'lightblue', marker='o',edgecolor='blue', alpha=0.3, s=exp(adc/max(adc)*vis_factor), label=str(int(adc)) )

            #leg1 = ax.legend(frameon=True, title='dE [MeV]', loc='best')
            #leg1 = ax.legend(frameon=True, title='n$_{adc}$(X)+n$_{adc}$(Y)', loc='best')
            #ax.add_artist(leg1)
            
            #Set the legend
            track_line = mlines.Line2D([], [], color='blue',  marker=None, linestyle='-',
                            markersize=10, label='reconstructed track')
            handles.append(track_line)
            # fit_mark = mlines.Line2D([], [], color='blue',  marker='+', linestyle='None',
            #                 markersize=10, label='intersections point')
            # handles.append(fit_mark)
            
            
            # purple_triangle = mlines.Line2D([], [], color='purple', marker='^', linestyle='None',
            #                   markersize=10, label='Purple triangles')

            ax.legend(handles=handles, fontsize='large',  loc=(0.5,0.85))#'upper right')#, pad=-10)
            ax.view_init(elev=10., azim=-60)



@dataclass
class EmissionSurface:
    """
    For plotting simulation event
    """
    shape : str
    size : float #in mm
    origin : np.ndarray #array-like, in meter
    def plot3D(self, ax):
        s = self.size
        X  = np.arange(self.origin[0], self.origin[0]+s, 10)
        Y = np.arange(self.origin[1], self.origin[1]+s, 10)
        X, Y = np.meshgrid(X, Y)
        if self.shape == "plane" : Z = np.ones(shape=X.shape)*self.origin[2]
        else: return None
        ax.plot_surface(X,Y,Z, alpha=0.2, color='blue' )
        return ax
        
        
        
        

if __name__ == '__main__':
     
    parser=argparse.ArgumentParser(
    description='''Plot event displays of random reconstructed events drawn in reco file.''', epilog="""All is well that ends well.""")
    parser.add_argument('--telescope', '-tel', required=True, help='Input telescope name (e.g "tel_SNJ"). It provides the associated configuration.',  type=str2telescope)
    parser.add_argument('--reco_dir', '-i', required=True, help="",  type=str)
    parser.add_argument('--out_dir', '-o', required=True, help="",  type=str)
    parser.add_argument('--nevts', '-n', help="number of events to display", type=int, default=1)
    
    args=parser.parse_args()
    tel = args.telescope
    
    start_time = time.time()
    print("Start: ", time.strftime("%H:%M:%S", time.localtime()))#start time
    tel = args.telescope
    reco_dir = Path(args.reco_dir)
    out_dir = Path(args.out_dir)
    
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    print(f'outdir: {out_dir}')
    
    recofileTomo = glob.glob(str(reco_dir/"*reco*"))[0]
    input_type = InputType.DATA
    evttype = EventType.MAIN
    
    recofile=RecoData(file=recofileTomo, 
                       telescope=tel,  
                         input_type=input_type, ) 
    df = recofile.df
    N= args.nevts 
    evtID_good= sample(list(df.index), N) 
    print(f"evtID = {evtID_good}")
    
    kwargs = {"delimiter":"\t", "index_col":0} 
    pl = RecoEvtDisplay(telescope=tel, recodir=reco_dir, label="", outdir=out_dir, kwargs=kwargs)
    for i, ev in enumerate(evtID_good):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d' )
        pl.get_points(evtID=ev)
        pl.plot3D(fig=fig, ax=ax, evtID=[ev], isReco=True, isInlier=True, isOutlier=True) 
        fout = str(out_dir/f"evt{ev}.png")
        plt.savefig(fout)
        plt.close()    
