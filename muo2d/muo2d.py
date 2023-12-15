#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Raphaël Bajou
"""
from dataclasses import dataclass, field
from typing import Union
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from mpl_toolkits.mplot3d.axes3d import Axes3D
from scipy.interpolate import griddata
from pathlib import Path
from matplotlib.gridspec import GridSpec
from mpl_toolkits.axes_grid1 import make_axes_locatable
import warnings
warnings.filterwarnings("ignore")
import pickle
#package module(s)
from reco import HitMap, EvtRate
from telescope import Telescope

@dataclass
class Estimate:

    estimate : dict = field(default_factory=lambda: dict({}))
    unc_sys  : dict = field(default_factory=lambda: dict({}))
    unc_stat : dict = field(default_factory=lambda: dict({}))

    def save(self, file:Union[Path, str]): 
        ''' 
        Save dictionaries in binary pickle file format.
        '''
        if isinstance(file, Path): file=str(file)
        with open(file, 'wb') as f:
            dict_out = {'estimate':self.estimate, 'unc_sys':self.unc_sys, 'unc_stat': self.unc_stat}
            pickle.dump(dict_out, f, pickle.HIGHEST_PROTOCOL)
            print(f"Save pickle file {file}")

    def load(self,file:Union[Path, str]):
        ''' 
        Load file containing dictionaries in binary pickle format.
        '''
        print(f"Load {file}")
        with open(file, 'rb') as f : 
            dict_in = pickle.load(f)
        keys = ['estimate', 'unc_sys', 'unc_stat']
        if all([k in list(dict_in.keys()) for k in keys]):
            self.estimate, self.unc_sys, self.unc_stat = dict_in['estimate'], dict_in['unc_sys'], dict_in['unc_stat']
        else :
            raise KeyError("Wrong dict keys in file")    


class Efficiency(Estimate):

    def __init__(self):
        Estimate.__init__(self)


@dataclass
class TransmittedFluxModel: 

    zenith: np.ndarray = field(default_factory=lambda: np.ndarray())
    flux: np.ndarray = field(default_factory=lambda: np.ndarray())
    opacity: np.ndarray = field(default_factory=lambda: np.ndarray())
    error: np.ndarray = field(default_factory=lambda: np.ndarray())


class Acceptance(Estimate): 


    def __init__(self, telescope:Telescope, hitmap:HitMap, flux:np.ndarray):
        
        self.hm = hitmap
        self.flux = flux
        self.sconfig = list(telescope.configurations.keys())
        Estimate.__init__(self)

    def compute(self, t_res:float=1.e-8) -> None:

        df = self.hm.df
        time = df['timestamp_s'] + df['timestamp_ns']*t_res #default clock time res=10ns
        time_sort = np.sort(time)
        dtime = np.diff(time_sort) 
        run_duration = np.sum(dtime[dtime < 3600])  # in s
        
        for conf, hitmap in self.hm.h_DXDY.items():
            
            if conf.startswith('3') : 
                flux = self.flux['3p']['mean']
            elif conf.startswith('4') : 
                flux = self.flux['4p']['mean']
            else : 
                raise ValueError("Unknown telescope configuration name")
                
            evt_rate = hitmap / run_duration 
            self.estimate[conf] = evt_rate / flux
            u_stat, u_dt = np.sqrt(hitmap), 1#s
            self.unc_stat[conf] = self.estimate[conf] *  np.sqrt((u_stat/hitmap)**2 + (u_dt/run_duration)**2)  

    def plot_fig_2d(self, **kwargs) -> plt.figure:

        fig = plt.figure(1, figsize= (12,7))
        gs = GridSpec(1, len(self.sconfig))#, left=0.02, right=0.98, wspace=0.1, hspace=0.5)

        for i, (conf,acc) in enumerate(self.estimate.items()):
            
            ax = fig.add_subplot(gs[0,i], aspect='equal')#, projection='3d')      
            w = self.hm.width[conf]
            [dxmin, dxmax], [dymin, dymax] = self.hm.rangeDXDY[conf] / w
           
            x, y = np.arange(dxmin, dxmax+1), np.arange(dymin, dymax+1)
            X, Y = np.meshgrid(x, y)
            max_acc = np.nanmax(acc)
            im = ax.pcolor(X, Y,  acc, vmin=0, vmax=max_acc, **kwargs)

            #color bar
            divider = make_axes_locatable(ax) #sized like figure
            cax = divider.append_axes("right", size="5%", pad=0.05)
            cbar = fig.colorbar(im, cax=cax, format='%.0e', orientation="vertical") # shrink=0.75,
            cbar.ax.tick_params(labelsize=8)
            cbar.set_label(label=u'Acceptance [cm².sr]')
            #ax.set_title(f'{conf} config')
        
        gs.tight_layout(fig)    

        return fig   
    

    def plot_fig_3d(self, ax:Axes3D, grid_x:np.ndarray, grid_y:np.ndarray, grid_z:np.ndarray, **kwargs) -> None:
        
        im = ax.plot_surface(
            grid_x,
            grid_y,
            grid_z,
            **kwargs
        )

        ax.set_xlabel('$\\Delta$X')
        ax.set_ylabel('$\\Delta$Y')
        ax.view_init(elev=15., azim=45)      

        cbar = plt.colorbar(im,  shrink=0.5, orientation="vertical")
        cbar.ax.tick_params(labelsize=12)
        cbar.set_label(label='Acceptance [cm².sr]', size=14)


class Flux(Estimate):

    def __init__(self):
        Estimate.__init__(self)

    def compute(self, hitmap:HitMap, acceptance:Acceptance, time:float, efficiency:Efficiency=None) -> None: 
        
        for (conf, hm) in hitmap.h_DXDY.items():
            unc_hm = np.sqrt(hm)
            acc, unc_acc = acceptance.estimate[conf], acceptance.unc_stat[conf]
            if efficiency is None : 
                eff, unc_eff = np.ones(acc.shape), np.zeros(acc.shape)
            else : 
                eff, unc_eff = efficiency[conf].estimate, efficiency[conf].unc_stat
            s = ((acc!=0.) & (eff!=0.))
            acc[~s], unc_acc[~s] = np.nan, np.nan 
            eff[~s], unc_eff[~s] = np.nan, np.nan 
            self.estimate[conf] = hm / ( time * acc * eff) 
            self.unc_stat[conf] = self.estimate[conf] * np.sqrt((unc_acc/acc)**2 + (unc_hm/hm)**2 + (unc_eff/eff)**2 ) 
            self.unc_sys[conf] = np.zeros(acc.shape)
   


class Opacity(Estimate):

    def __init__(self):
        Estimate.__init__(self)

    def compute(self, model:TransmittedFluxModel, zenith:dict, flux:Flux,  mask:dict, *args,**kwargs) -> None: 
        
        points, range_op_flat = np.array([model.zenith.flatten(), model.flux.flatten()]).T, model.opacity.flatten() #np.exp( np.log(10) * op ) 
        range_flux_flat = model.flux.flatten()
        model_error_flat = model.error.flatten()
        zemin, zemax = np.min(points[:, 0]), np.max(points[:, 0])
        flmin, flmax = np.min(points[:, 1]), np.max(points[:, 1])
        
        for (conf, fl) in flux.estimate.items():
            shape = fl.shape
            m = mask[conf]
            grid_op = np.zeros(shape)
            in_range = np.logical_and((flmin < fl) & (fl < flmax), ~(np.isnan(fl)) & ~m)
            grid_tmp = np.zeros(shape)
            grid_x, grid_y = zenith[conf], fl
            grid_tmp[in_range] = griddata(points, range_op_flat, (grid_x[in_range], grid_y[in_range]), *args, **kwargs) 
            #print(f"(opacity_min, opacity_max)_{conf}  =  ({np.nanmin(grid_tmp[in_range]):.3e}, {np.nanmax(grid_tmp[in_range]):.3e}) mwe")
            grid_tmp[~in_range] = np.nan
            grid_op = grid_tmp 
            uflux_data = np.sqrt(flux.unc_stat[conf]**2 + flux.unc_sys[conf]**2).flatten()
            self.unc_stat[conf], self.unc_sys[conf] = np.zeros(shape[0]*shape[1]), np.zeros(shape[0]*shape[1])
            self.estimate[conf] = grid_op
            for i,o in enumerate(grid_op.flatten()):
                ix = np.argmin(abs(range_op_flat-o))
                do = range_op_flat[ix] - range_op_flat[ix+1] 
                df = range_flux_flat[ix] - range_flux_flat[ix+1]
                b = df/do
                self.unc_stat[conf][i] = 1/abs(b) *  uflux_data[i]
                self.unc_sys[conf][i] = 1/abs(b) *  model_error_flat[ix]
                #sig_flux = np.sqrt(uflux_data[i]**2 + model.error[ix]**2)

            self.unc_stat[conf] = self.unc_stat[conf].reshape(shape)
            self.unc_stat[conf][m] = np.nan
            self.unc_sys [conf] = self.unc_sys[conf].reshape(shape)
            self.unc_sys [conf][m] = np.nan


class MeanDensity(Estimate):

    def __init__(self):
        Estimate.__init__(self)

    def compute(self, opacity:Opacity, thickness:dict, res:float=5.) -> None: 
        
        for conf, thick in thickness.items():
            op = opacity.estimate[conf]
            rho_mean = op/thick
            rho_mean[rho_mean==0] = np.nan
            self.estimate[conf] = rho_mean
            self.unc_stat[conf] = np.sqrt(  (res * op / thick**2 )**2  +  (opacity.unc_stat[conf] / thick)**2)
            self.unc_sys [conf] = np.sqrt(  ( res * op / thick**2 )**2  +  (opacity.unc_sys[conf] / thick)**2)



class Muo2D:
    
    
    def __init__(self, telescope:Telescope, hitmap:HitMap, acceptance:Acceptance, evtrate:EvtRate, model:TransmittedFluxModel, thickness:dict, *args, **kwargs):
        
        self.tel = telescope
        self.hitmap = hitmap
        self.acceptance = acceptance

        self.flux = Flux()
        time = evtrate.run_duration
        self.flux.compute( hitmap=hitmap, 
                          acceptance=acceptance, 
                          time=time)

        self.opacity = Opacity()
        mask = {conf: np.isnan(thick) for conf, thick in thickness.items()}
        self.opacity.compute(model=model, 
                             zenith=self.tel.zenithMatrix, 
                             flux=self.flux, 
                             mask=mask)

        self.mean_density = MeanDensity()
        self.mean_density.compute(opacity=self.opacity, 
                                  thickness=thickness)


    def plot_map_2d(self, fig:Figure, ax:Axes, grid_x:np.ndarray, grid_y:np.ndarray, grid_z:np.ndarray, **kwargs ) -> None:
    
        im = ax.pcolor(grid_x, grid_y, grid_z, **kwargs)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        cbar = fig.colorbar(im, cax=cax, orientation="vertical")
        if 'label' in kwargs.keys():  cbar.set_label(label=kwargs['label'])
        fig.subplots_adjust(right=0.85)
        ax.set_xlabel('Azimuth $\\varphi$ [deg]')
        ax.set_ylabel('Zenith $\\theta$ [deg]')
        ax.invert_yaxis()
        ax.set_ylim(90, 50)
        ax.set(frame_on=True)
        


if __name__ == "__main__":
    pass