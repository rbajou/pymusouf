#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable
import pandas as pd

#package module(s)
from telescope import Telescope

params = {'legend.fontsize': 'xx-large',
         'axes.labelsize': 'xx-large',
         'axes.titlesize':'xx-large',
         'xtick.labelsize':'xx-large',
         'ytick.labelsize':'xx-large',
         'axes.labelpad':10,
         'mathtext.fontset': 'stix',
         'font.family': 'STIXGeneral',
         }
plt.rcParams.update(params)


class HitMap: 
    
    
    def __init__(self, telescope:Telescope, df:pd.DataFrame):
        
        self.tel = telescope
        self.panels = self.tel.panels
        self.df = df.copy()
        self.sconfig = list(self.tel.configurations.keys()) #panels config in telescope (i.e 4-panel configuration = 2*3p + 1*4p)
        self.XY, self.DXDY, self.hDXDY = {}, {}, {}

        #brut (X,Y) hit maps
        self.binsXY = {pan.position.loc : [pan.matrix.nbarsX, pan.matrix.nbarsY] for pan in self.panels}
        self.rangeXY = { pan.position.loc : [  [ 0, int(pan.matrix.nbarsX) * float(pan.matrix.scintillator.width) ], [0, int(pan.matrix.nbarsY) * float(pan.matrix.scintillator.width)]] for pan in self.panels }
        
        #(DX,DY) maps : hits per telescope pixel (=line-of-sight) r_(DX,DY) 
        self.rayMatrix = { conf :  self.tel.get_ray_matrix(front_panel=panels[0], rear_panel=panels[-1]) for conf,panels in self.tel.configurations.items()}
        self.binsDXDY  = { conf : [m.shape[0], m.shape[1]] for conf, m in self.rayMatrix.items()}
        self.width = {conf: panels[0].matrix.scintillator.width for conf, panels in self.tel.configurations.items()}
        self.rangeDXDY = { conf : np.array([ [ np.min(los[:,:,0])*self.width[conf], np.max(los[:,:,0])*self.width[conf] ], [np.min(los[:,:,1])*self.width[conf], np.max(los[:,:,1])*self.width[conf] ]])  for conf,los in self.rayMatrix.items()}

        self.fill_dxdy()
    
        
    def fill_dxdy(self, dict_filter:dict=None):

        self.h_DXDY  = { conf : None for conf in self.sconfig}
        self.df_DXDY = { conf : None for conf in self.sconfig}
        
        for pan in self.panels : 
            #raw hit panel maps
            key = pan.position.loc
            xpos, ypos = f"X_{key}", f"Y_{key}"
            ((Xmin, Xmax), (Ymin, Ymax)) = self.rangeXY[key]
            sel =  ( (Xmin< self.df[xpos]) & (self.df[xpos]<Xmax) &  (Ymin< self.df[ypos]) & (self.df[ypos]<Ymax) )
            self.XY[key] = [self.df[sel][xpos].values, self.df[sel][ypos].values]

        for conf, panels in self.tel.configurations.items():
            #hit tel config maps
            front, rear = panels[0].position.loc, panels[-1].position.loc
            ((Xminf, Xmaxf), (Yminf, Ymaxf)) = self.rangeXY[front]
            ((Xminr, Xmaxr), (Yminr, Ymaxr)) = self.rangeXY[rear]
            xposf, yposf = f"X_{front}", f"Y_{front}"
            xposr, yposr = f"X_{rear}", f"Y_{rear}"
            sfront =  ( (Xminf< self.df[xposf]) & (self.df[xposf]<Xmaxf) &  (Yminf< self.df[yposf]) & (self.df[yposf]<Ymaxf) )
            srear = ( (Xminr < self.df[xposr]) & (self.df[xposr]<Xmaxr) &  (Yminr< self.df[yposr]) & (self.df[yposr]<Ymaxr) )
            sel = (sfront & srear)

            ###apply filter on evt ids
            idx = None
            if dict_filter: 
                filter = dict_filter[conf]
                idx  = self.df[sel].loc[filter].index
                #idx  = df[df.index.isin(filter)][sel].index
            else : 
                idx  = self.df[sel].index

            dftmp = self.df.loc[idx] 
            ts, tns = dftmp['timestamp_s'].values, dftmp['timestamp_ns'].values
            DX, DY =  dftmp[xposf].values - dftmp[xposr].values, dftmp[yposf].values - dftmp[yposr].values
            dfconf = pd.DataFrame(index=idx, data=np.array([ts, tns, DX, DY]).T, columns=['timestamp_s', 'timestamp_ns', f'DX_{conf}', f'DY_{conf}'])
            self.df_DXDY[conf] = dfconf
            self.h_DXDY[conf] = np.histogram2d(DX, DY, bins=self.binsDXDY[conf], range=self.rangeDXDY[conf] )[0]
       

    def plot_xy_map(self, invert_yaxis:bool=False, transpose:bool=False):
        """Plot hit map for reconstructed primaries and all primaries"""
        fig, fax = plt.subplots(figsize=(8, 12), nrows=len(self.panels), ncols=1, sharex=True)
        for i, p in enumerate(self.panels):  
            ax = fax[i]
            ax.set_aspect('equal')#, adjustable='box')
            key = p.position.loc
            X, Y = self.XY[key]
            ax.set_title(f"{p.position.loc}")
            if i == len(self.panels)-1: ax.set_title("Rear")
            ax.grid(False)
            counts, xedges, yedges, im1 = ax.hist2d( X, Y,cmap='viridis', bins=self.binsXY[key], range=self.rangeXY[key] ) #im1 = ax.imshow(hXY[i])
            if transpose: 
                ax.hist2d(Y, X, cmap='viridis', bins=self.binsXY[key], range=self.rangeXY[key] ) #im1 = ax.imshow(hXY[i])
            if invert_yaxis: ax.invert_yaxis()
            if i == len(self.panels)-1 : ax.set_xlabel('Y')
            ax.set_ylabel('X')
            divider1 = make_axes_locatable(ax)
            cax1 = divider1.append_axes("right", size="5%", pad=0.05)
            cbar=fig.colorbar(im1, cax=cax1, format='%.0e')
            cbar.set_label(label='entries')
        fig.tight_layout()

        
    def plot_dxdy_map(self, invert_xaxis:bool=True, invert_yaxis:bool=False, transpose:bool=False, fliplr:bool=False, flipud:bool=False):
        """
        Args:
            invert_xaxis (bool, optional): _description_. Defaults to True.
            invert_yaxis (bool, optional): _description_. Defaults to False.
            transpose (bool, optional): _description_. Defaults to False.
            fliplr (bool, optional): _description_. Defaults to False.
            flipud (bool, optional): _description_. Defaults to False.
        """
        nconfigs = len(self.sconfig)
        fig, fax = plt.subplots(figsize=(8, 12), nrows=nconfigs, ncols=1, sharex=True)
        for i, conf in enumerate(self.sconfig):
            if nconfigs > 1 : ax = fax[i]
            else: ax=fax
            ax.set_aspect('equal')#, adjustable='box')
            ax.set_ylabel('$\\Delta$X [mm]')#, fontsize=16)
            ax.set_xlabel('$\\Delta$Y [mm]')#, fontsize=16)
            DX_min, DX_max = self.rangeDXDY[conf][0]
            DY_min, DY_max = self.rangeDXDY[conf][1]
            h = self.hDXDY[conf]
            if transpose: h = h.T
            if fliplr : h = np.fliplr(h)
            if flipud : h = np.flipud(h)
            h[h==0] = np.nan 
            im = ax.imshow(h, cmap='viridis', norm=LogNorm(vmin=1, vmax=np.max(h[~np.isnan(h)])), extent=[DX_min, DX_max, DY_min, DY_max] )
            ax.grid(False)
            #hist, xedges, yedges, im1 = ax1.hist2d( DY[c], DX[c], edgecolor='black', linewidth=0., bins=self.binsDXDY[c], range=self.rangeDXDY[c], weights=None, cmap='viridis', norm=LogNorm(vmin=1, vmax=np.max(self.hDXDY[c]) ) ) #    
            if invert_xaxis:  ax.invert_xaxis()
            if invert_yaxis:  ax.invert_yaxis()
            divider1 = make_axes_locatable(ax)
            cax1 = divider1.append_axes("right", size="5%", pad=0.05)
            cbar = fig.colorbar(im, cax=cax1, extend='max')
            cbar.set_label(label='entries')
        fig.tight_layout()


if __name__ == "__main__":
    pass