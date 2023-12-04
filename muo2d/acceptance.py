#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 08 2021
@author: Raphaël Bajou
"""
# Librairies
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
from datetime import datetime, timezone
import time 
import os
from pathlib import Path
import sys
from matplotlib.gridspec import GridSpec
from mpl_toolkits.axes_grid1 import make_axes_locatable
import argparse
import re
#import seaborn as sns
# #sns.set()
# #sns.set_theme(style="whitegrid", palette="pastel")
import logging
#personal modules
#Get location of script

from reco import EventType,  AnaHitMap, AnaBase, Cut
from telescope import dict_tel, str2telescope, Telescope
from tracking import InputType
from utils.tools import create_subtitle


class GeometricalAcceptance:
    def __init__(self, telescope:Telescope, configuration:str):
        """
        
        """
        self.tel=telescope
        self.dxdy =  self.tel.los[configuration]
        front_panel, rear_panel = self.tel.configurations[configuration][0], self.tel.configurations[configuration][-1]
        self.mat_xy = self.tel.get_pixel_xy(front_panel=front_panel, rear_panel=rear_panel)
        #print(f"mat_xy = {mat_xy}, {mat_xy.shape}")
        #print(f"mat_dxdy = {self.dxdy}, {self.dxdy.shape}")
        front_width, rear_width = front_panel.matrix.scintillator.width, rear_panel.matrix.scintillator.width ##in mm 
        if front_width != rear_width : raise ValueError("front_width != rear_width : Current acceptance computation only valid for identical front-rear matrices.")
        Nx, Ny = front_panel.matrix.nbarsX, front_panel.matrix.nbarsY
        if Nx != Ny : raise ValueError("Nx != Ny : Current acceptance computation only valid for equal scintillator numbers in X and Y.")
        length = self.tel.configurations[configuration][-1].position.z - self.tel.configurations[configuration][0].position.z #mm
#        print(f"GeometricalAcceptance length = {length}")
        self.length, self.width = length, front_width ##in mm 
        self.Nxy = Nx

    
    def acceptance_axis_approx(self, dxdy):
        """
        Annexe B Kevin Jourde's Thesis
        Integrated acceptance [cm^2.sr] computation (simplified to Taylor expansion order 0 since width << length)
        A telescope line-of-sight is referenced as a (delta_x, delta_y) couple
        """
        delta_x, delta_y = dxdy
        Nxy = self.Nxy
        L = self.length*1e-1 # mm -> cm
        w = self.width*1e-1 # mm -> cm
        ####Number of pixel couples for given (delta_x, delta_y) direction
        couple_dxdy = Nxy**2 - Nxy * ( np.abs(delta_x) + np.abs(delta_y) )  + np.abs( delta_x * delta_y ) #ok
        alpha = np.arctan(w * np.sqrt((delta_x)**2+(delta_y)**2) /(2*L)) #ok ####Erreur in Kevin's thesis???? np.arctan(w * np.sqrt((delta_x)**2+(delta_y)**2) / 2*L)
        t_dxdy = w**4 * np.cos(alpha) / (4*L**2 + w**2 * ((delta_x)**2+(delta_y)**2) ) #ok 
        acc_dxdy = couple_dxdy * t_dxdy
        return acc_dxdy
    
    
    
    def acceptance_willis(self, x1, x2, y1, y2):
        z = self.length * 1e-1
        G = (z**2 + 2*(x1 + x2)**2) / (2*np.sqrt(z**2 + (x1 + x2)**2)) * ( (y1+y2)*np.arctan((y1+y2)/np.sqrt(z**2+(x1+x2)**2))  -  (y2-y1)*np.arctan((y2-y1)/np.sqrt(z**2+(x1+x2)**2))  )
        G-= (z**2 + 2*(x2 - x1)**2) / (2*np.sqrt(z**2 + (x2 - x1)**2)) * ( (y1+y2)*np.arctan((y1+y2)/np.sqrt(z**2+(x2-x1)**2))  -  (y2-y1)*np.arctan((y2-y1)/np.sqrt(z**2+(x2-x1)**2))  )
        G+= (z**2 + 2*(y1 + y2)**2) / (2*np.sqrt(z**2 + (y1 + y2)**2)) * ( (x1+x2)*np.arctan((x1+x2)/np.sqrt(z**2+(y1+y2)**2))  -  (x2-x1)*np.arctan((x2-x1)/np.sqrt(z**2+(y1+y2)**2))  )
        G-= (z**2 + 2*(y2 - y1)**2) / (2*np.sqrt(z**2 + (y2 - y1)**2)) * ( (x1+x2)*np.arctan((x1+x2)/np.sqrt(z**2+(y2-y1)**2))  -  (x2-x1)*np.arctan((x2-x1)/np.sqrt(z**2+(y2-y1)**2))  )
        return G


class Acceptance():
    def __init__(self, hitmap:AnaHitMap, outdir:str, opensky_flux:np.ndarray, evttype:EventType, theoric:dict=None ):
        #super().__init__(process_file=process_file, evttype=evttype,  label=label)
        self.hm = hitmap
        self.df = self.hm.df_DXDY
        self.label = self.hm.label
        self.outdir = outdir
        self.os_flux = opensky_flux
        self.evttype = evttype.name
        self.sconfig = [ c[3:] for c in self.df.columns if re.search('DX_(.+?)', c)]
        self.acc_dir=outdir
        self.acceptance = {conf: np.zeros(shape=self.hm.binsDXDY[conf])  for conf  in self.sconfig}
        self.unc = {conf: np.zeros(shape=self.hm.binsDXDY[conf])  for conf  in self.sconfig}

        self.ts = np.array(self.hm.df['timestamp_s'].values)
        self.tns = np.array(self.hm.df['timestamp_ns'].values)
        self.compute_acceptance()
        self.theoric = theoric

        #self.ratio = {conf: np.zeros(shape=self.hm.binsDXDY[c])  for c,conf in enumerate(sconfig)}

        logging.basicConfig(filename=os.path.join(self.outdir,f'acceptance_{self.label}_{time.strftime("%d%m%Y")}.log'), level=logging.INFO, filemode='w')
        timestr = time.strftime("%d%m%Y-%H%M%S")
        logging.info(timestr)
        logging.info(sys.argv)
        
    def compute_acceptance(self) -> None:
        time =self.ts + self.tns*10**(-8)
        time_sort = np.sort(time)
        dtime = np.diff(time_sort) 
        runDuration = np.sum(dtime[dtime < 3600])  # in second
        for conf, hDXDY in self.hm.hDXDY.items():
            evt_rate = hDXDY / runDuration
            self.acceptance[conf]= evt_rate/self.os_flux[conf]
            u_DXDY = np.sqrt(hDXDY)
            u_dt = 1 #s
            self.unc[conf] = self.acceptance[conf] *  np.sqrt((u_DXDY/hDXDY)**2 + (u_dt/runDuration)**2) 
            np.savetxt(os.path.join(self.acc_dir, '', f'acceptance_{conf}.txt'), self.acceptance[conf], delimiter='\t', fmt='%.5e')
            np.savetxt(os.path.join(self.acc_dir, '', f'unc_acc_{conf}.txt'), self.unc[conf], delimiter='\t', fmt='%.5e')
        # ####format dict of np.ndarray to dict of list to save in json
        # dict_out= { k:v.tolist() for k, v in self.unc.items()}
        # json_err = json.dumps(dict_out)
        # with open(self.outjson, 'w') as fp:
        #     fp.write(json_err)
    
    def plot_acceptance(self, az, ze):
        #if self.acceptance.size == 0: raise Exception("Please fill acceptance vector first.")
        fig = plt.figure(1, figsize= (16,9))
        nconfigs= self.acceptance.shape[0]
        gs = GridSpec(1,nconfigs)#, left=0.02, right=0.98, wspace=0.1, hspace=0.5)
        # sns.set_style("whitegrid")
        # max_acc = np.max([np.max(acc) for acc in self.acceptance])
        create_subtitle(fig, gs[0, ::], f'Experimental acceptances: {self.label}')
    
        for i, (conf,acc) in enumerate(self.acceptance.items()):
            max_acc = np.nanmax(acc)
            ax = fig.add_subplot(gs[0,i], aspect='equal')#, projection='3d') 
            #ax = Axes3D(fig)
            #ax.view_init(elev=25., azim=45)       
     
            im1 = ax.pcolor(np.arange(0,self.hm.binsDXDY[conf][0]), np.arange(0,self.hm.binsDXDY[conf][1]), 
                           acc,edgecolor='black', linewidth=0.5, cmap='viridis',  
                           shading='auto', vmin=0, vmax=max_acc) #norm=LogNorm(vmin=np.min(acc) , vmax=np.max(acc)))
            # im1 = ax.plot_surface(
            #     ze[i],
            #     az[i],
            #     acc,
            #     cmap="jet", #cm.coolwarm
            #     linewidth=0,
            #     antialiased=False,
            #     alpha=1
            # )
            
           
            ####2D plot
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            cbar = fig.colorbar(im1, cax=cax, format='%.0e', orientation="vertical") # shrink=0.75,
            cbar.ax.tick_params(labelsize=8)
            cbar.set_label(label=u'Acceptance (cm².sr)', size=12)
        
            
            ax.set_xlabel('zenith $\\theta$ ($^{\circ}$)', fontsize=12)
            ax.set_ylabel('azimuth $\\phi$ ($^{\circ}$)', fontsize=12)
            ax.set_title(f'{self.sconfig[i]} config')
        
        gs.tight_layout(fig)       
        #plt.figtext(.5,.95, f"Experimental acceptances : {self.label}", fontsize=12, ha='center')
        plt.savefig(
            os.path.join(self.acc_dir,"", f"acceptance.png")
        )
        
        
    
    def plot_acceptance_3D(self, acc_exp, AZ, ZE, acc_th=None,label=None):
        
        if label is None: label=self.label
        #if self.acceptance.size == 0: raise Exception("Please fill acceptance vector first.")
        fig = plt.figure(1, figsize= (16,9))
        #, left=0.02, right=0.98, wspace=0.1, hspace=0.5)
        #gs = GridSpec(1,2)#, left=0.02, right=0.98, wspace=0.1, hspace=0.5)
        # sns.set_style("whitegrid")
       # max_acc = np.max([np.max(acc) for acc in self.acceptance])
        #create_subtitle(fig, gs[0, ::], f'Experimental acceptances: {self.label}')
        
       # print(acc)
        max_acc_exp = np.nanmax(acc_exp)
        max_acc = max_acc_exp
        
        gs = GridSpec(1,2)
        
        if acc_th is not None:
           
            max_acc_th =  np.nanmax(acc_th)
            ratio = acc_exp / acc_th
            #u_ratio = ratio * err / acc_exp
            variance = acc_exp*(acc_th-acc_exp)/(acc_th**3) ####Binomial variance
            nz = (variance>=0)
            wi = 1/variance[nz]
            #print(variance)
            mean_ratio = np.sum(ratio[nz] * wi ) / np.sum(wi)
            #u_ratio = np.sqrt(variance) ##poisson unc
            mean_u_ratio = 1/np.sqrt(np.sum(wi))
            str_max = f'max_A_exp/max_A_th = {max_acc_exp:.2f}/{max_acc_th:.2f} = {max_acc_exp/max_acc_th:.2f}' 
            str_mean = f'<A_exp/A_th> +/- = {mean_ratio:.4f} +/- {2*mean_u_ratio:.4f} 95% C.L.'
            print(str_max, '\n', str_mean)
            logging.info(str_max)
            logging.info(str_mean)
            ax0 = fig.add_subplot(gs[0,0], projection='3d') 
            
            #ax0 = Axes3D(fig)
            ax0.view_init(elev=15., azim=45)       
           
            im0 = ax0.plot_surface(
                ZE,
                AZ,
                acc_th,
                cmap="jet", #cm.coolwarm
                linewidth=0,
                antialiased=False,
                alpha=1
            )
            max_acc= max_acc_th
            ax0.set_xlabel('zenith $\\theta$ ($^{\circ}$)', fontsize=14)
            ax0.set_ylabel('azimuth $\\varphi$ ($^{\circ}$)', fontsize=14)

        
        ax1 = fig.add_subplot(gs[0,1], projection='3d') 
        ax1.view_init(elev=15., azim=45)      
        im1 = ax1.plot_surface(
            ZE,
            AZ,
            acc_exp,
            cmap="jet", #cm.coolwarm
            linewidth=0,
            antialiased=False,
            alpha=1,
            vmin=0, vmax=max_acc
        )
    
        #3D plot 
        ax1.set_zlim(0, max_acc )
        #ax1.get_zaxis().set_visible(False)
        cbar = plt.colorbar(im1,  shrink=0.5, orientation="vertical")
        cbar.ax.tick_params(labelsize=12)
        if acc_th is not None: cbar.set_label(label='Integrated acceptance (cm².sr)', size=14)
        else: cbar.set_label(label='Experimental Acceptance (cm².sr)', size=14)

        
        ax1.set_xlabel('zenith $\\theta$ ($^{\circ}$)', fontsize=14)
        ax1.set_ylabel('azimuth $\\varphi$ ($^{\circ}$)', fontsize=14)
            
        gs.tight_layout(fig)       
        plt.savefig(
            os.path.join(self.acc_dir,"", f"acceptance_3D_{label}_{self.evttype}.png")
        )
        plt.close()
    
    
    def plot_ratio_acc(self, acc_exp, acc_th, az=None, ze=None, label=None):
        """Ratio maps between experimental and theoritical acceptances""" 
        if label is None: label=self.label
        r =  acc_exp/acc_th 

        #var = [ k*(n-k)/(n**3) for k, n in zip(acc_exp,acc_th) ] 
        #weight = [1/v for v in var]
        
        #nnan = [ ((~np.isnan(r)) & (r!=np.inf) & (~np.isnan(wi)) & (wi!=np.Inf)) for r, wi in zip(self.ratio, weight) ]
        #ratio_mean = [ np.sum(r[c] * wi[c]) / np.sum(wi[c]) for r, wi, c in zip(self.ratio, weight, nnan) 
        #u_ratio =   [ 1/np.sqrt(np.sum(wi[c])) for wi, c in zip( weight, nnan)]
        #str_eff = f'r_acc={eff_mean:.4f}+/-{u_eff:.4f}'
        
        fig= plt.figure(14, figsize= (16,9))
        gs = GridSpec(1, 1, left=0.05, right=0.95, wspace=0.2, hspace=0.5)
        #fig, axs = plt.subplots(nrows=1, ncols=len(ratios))
        fontsize = 36
        ticksize = 26
        labelbarsize = 40
     
        ax = fig.add_subplot(gs[0,0], aspect='equal')
        im = ax.imshow(r, cmap='jet',  vmin=0 , vmax=1 ) #shading='auto',
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        cbar = plt.colorbar(im, cax=cax)# format='%.0e')
        cbar.ax.tick_params(labelsize=ticksize)
        #cbar.set_label(label='$\\mathbf{\\mathcal{T}_{exp}}$ / $\\mathbf{\\mathcal{T}_{th}}$', size=labelbarsize)
        cbar.set_label(label='$\\mathcal{T}_{exp}$ / $\\mathcal{T}_{th}$', size=labelbarsize)
        ax.grid(False)
        
        locs = ax.get_xticks()[1:-1]  # Get the current locations and labels.
        new_x = [str(int(az[int(l)])) for l in locs]
        new_y = [str(int(ze[int(l)])) for l in locs]
        ax.set_xticks(locs)
        ax.set_xticklabels(new_x)
        ax.set_yticks(locs)
        ax.set_yticklabels(new_y)
        ax.tick_params(axis='both', labelsize=ticksize)
        ax.set_xlabel('azimuth $\\varphi$ [deg]', fontsize=fontsize)
        ax.set_ylabel('zenith $\\theta$ [deg]', fontsize=fontsize)
        gs.tight_layout(fig)
        plt.savefig(
            os.path.join(self.acc_dir,"", f"ratio_acc_exp_vs_theo.pdf")
        )
        plt.close()
        
if __name__ == '__main__':
    pass

    
    
    
    
   
    