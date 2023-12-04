#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Raphaël Bajou
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.markers import MarkerStyle
from scipy.interpolate import griddata
import scipy.ndimage
import os
from pathlib import Path
from matplotlib.gridspec import GridSpec
from matplotlib.colors import LogNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable
import warnings
warnings.filterwarnings("ignore")
#personal modules
from telescope import Telescope


class Muography:
    def __init__(self, telescope:Telescope,  hitmap:dict=None, label:str=None, outdir:str=None, info:dict=None, acceptance:dict=None, mask:dict=None, topography:dict=None):
        self.tel = telescope
        self.label = label
        self.acceptance = acceptance
        self.mask = mask
        self.topography = topography
        self.sconfig = list(telescope.configurations.keys())
        if outdir is not None:
            self.outdir = outdir
            self.flux_dir = os.path.join(self.outdir, "flux")
            self.op_dir   = os.path.join(self.outdir, "opacity")
            self.de_dir   = os.path.join(self.outdir, "density")
            Path(self.flux_dir).mkdir(parents=True, exist_ok=True)
            Path(self.op_dir).mkdir(parents=True, exist_ok=True)
            Path(self.de_dir).mkdir(parents=True, exist_ok=True)
        if hitmap is not None:
            self.hm = hitmap
            self.pixels  = { conf : self.tel.los[conf].shape[:-1] for conf in self.sconfig}
            self.flux    = { conf : np.zeros(shape=self.pixels[conf]) for conf in self.sconfig } 
            self.unc_flux    = { conf : np.zeros(shape=self.pixels[conf]) for conf in self.sconfig } 
            self.unc_flux_tot = { conf : np.zeros(shape=self.pixels[conf]) for conf in self.sconfig } 
            self.opacity = { conf : np.zeros(shape=self.pixels[conf]) for conf in self.sconfig } 
            self.unc_opacity_stat = { conf : np.zeros(shape=self.pixels[conf]) for conf in self.sconfig } 
            self.unc_opacity_sys = { conf : np.zeros(shape=self.pixels[conf]) for conf in self.sconfig } 
            self.unc_opacity_tot = { conf : np.zeros(shape=self.pixels[conf]) for conf in self.sconfig } 
            self.density = { conf : np.zeros(shape=self.pixels[conf]) for conf in self.sconfig } 
            self.unc_density_stat = { conf : np.zeros(shape=self.pixels[conf]) for conf in self.sconfig } 
            self.unc_density_sys = { conf : np.zeros(shape=self.pixels[conf]) for conf in self.sconfig } 
            self.unc_density_tot = { conf : np.zeros(shape=self.pixels[conf]) for conf in self.sconfig } 
        if info is not None: 
            self.runDuration = float(info['run_duration'])
            print(f'run_duration={self.runDuration/(24*3600):.1f} jours')

    def compute_flux(self, efficiency:dict=None, unc_eff:dict=None, hitmap:dict=None,  dT:float=None, acceptance:dict=None, unc_acc:dict=None):   
        if efficiency is None: efficiency = {c:np.ones(shape=self.pixels[c]) for c in self.sconfig}
        if hitmap is None: hitmap = self.hm
        if dT is None: dT = self.runDuration
        if acceptance is None:   
            acceptance=self.acceptance
        for ( (c, n), (_, acc), (_, eff) ) in zip(hitmap.items(), acceptance.items(), efficiency.items()):
            s = ((acc!=0.) & (eff!=0.))
            acc[~s], unc_acc[c][~s] = np.nan, np.nan 
            eff[~s], unc_eff[c][~s] = np.nan, np.nan 
            self.flux[c]= n / ( dT * acc * eff) 
            np.savetxt(os.path.join(self.flux_dir, '', f'flux_{c}.txt'), self.flux[c], delimiter='\t', fmt='%.5e')
            self.unc_flux[c] = self.flux[c] * np.sqrt((unc_acc[c]/acc)**2 + (np.sqrt(n)/n)**2 )#+ (unc_eff[c]/eff)**2 ) 
            np.savetxt(os.path.join(self.flux_dir, '', f'unc_flux_{c}.txt'), self.unc_flux[c], delimiter='\t', fmt='%.5e')
   

    def plot_flux(self, flux:dict, range:tuple, az:dict, ze:dict, topography:dict=None, mask:dict=None, colorbar:bool=True, sigma:list=[1,1], mode:str='constant', outdir:str=None, label:str=None):
        fontsize = 22#36
        ticksize = 16#26
        legendsize = 18#40
        if outdir is None: outdir= self.flux_dir
        if label is None: label=self.label
        for i, (c,f) in enumerate(flux.items()):
            fig,ax = plt.subplots(figsize= (12,8))
            ax.grid(False)
            f[~np.isfinite(f)] =  np.nan
            if mask is not None : f[mask[c]] = np.nan
            # Diplay filtered array
            fmin,fmax  = range
            a, z = az[c], ze[c]
            im = ax.pcolor(a,z, f, cmap='jet_r',  shading='auto', norm=LogNorm(vmin=fmin, vmax=fmax)) #norm=LogNorm(vmin=np.min(f), vmax=np.max(f))#vmin=ZTrue_Tomo_Flux_3p1.min(), vmax=ZTrue_Tomo_Flux_3p1.max(),
            ax.set(ylim=[50, 90]) #deg
            if c=='4p': 
                ax.set(ylim=[55, 85]) #deg
            if  topography is not None: 
                ax.plot(topography[c][:,0], topography[c][:,1], linewidth=3, color='black')
            if colorbar : 
                #cax =  fig.add_axes([0.87, 0.15, 0.03, 0.7]) #[0.15, 0.15, 0.5, 0.05]) #[left, bottom, length/width, height]
                divider0 = make_axes_locatable(ax)
                cax = divider0.append_axes("right", size="5%", pad=0.05)
                cbar = fig.colorbar(im, cax=cax, orientation="vertical")
                cbar.ax.tick_params(labelsize=ticksize)
                cbar.set_label(label='Flux [cm$^{-2}$.s$^{-1}$.sr$^{-1}$]', size=fontsize)
                fig.subplots_adjust(right=0.85)
        
            ax.set_xlabel('azimuth $\\varphi$ [deg]', fontsize=fontsize)
            ax.set_ylabel('zenith $\\theta$ [deg]', fontsize=fontsize)
            ax.tick_params(axis='both', which='major', labelsize=ticksize)
            ax.invert_yaxis()
            ax.set(frame_on=False)
            fig.savefig(
                os.path.join(outdir,"", f"tomo_flux_{c}.png")
            )
        
            

    def plot_thickness(self, az:dict, ze:dict, app_thick:dict):
        fig = plt.figure(figsize= (12,8))
        gs = GridSpec(1, len(self.opacity))#, left=0.04, right=0.99, wspace=0.1, hspace=0.5)
        thick_min = min([ t[~np.isnan(t)].min() for _,t in app_thick.items()])
        thick_max = min([ t[~np.isnan(t)].max() for _,t in app_thick.items()])
        for i, (c,thick) in enumerate(app_thick.items()):
            ax = fig.add_subplot(gs[0,i],aspect="equal")
            ax.grid(False)
            a, z = az[c], ze[c]
            A, Z = np.meshgrid(np.linspace(a.min(), a.max(), 31  )  , np.linspace(z.min(), z.max(), 31  ))
            c = ax.pcolor(a, z, thick, cmap='jet', shading='auto', vmin=thick_min , vmax=thick_max )
            ax.invert_yaxis()
            cbar = fig.colorbar(c, ax=ax, shrink=0.75, format='%.0e', orientation="horizontal")
            cbar.ax.tick_params(labelsize=8)
            cbar.set_label(label=u'thickness [m]', size=12)
            if i==0 :  ax.set_ylabel('zenith $\\theta$ [deg]', fontsize=12)
            ax.set_xlabel('azimuth $\\varphi$ [deg]', fontsize=12)
            ax.set_title(f'{self.sconfig[i]} config')
            ax.set(frame_on=False)
        gs.tight_layout(fig)
        plt.figtext(.5,.95, f"thickness : {self.label}", fontsize=12, ha='center')
        plt.savefig(
            os.path.join(self.op_dir,"", f"thickness.png")
        )
        plt.close()
        
    def interpolate_opacity(self, range_ze:np.ndarray, range_flux:np.ndarray, sig_model:np.ndarray, range_op:np.ndarray, tomo_ze:dict, tomo_flux:dict, app_thick:dict, unc_tl:float=1, *args,**kwargs):
        points, range_op_flat = np.array([range_ze.flatten(), range_flux.flatten()]).T, range_op.flatten() #np.exp( np.log(10) * op ) 
        zmin, zmax = np.min(points[:, 0]), np.max(points[:, 0])
        fmin, fmax = np.min(points[:, 1]), np.max(points[:, 1])
        range_flux_flat = range_flux.flatten()
        for (conf,ze), (_,flux), (_,tl) in zip(tomo_ze.items(), tomo_flux.items(), app_thick.items()):  
            shape= (self.tel.los[conf].shape[0], self.tel.los[conf].shape[1])
            grid_op =  np.zeros(shape=shape)
            in_range = np.logical_and((fmin < flux) & (flux < fmax), ~(np.isnan(flux)) & ~(np.isnan(tl)))
            grid_tmp = np.zeros(shape=flux.shape)
            grid_x, grid_y = ze, flux
            grid_tmp[in_range] = griddata(points, range_op_flat, (grid_x[in_range], grid_y[in_range]), *args, **kwargs) 
            print(f"(opacity_min, opacity_max)_{conf}  =  ({np.nanmin(grid_tmp[in_range]):.3e}, {np.nanmax(grid_tmp[in_range]):.3e}) mwe")
            grid_tmp[~in_range] = np.nan
            grid_op = grid_tmp 
            #std_op  = np.nanstd(grid_op)
            uflux_data = self.unc_flux[conf].flatten()
            self.unc_opacity_stat[conf] = self.unc_opacity_stat[conf].flatten()
            self.unc_opacity_sys[conf] = self.unc_opacity_sys[conf].flatten()
            self.unc_opacity_tot[conf] = self.unc_opacity_tot[conf].flatten()
            for i,o in enumerate(grid_op.flatten()):
                ix = np.argmin(abs(range_op_flat-o))
                do = range_op_flat[ix] - range_op_flat[ix+1] 
                df = range_flux_flat[ix] - range_flux_flat[ix+1]
                b = df/do
                self.unc_opacity_stat[conf][i] = 1/abs(b) *  uflux_data[i]
                self.unc_opacity_sys[conf][i] = 1/abs(b) *  sig_model[ix]
                sig_flux = np.sqrt(uflux_data[i]**2 + sig_model[ix]**2)
                self.unc_opacity_tot[conf][i] = 1/abs(b) * sig_flux

            self.unc_opacity_stat[conf] = self.unc_opacity_stat[conf].reshape(self.pixels[conf])
            self.unc_opacity_stat[conf][np.isnan(tl)]=np.nan
            self.unc_opacity_sys[conf] = self.unc_opacity_sys[conf].reshape(self.pixels[conf])
            self.unc_opacity_sys[conf][np.isnan(tl)]=np.nan
            self.unc_opacity_tot[conf] = self.unc_opacity_tot[conf].reshape(self.pixels[conf])
            self.unc_opacity_tot[conf][np.isnan(tl)]=np.nan
            self.opacity[conf] = grid_op
            self.mask = in_range
            rho_mean = grid_op/tl
            rho_mean[rho_mean==0] = np.nan
            self.density[conf] = rho_mean 
            self.unc_density_stat[conf] = np.sqrt(  ((unc_tl*self.opacity[conf]) / tl**2 )**2  +   (self.unc_opacity_stat[conf]/tl)**2)
            self.unc_density_sys[conf] =  np.sqrt(  ((unc_tl*self.opacity[conf]) / tl**2 )**2  +   (self.unc_opacity_sys[conf]/tl)**2)
            self.unc_density_tot[conf]     =  np.sqrt(  ((unc_tl*self.opacity[conf]) / tl**2 )**2  +  (self.unc_opacity_tot[conf]/tl)**2)

    def plot_mean_density(self, quantity:str, val:dict, range:tuple, az:dict, ze:dict, topography:dict=None, sigma:tuple=None, mask:dict=None, outdir:str=None, label:str=None, cmap:str="jet", lognorm:bool=False, mode:str='mirror', threshold:float=None, crater:dict=None):
        """ Mean density maps """
        fontsize = 22#36
        ticksize = 16#26
        legendsize = 18#40
        vmin, vmax = range
        if outdir is None: outdir= self.op_dir
        if label is None: label=self.label
        for i, (c,rho) in enumerate(val.items()):
           
            fig,ax = plt.subplots(figsize= (12,8))
            a, z = az[c], ze[c]
            ax.grid(False)
            dphi = 35 #deg
            xlim, ylim = [np.median(a)-dphi, np.median(a)+dphi],[50, 90]
            ax.set(xlim=xlim, ylim=ylim) #deg
            if mask is not None: 
                if threshold is not None: 
                    aberrant = (rho >= threshold) 
                    rho[aberrant] = np.nan

                #https://stackoverflow.com/questions/18697532/gaussian-filtering-a-image-with-nan-in-python
                rho[mask[c]] = np.nan
                rho.reshape(self.tel.los[c].shape[0], self.tel.los[c].shape[1])
 
            
            if lognorm : 
                im = ax.pcolor(a,z, rho, cmap=cmap,  shading='auto', norm=LogNorm(vmin=vmin, vmax=vmax))#vmin=ZTrue_Tomo_Flux_3p1.min(), vmax=ZTrue_Tomo_Flux_3p1.max(),
            else: 
                im = ax.pcolor(a,z, rho, cmap=cmap,  shading='auto', vmin=vmin, vmax=vmax)
            if  topography is not None: 
                topo = topography[c]
                ax.plot(topo[:,0],topo[:,1], linewidth=3, color='black')
            if crater is not None:
                marker=MarkerStyle("*")
                dz=-0.5
                size=35
                for key, value in crater.items():    
                    if key =="SC": mark_col = "magenta"
                    elif key =="TAR": mark_col="white"
                    elif key =="G56": mark_col="lightgrey"
                    elif key =="BLK": mark_col="grey"
                    else: mark_col="black"
                    az_cr, ze_cr =  value[0], value[1]
                    ax.plot(az_cr,ze_cr+dz, marker=marker, color=mark_col, markersize=size, markeredgewidth=1.5, markeredgecolor="black")
                    ax.annotate(key, ((az_cr+0.5, ze_cr+dz-1)), fontsize=14)
            
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            cbar = fig.colorbar(im, cax=cax, orientation="vertical")
            cbar.ax.tick_params(labelsize=ticksize)
            cbar.set_label(label=quantity, size=fontsize)
            fig.subplots_adjust(right=0.85)
            
            ax.set_xlabel('azimuth $\\varphi$ [deg]', fontsize=fontsize)
            ax.set_ylabel('zenith $\\theta$ [deg]', fontsize=fontsize)
            ax.tick_params(axis='both', which='major', labelsize=ticksize)
            ax.grid(True, which='both',linestyle='-', linewidth="0.25", color='grey')
            ax.invert_yaxis()
            ax.set(frame_on=True)
            fig.savefig(
                os.path.join(outdir,"", f"mean_density_{c}.pdf")
            )
        plt.close()
        
        
  
    def plot_mean_density_gausfilter(self, quantity:str, val:dict, range:tuple, az:dict, ze:dict, topography:dict=None, sigma:tuple=None, mask:dict=None, outdir:str=None, label:str=None, cmap:str="jet", lognorm:bool=False, mode:str='mirror', threshold:float=None, crater:dict=None):
        """ Mean density maps with gaussian filter applied """
        fontsize = 22#36
        ticksize = 16#26
        legendsize = 18#40
        vmin, vmax = range
        if outdir is None: outdir= self.op_dir
        if label is None: label=self.label
        for i, (c,rho) in enumerate(val.items()):
            fig,ax = plt.subplots(figsize= (12,8))
            a, z = az[c], ze[c]
            ax.grid(False)
            dphi = 35 #deg
            xlim, ylim = [np.median(a)-dphi, np.median(a)+dphi],[50, 90]
            ax.set(xlim=xlim, ylim=ylim) #deg
            if mask is not None: 
                if threshold is not None: 
                    aberrant = (rho >= threshold) 
                    rho[aberrant] = np.nan

                #https://stackoverflow.com/questions/18697532/gaussian-filtering-a-image-with-nan-in-python
                rho[mask[c]] = np.nan
                rho_cp = np.copy(rho)
                rho_cp[np.isnan(rho)]=0
                rho_cp2 = scipy.ndimage.filters.gaussian_filter(rho_cp, sigma, mode=mode)
                rho_cp3= 0*rho.copy()+1
                rho_cp3[np.isnan(rho)]=0
                rho_cp4=scipy.ndimage.gaussian_filter(rho_cp3,sigma=sigma, mode=mode)
                rho_gaus=rho_cp2/rho_cp4
                rho.reshape(self.tel.los[c].shape[0], self.tel.los[c].shape[1])
                rho_gaus.reshape(self.tel.los[c].shape[0], self.tel.los[c].shape[1])
                rho_gaus[mask[c]] = np.nan
            else: rho_gaus = scipy.ndimage.filters.gaussian_filter(rho, sigma, mode=mode)
            
            
            if lognorm : 
                im = ax.pcolor(a,z, rho_gaus, cmap=cmap,  shading='auto', norm=LogNorm(vmin=vmin, vmax=vmax))#vmin=ZTrue_Tomo_Flux_3p1.min(), vmax=ZTrue_Tomo_Flux_3p1.max(),
            else: 
                im = ax.pcolor(a,z, rho_gaus, cmap=cmap,  shading='auto', vmin=vmin, vmax=vmax)
            np.savetxt(os.path.join(outdir, '', f'mean_density_gausfilter_{c}.txt'), rho_gaus , fmt='%.5e', delimiter='\t') 
            if  topography is not None: 
                topo = topography[c]
                ax.plot(topo[:,0], topo[:,1], linewidth=3, color='black')
            if crater is not None:
                marker=MarkerStyle("*")
                dz=-0.5
                size=35
                for key, value in crater.items():    
                    if key =="SC": mark_col = "magenta"
                    elif key =="TAR": mark_col="white"
                    elif key =="G56": mark_col="lightgrey"
                    elif key =="BLK": mark_col="grey"
                    else: mark_col="black"
                    az_cr, ze_cr =  value[0], value[1]
                    ax.plot(az_cr,ze_cr+dz, marker=marker, color=mark_col, markersize=size, markeredgewidth=1.5, markeredgecolor="black")
                    ax.annotate(key, ((az_cr+0.5, ze_cr+dz-1)), fontsize=14)
                   
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            cbar = fig.colorbar(im, cax=cax, orientation="vertical")
            cbar.ax.tick_params(labelsize=ticksize)
            cbar.set_label(label=quantity, size=fontsize)
            fig.subplots_adjust(right=0.85)
            
            ax.set_xlabel('azimuth $\\varphi$ [deg]', fontsize=fontsize)
            ax.set_ylabel('zenith $\\theta$ [deg]', fontsize=fontsize)
            ax.tick_params(axis='both', which='major', labelsize=ticksize)
            ax.grid(True, which='both',linestyle='-', linewidth="0.25", color='grey')
            ax.invert_yaxis()
            ax.set(frame_on=True)
            fig.savefig(
                os.path.join(outdir,"", f"mean_density_gausfilter_{c}.pdf")
            )
        plt.close()



        
    
    
    # def plot_density_variations(self, density:dict, range:tuple, az:dict, ze:dict, rho0:float=2.1, sigma:tuple=(1,1), outdir:str=None, label:str=None):
    #     """ Average density [g.cm$^{-3}$] maps """
        
    #     fig = plt.figure(figsize= (12,8))
    #     gs = GridSpec(1, len(density))#, left=0.04, right=0.99, wspace=0.1, hspace=0.5)
    #  
        
        
    #     #D_min = min([ np.min(D[~np.isnan(D)]) for _,D in self.density.items()])
    #     #D_max = min([ np.max(D[~np.isnan(D)]) for _,D in self.density.items()])
  
    #     vmin, vmax = range
    #     if outdir is None: outdir= self.op_dir
    #     if label is None: label=self.label
        
    #     for i, (conf,rho) in enumerate(density.items()):
    #         ax = fig.add_subplot(gs[0,i], aspect="equal")
    #         ax.grid(False)
    #         a, z = az[conf], ze[conf]
    #         m = ( ( np.isnan(rho) ) & ( ~np.isfinite(rho) ) )
    #         rho[m] = np.nan
    #         #c=ax.imshow(y, cmap='jet', interpolation='nearest')
    #         #A, Z = np.meshgrid(np.linspace(a.min(), a.max(), 31  )  , np.linspace(z.min(), z.max(), 31  ))
    #         rel_var = rho-rho0/rho0
    #         rel_var[m] = np.nan
    #         rv_gaus = scipy.ndimage.filters.gaussian_filter(rel_var, sigma, mode='constant')
    #         np.savetxt(os.path.join(self.op_dir, '', f'relvar_density_{self.label}_{self.evttype}_{conf}.txt'), rel_var , fmt='%.5e', delimiter='\t')
    #         # Diplay filtered array
    #         #rv_gaus[rv_gaus<=var_min]=np.nan
    #         #var_min, var_max = np.min(rel_var[~np.isnan(rel_var)]), np.max(rel_var[~np.isnan(rel_var)])
    #         c = ax.pcolor(a, z, rv_gaus, cmap='jet', shading='auto', vmin=vmin , vmax=vmax )
    #         if self.topography is not None: 
    #             ax.plot(self.topography[conf][:,0], self.topography[conf][:,1], linewidth=2, color='red')
    #         #ax.invert_yaxis()
    #         ax.set(ylim=[50, 90]) #deg
    #         #ticks = np.linspace(np.around(d_min,0),d_max, 6 )
    #         ticks = np.linspace(vmin, vmax,10)
    #         cbar = fig.colorbar(c, ax=ax, shrink=0.75, format='%.0e', orientation="horizontal", ticks=ticks )
    #         labels = [f"{t:.1f}" for t in ticks]
    #         cbar.ax.set_xticklabels(labels, fontsize=12)
    #         cbar.ax.tick_params(labelsize=12)
    #         #cbar.set_label(label=u'mean density $\\varrho/L$ [g.cm$^{-3}$]', size=12)#[mwe.m$^{-1}$]', size=12)
    #         cbar.set_label(label='relative variation mean density ($\\overline{\\rho}$-$\\rho_{0}$)/$\\rho_{0}$', size=12)#[mwe.m$^{-1}$]', size=12)
    #         ax.set_ylabel('zenith $\\theta$ [deg]', fontsize=12)
    #         ax.set_xlabel('azimuth $\\varphi$ [deg]', fontsize=12)
    #         ax.set_title(f'{self.sconfig[i]} config')
    #         ax.invert_yaxis()
    #         ax.set(frame_on=False)
    #     gs.tight_layout(fig)    
    #     plt.figtext(.5,.95, 'Relative mean density $\\overline{\\rho}$ variation : '+f'{label}'+'\n$\\rho_{0}$='+f'{rho0}'+' g.cm$^{-3}$' , fontsize=12, ha='center')
    #     plt.savefig(
    #         os.path.join(outdir,"", f"relvar_mean_dens_{label}_{self.evttype}.pdf")
    #     )
    #     plt.close()
    
     
        
