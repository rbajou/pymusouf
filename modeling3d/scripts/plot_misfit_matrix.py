#!/usr/bin/env python3
# -*- coding: utf-8 -*-



from dataclasses import dataclass, field
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colours
from matplotlib.cm import ScalarMappable
from mpl_toolkits.mplot3d.art3d import PolyCollection, Poly3DCollection, Line3DCollection
from matplotlib.ticker import MultipleLocator,EngFormatter, ScalarFormatter, LogLocator
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.gridspec import GridSpec
from matplotlib.offsetbox import AnchoredText
import matplotlib.lines as mlines
from scipy.interpolate import griddata
import sys
import os
import scipy.io as sio
from pathlib import Path
import inspect
import time
from datetime import datetime, date, timezone
import logging
import glob
import yaml
import pickle
import json
import warnings
import logging
import pandas as pd
import mat73 #read v7.3 mat files
import palettable

#personal modules
from configuration import str2telescope, Telescope, dict_tel
from analysis import AcqVars, Observable
from plot_density_3d import VoxeledDome
from inversion import Inversion, KernelMatrix



params = {'legend.fontsize': 'xx-large',
            'axes.labelsize': 'xx-large',
            'axes.titlesize':'xx-large',
            'xtick.labelsize':'xx-large',
            'ytick.labelsize':'xx-large',
            'axes.labelpad':10}
plt.rcParams.update(params)  

if __name__=="__main__":
    data_dir = Path.home()/"data"
    run = "synthetic"
    tag_inv = "smoothing"
    out_dir = data_dir / "inversion" / run / tag_inv 
    #dict_run = { 1: out_dir / "19032023_1950_smooth_4det", }
    res = sys.argv[1]
    #rho0 = 2.0
    #
    # dict_run = { 1: out_dir /"28052023_172626_smooth_4det_32",}
    dict_run = { 1: out_dir /sys.argv[2],}
    rho0 = 1.8
    # dict_run = { 1: out_dir /"30052023_110352_smooth_4det_32",}
    # rho0 = 1.8
    # dict_run = { 1: out_dir /"29052023_232640_smooth_4det_32"}
    # rho0 = 1.8      
    
    for i, (_,run) in enumerate(dict_run.items()):
        mat_rho0 = np.loadtxt(run/ "rho0.txt",delimiter="\t")
        if mat_rho0.shape == ():  mat_rho0 = np.array([mat_rho0.mean()])
        
        ix_rho0 = np.argmin(abs(rho0-mat_rho0))
        
        m_sigma = np.loadtxt(run / "sigma_prior.txt",delimiter="\t")
        if m_sigma.shape == ():  m_sigma = np.array([m_sigma.mean()])
        m_length = np.loadtxt(run / "correlation_length.txt", delimiter="\t")
        if m_length.shape == ():  m_length = np.array([m_length.mean()])
        
        # m_misfit_d = np.loadtxt(run/ ".txt", delimiter="\t")
        # m_misfit_m = np.loadtxt(run/ ".txt", delimiter="\t")
        m_misfit_d = np.load(run/ f"misfit_data_rho0{rho0}_res{res}m.npy")
        #print(f"m_misfit_d = {m_misfit_d}")
        m_misfit_m = np.load(run/ f"misfit_model_rho0{rho0}_res{res}m.npy")
        m_res_d = np.loadtxt(run/ f"res_data_rho0{rho0}_res{res}m.txt")
        m_res_m = np.loadtxt(run/ f"res_model_rho0{rho0}_res{res}m.txt")
        
        m_std_post = np.load(run/ f"std_dev_res{res}m.npy")[ix_rho0]

      #  print(np.all(np.nanmean(m_std_post,axis=2)==m_misfit_m))
      

        if i==0 :
            m_sigma_all = m_sigma
            m_length_all = m_length
            m_misfit_d_all = m_misfit_d
            m_misfit_m_all = m_misfit_m
            m_res_d_all = m_res_d 
            m_res_m_all = m_res_m 
        else : 
            # m_sigma_all = np.sort(np.vstack(( m_sigma_all, m_sigma )) , axis=0)
            # m_length_all = np.sort(np.hstack(( m_length_all, m_length )), axis=1)
            m_sigma_all = np.vstack(( m_sigma_all, m_sigma )) 
            #print(m_sigma_all[0:m_sigma.shape[0],:]
            #m_sigma_all = np.repeat(m_sigma_all, np.ones(m_sigma.shape[0]), axis=0)

            #m_sigma_all = np.hstack(( m_sigma_all, m_sigma_all[0:m_sigma.shape[0],:]))
            #m_sigma_all = np.tile(m_sigma_all, reps=m_sigma.shape[0])
            m_length_all = np.hstack(( m_length_all, m_length ))
            m_misfit_d_all = np.vstack(( m_misfit_d_all, m_misfit_d ))  
            m_misfit_m_all = np.vstack(( m_misfit_m_all, m_misfit_m ))   
            m_res_d_all = np.vstack(( m_res_d_all, m_res_d ))  
            m_res_m_all = np.vstack(( m_res_m_all, m_res_m ))  
            

    X,Y,Z =  m_length_all, m_sigma_all, m_misfit_d_all
    print(X.shape,Y.shape,Z.shape)
    
    
    points = np.zeros(shape=(len(X.flatten()), 2))#60000
    points[:, 0] = X.flatten() #zenith 1D array
    points[:, 1] = Y.flatten()
    values = Z.flatten()
    x, y  = np.linspace(np.min(X), np.max(X), 100), np.linspace(np.min(Y), np.max(Y), 100)
    Xnew, Ynew = np.meshgrid(x,y)
    Znew= griddata(points, values.flatten(), (Xnew, Ynew))#, *args, **kwargs) 
    vmin, vmax = np.min(Znew), np.max(Znew) #max(np.max(misfit_d[ix_rho]), misfit_d[ix_rho]) # np.nanmin(misfit), np.nanmax(misfit)
    
    fig, ax = plt.subplots(figsize=(12,8))
    ax.xaxis.set_major_locator(MultipleLocator(1e2))
    ax.yaxis.set_major_locator(MultipleLocator(2e-1))
    cmap = 'jet'
    #im = ax.pcolormesh(X,Y,Z, norm=colours.LogNorm(vmin, vmax), cmap=cmap) # vmin=vmin, vmax=vmax,
    im = ax.pcolormesh(Xnew,Ynew,Znew, vmin=vmin, vmax=vmax, cmap=cmap) 
    ij = np.unravel_index(np.argmin(abs(Znew-1), axis=None), abs(Znew-1).shape) #np.argmin(abs(misfit_d[ix_rho0]-1), axis=1)
    xbf, ybf = Xnew[0, ij[1]], Ynew[ij[0],0]
    sout = f"best misfit_data(rho0={rho0}, l={xbf:.0f}m, sigma={ybf:.2f}g/cm3) = {Znew[ij]:.1f} "
    logging.info(sout)
    print(sout)
    color='floralwhite'
    ax.axvline(xbf, linestyle='dashed', color=color, linewidth=1.)
    ax.axhline(ybf, linestyle='dashed', color=color, linewidth=1.)
    ax.scatter(xbf, ybf ,marker='*', s=40, color=color)
    xlabel = "correlation length $l$ [m]"
    ylabel = "prior error $\\sigma$ [g/cm$^3$]"
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    #ax.set_xscale('log')
    #ax.set_yscale('log')
    ax.set_ylim(np.min(Y), np.max(Y))
    #ax.invert_yaxis()
    ax.set_title(f"$\\rho_0$ = {rho0}"+ " g/cm$^{3}$")
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = fig.colorbar(im, cax=cax, orientation="vertical")
    cbar.set_label("misfit data $\\phi_{d}$")#, size=14)
    fig.tight_layout()
    #plt.show()
    fout = out_dir / dict_run[1]/ f"misfit_data_rho0{rho0}_res{res}m"
    fig.savefig(f"{fout}.png", transparent=True)
    print(f"out: {fout}.png")
    plt.close()

    ######### misfit model in (lc, sigma)
    X,Y,Z =  m_length_all, m_sigma_all, m_misfit_m_all
    
    points = np.zeros(shape=(len(X.flatten()), 2))#60000
    points[:, 0] = X.flatten() #zenith 1D array
    points[:, 1] = Y.flatten()
    values = Z.flatten()
    x, y  = np.linspace(np.min(X), np.max(X), 100), np.linspace(np.min(Y), np.max(Y), 100)
    Xnew, Ynew = np.meshgrid(x,y)
    Znew= griddata(points, values.flatten(), (Xnew, Ynew))#, *args, **kwargs) 
    
    fig, ax = plt.subplots(figsize=(12,8))
    vmin, vmax = np.nanmin(Znew), np.nanmax(Znew)
    im = ax.pcolor(Xnew, Ynew,Znew, norm=colours.LogNorm(vmin, vmax), cmap=cmap)#, vmin=vmin, vmax=vmax) 
    ij = np.unravel_index(np.argmin(Znew, axis=None), Znew.shape) #np.argmin(abs(misfit_d[ix_rho0]-1), axis=1)
    xbf, ybf = Xnew[0, ij[1]], Ynew[ij[0],0] #best fit
    sout = f"best residual(rho0={rho0}, l={xbf:.0f}m, sigma={ybf:.2f}g/cm3) = {Znew[ij]:.1f} "
    logging.info(sout)
    print(sout)
    ax.xaxis.set_major_locator(MultipleLocator(1e2))
    ax.yaxis.set_major_locator(MultipleLocator(2e-1))
    ax.axvline(xbf, linestyle='dashed', color=color, linewidth=1.)
    ax.axhline(ybf, linestyle='dashed', color=color, linewidth=1.)
    ax.scatter(xbf, ybf ,marker='*', s=40, color=color)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    #ax.set_xscale('log')
    #ax.set_yscale('log')
    ax.set_ylim(np.min(Y), np.max(Y))
    #ax.invert_yaxis()
    ax.set_title(f"$\\rho_0$ = {rho0}"+ " g/cm$^{3}$")
    #ax.legend(loc="upper right")
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = fig.colorbar(im, cax=cax, orientation="vertical")
    #cbar.set_label("$|| \\rho_{true} - \\rho_{post} ||^{2}$")#, size=14)
    cbar.set_label("misfit model $\\phi_{m}$")
    fig.tight_layout()
    fout = out_dir/ dict_run[1]/ f"misfit_model_rho0{rho0}_res{res}m.png"
    fig.savefig(str(fout), transparent=True)
    print(f"save {fout}")
    plt.close()
    ############
    
    ############ misfit data in (lc, sigma)
    X,Y,Z =  m_length_all, m_sigma_all, m_misfit_d_all
    
    points = np.zeros(shape=(len(X.flatten()), 2))#60000
    points[:, 0] = X.flatten() #zenith 1D array
    points[:, 1] = Y.flatten()
    values = Z.flatten()
    x, y  = np.linspace(np.min(X), np.max(X), 100), np.linspace(np.min(Y), np.max(Y), 100)
    Xnew, Ynew = np.meshgrid(x,y)
    Znew= griddata(points, values.flatten(), (Xnew, Ynew))#, *args, **kwargs) 
    
    fig, ax = plt.subplots(figsize=(12,8))
    vmin, vmax = np.nanmin(Znew), np.nanmax(Znew)
    #im = ax.pcolor(Xnew, Ynew,Znew, norm=colours.LogNorm(vmin, vmax), cmap=cmap)#, vmin=vmin, vmax=vmax) 
    im = ax.pcolor(Xnew, Ynew,Znew, vmin=vmin, vmax=vmax, cmap=cmap)
    ij = np.unravel_index(np.argmin(abs(Znew-1), axis=None), Znew.shape) #np.argmin(abs(misfit_d[ix_rho0]-1), axis=1)
    xbf, ybf = Xnew[0, ij[1]], Ynew[ij[0],0] #best fit
    sout = f"best misfit data(rho0={rho0}, l={xbf:.0f}m, sigma={ybf:.2f}g/cm3) = {Znew[ij]:.1f} "
    logging.info(sout)
    print(sout)
    ax.xaxis.set_major_locator(MultipleLocator(1e2))
    ax.yaxis.set_major_locator(MultipleLocator(2e-1))
    #ax.axvline(xbf, linestyle='dashed', color=color, linewidth=1.)
    #ax.axhline(ybf, linestyle='dashed', color=color, linewidth=1.)
    ax.scatter(xbf, ybf ,marker='*', s=60, color=color)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    #ax.set_xscale('log')
    #ax.set_yscale('log')
    ax.set_ylim(np.min(Y), np.max(Y))
    #ax.invert_yaxis()
    #ax.set_title(f"$\\rho_0$ = {rho0}"+ " g/cm$^{3}$")
    #ax.legend(loc="upper right")
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = fig.colorbar(im, cax=cax, orientation="vertical")
    #cbar.set_label("$\\dfrac{1}{n_d}$ $|| d_{true} - d_{post} ||^{2}$")#, size=14)
    cbar.set_label("misfit data $\\phi_{d}$")
    fig.tight_layout()
    fout = out_dir/ dict_run[1]/ f"misfit_data_rho0{rho0}_res{res}m.png"
    fig.savefig(str(fout), transparent=True)
    print(f"save {fout}")
    plt.close()
    
    

    ############
    ################# residual data in (lc, sigma)
    X,Y,Z =  m_length_all, m_sigma_all, m_res_d_all
    
    points = np.zeros(shape=(len(X.flatten()), 2))#60000
    points[:, 0] = X.flatten() #zenith 1D array
    points[:, 1] = Y.flatten()
    values = Z.flatten()
    x, y  = np.linspace(np.min(X), np.max(X), 100), np.linspace(np.min(Y), np.max(Y), 100)
    Xnew, Ynew = np.meshgrid(x,y)
    Znew= griddata(points, values.flatten(), (Xnew, Ynew))#, *args, **kwargs) 
    
    fig, ax = plt.subplots(figsize=(12,8))
    vmin, vmax = np.nanmin(Znew), np.nanmax(Znew)
    im = ax.pcolor(Xnew, Ynew,Znew, norm=colours.LogNorm(vmin, vmax), cmap=cmap)#, vmin=vmin, vmax=vmax) 
    ij = np.unravel_index(np.argmin(abs(Znew-1), axis=None), Znew.shape) #np.argmin(abs(misfit_d[ix_rho0]-1), axis=1)
    xbf, ybf = Xnew[0, ij[1]], Ynew[ij[0],0] #best fit
    sout = f"best misfit data(rho0={rho0}, l={xbf:.0f}m, sigma={ybf:.2f}g/cm3) = {Znew[ij]:.1f} "
    logging.info(sout)
    print(sout)
    ax.xaxis.set_major_locator(MultipleLocator(1e2))
    ax.yaxis.set_major_locator(MultipleLocator(2e-1))
    ax.axvline(xbf, linestyle='dashed', color=color, linewidth=1.)
    ax.axhline(ybf, linestyle='dashed', color=color, linewidth=1.)
    ax.scatter(xbf, ybf ,marker='*', s=40, color=color)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_ylim(np.min(Y), np.max(Y))
    #ax.invert_yaxis()
    ax.set_title(f"$\\rho_0$ = {rho0}"+ " g/cm$^{3}$")
    #ax.legend(loc="upper right")
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = fig.colorbar(im, cax=cax, orientation="vertical")
    cbar.set_label("$\\dfrac{1}{n_d}$ $|| d_{true} - d_{post} ||^{2}$")
    fig.tight_layout()
    fout = out_dir/ dict_run[1]/ f"res_data_rho0{rho0}_res{res}m.png"
    fig.savefig(str(fout), transparent=True)
    print(f"save {fout}")
    plt.close()
    
    
    ################# residual model in (lc, sigma)
    X,Y,Z =  m_length_all, m_sigma_all, m_res_m_all
    
    points = np.zeros(shape=(len(X.flatten()), 2))#60000
    points[:, 0] = X.flatten() #zenith 1D array
    points[:, 1] = Y.flatten()
    values = Z.flatten()
    x, y  = np.linspace(np.min(X), np.max(X), 100), np.linspace(np.min(Y), np.max(Y), 100)
    Xnew, Ynew = np.meshgrid(x,y)
    Znew= griddata(points, values.flatten(), (Xnew, Ynew))#, *args, **kwargs) 
    
    fig, ax = plt.subplots(figsize=(12,8))
    vmin, vmax = np.nanmin(Znew), np.nanmax(Znew)
    im = ax.pcolor(Xnew, Ynew,Znew, norm=colours.LogNorm(vmin, vmax), cmap=cmap)#, vmin=vmin, vmax=vmax) 
    ij = np.unravel_index(np.argmin(abs(Znew-1), axis=None), Znew.shape) #np.argmin(abs(misfit_d[ix_rho0]-1), axis=1)
    xbf, ybf = Xnew[0, ij[1]], Ynew[ij[0],0] #best fit
    sout = f"best misfit data(rho0={rho0}, l={xbf:.0f}m, sigma={ybf:.2f}g/cm3) = {Znew[ij]:.1f} "
    logging.info(sout)
    print(sout)
    ax.xaxis.set_major_locator(MultipleLocator(1e2))
    ax.yaxis.set_major_locator(MultipleLocator(2e-1))
    #ax.axvline(xbf, linestyle='dashed', color=color, linewidth=1.)
    #ax.axhline(ybf, linestyle='dashed', color=color, linewidth=1.)
    #ax.scatter(xbf, ybf ,marker='*', s=40, color=color)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_ylim(np.min(Y), np.max(Y))
    #ax.invert_yaxis()
    ax.set_title(f"$\\rho_0$ = {rho0}"+ " g/cm$^{3}$")
    #ax.legend(loc="upper right")
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = fig.colorbar(im, cax=cax, orientation="vertical")
    cbar.set_label("$\\dfrac{1}{n_m}$$|| \\rho_{true} - \\rho_{post} ||^{2}$")
    fig.tight_layout()
    fout = out_dir/ dict_run[1]/ f"res_model_rho0{rho0}_res{res}m.png"
    fig.savefig(str(fout), transparent=True)
    print(f"save {fout}")
    plt.close()
    
    
    
    ###|| rho_post - rho_true ||^2 vs || d_post - d_syn ||^2
    X,Y,Z =  m_res_d_all, m_res_m_all, m_sigma_all
    print(X.shape,Y.shape)

    cmap_sigma = 'jet'#palettable.scientific.sequential.Batlow_20.mpl_colormap 
    length_min, length_max = np.min(m_sigma_all), np.max(m_sigma_all)
    range_val_sigma = np.linspace(length_min, length_max, 100)
    norm_sigma = colours.Normalize(vmin=length_min, vmax=length_max)
    try :
        color_scale_sigma =  cmap_sigma(norm_sigma(range_val_sigma))
    except :
        color_scale_sigma = plt.colormaps[cmap_sigma](norm_sigma(range_val_sigma))
        
    arg_col =  [np.argmin(abs(range_val_sigma-v))for v in m_sigma_all.flatten()]    
    color_marker = color_scale_sigma[arg_col]
    visfactor = 5.5
    size_length = np.exp(m_length_all / np.max(m_length_all) *visfactor )
    
    fig, ax = plt.subplots(figsize=(12,8))
    vmin, vmax = np.nanmin(Znew), np.nanmax(Znew)
    x, y = m_misfit_d_all.flatten(), m_misfit_m_all.flatten()
    ax.scatter(x, y, s=size_length, c=color_marker)
    ij = np.unravel_index(np.argmin(Y, axis=None), Y.shape) 
    xbf, ybf = X[0, ij[1]], Y[ij[0],0] #best fit
    print(xbf, ybf)
    ax.scatter(xbf, ybf ,marker='*', s=40, color=color)
    ax.grid(True, which='both',linestyle='dotted', linewidth="0.3 ", color='grey')
    #ax.set_title(f"$\\rho_0$ = {rho0}"+ " g/cm$^{3}$")
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = fig.colorbar(ScalarMappable(norm=norm_sigma, cmap=cmap_sigma), cax=cax)
    cbar.set_label("prior error $\\sigma$ [g/cm$^3$]")#, size=14)
    fig.tight_layout()
    fout = out_dir / dict_run[1]/ f"res_model_vs_res_data_{rho0}_res{res}m.png"
    lval= [50, 100, 200, 300]#m_length_all[0,:][::4]
    lhdl = []
    for l in lval :
        s = np.exp(l / np.max(m_length_all) * visfactor ) #size_length[np.argmin(abs(l-m_length_all.flatten()))][0]
        h=mlines.Line2D([], [], color='grey', marker='o', linestyle='none', fillstyle='none',
                          markersize=s, label=f'{l}')    
        lhdl.append(h)
    ax.legend(handles=lhdl, title="correlation length [m]", loc="lower right")
    fig.savefig(str(fout), transparent=True)
    print(f"save {fout}")
    plt.close()
    
    
    ###misfit_model vs misfit_data
    x, y = m_misfit_d_all.flatten(), m_misfit_m_all.flatten()
    cmap_sigma = 'jet'#palettable.scientific.sequential.Batlow_20.mpl_colormap 
    length_min, length_max = np.min(m_sigma_all), np.max(m_sigma_all)
    range_val_sigma = np.linspace(length_min, length_max, 100)
    norm_sigma = colours.Normalize(vmin=length_min, vmax=length_max)
    try :
        color_scale_sigma =  cmap_sigma(norm_sigma(range_val_sigma))
    except :
        color_scale_sigma = plt.colormaps[cmap_sigma](norm_sigma(range_val_sigma))
        
    arg_col =  [np.argmin(abs(range_val_sigma-v))for v in m_sigma_all.flatten()]    
    color_marker = color_scale_sigma[arg_col]
    size_length = np.exp(m_length_all / np.max(m_length_all) *visfactor )
    fig, ax = plt.subplots(figsize=(12,8))
    vmin, vmax = np.nanmin(Znew), np.nanmax(Znew)    
    ax.scatter(x, y, s=size_length, c=color_marker, marker='o',)
    #ij = np.unravel_index(np.argmin(x-1, axis=None)[0], Y.shape) #np.argmin(abs(misfit_d[ix_rho0]-1), axis=1)
    #xbf, ybf = X[0, ij[1]], Y[ij[0],0] #best fit
    #print(xbf, ybf)
    #ax.scatter(xbf, ybf ,marker='*', s=40, color=color)
    ax.set_xlabel("misfit data $\\phi_{d}$")
    ax.set_ylabel("misfit model $\\phi_{m}$")
    ax.grid(True, which='both',linestyle='dotted', linewidth="0.3 ", color='grey')    
    ax.set_title(f"$\\rho_0$ = {rho0}"+ " g/cm$^{3}$")
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = fig.colorbar(ScalarMappable(norm=norm_sigma, cmap=cmap_sigma), cax=cax)
    cbar.set_label("prior error $\\sigma$ [g/cm$^3$]")#, size=14)
    fig.tight_layout()
    fout = out_dir/ dict_run[1]/ f"mf_data_vs_mf_model_{rho0}_res{res}m.png"
    lhdl = []
    for l in lval :
        s = np.exp(l / np.max(m_length_all) * visfactor ) #size_length[np.argmin(abs(l-m_length_all.flatten()))][0]
        h=mlines.Line2D([], [], color='grey', marker='o', linestyle='none', fillstyle='none',
                          markersize=s, label=f'{l}')    
        lhdl.append(h)
    ax.legend(handles=lhdl, title="correlation length [m]", loc="lower right")
   # ax.set_ylim(0,10)
    fig.savefig(str(fout), transparent=True)
    print(f"save {fout}")
    plt.close()
    
    
    
    ### sigma posterior vs || d_post - d_syn ||^2
    X = m_misfit_d_all #np.nanmean(m_std_post, axis=2)
    x, y = X.flatten(), np.nanmean(m_std_post, axis=2).flatten()
    cmap_sigma = 'jet'#palettable.scientific.sequential.Batlow_20.mpl_colormap 
    
    #reg. param
    lambda_1, lambda_2 = m_sigma_all.flatten(), m_length_all.flatten()
    
    length_min, length_max = np.min(m_sigma_all), np.max(m_sigma_all)
    range_val_sigma = np.linspace(length_min, length_max, 100)
    norm_sigma = colours.Normalize(vmin=length_min, vmax=length_max)
    try :
        color_scale_sigma =  cmap_sigma(norm_sigma(range_val_sigma))
    except :
        color_scale_sigma = plt.colormaps[cmap_sigma](norm_sigma(range_val_sigma))
        
    arg_col =  [np.argmin(abs(range_val_sigma-v))for v in m_sigma_all.flatten()]    
    color_marker = color_scale_sigma[arg_col]

    size_length = np.exp(m_length_all.flatten() / np.max(m_length_all.flatten()) *visfactor )
    
    fig, ax = plt.subplots(figsize=(12,8))
    vmin, vmax = np.nanmin(Znew), np.nanmax(Znew)
    #im = ax.pcolor(Xnew, Ynew,Znew, norm=colours.LogNorm(vmin, vmax), cmap=cmap)#, vmin=vmin, vmax=vmax) 
    
    degree=3
    m = (x<1.1) & (y < 0.4 )
    model_chi2min = np.poly1d(np.polyfit(x[m], y[m], degree))
    newx = np.linspace(np.min(x[m]), np.max(x[m]), 100)
    newy = model_chi2min(newx)



    ymin, ymax = np.min(newy), np.max(newy)
    x_ymin, x_ymax = newx[np.argmin(newy)], newx[np.argmax(newy)]
    #r2 = adjR(x,y, degree) #https://stats.stackexchange.com/questions/41138/calculating-adjusted-r2-in-polynomial-linear-regression-with-single-variable
    #ax.plot(newx, newy, color='magenta', linestyle='dashed', linewidth=2, marker=None, )#label=f'fit polynom(deg={degree}),'+' $R^2_{adj}$'+f'$={r2:.3f}$')
    

    from scipy.interpolate import UnivariateSpline

    # order = np.argsort(x)
    # y_spl = UnivariateSpline(x[order],y[order],s=None,k=4)
    # m = (x<1.2) & (y < 0.4 )
    # x_range = np.linspace(np.min(x[m]), np.max(x[m]), 100)
    # ax.plot(x_range, y_spl(x_range), color='red', linestyle='dashed', linewidth=2, marker=None, )#label=f'fit polynom(deg={degree}),'+' $R^2_{adj}$'+f'$={r2:.3f}$')
   
    #mder  = (0.95<x) & (x < 1.05)
    mder  = (0.9<x) & (x < 1.15)
    print(f"len(mask) = {len(mder[mder==True])}")
    dx=np.diff(x[mder],1)
    dy=np.diff(y[mder],1)
    yfirst=dy/dx
    ###And the corresponding values of x are :
    xfirst=0.5*(x[mder][:-1]+x[mder][1:])
    print(xfirst)
    print(f"len(xfirst) = {len(xfirst)}")
    
    ##For the second order, do the same process again :
    dxfirst=np.diff(xfirst,1)
    print(f"len(dxfirst) = {len(dxfirst)}")
    dyfirst=np.diff(yfirst,1)
    print(f"len(dyfirst) = {len(dyfirst)}")
    ysecond=dyfirst/dxfirst
    print(f"len(ysecond) = {len(ysecond)}")
    xsecond=0.5*(xfirst[:-1]+xfirst[1:])
    curvature = ysecond / (1+yfirst[:-1]**2)**3/2  #https://fr.wikipedia.org/wiki/Courbure_d%27un_arc
    #curvature = 2* ( (dxfirst[:-1]*ysecond) - (xsecond*yfirst[:-1]) ) / ( xfirst[:-1]**2 + yfirst[:-1]**2 )**(3/2)
    print(curvature)
    
    print(f"xfirst = {xfirst}")
    
     
    xinterp = xfirst[:-1]
    order = np.argsort(xinterp)
    xinterp = xinterp[order]
    curv_sort = curvature[order]
    #minterp = (0.90<xinterp) & (xinterp < 1.15)
    
    #print(f"xinterp = {xinterp}")
    
    xf = np.linspace(np.min(xinterp), np.max(xinterp), 100)
  
    func_curv = np.interp(xf,  xinterp, curv_sort)
    #func_lambda1 = np.interp(xnew,  xinterp, curvature)
    
    print(f"curvature = {curv_sort}")
    print(f"func_curv = {func_curv}")
    # fig, ax = plt.subplots()
    # ax.plot(xnew, func_curv)
    # plt.show()
    
    
    print(f"len(curvature) = {len(curvature)}")
    print(len(mder[mder==True]), len(curvature), len(x), len(m_sigma_all.flatten()) )
    
    #print("xsecond, ysecond = ",xsecond[np.argmax(curvature)], ysecond[np.argmax(curvature)], )
    ij = np.argmax(func_curv)
    
    #print(f"L-curve max curvature : phi_d, sigma_prior, lc = {x[ij]:.3f}, {m_sigma_all.flatten()[ij]} g/cm3, {m_length_all.flatten()[ij]} m")
    
    
    #m = (m_length_all.flatten() == 300)
    #size_length[~m] = 0
    sc = ax.scatter(x, y, s=size_length, c=color_marker, marker='o',)
    
    
    #ij = np.argmin(abs(x-1.0), axis=None)
    #xbf, ybf = X[0, ij[1]], ybf[ij[0],0] #best fit
    #print(f"ij={ij}")
    #print(f"L-curve misfit~1 : phi_d, sigma_prior, lc = {x[ij]:.3f}, {m_sigma_all.flatten()[ij]} g/cm3, {m_length_all.flatten()[ij]} m")
    
    #exit()
    #xbf, ybf = X[ij[1]], ybf[ij[0],0] #best fit
    #xbf, ybf = x[ij], model_chi2min(x[ij])
    xbf, ybf = xf[ij], func_curv[ij]
    print(xbf, ybf)
    #ax.scatter(xbf, ybf ,marker='*', s=300, color="floralwhite", edgecolor="black")
    ax.axvline(xbf, linestyle='dashed', color=color, linewidth=1.)
    #ax.axhline(ybf, linestyle='dashed', color=color, linewidth=1.)
    ax.set_xlabel("misfit data $\\phi_{d}$")
    ax.set_ylabel("posterior standard deviation $\\~\\sigma$")
    ax.set_ylim(0,0.4)
    ax.grid(True, which='both',linestyle='dotted', linewidth="0.3 ", color='grey')
    #ax.set_title(f"$\\rho_0$ = {rho0}"+ " g/cm$^{3}$")
    s =f"$\\rho_0$ = {rho0}"+ " g/cm$^{3}$"
    anchored_text = AnchoredText(s, loc="lower left", frameon=True,
                                 prop=dict(fontsize='x-large'))
    anchored_text.patch.set_boxstyle("round,pad=0")
    ax.add_artist(anchored_text)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = fig.colorbar(ScalarMappable(norm=norm_sigma, cmap=cmap_sigma), cax=cax)
    cbar.set_label("prior error $\\sigma$ [g/cm$^3$]")#, size=14)
    fig.tight_layout()
    fout = out_dir/ dict_run[1]/ f"std_post_vs_mf_data_{rho0}_res{res}m.png"


    handles, _ = sc.legend_elements(prop="sizes", num="auto")
   
    labels = np.sort(list(set(m_length_all.flatten())))
    handles = np.asarray(handles)[1::2]
    handles = handles.tolist()
    labels = [f"{l:.0f}" for l in labels[1::2].tolist()]
    ax.legend(handles=handles, labels=labels, title="correlation length $l_c$ [m]")#, num=5))
    #print(f"l={l}m, s={size_length[m_length_all.flatten()==300]}" )
    
    fig.savefig(str(fout), transparent=True)
    print(f"save {fout}")
    plt.close()
    
    