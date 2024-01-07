#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredText
from scipy.optimize import curve_fit
import pandas as pd
import pylandau

#package module(s)
from telescope import Telescope, Panel
from survey.data import DataType
from utils import functions

params = {'legend.fontsize': 'large',
         'axes.labelsize': 'xx-large',
         'axes.titlesize':'xx-large',
         'xtick.labelsize':'xx-large',
         'ytick.labelsize':'xx-large',
         'axes.labelpad':10,
         'mathtext.fontset': 'stix', #latex style
         'font.family': 'STIXGeneral',  #latex style
         'axes.grid' : True,
         'grid.linestyle' : 'dotted',
         }
plt.rcParams.update(params)

class Charge:
    
    def __init__(self, df:pd.DataFrame, telescope:Telescope, input_type:DataType):
    
        self.df = df
        self.panels = telescope.panels
        self.binsDX, self.binsDY = 2*self.panels[0].matrix.nbarsX-1, 2*self.panels[0].matrix.nbarsY-1
        self.ADC_XY  = [] 

        self.fill_charge_arrays()

        self.input_type = input_type
        if self.input_type == DataType.real: 
            self.lpar_fit = ['MPV', 'eta', 'sigma', 'A']##parameters landauxgaussian distribution
        elif self.input_type == DataType.mc: 
            self.lpar_fit = ['MPV', 'eta', 'A'] ##parameters landau distribution
        else : raise ValueError()

        lmes = ['value', 'error']
        lpar = self.lpar_fit.copy()
        lpar.extend(['xmin', 'xmax', 'entries'])
        mix = pd.Index([pan.ID for pan in self.panels], name="panel_id")
        cols = pd.MultiIndex.from_tuples([(par, mes) for par in lpar for mes in lmes])
        self.df_fit = pd.DataFrame(columns=cols,index=mix)
        self.df_perc = pd.DataFrame(index=np.arange(1, 101), columns=[f"panel_{panel.ID}" for panel in self.panels])


    def fill_charge_arrays(self):
       
        sumADC_XY = self.df['ADC_X'] + self.df['ADC_Y'] 
        self.df = self.df.assign(ADC_SUM=pd.Series(sumADC_XY).values)        
        self.dict_ADC_XY = { pan.ID : np.array(self.df.loc[self.df['Z'] ==pan.position.z]["ADC_SUM"].values) for pan in self.panels}


    def fit_charge_distrib(self,  panel:Panel, q:np.ndarray=None, nbins:int=100, is_scaling:bool=False):    
       
        fscale= 1 
        if q is None : q = self.dict_ADC_XY[panel.ID]
        if is_scaling:   fscale = 1e3 ###needed to fit with Landau function from pylandau
        q = q*fscale
        xmax_fig = np.mean(q) + 5*np.std(q)
        xmax_fit = xmax_fig
        entries, bins = np.histogram(q,  range=(0,  xmax_fit), bins =  nbins)
        widths = np.diff(bins)
        bin_centers = np.array([ (bins[i+1]+bins[i])/2 for i in range(len(bins)-1)])
        N = len(bin_centers)     
        mean = sum( bin_centers * entries) / sum(entries)#(n*max(nentries))#sum(np.multiply(bin_centers, nentries)) / n
        sigma = np.sqrt( sum( entries*(bin_centers - mean)**2  )  /  ((N-1)/N*sum(entries))  )
        rough_max = np.max( bin_centers[bin_centers>0][entries.argmax()] )#bin_centers[np.where(entries==max(entries))] )
        #fitrange =  ( ( rough_max*0.2 < bin_centers ) & (bin_centers< 3*rough_max ) )
        fitrange =  ( ( rough_max*0.2 < bin_centers ) & (bin_centers < xmax_fit ) )
        yerr = np.array([np.sqrt(n) for n in entries[fitrange] ]) 
        yerr[entries[fitrange]<1] = 1
        xfit = bin_centers[fitrange]
        yfit = entries[fitrange]
        bin_w = np.diff(bin_centers[fitrange] )
        mpv, eta, amp = int(rough_max), sigma, np.max(entries)
        print(mpv, eta, sigma, amp)
    
        if self.input_type == DataType.mc : 
            values, pcov = curve_fit(pylandau.landau, xfit, yfit,
                sigma=yerr,
                absolute_sigma=False,
                p0=(mpv, eta, amp)
                )
            errors = np.sqrt(np.diag(pcov))
            values[0], values[1] =  values[0]/fscale, values[1]/fscale
            errors[0], errors[1] =  errors[0]/fscale, errors[1]/fscale
        else :  
            values, errors, m = functions.fit_landau_migrad(
                                            xfit,
                                            yfit,
                                            p0=[mpv, eta, sigma, amp],#
                                            limit_mpv=(rough_max*0.8,rough_max*1.2), #(10., 100.)
                                            limit_eta=(0.3*eta,1.5*eta), #(0.8*eta,1.2*eta)
                                            limit_sigma=(0.3*sigma,1.5*sigma), #(0.8*sigma,1.2*sigma)
                                            limit_A=(0.8*amp,1.2*amp) #(0.8*amp,1.2*amp)
                                            ) 
            values[0], values[1], values[2] =  values[0]/fscale, values[1]/fscale, values[2]/fscale
            errors[0], errors[1], errors[2] =  errors[0]/fscale, errors[1]/fscale, errors[2]/fscale 
     
        xfit, yfit = xfit/fscale, yfit/fscale

        xyrange = [[np.nanmin(xfit), np.nanmax(xfit)], [np.nanmin(yfit), np.nanmax(yfit)]]
        
        
        [xmin, xmax], [ymin, ymax] = xyrange
        sym_err = [np.max(np.abs(e)) for e in errors]
        
        for par, val, err in zip(self.lpar_fit, values, sym_err):
            self.df_fit.loc[panel.ID][par] = [np.around(val,3),np.around(err,3)] 
        
        self.df_fit.loc[panel.ID]['xmin'] = [np.around(xmin,3), 0] 
        self.df_fit.loc[panel.ID]['xmax'] = [np.around(xmax,3), 0]  
        return values, errors, xyrange


    def plot_charge_panel(self, ax,  panel:Panel, q:np.ndarray=None, nbins:int=100, fcal:float=None, ufcal:float=None, is_scaling:bool=False, do_fit:bool=False, **kwargs):
        
        if q is None : q = self.dict_ADC_XY[panel.ID]
        if fcal is None: fcal=1
        q = q/fcal
        xmax_fig = np.mean(q) + 5*np.std(q)
        ax.set_xlim(0, xmax_fig)
        entries, bins = np.histogram(q,  range=(0,  xmax_fig), bins =  nbins)
        widths = np.diff(bins)
        ax.bar(bins[:-1], entries, widths, **kwargs)
        ax.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
        ax.tick_params(axis='both')
        for per in range(1,101) : self.df_perc.loc[per][f"panel_{panel.ID}"] = np.around(np.percentile(q,per),3)
        ####if calibration constante C_ADC/MIPfraction is parsed (measured with 'golden' events)
        # ax.axvspan(1-unc_fcal[panel.ID]/fcal[panel.ID], 1+unc_fcal[panel.ID]/fcal[panel.ID], color='orange', alpha=0.2,
        #             label = "Gold evts peak")


    def plot_fit_panel(self, ax, panel:Panel, serie:pd.Series=None, x:np.ndarray=None, **kwargs):
        
        label=""
        values = []
        if serie is None: serie = self.df_fit.loc[panel.ID]
        for par in self.lpar_fit:
            (val, err) = serie[par]
            values.append(val)
            label+='{}={:0.1f}$\\pm${:0.1f}\n'.format(par, val, err)
        xmin, xmax = serie['xmin']['value'], serie['xmax']['value']
        if x is None: x = np.linspace(xmin, xmax, 100)
        if self.input_type == DataType.mc : 
            y = pylandau.landau(x, *values)
        else:
            y = pylandau.langau(x, *values)
        kwargs['label'] = label
        ax.plot(x, y, **kwargs)


    def scatter_plot_dQ(self, fig, gs, dQx:dict, dQy:dict, rangex:tuple=None, rangey:tuple=None, nbins:int=100) : 
       
        for i, (((keyx, colorx, do_fitx), valx), ((keyy, colory, do_fity),valy))  in enumerate( zip(dQx.items(), dQy.items() )) :

            ax = fig.add_subplot(gs[1, 0])
            ax_histx = fig.add_subplot(gs[0, 0], sharex=ax)
            #if fcal[panel.ID]!=1 : 
                #ax.set_xlabel('dQ [MIP fraction]', fontsize=fontsize) 
            atx = AnchoredText('dQ_front',
                        prop=dict(size=14), frameon=True,
                        loc='upper right',
                        )
            atx.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
            ax_histx.add_artist(atx) 
            ax_histy = fig.add_subplot(gs[1, 1], sharey=ax)
            aty = AnchoredText('dQ_rear',
                        prop=dict(size=14), frameon=True,
                        loc='upper right',
                        )
            aty.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
            ax_histy.add_artist(aty) 
            ax_histx.tick_params(axis="x", labelbottom=False)
            ax_histx.set_ylabel("entries", fontsize=10)
            ax_histx.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
            ax_histx.set_yscale('log')
            ax_histy.tick_params(axis="y", labelleft=False)
            ax_histy.set_xlabel("entries", fontsize=10)
            ax_histy.ticklabel_format(axis="x", style="sci", scilimits=(0,0))
            ax_histy.set_xscale('log')            
            if rangex is None and rangey is None : 
                xmax_2d = np.mean(valx) + 5*np.std(valx)
                ymax_2d = np.mean(valy) + 5*np.std(valy)
                rangex, rangey= (0,xmax_2d), (0,ymax_2d)
               
            entries_x, bins_x = np.histogram(valx,  range=rangex, bins =  nbins)#ax_histx.hist(valx, bins=nbins, range =rangex, color ='lightgreen', alpha=1., label='X', edgecolor='none')
            widths_x = np.diff(bins_x)
            ax_histx.bar(bins_x[:-1], entries_x, widths_x,color='None', edgecolor='lightgreen', label=f"X")
            ax_histx.set_xlim(rangex)
            
            entries_y, bins_y = np.histogram(valy,  range=rangey, bins =  nbins)
            widths_y = np.diff(bins_y)
            ax_histy.barh(bins_y[:-1], entries_y, widths_y,color='None', edgecolor='lightgreen', label=f"Y")
            ax_histy.set_ylim(rangey)
           
            h, xedges, yedges =  np.histogram2d(valx, valy, bins=nbins, range=[rangex, rangey])
            Z = np.ma.masked_where(h < 1, h).T
            xc, yc = xedges, yedges
            X, Y = np.meshgrid(xc, yc)
            im = ax.pcolormesh(X,  Y, Z, shading='auto', cmap='jet')
        return ax, h, xedges, yedges





'''
class RansacCharge(RansacData):
    """
    Class to checkout charge distributions per panel
    """
    def __init__(self, file:Union[str, List], telescope:Telescope, kwargs:dict={"index_col":0, "delimiter":'\t'}):
        RansacData.__init__(self, file, telescope, kwargs)
        self.panels = telescope.panels
        self.binsDX = 2*self.panels[0].matrix.nbarsX-1
        self.binsDY = 2*self.panels[0].matrix.nbarsY-1
        self.ADC_XY_inlier  = [] #sum(ADC_X+ADC_Y) 
        self.ADC_XY_outlier = []
        self.fill_charge_arrays()
'''
    
 


if __name__ == '__main__':
    pass