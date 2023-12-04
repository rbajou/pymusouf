# -*- coding: utf-8 -*-
#!/usr/bin/env python3

import numpy as np
import os
from pathlib import Path
import scipy.io as sio 
from scipy.interpolate import griddata, RegularGridInterpolator
from scipy import optimize 
import sys
import time
from dataclasses import dataclass, field
from abc import abstractmethod

#personal modules
from telescope import Telescope, dict_tel
    
class AcqVars : 
    '''
    Read former 'acqVars.mat' matlab files
    '''
    def __init__(self, telescope:Telescope, dir:str, mat_files:dict=None, tomo:bool=False):
        self.tel = telescope
        self.dir = dir
        if not self.dir.exists(): raise ValueError("Input directory not found.")
        barwidths = { p.ID :  float(p.matrix.scintillator.width) for p  in self.tel.panels }
        self.sconfig= telescope.configurations
        self.thickness = {}
        self.topography = {} 
        self.interception_in = {} 
        self.interception_out = {} 
        if tomo: 
            acqVars_3p_file = Path(self.dir) / 'acqVars_3p.mat'
            acqVars_4p_file = Path(self.dir) / 'acqVars_4p.mat'
            if mat_files is not None: 
                if "3p" in mat_files: 
                    acqVars_3p_file = str(self.dir / mat_files["3p"]) 
                if "4p" in mat_files:  
                    acqVars_4p_file = str(self.dir / mat_files["4p"]) 
            
        self.az_os,  self.ze_os,  self.AZ_OS_MESH, self.ZE_OS_MESH={},{},{},{}
        self.az_tomo,  self.ze_tomo,  self.AZ_TOMO_MESH, self.ZE_TOMO_MESH={},{},{},{}
        for conf, l_panel in self.sconfig.items():
            first_panel = l_panel[0]
            last_panel = l_panel[-1]
            nbars = int(first_panel.matrix.nbarsX)
            panel_side = barwidths[first_panel.ID] * nbars
            #length= abs(first_panel.position[1] - last_panel.position[1] )
            length= abs(first_panel.position.z - last_panel.position.z )
            xlos = 2*first_panel.matrix.nbarsX-1
            ylos = 2*first_panel.matrix.nbarsY-1
            #OPEN-SKY angles values
            self.az_os[conf] = np.linspace(-180, 180, xlos)
            ze_max = np.around(np.arctan(panel_side/length)*180/np.pi, 1)
            self.ze_os[conf] = np.linspace(-ze_max, ze_max, ylos)
            self.AZ_OS_MESH[conf], self.ZE_OS_MESH[conf] = np.meshgrid(self.az_os[conf], self.ze_os[conf])
        
            if not tomo: continue 
            #TOMO angles values   
            if conf.startswith("3p"): 
                acqVars_mat = sio.loadmat(acqVars_3p_file) 
            
            elif conf.startswith("4p"): 
                acqVars_mat = sio.loadmat(acqVars_4p_file) 
            else : raise ValueError("Error in 'AcqVars'. Wrong configuration name.")
           
            az_tomo, ze_tomo = acqVars_mat['azimutAngleMatrix']*180/np.pi, acqVars_mat['zenithAngleMatrix']*180/np.pi
            if "apparentThickness" in acqVars_mat:
                self.thickness[conf] = acqVars_mat["apparentThickness"] #meter
            if "topography" in acqVars_mat:
                self.topography[conf] = acqVars_mat["topography"]* 180/np.pi
            if 'interceptionInMatrix' in acqVars_mat:
                self.interception_in[conf] = acqVars_mat['interceptionInMatrix']
            if 'interceptionOutMatrix' in acqVars_mat:
                self.interception_out[conf] = acqVars_mat['interceptionOutMatrix']
            
            self.az_tomo[conf] = az_tomo
            self.ze_tomo[conf] = ze_tomo
            self.AZ_TOMO_MESH[conf], self.ZE_TOMO_MESH[conf] = np.meshgrid(self.az_tomo[conf],  self.ze_tomo[conf])
       