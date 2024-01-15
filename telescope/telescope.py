#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from dataclasses import dataclass, field
from abc import abstractmethod
from typing import List, Union
from enum import Enum, auto
import numpy as np 
from pathlib import Path
import argparse
import json
import matplotlib.axes 

#package module(s)
from config import SURVEY_DIR
from utils import tools


@dataclass(frozen=True)
class Scintillator:
    type: str 
    #dimensions in mm
    length: float 
    width: float
    thickness: float
    def __str__(self): return f"{self.type}"


@dataclass
class ChannelMap:
    file: str
    
    def map(self):
        """
        Read the .dat mapping file and fill a dictionary 'dict_ch_to_bar' :
        { keys = channel No (in [0;63]) : values = X or Y coordinate (e.g 'X01') }
        """
        ####convert ch to chmap
        with open(self.file, 'r') as fmap:
           self.dict_ch_to_bar = json.loads(fmap.read())
        self.dict_ch_to_bar = {int(k):v for k,v in self.dict_ch_to_bar.items()}
        self.dict_bar_to_ch = { v: k for k,v in self.dict_ch_to_bar.items()  }
        self.channels = list(self.dict_ch_to_bar.keys())
        self.bars = list(self.dict_ch_to_bar.values())
                
    def __post_init__(self):
        self.map()

    def __str__(self):
        return f'ChannelMap: {self.dict_ch_to_bar}'
    

@dataclass(frozen=True)
class Matrix:
    version : int
    scintillator : Scintillator
    nbarsX : int
    nbarsY : int  
    wls_type : str
    fiber_out : str
    def __str__(self):
        return f"v{self.version} with ({self.nbarsX}, {self.nbarsY}) {self.scintillator} scintillators"


class PositionEnum(Enum):
    Front = auto()
    Middle1 = auto()
    Middle2 = auto()
    Rear = auto()


class Position:
    def __init__(self, loc:PositionEnum, z:float):
        self.loc = loc.name 
        self.z = z  #in mm


@dataclass(frozen=True)
class Panel:
    matrix : Matrix 
    ID : int
    channelmap : ChannelMap 
    position: Position #Tuple[PositionEnum, float] 
    def __str__(self,):
        return self.position.loc


@dataclass(frozen=True)
class PMT:
    ID : int
    panel : List[Panel] 
    channelmap : ChannelMap
    type : str = field(default='MAPMT')


@dataclass
class PanelConfig:
   
    name : str
    panels : List[Panel]
    pmts : List[PMT] = field(default_factory=list)

    def __post_init__(self):

        z_front, z_rear =  self.panels[0].position.z, self.panels[-1].position.z
        object.__setattr__(self, 'length',  abs(z_front-z_rear) )
        # if len(self.panels) == 3:
        #     object.__setattr__(self, 'configurations',  {'3p1':self.panels} )
        # elif len(self.panels) == 4:
        #     object.__setattr__(self, 'configurations',  {'3p1':self.panels[:3], 
        #                                                  '3p2':self.panels[1:], 
        #                                                  '4p':self.panels} )
        # else: raise ValueError("Unknown panel configuration...")

        object.__setattr__(self, 'rays', None)
        #object.__setattr__(self, 'pixel_xy', {conf: self.get_pixel_xy(pan[0],pan[-1]) for conf, pan in self.configurations.items() }
    
    def __str__(self):
        matrices = [p.matrix for p in self.panels]
        versions = [m.version for m in matrices]
        v = {f'v{v}': versions.count(v) for v in  set(versions)}
        sout =f"Config {self.name}: {len(self.panels)} panels " + "-".join([p.position.loc for p in self.panels]) + "\n"
        return sout

@dataclass
class Telescope:
    
    name : str
    utm : np.ndarray = field(default_factory=lambda: np.ndarray(shape=(3,))) #coordinates (easting, northing, altitude)
    altitude : float =  field(default_factory=lambda: float)
    azimuth : float = field(default_factory=float)
    zenith : float = field(default_factory=float)
    elevation : float = field(default_factory=float)
    color : str = field(default_factory=lambda: "")
    site : str = field(default_factory=lambda: "")
    survey : str = field(default_factory=lambda: "")
    
    def __post_init__(self, ): 
        self.configurations = {}
        self.panels = List[Panel]
        self.rays = None
      
    def __setitem__(self, name:str, configuration:PanelConfig): 
        self.configurations[name] = configuration

    def __getitem__(self, name:str): 
        config = self.configurations[name]
        return config

    def __str__(self):
        
        sout = f"Telescope: {self.name}\n "
        if self.site : sout += f"- Site: {self.site}\n "
        if np.all(self.utm != None) :  sout += "- UTM (easting, northing, altitude): ("+ ', '.join([f'{i:.0f}' for i  in self.utm]) + ") m\n "
        if self.azimuth and self.elevation: sout += f"- Orientation (azimuth,elevation): ({self.azimuth}, {self.elevation}) deg\n "
        sout += "- Configurations:\n" 
        for _, conf in self.configurations.items(): sout += "\t" + conf.__str__() 
        return sout
    

    def get_ray_matrix(self, front_panel:Panel, rear_panel:Panel):
        """
        Ray paths referenced as (DX,DY) couples
        """
        nbarsXf, nbarsYf  = front_panel.matrix.nbarsX,front_panel.matrix.nbarsY
        nbarsXr, nbarsYr  = rear_panel.matrix.nbarsX,rear_panel.matrix.nbarsY
        barNoXf, barNoYf = np.arange(1, nbarsXf+1),np.arange(1, nbarsYf+1)
        barNoXr, barNoYr = np.arange(1, nbarsXr+1),np.arange(1, nbarsYr+1)
        DX_min, DX_max = np.min(barNoXf) - np.max(barNoXr) ,  np.max(barNoXf) - np.min(barNoXr) 
        DY_min, DY_max = np.min(barNoYf) - np.max(barNoYr) ,  np.max(barNoYf) - np.min(barNoYr) 
        mat_rays = np.mgrid[DX_min:DX_max+1:1, DY_min:DY_max+1:1].reshape(2,-1).T.reshape(2*nbarsXf-1,2*nbarsYf-1,2) 
        return mat_rays


    def get_ray_paths(self, front_panel:Panel, rear_panel:Panel, rmax:float=600,): 
        """
        Compute telescope ray paths (or line of sights)
        """
        front = front_panel
        rear = rear_panel
        L = (rear.position.z - front.position.z)  * 1e-3
        w = front.matrix.scintillator.length * 1e-3
        nx, ny = front.matrix.nbarsX, front.matrix.nbarsY
        step = w/nx
        #mat_rays = self._get_ray_matrix(front_panel=front, rear_panel=rear)

        barNoXf, barNoYf = np.arange(1, nx+1),np.arange(1, ny+1)
        barNoXr, barNoYr = np.arange(1, nx+1),np.arange(1, ny+1)
        DX_min, DX_max = np.min(barNoXf) - np.max(barNoXr) ,  np.max(barNoXf) - np.min(barNoXr) 
        DY_min, DY_max = np.min(barNoYf) - np.max(barNoYr) ,  np.max(barNoYf) - np.min(barNoYr) 
        arrDX, arrDY = np.arange(DX_min, DX_max+1), np.arange(DY_min, DY_max+1)

        nrays = (2*nx-1) * (2*ny-1)
        mat_pixel = np.zeros(shape=(nrays,3))

        delta_az, delta_incl = 0,0
        azimuth =  self.azimuth + delta_az
        elevation = self.elevation + delta_incl
        alfa, beta = (360 - azimuth)*np.pi/180, (90 + elevation)*np.pi/180 
        gamma = 0
        Rinv_alfa = np.array([[np.cos(alfa),-np.sin(alfa),0 ], [np.sin(alfa),np.cos(alfa),0], [0,0,1]]) 
        Rinv_beta = np.array([[1,0,0], [0,np.cos(beta),-np.sin(beta)], [0,np.sin(beta),np.cos(beta)]]) 
        Rinv_gamma = np.array([[np.cos(gamma),-np.sin(gamma),0], [np.sin(gamma),np.cos(gamma),0], [0,0,1]])
        Rinv = np.matmul(Rinv_alfa, np.matmul(Rinv_beta, Rinv_gamma))
        self.rays = np.zeros(shape=(nrays,2,3))
        k=0
        for dy in arrDY:
            ycoord = dy*step
            for dx in arrDX:
                xcoord = dx*step
                mat_pixel[k,:] = np.matmul(Rinv,np.array([xcoord, ycoord, -L]).T)  #rotation
                self.rays[k,:] = np.array([self.utm[:], self.utm[:] + mat_pixel[k,:] * rmax])
                k += 1
        self.rays = np.flipud(self.rays)

        
    def plot_ray_paths(self, ax:matplotlib.axes.Axes, front_panel:Panel, rear_panel:Panel, mask:np.ndarray=None, rmax:float=600, **kwargs):
        """
        Plot telescope ray paths (or line of sights) on given 'axis' (matplotlib.axes.Axes)
        INPUTS: 
        - color_map (np.ndarray) : RGBA array (n, 4)  
        - mask (np.ndarray) : bool array (n,)  
        """
        if self.rays is None : self.get_ray_paths(front_panel=front_panel, rear_panel=rear_panel, rmax=rmax)
        if mask is not None: self.rays[mask,:] = [np.ones(3)*np.nan, np.ones(3)*np.nan]
        for k in range(self.rays.shape[0]):      
            ax.plot(self.rays[k,:, 0], self.rays[k,:, 1], self.rays[k,:, 2], **kwargs)
        
    def plot_ray_values(self, ax:matplotlib.axes.Axes, color_values:np.ndarray, front_panel:Panel, rear_panel:Panel, mask:np.ndarray=None, rmax:float=600, **kwargs):
        """
        Plot array of values associated to n ray paths at the end of paths (from the telescope position)
        INPUTS: 
        - color_map (np.ndarray) : RGBA array (n, 4)  
        - mask (np.ndarray) : bool array (n,)  
        """
        if self.rays is None : self.get_ray_paths(front_panel=front_panel, rear_panel=rear_panel, rmax=rmax)
        if mask is not None: self.rays[mask,:] = [np.ones(3)*np.nan, np.ones(3)*np.nan]
        for k in range(self.rays.shape[0]):       
            ax.scatter(self.rays[k,-1, 0], self.rays[k,-1, 1], self.rays[k,-1, 2], c=color_values[k], **kwargs)    

    def compute_angle_matrix(self):
        
        self.azimuthMatrix, self.zenithMatrix = {}, {}

        for _,conf in self.configurations.items():
            panels = conf.panels
            nx = panels[0].matrix.nbarsX
            wx = panels[0].matrix.scintillator.width
            z_front, z_rear =  panels[0].position.z, panels[-1].position.z
            L = abs(z_front-z_rear)
            azimuthMatrix = np.zeros(shape=(nx*2-1,nx*2-1)) # different observation axis (in rad)
            zenithMatrix = np.zeros(shape=(nx*2-1,nx*2-1)) # different observation axis (in rad)
            rhoMatrix = np.zeros(shape=(nx*2-1,nx*2-1))
            MrotZ  = np.array([[np.cos(self.azimuth*np.pi/180), -np.sin(self.azimuth*np.pi/180), 0],
                        [np.sin(self.azimuth*np.pi/180), np.cos(self.azimuth*np.pi/180), 0], 
                        [0                 ,     0           ,             1 ]])
            MrotY  = np.array([[np.cos((-90+self.zenith)*np.pi/180), 0, np.sin((-90+self.zenith)*np.pi/180)],
                        [0            ,            1, 0 ],                    
                        [-np.sin((-90+self.zenith)*np.pi/180), 0, np.cos((-90+self.zenith)*np.pi/180) ]])
            for deltaX in range(-(nx-1),(nx-1)+1): 
                for deltaY  in range(-(nx-1), (nx-1)+1) : 
                    #print(deltaX,deltaY)
                    cartTelescope = np.array([L, deltaY*wx, deltaX*wx])
                    cartTelescope = MrotZ @ MrotY @ np.transpose(cartTelescope)
                    rho = np.sqrt(cartTelescope[0]**2 + cartTelescope[1]**2 + cartTelescope[-1]**2)
                    phi = np.arccos(cartTelescope[-1]/rho)
                    
                    if (cartTelescope[1] >= 0) :
                        theta = np.arccos(cartTelescope[0]/np.sqrt(cartTelescope[0]**2 + cartTelescope[1]**2))
                    
                    if (cartTelescope[1] < 0) :
                        theta = 2*np.pi - np.arccos(cartTelescope[0]/np.sqrt(cartTelescope[0]**2 + cartTelescope[1]**2))
                    azimuthMatrix[nx-deltaX-1,nx-deltaY-1] = theta
                    zenithMatrix[nx-deltaX-1,nx-deltaY-1] = phi
                    rhoMatrix[deltaX+nx-1,deltaY+nx-1] = rho
            
            self.azimuthMatrix[conf.name] = tools.wrapToPi(azimuthMatrix.T).T #rad
            self.zenithMatrix[conf.name] = tools.wrapToPi(zenithMatrix.T).T #rad


    def get_pixel_xy(self):
        """
        Position XY pixels
        """
        func = lambda xf,xr: xf-xr
        front_panel, rear_panel = self.panels[0], self.panels[-1]
        nbarsXf, nbarsYf  = front_panel.matrix.nbarsX,front_panel.matrix.nbarsY
        nbarsXr, nbarsYr  = rear_panel.matrix.nbarsX,rear_panel.matrix.nbarsY
        barNoXf, barNoYf = np.arange(1, nbarsXf+1),np.arange(1, nbarsYf+1)
        barNoXr, barNoYr = np.arange(1, nbarsXr+1),np.arange(1, nbarsYr+1)
        DX_min, DX_max = np.min(barNoXf) - np.max(barNoXr) ,  np.max(barNoXf) - np.min(barNoXr) 
        DY_min, DY_max = np.min(barNoYf) - np.max(barNoYr) ,  np.max(barNoYf) - np.min(barNoYr) 
        res_dx = np.tile(np.mgrid[DX_min:DX_max+1:1],  (2*nbarsXf-1, 1)).T
        res_dy = np.tile(np.mgrid[DY_min:DY_max+1:1],  (2*nbarsYf-1, 1))
        mat_dx, mat_dy = np.zeros(shape=(res_dx.shape[0],res_dx.shape[1],2)), np.zeros(shape=(res_dy.shape[0],res_dy.shape[1],2))
        for i in range(1,nbarsXf+1): 
            for j in np.flip(range(1,nbarsXf+1)): 
                mat_dx[res_dx==func(i,j),:] = [i,j]
                mat_dy[res_dy==func(i,j),:] = [i,j]
        mat = np.concatenate((mat_dx, mat_dy), axis=2)
        return mat
    

    def plot3D(self, fig, ax, position:np.ndarray=np.zeros(3)):
        '''
        Input:
        - 'ax' (plt.Axes3D) : e.g 'ax = fig.add_subplot(111, projection='3d')'
        - 'position' (np.ndarray)
        '''
        zticks=[]
        for p in self.panels:
            w  = float(p.matrix.scintillator.width)
            nbarsX = int(p.matrix.nbarsX)
            nbarsY = int(p.matrix.nbarsY)
            sx = w*nbarsX 
            sy = w*nbarsY
            x, y = np.linspace(position[0], position[0]+sx , nbarsX+1 ), np.linspace(position[0], position[0]+sy , nbarsX+1 )
            X, Y = np.meshgrid(x, y)
            zpos = position[2]
            Z = np.ones(shape=X.shape)*(zpos + p.position.z)
            ax.text(0, 0, zpos + p.position.z, p.position.loc, 'y', alpha=0.5, color='grey')#, rotation_mode='default')
            ax.plot_surface(X,Y,Z, alpha=0.2, color='greenyellow', edgecolor='turquoise' )
            zticks.append(Z[0,0])
        ###shield panel
        X = np.linspace(position[0], position[0]+sx ,2 )
        Y = np.linspace(position[0], position[0]+sy ,2 )
        X, Y = np.meshgrid(X, Y)
        zshield = self.panels[-1].position.z/2
        Z = np.ones(shape=X.shape)*(zpos + zshield)
        ax.plot_surface(X,Y,Z, alpha=0.1, color='none', edgecolor='tomato' )
        ax.text(0, 0, zshield, "Shielding", 'y', alpha=0.5, color='grey')
        panel_side=float(self.panels[0].matrix.nbarsX)*float(self.panels[0].matrix.scintillator.width)
        ax.set_xlabel("X [mm]")
        ax.set_ylabel("Y [mm]")
        ax.set_xticks(np.linspace(0, panel_side, 3))
        ax.set_yticks(np.linspace(0, panel_side, 3))
        ax.w_xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
        ax.invert_zaxis()
        ax.set_zticks(zticks)
        ax.set_zticklabels([])
        ax = fig.add_axes(tools.MyAxes3D(ax, 'l'))
        ax.set_zticks(zticks)
        ax.set_zticklabels([])
        ax.annotate("Z", xy=(0.5, .5), xycoords='axes fraction', xytext=(0.04, .78),)
        
        return ax

scint_Fermi = Scintillator(type="Fermilab", length=800, width=50, thickness=7 )
scint_JINR = Scintillator(type="JINR", length=800, width=25, thickness=7 )
matrixv1_1 = Matrix(version="1.1",  scintillator=scint_Fermi, nbarsX=16, nbarsY= 16, wls_type="BCF91A",fiber_out="TR644 POM")
matrixv2_0 = Matrix(version="2.0",  scintillator=scint_JINR, nbarsX=32, nbarsY= 32, wls_type="Y11",fiber_out="TR644 POM")

survey_path = SURVEY_DIR
souf_tel_path = survey_path / "soufriere" / "telescope"
cop_tel_path  = survey_path / "copahue" / "telescope"

##BR: BaronRouge GW Rocher Fendu 2015-2019 3 matrices = 1 * v1.1 + 2 * v2.0 (mapping might be wrong)
tel_name = 'BR'
tel_BR = Telescope(name=tel_name)

tel_BR.utm = np.array([643345.81, 1774030.46,1267])
tel_BR.altitude = tel_BR.utm[-1]
tel_BR.azimuth = 297.0#295.
tel_BR.zenith = 80.0#16.
tel_BR.elevation = round(90.0-tel_BR.zenith, 1) #16
tel_BR.site = "Rocher Fendu - Soufrière"
tel_BR.color = "red"

chmap32 = ChannelMap(file=str( souf_tel_path / tel_name / "channel_bar_map" / "mapping32x32.json"))
chmap16 = ChannelMap(file=str( souf_tel_path / tel_name / "channel_bar_map" / "mapping16x16.json"))
front_panel = Panel(matrix = matrixv2_0, ID=0, position=Position(PositionEnum.Front,0), channelmap=chmap32)
front_pmt = PMT(ID=0, panel=front_panel, channelmap=chmap32) 
middle1_panel = Panel(matrix = matrixv1_1, ID=1, position=Position(PositionEnum.Middle1,600), channelmap=chmap16)
middle1_pmt = PMT(ID=1, panel=middle1_panel, channelmap=chmap16) 
rear_panel = Panel(matrix = matrixv2_0, ID=2, position=Position(PositionEnum.Rear,1200), channelmap=chmap32)
rear_pmt = PMT(ID=2, panel=rear_panel, channelmap=chmap32) 
conf_name = '3p1'
Config_3p1_32x32 = PanelConfig(name = conf_name, 
                               panels=[front_panel, middle1_panel, rear_panel],
                               pmts=[front_pmt, middle1_pmt, rear_pmt])
tel_BR[conf_name] = Config_3p1_32x32
tel_BR.panels = Config_3p1_32x32.panels

#####COPAHUE 
tel_name = 'COP'
tel_COP = Telescope(name=tel_name)

tel_COP.utm = np.array([310722.14, 5808130.41, 2543.]) 
tel_COP.altitude = tel_COP.utm[-1]
tel_COP.azimuth = 262.0#295
tel_COP.zenith = 86.0#16
tel_COP.elevation = round(90.0-tel_COP.zenith,1)#16
tel_COP.site = "Copahue"
tel_COP.color = "grey"

chmap32 = ChannelMap(file=str( cop_tel_path / tel_name / "channel_bar_map" / "mapping32x32.json"))
chmap16 = ChannelMap(file=str( cop_tel_path / tel_name / "channel_bar_map" / "mapping16x16.json"))
front_panel = Panel(matrix = matrixv2_0, ID=0, position=Position(PositionEnum.Front,0), channelmap=chmap32)
front_pmt = PMT(ID=0, panel=front_panel, channelmap=chmap32) 
middle1_panel = Panel(matrix = matrixv1_1, ID=1, position=Position(PositionEnum.Middle1,600), channelmap=chmap16)
middle1_pmt = PMT(ID=1, panel=middle1_panel, channelmap=chmap16) 
rear_panel = Panel(matrix = matrixv2_0, ID=2, position=Position(PositionEnum.Rear,1200), channelmap=chmap32)
rear_pmt = PMT(ID=2, panel=rear_panel, channelmap=chmap32) 
conf_name = '3p1'
Config_3p1_32x32 = PanelConfig(name = conf_name, 
                               panels=[front_panel, middle1_panel, rear_panel],
                               pmts=[front_pmt, middle1_pmt, rear_pmt])
tel_COP[conf_name] = Config_3p1_32x32
tel_COP.panels = Config_3p1_32x32.panels

#####OM: OrangeMecanique GW Fente du Nord 2017-2019 3 matrices = 1 * v1.1 + 2 * v2.0
tel_name = 'OM'
tel_OM = Telescope(name=tel_name)

tel_OM.utm = np.array([642954.802937528, 1774560.94061667, 1344.6 + 1.5])
tel_OM.altitude = tel_OM.utm[-1]
tel_OM.azimuth = 192.0
tel_OM.zenith = 76.1
tel_OM.elevation = round(90.-tel_OM.zenith,1)
tel_OM.site = "Fente du Nord - Soufrière"
tel_OM.color = "orange"

chmap32 = ChannelMap(file=str( souf_tel_path / tel_name / "channel_bar_map" / "mapping32x32.json"))
chmap16 = ChannelMap(file=str( souf_tel_path / tel_name / "channel_bar_map" / "mapping16x16.json"))
front_panel = Panel(matrix = matrixv2_0, ID=0, position=Position(PositionEnum.Front,0), channelmap=chmap32)
front_pmt = PMT(ID=0, panel=front_panel, channelmap=chmap32) 
middle1_panel = Panel(matrix = matrixv1_1, ID=1, position=Position(PositionEnum.Middle1,600), channelmap=chmap16)
middle1_pmt = PMT(ID=1, panel=middle1_panel, channelmap=chmap16) 
rear_panel = Panel(matrix = matrixv2_0, ID=2, position=Position(PositionEnum.Rear,1200), channelmap=chmap32)
rear_pmt = PMT(ID=2, panel=rear_panel, channelmap=chmap32) 
conf_name = '3p1'
Config_3p1_32x32 = PanelConfig(name = conf_name, 
                               panels=[front_panel, middle1_panel, rear_panel],
                               pmts=[front_pmt, middle1_pmt, rear_pmt])
tel_OM[conf_name] = Config_3p1_32x32
tel_OM.panels = Config_3p1_32x32.panels

#####SB: SacreBleu GW Savane-à-mulets (ouest NJ) 2017-2019 3 matrices = 1 * v1.1 + 2 * v2.0
tel_name = 'SB'
tel_SB = Telescope(name=tel_name)

tel_SB.utm = np.array([642611.084416928, 1773797.5200942, 1185.])
tel_SB.altitude = tel_SB.utm[-1]
tel_SB.azimuth = 40.0#44.9
tel_SB.zenith = 79.0#75
tel_SB.elevation = round(90 - tel_SB.zenith ,1)
tel_SB.site = "Savane-à-mulets - Soufrière"
tel_SB.color = "blue"

chmap32 = ChannelMap(file=str( souf_tel_path / tel_name / "channel_bar_map" / "mapping32x32.json"))
chmap16 = ChannelMap(file=str( souf_tel_path / tel_name / "channel_bar_map" / "mapping16x16.json"))
front_panel = Panel(matrix = matrixv2_0, ID=26, position=Position(PositionEnum.Front,0), channelmap=chmap32)
front_pmt = PMT(ID=0, panel=front_panel, channelmap=chmap32) 
middle1_panel = Panel(matrix = matrixv1_1, ID=27, position=Position(PositionEnum.Middle1,600), channelmap=chmap16)
middle1_pmt = PMT(ID=1, panel=middle1_panel, channelmap=chmap16) 
rear_panel = Panel(matrix = matrixv2_0, ID=28, position=Position(PositionEnum.Rear,1200), channelmap=chmap32)
rear_pmt = PMT(ID=2, panel=rear_panel, channelmap=chmap32) 
conf_name = '3p1'
Config_3p1_32x32 = PanelConfig(name = conf_name, 
                               panels=[front_panel, middle1_panel, rear_panel],
                               pmts=[front_pmt, middle1_pmt, rear_pmt])
tel_SB[conf_name] = Config_3p1_32x32
tel_SB.panels = Config_3p1_32x32.panels

####SNJ: SuperNainJaune GW Parking 2019
tel_name = 'SNJ'
tel_SNJ = Telescope(name=tel_name)

tel_SNJ.utm = np.array([642782.001377887, 1773682.54931093, 1143.])
tel_SNJ.altitude = tel_SNJ.utm[-1]
tel_SNJ.azimuth = 18.0#20.5#20
tel_SNJ.zenith = 74.9#16
tel_SNJ.elevation = round(90-tel_SNJ.zenith, 1)
tel_SNJ.site = "Parking - Soufrière"
tel_SNJ.color = "yellow"

channelmap = ChannelMap(file=str( souf_tel_path / tel_name / "channel_bar_map" / "mapping.json"))
front_panel = Panel(matrix = matrixv1_1, ID=9, position=Position(PositionEnum.Front,0), channelmap=channelmap)
front_pmt = PMT(ID=9, panel=front_panel, channelmap=channelmap) 
middle1_panel = Panel(matrix = matrixv1_1, ID=10, position=Position(PositionEnum.Middle1,600), channelmap=channelmap)
middle1_pmt = PMT(ID=10, panel=middle1_panel, channelmap=channelmap) 
middle2_panel = Panel(matrix = matrixv1_1, ID=11, position=Position(PositionEnum.Middle2,1200), channelmap=channelmap)
middle2_pmt = PMT(ID=11, panel=middle2_panel, channelmap=channelmap) 
rear_panel = Panel(matrix = matrixv1_1, ID=12, position=Position(PositionEnum.Rear,1800), channelmap=channelmap)
rear_pmt = PMT(ID=12, panel=rear_panel, channelmap=channelmap) 
conf_name = '3p1'
Config_3p1_16x16 = PanelConfig(name = conf_name, 
                               panels=[front_panel, middle1_panel, middle2_panel],
                               pmts=[front_pmt, middle1_pmt, middle2_pmt])
tel_SNJ[conf_name] = Config_3p1_16x16
conf_name = '3p2'
Config_3p2_16x16 = PanelConfig(name = conf_name,
                               panels=[middle1_panel, middle2_panel, rear_panel],
                               pmts=[middle1_pmt, middle2_pmt, rear_pmt])
tel_SNJ[conf_name] = Config_3p2_16x16
conf_name = '4p'
Config_4p_16x16 = PanelConfig(name = conf_name, 
                              panels=[front_panel, middle1_panel, middle2_panel, rear_panel],
                              pmts=[front_pmt, middle1_pmt, middle2_pmt, rear_pmt])
tel_SNJ[conf_name] = Config_4p_16x16
tel_SNJ.panels = Config_4p_16x16.panels

##SBR: SuperBaronRouge GW Rocher Fendu 2021-2022 4 matrices = 4 * v1.1
# tel_name = 'SBR'
# tel_SBR = Telescope(name=tel_name)

# tel_SBR.utm = np.array([643345.81, 1774030.46,1267])
# tel_SBR.altitude = tel_BR.utm[-1]
# tel_SBR.azimuth = 297.0 #?
# tel_SBR.zenith = 80.0 #?
# tel_SBR.elevation = round(90.0-tel_BR.zenith, 1) #16
# tel_SBR.site = "Rocher Fendu - Soufrière"

# channelmap = ChannelMap(file=str( souf_tel_path / tel_name / "channel_bar_map" / "mapping.json"))

# front_panel = Panel(matrix = matrixv1_1, ID=0, position=Position(PositionEnum.Front,0), channelmap=channelmap)
# middle1_panel = Panel(matrix = matrixv1_1, ID=2, position=Position(PositionEnum.Middle1,600), channelmap=channelmap)
# front_middle1_pmt = PMT(ID=6, panel=[front_panel, middle1_panel], channelmap=channelmap)

# middle2_panel = Panel(matrix = matrixv1_1, ID=1, position=Position(PositionEnum.Middle2,1200), channelmap=channelmap)
# rear_panel = Panel(matrix = matrixv1_1, ID=3, position=Position(PositionEnum.Rear,1800), channelmap=channelmap)
# middle2_rear_pmt = PMT(ID=7, panel=[middle2_panel, rear_panel], channelmap=channelmap) 
# conf_name = '3p1'

# Config_3p1_16x16 = PanelConfig(name = conf_name, 
#                                panels=[front_panel, middle1_panel, middle2_panel],
#                                pmts=[front_pmt, middle1_pmt, middle2_pmt])
# tel_SBR[conf_name] = Config_3p1_16x16
# conf_name = '3p2'
# Config_3p2_16x16 = PanelConfig(name = conf_name,
#                                panels=[middle1_panel, middle2_panel, rear_panel],
#                                pmts=[middle1_pmt, middle2_pmt, rear_pmt])
# tel_SBR[conf_name] = Config_3p2_16x16
# conf_name = '4p'
# Config_4p_16x16 = PanelConfig(name = conf_name, 
#                               panels=[front_panel, middle1_panel, middle2_panel, rear_panel],
#                               pmts=[front_pmt, middle1_pmt, middle2_pmt, rear_pmt])
# tel_SBR[conf_name] = Config_4p_16x16


DICT_TEL = { 'SNJ': tel_SNJ, 'BR': tel_BR, 'OM': tel_OM, 'SB': tel_SB, 'COP' : tel_COP } #'SBR': tel_SBR, 


def str2telescope(v):
   
    if isinstance(v, Telescope):
       return v

    if v in list(DICT_TEL.keys()):
        return DICT_TEL[v]
    elif v in [f"tel_{k}" for k in list(DICT_TEL.keys()) ]:
        return DICT_TEL[v[4:]]
    elif v in [ k.lower() for k in list(DICT_TEL.keys())]:
        return DICT_TEL[v.upper()]
    elif v in [f"tel_{k.lower()}" for k in list(DICT_TEL.keys()) ]:
        return DICT_TEL[v[4:].upper()]
    else:
        raise argparse.ArgumentTypeError('Input telescope does not exist.')


if __name__ == '__main__':
    print(tel_SNJ)



