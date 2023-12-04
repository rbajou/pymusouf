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
#personal modules
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
class Configuration:
    '''
    not used
    '''
    name : str 
    panels : List[Panel] = field(default_factory=list)

class ConfigurationEnum(Enum):
    '''
    not used
    '''
    ThreePanel1 = Configuration(name="3p1")
    ThreePanel2 = Configuration(name="3p2")
    FourPanel = Configuration(name="4p")

@dataclass
class Telescope:
    name : str
    panels : List[Panel]
    PMTs : List[PMT] = field(default_factory=list)
    utm : np.ndarray = field(default_factory=lambda: np.ndarray(shape=(3,))) #coordinates (easting, northing, altitude)
    azimuth : float = field(default_factory=float)
    zenith : float = field(default_factory=float)
    elevation : float = field(default_factory=float)
    color : str = field(default_factory=lambda: "black")
    site : str = field(default_factory=lambda: "")


    def __str__(self):
        matrices = [p.matrix for p in self.panels]
        versions = [m.version for m in matrices]
        v = {f'v{v}': versions.count(v) for v in  set(versions)}
        return f"Telescope {self.name}: {len(self.panels)} matrices {v}"#type {','.join(set([m.version for m in matrices]))} "
    def __post_init__(self):

        z_front, z_rear =  self.panels[0].position.z, self.panels[-1].position.z
        object.__setattr__(self, 'length',  abs(z_front-z_rear) )
        if len(self.panels) == 3:
            object.__setattr__(self, 'configurations',  {'3p1':self.panels} )
        elif len(self.panels) == 4:
            object.__setattr__(self, 'configurations',  {'3p1':self.panels[:3], 
                                                         '3p2':self.panels[1:], 
                                                         '4p':self.panels} )
        else: raise ValueError("Unknown panel configuration...")

        object.__setattr__(self, 'rays', None)
        #object.__setattr__(self, 'pixel_xy', {conf: self.get_pixel_xy(pan[0],pan[-1]) for conf, pan in self.configurations.items() })
    
    def _get_ray_matrix(self, front_panel:Panel, rear_panel:Panel):
        """
        Ray paths (lines of sight) matrix referenced as (DX,DY) couples
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
        Compute telescope ray paths or line of sights
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
        for conf,panels in self.configurations.items():
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
            
            self.azimuthMatrix[conf] = tools.wrapToPi(azimuthMatrix.T).T #rad
            self.zenithMatrix[conf] = tools.wrapToPi(zenithMatrix.T).T #rad



    def get_pixel_xy(self,front_panel, rear_panel):
        """
        Position XY pixels
        """
        func = lambda xf,xr: xf-xr
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

tel_path  = Path(__file__).parents[1] / "files" / "telescopes"

##BR: BaronRouge GW Rocher Fendu 2015-2019 3 matrices = 1 * v1.1 + 2 * v2.0 (mapping might be wrong)
name = "BR"
ChMap32 = ChannelMap(file=str( tel_path / name / "channel_bar_map" / "mapping32x32.json"))
ChMap16 = ChannelMap(file=str( tel_path / name / "channel_bar_map" / "mapping16x16.json"))
ChMap_BR = [ChMap32, ChMap16, ChMap32]
matrix_BR = [matrixv2_0, matrixv1_1, matrixv2_0]
pos_BR = [ Position(PositionEnum.Front,0), Position(PositionEnum.Middle1, 600), Position(PositionEnum.Rear, 1200)]
panel_id_BR = [23, 24, 25] 
panels_BR = [ Panel(ID=panID, matrix=m, position=pos, channelmap=map) for (m, pos, panID, map) in zip(matrix_BR, pos_BR, panel_id_BR, ChMap_BR ) ]
pmt_BR = [ PMT(ID=int(pan.ID), panel=pan, channelmap=pan.channelmap) for pan in panels_BR ]
tel_BR= Telescope(name = name, panels=panels_BR, PMTs = pmt_BR, color="red")
tel_BR.utm = np.array([643345.81, 1774030.46,1267])
tel_BR.azimuth = 297.0#295.
tel_BR.zenith = 80.0#16.
tel_BR.elevation = round(90.0-tel_BR.zenith, 1) #16
tel_BR.site = "East"

#####COPAHUE 
name = "COP"
ChMap32 = ChannelMap(file=str( tel_path / name / "channel_bar_map" / "mapping32x32.json"))
ChMap16 = ChannelMap(file=str( tel_path / name / "channel_bar_map" / "mapping16x16.json"))
ChMap_COP = [ChMap32, ChMap16, ChMap32]
npanels_COP = 3
matrix_COP = [matrixv2_0, matrixv1_1, matrixv2_0]
pos_COP = [ Position(PositionEnum.Front,0), Position(PositionEnum.Middle1, 600), Position(PositionEnum.Rear, 1200)]
panel_id_COP = [0, 1, 2] 
panels_COP = [ Panel(ID=panID, matrix=m, position=pos, channelmap=map) for (m, pos, panID, map) in zip(matrix_COP, pos_COP, panel_id_COP, ChMap_COP ) ]
pmt_COP =  [ PMT(ID=int(pan.ID), panel=pan, channelmap=pan.channelmap) for pan in panels_COP ]
tel_COP= Telescope(name = name, panels=panels_COP, PMTs = pmt_COP)
tel_COP.utm = np.array([310722.14, 5808130.41,2542.98]) 
tel_COP.azimuth = 262.0#295
tel_COP.zenith = 86.0#16
tel_COP.elevation = round(90.0-tel_COP.zenith,1)#16
tel_COP.site = "Copahue"


#####OM: OrangeMecanique GW Fente du Nord 2017-2019 3 matrices = 1 * v1.1 + 2 * v2.0
name = "OM"
ChMap32 = ChannelMap(file=str( tel_path / name / "channel_bar_map" / "mapping32x32.json"))
ChMap16 = ChannelMap(file=str( tel_path / name / "channel_bar_map" / "mapping16x16.json"))
ChMap_OM = [ChMap32, ChMap16, ChMap32]
matrix_OM = [matrixv2_0, matrixv1_1, matrixv2_0]
pos_OM = [Position(PositionEnum.Front,0), Position(PositionEnum.Middle1, 600), Position(PositionEnum.Rear, 1200)]
panel_id_OM = [0, 1, 2]
panels_OM = list(Panel(ID=panID, matrix=m, position=pos, channelmap=map) for(m, pos, panID, map) in zip(matrix_OM, pos_OM, panel_id_OM, ChMap_OM ) )
pmt_OM = [ PMT(ID=int(pan.ID), panel=pan, channelmap=pan.channelmap) for pan in panels_OM ]
tel_OM= Telescope(name = name, panels=panels_OM, PMTs = pmt_OM, color="orange")
tel_OM.utm = np.array([642954.802937528, 1774560.94061667, 1344.6 + 1.5])
tel_OM.azimuth = 192.0
tel_OM.zenith = 76.1
tel_OM.elevation = round(90.-tel_OM.zenith,1)
tel_OM.site = "North"

#####SB: SacreBleu GW Savane-à-mulets (ouest NJ) 2017-2019 3 matrices = 1 * v1.1 + 2 * v2.0
name = "SB"
ChMap32 = ChannelMap(file=str( tel_path / name / "channel_bar_map" / "mapping32x32.json"))
ChMap16 = ChannelMap(file=str( tel_path / name / "channel_bar_map" / "mapping16x16.json"))
ChMap_SB = [ChMap32, ChMap16, ChMap32]
matrix_SB = [matrixv2_0, matrixv1_1, matrixv2_0]
pos_SB = [Position(PositionEnum.Front,0), Position(PositionEnum.Middle1, 600), Position(PositionEnum.Rear, 1200)]
panel_id_SB = [26, 27, 28]
panels_SB = list(Panel(ID=panID, matrix=m, position=pos, channelmap=map) for(m, pos, panID, map) in zip(matrix_SB, pos_SB, panel_id_SB, ChMap_SB ) )
pmt_SB = [ PMT(ID=int(pan.ID), panel=pan, channelmap=pan.channelmap) for pan in panels_SB ]
tel_SB= Telescope(name = name, panels=panels_SB, PMTs = pmt_SB, color="blue")
tel_SB.utm = np.array([642611.084416928, 1773797.5200942, 1185])
tel_SB.azimuth = 40.0#44.9
tel_SB.zenith = 79.0#75
tel_SB.elevation = round(90 - tel_SB.zenith ,1)
tel_SB.site = "South-West"




#####SBR: SuperBaronRouge GW Rocher Fendu 2021-2022 4 matrices = 4 * v1.1
name = "SBR"
channelmap = ChannelMap(file=str( tel_path / name / "channel_bar_map" / "mapping.json"))
npanels_SBR = 4
matrix_SBR = [matrixv1_1 for _ in range(npanels_SBR)]
pos_SBR = [ Position(PositionEnum.Front,0), Position(PositionEnum.Middle1, 600), Position(PositionEnum.Middle2, 1200), Position(PositionEnum.Rear, 1800) ]
panel_id_SBR = [0, 2, 1, 3]
panels_SBR = [ Panel(matrix=m, ID=panID, position=pos, channelmap=channelmap) for (m, pos, panID) in zip(matrix_SBR, pos_SBR, panel_id_SBR) ]
pmt_SBR =  [ PMT(ID=6, panel=[panels_SBR[0], panels_SBR[1]], channelmap=channelmap), PMT(ID=7, panel=[panels_SBR[2],panels_SBR[-1]], channelmap=channelmap) ]
tel_SBR= Telescope(name = name, panels=panels_SBR, PMTs = pmt_SBR, color="red")


####SNJ: SuperNainJaune GW Parking 2019
name = "SNJ"
channelmap = ChannelMap(file=str( tel_path / name / "channel_bar_map" / "mapping.json"))
npanels_SNJ=4
matrix_SNJ = [matrixv1_1 for _ in range(npanels_SNJ)]
pos_SNJ = [ Position(PositionEnum.Front,0), Position(PositionEnum.Middle1, 600), Position(PositionEnum.Middle2, 1200), Position(PositionEnum.Rear, 1800)]
panels_SNJ = [ Panel(matrix=m, ID=9+i, position=pos, channelmap=channelmap) for i, (m, pos) in enumerate(zip(matrix_SNJ, pos_SNJ)) ]
pmt_SNJ =  [ PMT(ID=i+9, panel=pan, channelmap=channelmap) for i, pan in enumerate(panels_SNJ) ]
tel_SNJ= Telescope(name = name, panels=panels_SNJ, PMTs=pmt_SNJ, color="gold")
tel_SNJ.utm = np.array([642782.001377887, 1773682.54931093, 1143])
tel_SNJ.azimuth = 18.0#20.5#20
tel_SNJ.zenith = 74.9#16
tel_SNJ.elevation = round(90-tel_SNJ.zenith, 1)
tel_SNJ.site = "South-West"

dict_tel = { 'SNJ': tel_SNJ, 'BR': tel_BR, 'OM': tel_OM, 'SB': tel_SB } #'SBR': tel_SBR, 'COP : tel_COP


##GV: GeantVert GW Matylis 2015-2019  (mapping might be wrong)
# matrix_GV = [matrixv1_1 for _ in range(3)]
# pos_GV = [PositionEnum.Front, PositionEnum.Middle1, PositionEnum.Rear]
# panels_GV = list(Panel(matrix=m, ID=9+i, position=pos, channelmap=ChMap16) for i, (m, pos) in enumerate(zip(matrix_GV, pos_GV)) )
# tel_GV = Telescope(name = "GeantVert", panels=panels_GV, spacing=600)
# channelmaps_GV= { p.ID :  p.channelmap.map() for p  in panels_GV}
# barwidths_GV = { p.ID :  int(m.scintillator.width) for p,m  in zip(panels_GV, matrix_GV)} #in mm
# zpos_GV = { p.ID : tel_GV.spacing * i for i, p in enumerate(panels_GV)}


##ND: NoirDesir
# matrix_ND = [matrixv2_0, matrixv1_1, matrixv2_0]
# pos_ND = [PositionEnum.Front, PositionEnum.Middle1, PositionEnum.Rear]
# panels_ND = list(Panel(matrix=m, ID=9+i, position=pos, channelmap=ChMap32) for i, (m, pos) in enumerate(zip(matrix_ND, pos_ND)) )
# tel_ND = Telescope(name = "NoirDesir", panels=panels_ND, spacing=600)
# channelmaps_ND= { p.ID :  p.channelmap.map() for p  in panels_ND}
# barwidths_ND = { p.ID :  int(m.scintillator.width) for p,m  in zip(panels_ND, matrix_ND)} #in mm
# zpos_ND = { p.ID : tel_ND.spacing * i for i, p in enumerate(panels_ND)}


def str2telescope(v):
    if isinstance(v, Telescope):
       return v
    #print(v)
    if v in list(dict_tel.keys()):
        return dict_tel[v]
    elif v in [f"tel_{k}" for k in list(dict_tel.keys()) ]:
        return dict_tel[v[4:]]
    elif v in [ k.lower() for k in list(dict_tel.keys())]:
        return dict_tel[v.upper()]
    elif v in [f"tel_{k.lower()}" for k in list(dict_tel.keys()) ]:
        return dict_tel[v[4:].upper()]
    else:
        raise argparse.ArgumentTypeError('Input telescope does not exist.')


if __name__ == '__main__':
    pass