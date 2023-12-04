#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from dataclasses import dataclass, field
from abc import abstractmethod
from typing import List, Dict, Union
from enum import Enum, auto
import numpy as np
import os
from pathlib import Path
import inspect
filename = inspect.getframeinfo(inspect.currentframe()).filename
script_path = os.path.dirname(os.path.abspath(filename))
from scipy.signal import find_peaks
from scipy.optimize import curve_fit
from dataclasses import dataclass, field
import gzip
from skimage.measure import ransac, LineModelND
from sklearn.linear_model import RANSACRegressor,LinearRegression
import pandas as pd
import logging
import glob
import requests #url
from bs4 import BeautifulSoup #fetch data 
from itertools import combinations

#personal modules
from telescope import dict_tel, Telescope, ChannelMap

class InputType(Enum):
    REAL = auto()
    DATA = auto()
    SIM = auto()
    MC = auto()
    PRIMARY = auto()

class Dataset:
    @abstractmethod
    def dataset(self, dataset):
        pass

class Data:
    def __init__(self, telescope:Telescope, input:str, type:InputType, label:str, url:str=None, save_path:str=None, max_nfiles:int=None):
        self.telescope = telescope
        self.input = input
        self.type = type
        self.max_nfiles = max_nfiles
        self.label = label
        self.url = url #data files location
        self.save_path = save_path
        self.dataset = Dataset
        self.basename = str
        
    def listFilesUrl(self):
        """
        @params:
            url (e.g 'https://cours.ip2i.in2p3.fr/marteau/muography/ZENITH/IFPEN/')
        List files url to be fetched
        """
        page = requests.get(self.url).text
        soup = BeautifulSoup(page, 'html.parser')
        listUrl = [self.url + '/' + node.get('href') for node in soup.find_all('a') if node.get('href').endswith('')]
        return listUrl


 
    def fetchFiles(self, nfiles):
        '''
        Fetch datafiles online at 'url'
        '''
        Path(self.save_path).mkdir(parents=True, exist_ok=True)   
         #with extension '.dat.gz'
        for i, url in enumerate(self.listFilesUrl()):
            if i > nfiles: break 
            file_basename = os.path.basename(url)
            if not os.path.isfile(os.path.join(self.save_path, "", file_basename)):
            #check if already existing file 
                file = requests.get(url)
                with open(os.path.join(self.save_path, "" ,file_basename), 'wb') as f:
                    f.write(file.content)
                f.close()
        else : pass 
   
    def outname(self):
        self.basename  = os.path.basename(self.dataset[0])   
        if self.basename.endswith(".dat.gz"): self.basename =  self.basename[:-7]
        elif self.basename.endswith(".dat"): self.basename =  self.basename[:-4]
        
    def builddataset(self):
       
        if self.url is not None:
            self.input = self.save_path
            nfiles = len(os.listdir(self.input))
            if self.max_nfiles is not None: nfiles = self.max_nfiles 
            self.dataset = [f for n, f in enumerate(glob.glob(os.path.join(self.input , '', '*.dat*'))) if n < nfiles ]
        if type(self.input) is list:
            #data files must be in same directory (i.e belonged to the same run)
            self.dataset = self.input
        elif os.path.isfile(self.input):
            self.dataset = [self.input]
        elif os.path.isdir(self.input): #whole data directory
            if self.input.endswith("/"): self.input=self.input[:-1]
            nfiles = len(os.listdir(self.input))
            if self.max_nfiles is not None: nfiles = self.max_nfiles 
            self.dataset = [f for n, f in enumerate(glob.glob(os.path.join(self.input , '', '*.dat*'))) if n < nfiles ]
        else: raise Exception("Problem with input data...")
        self.outname()
        
    
    def readfile(self, file):
        datalines = list()
        try : 
            if file.endswith('.gz') : 
                with gzip.open(f"{file}", 'rt') as f:
                    for l in f:
                        if l == "\n": continue
                        datalines.append(l) 
                f.close()
                        
            elif file.endswith('.dat') or file.endswith('.txt'): 
                with open(f"{file}", 'rt') as f :
                    for l in f:
                        if l == "\n": continue
                        datalines.append(l) 
                f.close()
            else : raise ValueError("Data files should be either '.dat' format, or in compressed gunzip form '.dat.gz' ")
        except OSError: 
            raise ValueError("Data files should be either '.dat' format, or in compressed gunzip form '.dat.gz' ")
        return datalines
    
            

@dataclass
class Hit:
    channelNo : int
    barNo : int
    adc : int
    panelID : int = field(default=0)
    

@dataclass 
class Impact:
    """XY hits collection on panel"""
    line : str
    zpos : float
    panelID :int= field(default=None)
    evtID: int= field(init=False)
    timestamp_s : float= field(init=False)
    timestamp_ns : float= field(init=False)
    nhits: int = field(init=False)
    hits : List[Hit] = field(default_factory=list)
    
    def __post_init__(self):
        self._Channels = []
        self._ADC = []
        self.readline()
        
    def readline(self):
        l = self.line.split()
        if "\t" in self.line : l= self.line.split("\t")
        try:
            timestamp_s, self.evtID, timestamp_ns  = float(l[0]), int(l[1]), float(l[2])
        except:
            raise ValueError(f"{self.line}\n{l}")
        if self.panelID == None: self.panelID = int(l[5])
        self.nhits = int(l[8])
        self.timestamp_s  = timestamp_s
        self.timestamp_ns = timestamp_ns #10^-8 s
        self._Channels = [int(l[9 + 2 * i]) for i in range(self.nhits)]
        self._ADC = [float(l[10 + 2 * i]) for i in range(self.nhits)]
       
        
    def convert_ch_to_bar(self, channelmap):
        chmap  = channelmap.dict_ch_to_bar
        keys = list(chmap.keys())
        _barNo = [ chmap[ch] if ch in keys else float('nan') for ch in self._Channels ]
        self.hits.extend([Hit(channelNo = int(ch), barNo= bar, adc=adc) for ch, bar, adc in zip(self._Channels, _barNo, self._ADC)])
     

@dataclass 
class ImpactPM: 
    """Hit collections on one PMT
    Useful for configuration where one PMT is connected to two detection panels"""
    line : str
    evtID : int = field(init=False)
    timestamp_s : float= field(init=False)
    timestamp_ns : float= field(init=False)
    nhits: int = field(init=False)
    impacts : Dict[int, Impact] = field(default_factory=dict)
    
    def __post_init__(self):
        self._Channels = []
        self._ADC = []
        self.readline()
        
    def readline(self):
        l = self.line.split()
        #print(l)
        if "\t" in self.line : l= self.line.split("\t")
        try:
            timestamp_s, self.evtID, timestamp_ns  = float(l[0]), int(l[1]), float(l[2])
        except:
            raise ValueError(f"{self.line}\n{l}")
        self.pmID = int(l[5])
        self.nhits = int(l[8])
        self.timestamp_s  = timestamp_s
        self.timestamp_ns = timestamp_ns #10^-8 s
        self._Channels = [int(l[9 + 2 * i]) for i in range(self.nhits)]  
        self._ADC = [float(l[10 + 2 * i]) for i in range(self.nhits)]
       
    def fill_panel_impacts(self,channelmap:ChannelMap, nPM:int, zpos:dict, minPlan:int=6, scint_eff:float=None, langau:dict=None):
        chmap  = channelmap.dict_ch_to_bar
        ch_keys = list(chmap.keys())
        l_panelID = []
        for adc, ch in zip(self._ADC, self._Channels):
            if ch not in ch_keys: continue
            if nPM ==2:
                ####SBR config
                if ch%8 <= 3 : panID = 2*(self.pmID - minPlan) 
                elif ch%8 >= 4 : panID = 2*(self.pmID - minPlan)+1  
                else : raise "Channel issue"
            elif nPM == 3 or nPM == 4:
                panID = self.pmID
            else: return "Unknown PMT configuration"    
            
            if panID not in l_panelID:     
                l_panelID.append(panID)
                self.impacts[panID] =  Impact(line=self.line, panelID=panID, zpos=zpos[int(panID)]) 
            
            hit = Hit(channelNo=ch, barNo=chmap[ch], adc=adc, panelID=panID)
            if scint_eff is None: 
                self.impacts[panID].hits.append(hit)
                continue
            else:
                p = np.random.uniform(low=0.0, high=1., size=None)
                if p <= scint_eff : 
                    self.impacts[panID].hits.append(hit)
                    continue
                else: pass
            
            
            ####Hit selection based on adc charge, 
            if langau is None: 
                self.impacts[panID].hits.append(hit)
                continue
            else:
                p = np.random.uniform(low=0.0, high=1., size=None)
                thres = langau[panID](np.array([adc]).astype('double')) 
                if p <= thres : 
                    self.impacts[panID].hits.append(hit)
                    continue
                else: pass
            

@dataclass
class Event:

    ID : int 
    timestamp_s : float = field(init=False)
    timestamp_ns : float = field(init=False)
    tof : float = field(init=False) #time-of-flight
    xyz : np.ndarray = field(init=False)
    npts : int = 0
    adc : np.ndarray = field(init=False)
 #   impacts : List[Impact] = field(default_factory=list) 
    impacts : Dict[int, Impact] = field(default_factory=dict) 
    gold : bool = 0 #if an event forms exactly 1 hit per scintillation layer
    
    def xyz_bar(self):
        """
        Combinations of 1D hit couples to retrieve XY coordinate for each impact

        :return: type: numpy array of XYZ points.
        """
        barNo = []
        adc = []
        for _, imp in self.impacts.items():  # loop on impacts (=touched panels)
            z = imp.panelID
            #coord.extend([hit.channelNo for hit in imp.hits])
            #adc = [hit.adc for hit in imp.hits]
            l_hits= [ hit for hit in imp.hits if type(hit.barNo) == str and hit.adc != 0]
            l_comb = combinations(l_hits, 2)
            for hit1, hit2 in l_comb : 
                if hit1.barNo[0] == hit2.barNo[0] : continue    
                barNo.extend( [( int(hit1.barNo[1:]), int(hit2.barNo[1:]), z) if (hit1.barNo[0] == 'X') else (int(hit2.barNo[1:]), int(hit1.barNo[1:]), z) ] )
                adc.extend( [( float(hit1.adc), float(hit2.adc), z )  if (hit1.barNo[0] == 'X') else (float(hit2.adc), float(hit1.adc), z ) ] ) 
        arr_xyz_bar = np.asarray(barNo)
        self.adc = np.asarray(adc)
        return arr_xyz_bar

    def xyz_mm(self, width:dict={}, zpos:dict={}):
        XYZ = self.xyz_bar()
        arr_xyz_mm = np.ones(XYZ.shape)
        for i, xyz in enumerate(XYZ):
            z = xyz[2]
            scint_width = width[z] #mm
            z_coord = zpos[z] #mm
            x_bar = xyz[0] #bar num
            y_bar = xyz[1]
            #on prend le milieu du scintillateur comme position en mm 
            arr_xyz_mm[i] =  np.array([x_bar-1/2, y_bar-1/2, 1]) * np.append( np.ones(2)*scint_width,  z_coord  )
            #arr_xyz_mm[i] = np.multiply(arr_xyz_mm[i], np.array([-1,-1,-1]))
            #arr_xyz_mm[i] = np.add(arr_xyz_mm[i], np.array([0,0,-900]))
        return arr_xyz_mm
        
    def get_adc(self): 
        """
        :return: type: numpy array of nadc values X and Y hits for each XYZ point.
        """
        return self.adc
    
    def get_xyz(self, in_mm:bool=True, width:dict={}, zpos:dict={}):
        if in_mm: self.xyz = self.xyz_mm(width, zpos)
        else : self.xyz = self.xyz_bar()
        if len(self.xyz) != 0: self.npts = len(np.unique(self.xyz, axis=0))
   
    def get_time_of_flight(self):
        # imp = list(self.impacts.values())
        # dt = float(imp[-1].timestamp_ns) - float(imp[0].timestamp_ns)
        # self.tof= dt#in 10ns
        l_impacts= list(self.impacts.values())
        l_z = [imp.zpos for imp in l_impacts]
        imp_front, imp_rear =  l_impacts[np.argmin(l_z)], l_impacts[np.argmax(l_z)]
        ts_front, ts_rear = imp_front.timestamp_ns, imp_rear.timestamp_ns
        self.tof = (ts_rear-ts_front) #in 10ns
        

class Model:
    @abstractmethod
    def model(self, model):
        pass

class Inliers:
    @abstractmethod
    def inliers(self, inliers):
        pass

class Intersection:
    def __init__(self, model:Union[LineModelND, LinearRegression], z:np.ndarray, axis:int=2, xlim:tuple=(0,800), ylim:tuple=(0,800)):
        self.model = model
        self.z = z #1D panel coordinate for which we want to get straight line get_intersection points
        self.axis = axis
        self.xlim = xlim
        self.ylim = ylim
        self.points = np.zeros(shape=(len(z), 3))
        self.in_panel =  np.zeros(shape=(len(z)), dtype=bool)
        self.get_intersection()
    def get_intersection(self):
        self.points =  self.model.predict(self.z, axis=self.axis)
        for i, xyz in enumerate(self.points): 
            self.in_panel[i] = self.is_point_on_panel(xyz)
            
    def is_point_on_panel(self, xyz:np.ndarray) : 
        xmin, xmax = self.xlim
        ymin, ymax = self.ylim
        if xmin <= xyz[0] and xyz[0] <= xmax and ymin <= xyz[1] and xyz[1] <= ymax: return True
        else: return False
          
      
class ReconstructedParticle(Event):
    def __init__(self, evt:Event):
        self.event = evt 
        self._xyz = self.event.xyz
        is_xyz_empty = self._xyz.size == 0 == 0 
        self.is_reco = True
        if is_xyz_empty: 
            self.is_reco = False
        self.model_robust = Model #LineModelND
        self.inliers = Inliers #list()
        self.outliers = Inliers
        self.gof = float #goodness-of-fit
        if self._xyz.size == 0 : self.is_reco = False
        self._adc = self.event.adc
        self.pixel = tuple  #(DX, DY) or (Phi, Theta)
        #RANSAC parameters
        self.residual_threshold = float #in same unit as xyz coordinates
        self.min_samples = int
        self.max_trials = int 
        
        if len(self.event.impacts) < 3 : 
            #print("RANSAC fit cannot be performed.")
            self.is_reco = False
        #if  self.is_reco is False:  print("Reconstruction is not possible.")
        t = 50 #in mm
        p_success = 0.99 #probability of getting a pure inlier sample 
        p_inlier = 0.5 #expected proportion of inliers in data points, assumption
        ms = 0
        if not is_xyz_empty:  ms = len(set( self._xyz [:,-1] ) ) -1
        N = 100
        #if ms != 0 :  
        #    N = int( np.log(1-p_success)/ np.log(1-p_inlier**ms) ) 
        self.default_ransac_param = {'residual_threshold': t, "min_samples": ms, "max_trials": N, "stop_probability":0.99} 
    
    def ransac_reco(self, **kwargs)->None:
        ransac_params =  kwargs
        for key,par in self.default_ransac_param.items():    
            if key not in list(ransac_params.keys()): ransac_params[key] = par
               
        self.model_robust, self.inliers = None, np.array([])
        if self.is_reco is False:
            return None
        try:
        #print(f"evt{self.event.ID}")
        #wi = np.ones(shape=self._xyz.shape)
            self.model_robust, self.inliers = ransac(
                self._xyz,
                LineModelND,
                min_samples=ransac_params['min_samples'], #ransac_params['min_samples'],  # The minimum number of data points to fit a model to.
                residual_threshold=ransac_params['residual_threshold'],  # Maximum distance for a data point to be classified as an inlier.
                max_trials=ransac_params['max_trials'],
                #stop_probability=ransac_params['stop_probability'],
                #stop_n_inliers=stop_n_inliers 
            )
        
            
        except:
            ValueError("ERROR",self.model_robust)

    def is_valid(self,max_outliers:dict=None)->bool:        
        model   = self.model_robust
        inliers = np.asarray(self.inliers)
        
        if model is None or inliers.size ==0:
            return False

        xyz = self.event.xyz
        nimp= len(self.event.impacts)
        if inliers[inliers==False].size == inliers.size:
            #if 0 inliers
            return False
    
        if model.params[1][2] == 0: 
            #parallel track
            return False
        
        xyz_inliers = np.array([xyz[i,:] for i in np.where(inliers == True)])[0,:,:]
        xyz_outliers = np.array([xyz[i,:] for i in np.where(inliers == False)])[0,:,:]
        
        
        if len(set(xyz_inliers[:,-1]) ) < 3: 
            #if less than 3 impacts containing inliers
            return False
        self.outliers = self.inliers == False #outlier pts are 'false' inliers
        
        zsort = np.sort(list(set(xyz[:,-1])))
        
        if max_outliers is None: pass
        else: 
            if len(xyz_outliers) != 0 : 
                if len(zsort) == 3: 
                    xyz_outliers_rear_panel = xyz_outliers[xyz_outliers[:,-1]==zsort[-1]]
                    if  len(xyz_outliers_rear_panel) > max_outliers['3p']: 
                        return False
                elif len(zsort) == 4: 
                    xyz_outliers_middle_panel = xyz_outliers[xyz_outliers[:,-1]==zsort[-2]]
                    xyz_outliers_rear_panel = xyz_outliers[xyz_outliers[:,-1]==zsort[-1]]
                    if  len(xyz_outliers_middle_panel) > max_outliers['4p'] or len(xyz_outliers_rear_panel) > max_outliers['4p']: 
                        return False
                else: raise ValueError
            
            else: pass 
        
        self.goodness_of_fit(xyz)
        return True
    
    def goodness_of_fit(self, xyz, sigma=5):
        self.residuals = self.model_robust.residuals(xyz)
        #self.expected_xyz = Intersection(self.model_robust, xyz[:, -1] ).points[0][:,:-1]
        #self.obs_exp = self._xyz[:,:-1] - self.expected_xyz
        #self.chi2 = np.sum(np.divide(np.power(self.obs_exp,2),self.expected_xyz))/sigma**2
        self.ndof = len(self.residuals) - (len(self.model_robust.params[1])-1)
        #rss = np.sum(np.power(self.residuals, 2))
        #self.rchi2 = self.chi2/self.ndof    
        self.quadsumres = np.around(np.sum(np.power(self.residuals,2)),3)
        self.gof = 0.#np.around(self.rchi2, 3)
        


    
class Processing: 
    def __init__(self, data:Data, outdir:str):
        self.data   = data
        self.outdir = outdir
        self.input_type = data.type
        self.tel = self.data.telescope
        self.PMTs = { pm.ID : pm for pm in self.tel.PMTs}
        self.panels = { p.ID : p for p in self.tel.panels}
        npanels = len(self.panels)
        self.nreco, self.nevt_tot, self.ngold = {f'{npanels-1}p':0, f'{npanels}p':0}, 0, 0
        self.zpos = { i : p.position.z  for i,p in self.panels.items()}
        self.zcoord = np.array(list(self.zpos.values()))
        if self.input_type == InputType.DATA or self.input_type == InputType.MC:
            self.col_reco  = ['evtID', 'gold', 'timestamp_s', 'timestamp_ns','time-of-flight','npts','nimp', 'quadsumres' ]  
            self.col_coord = ["X_"+ str(p.position.loc) for _,p  in self.panels.items()] 
            self.col_coord.extend(["Y_"+ str(p.position.loc) for _,p  in self.panels.items()] )
            self.col_reco.extend(self.col_coord)
            self.col_reco.extend(['ninl', 'noutl'])
            npanels= len(self.tel.panels)
            self.sel_signal = {f'{npanels-1}p':[], f'{npanels}p':[]}
            self.df_reco = pd.DataFrame(columns = self.col_reco)
            self.col_inlier = ['evtID', 'timestamp_s', 'timestamp_ns', 'inlier', 'gold', 'X', 'Y', 'Z', 'ADC_X', 'ADC_Y']
            self.df_inlier = pd.DataFrame(columns = self.col_inlier)
        else: raise Exception("Unknown 'InputType'")
        
            
    def to_csv(self):
        """Save dataframes in .csv files"""
        l = self.data.label
        if self.input_type == InputType.DATA or self.input_type == InputType.MC:
            outfile_reco = os.path.join(self.outdir, 'reco.csv.gz')
            outfile_inlier = os.path.join(self.outdir, 'inlier.csv.gz')
            self.df_reco.to_csv(outfile_reco, compression='gzip',index=False, sep='\t')
            self.df_inlier.to_csv(outfile_inlier, compression='gzip', sep='\t')
        else: raise Exception("Unknown 'InputType'")
    
    def reinit_evt(self, old_evt:Event, impact_pmt:ImpactPM) -> Event:
        del old_evt
        #new event
        new_evtID = impact_pmt.evtID
        evt = Event(ID=new_evtID)
        for pid, impan in impact_pmt.impacts.items() : evt.impacts[pid] = impan
        evt.timestamp_s, evt.timestamp_ns= impact_pmt.timestamp_s, impact_pmt.timestamp_ns
        return evt
    
    
    def filter(self, evt:Event)->bool: #evd:eventdisplay.RawEvtDisplay=None
        '''
        Here you can add filters to event before reconstructing the trajectory
        '''
        
        iscut= False             
        tag = ""
        
        npanels = len(self.panels)

        if len(evt.xyz) ==0 : 
            iscut= True
            tag = 'xyz'
            return iscut, tag
       
        ztouched_panels = list(set(evt.xyz[:,-1]))
        ntouched_panels = len(ztouched_panels) 
        #Capture selected events to be reconstructed
        if  ntouched_panels  < npanels-1 : 
            iscut = True 
            tag = "panels"
            return iscut, tag   
        
        #check hit multiplicity on each impact (=touched panel)   
        max_multiplicity = max([len(i.hits) for _,i in evt.impacts.items()])
        if max_multiplicity > 10: 
            iscut=True
            tag = "multiplicity"
            return iscut, tag   
          
        nhits = sum([len(i.hits) for _,i in evt.impacts.items()])
        ####is evt gold ?
        if ntouched_panels == npanels and nhits == 2*npanels : 
            evt.gold = 1 
            self.ngold += 1
        return iscut, tag
        
        
        


    def ransac_reco_pmt(self, residual_threshold:float, min_samples:int, max_trials:int, scint_eff:float=None, langau:dict=None, max_outliers:dict=None, is_fit_intersect:bool=False)-> None :
        '''
        RANSAC processing of data files event-by-event
        '''
        nPM = len(self.tel.PMTs)
        minPlan = np.min([pm.ID for pm in self.tel.PMTs])
        last_evtID = 0
        barwidths = { i :  float(p.matrix.scintillator.width) for i,p  in self.panels.items() }
        headers= list(self.df_reco.keys())
        npanels = len(self.panels)
        nsel= {f'{npanels-1}p':0, f'{npanels}p':0}
        ninliers = {f'{npanels-1}p':[], f'{npanels}p':[]}
        noutliers = {f'{npanels-1}p':[], f'{npanels}p':[]}
        npts = {f'{npanels-1}p':[], f'{npanels}p':[]}
        
        for n, file in enumerate(self.data.dataset):
            lines= self.data.readfile(file)
            nlines = len(lines)
            #init : 1st impact on PMT
            impm = ImpactPM(line=lines[0])
            #create pmt impact 
            pmt = self.PMTs[impm.pmID]
            channelmap = pmt.channelmap
            #create panel impacts
            impm.fill_panel_impacts(channelmap, nPM, self.zpos, minPlan, scint_eff, langau)
            evt = Event(ID = impm.evtID)
            #Add impacts (touched panels) to evt
            for pid, imp in impm.impacts.items() : evt.impacts[pid] = imp
            evt.timestamp_s, evt.timestamp_ns= impm.timestamp_s, impm.timestamp_ns
            
            last_evtID = evt.ID
            out_matrix = np.zeros(shape=(nlines, len(headers)))
            
           
            
            for i, l in enumerate(lines[1:]) :
                #print(f'--->EVT{last_evtID}')
                impm = ImpactPM(line=l)
                pmt = self.PMTs[impm.pmID]
                channelmap = pmt.channelmap
                impm.fill_panel_impacts(channelmap, nPM, self.zpos, minPlan, scint_eff, langau)
                if impm.evtID == last_evtID:
                    for pid, imp in impm.impacts.items() : evt.impacts[pid] = imp
                    if i == len(lines)-1: pass #last line of file
                    else: continue
                #if new evtID, retrieve the last evtID and reconstruct it 
                #get coordinates 
                evt.get_xyz(in_mm=True, width=barwidths, zpos=self.zpos)

                iscut, tag = self.filter(evt)
                if iscut:
                    evt = self.reinit_evt(old_evt=evt, impact_pmt=impm)
                    last_evtID = evt.ID
                    if i != len(lines)-1: self.nevt_tot += 1
                    continue
                else: 
                    for _,imp in evt.impacts.items():
                        s ="{},{},{}".format(evt.ID,imp.panelID, ','.join(str(h.adc) for h in imp.hits))
                        ####Get number of 3 and 4panels events
                        ntouched_panels = len(list(set(evt.xyz[:,2]))) 
                        self.sel_signal[f'{ntouched_panels}p'].append(s)
                    nsel[f'{ntouched_panels}p'] += 1    
                try:
                    evt.get_time_of_flight()
                except: 
                    l_imp=list(impm.impacts.values())
                    raise ValueError(f"{file}\n{evt.ID}\nError during 'evt.get_time_of_flight()'\nl_impacts{l_imp}\nl_z={[imp.zpos for imp in l_imp]}")            

                # if iscut and tag=="multiplicity": 
                #     print(iscut, tag)
                #     evd = eventdisplay.RawEvtDisplay(telescope=self.tel, label="", max_nevt=3)
                #     evd.addEvt(evt=evt, color='red')
                #     evd.show()
                
                
                reco = ReconstructedParticle(evt)
                reco.ransac_reco(residual_threshold=residual_threshold, 
                                 min_samples=min_samples, 
                                 max_trials=max_trials) 

                if reco.is_valid() :
                    nin, nout = len(np.unique(evt.xyz[reco.inliers==True], axis=0)), 0
                    if np.any(reco.outliers==True): nout=len(np.unique(evt.xyz[reco.outliers==True], axis=0))
                    ninliers[f'{ntouched_panels}p'].append(nin)
                    noutliers[f'{ntouched_panels}p'].append(nout)
                    npts[f'{ntouched_panels}p'].append(nin+nout)
    
                    
                    inter_pts = Intersection(reco.model_robust, self.zcoord ).points
                    reco_XYZ_inter = np.around(inter_pts,1)
                    
                    if is_fit_intersect : 
                        X, Y, _ = reco_XYZ_inter.T
                    else : 
                        xyz_inliers = evt.xyz[reco.inliers]
                        xyz_inliers_sort = xyz_inliers[xyz_inliers[:,-1].argsort()]
                        z_touched_panels = set(xyz_inliers_sort[:,2])
                        id_nearest = np.array([np.argmin(np.array([ np.linalg.norm(xyz-reco_XYZ_inter[reco_XYZ_inter[:,2]==z]) for xyz in xyz_inliers_sort ]) ) if z in z_touched_panels else None  for z in self.zcoord ])
                        close_XYZ = np.array([ xyz_inliers_sort[ix] if ix is not None else np.zeros(3) for ix in id_nearest ] ) 
                        X, Y, _ = close_XYZ.T
                        X[X==0.], Y[Y==0.] = reco_XYZ_inter[np.where(X==0)[0], 0], reco_XYZ_inter[np.where(Y==0)[0], 1]
                
                    line =  np.concatenate(([evt.ID, evt.gold, evt.timestamp_s, evt.timestamp_ns, evt.tof, evt.npts, ntouched_panels, reco.quadsumres], X, Y, [nin, nout]), axis=0)
                    out_matrix[i, :] = line
                    self.nreco[f'{ntouched_panels}p']+=1     
                    self.process_inlier(reco)
                    
                evt  = self.reinit_evt(old_evt=evt, impact_pmt=impm)
                last_evtID = evt.ID
                if i != len(lines)-1: self.nevt_tot += 1                
            
            out_matrix =  out_matrix[~np.all( (out_matrix == 0.), axis=1)]
            #self.df_reco  = self.df_reco.append(pd.DataFrame(out_matrix, columns=headers))
            self.df_reco = pd.concat([self.df_reco, pd.DataFrame(out_matrix, columns=self.df_reco.columns)])
         ####format columns 
        for col in list(set(self.df_reco.columns) - set(self.col_coord)):
            self.df_reco[col] = np.ndarray.astype(self.df_reco[col].values, dtype=int)
        for col in self.col_coord:
            self.df_reco[col] = np.ndarray.astype(self.df_reco[col].values, dtype=float)    
        for col in  self.col_inlier :
            dtype=int
            if col == 'ADC_X' or col =='ADC_Y':    dtype=float
            self.df_inlier[col] = np.ndarray.astype(self.df_inlier[col].values, dtype=dtype)
    
        self.df_inlier = self.df_inlier.set_index(['evtID', 'timestamp_s', 'timestamp_ns'])

        nreco_tot = sum([val for _, val in self.nreco.items()])
        sout0 = f"RANSAC output:\n(nreco/nevt)_tot = {nreco_tot}/{self.nevt_tot} = {nreco_tot/self.nevt_tot:.2f}\n"
        sout1 = "".join([f"(nreco/nselect)_{key} = {self.nreco[key]}/{nsel[key]} = {self.nreco[key]/nsel[key] :.2f}\n" for key in list( nsel.keys() ) if nsel[key] != 0  ] )  
        sninl = "".join([f"<f_inliers>_{key} = {np.mean(np.divide(ninliers[key],npts[key])):.2f} \n" for key in list( nsel.keys() ) if nsel[key] != 0  ] )
        sfinl = "".join([f"<ninliers>_{key} = {np.mean(ninliers[key]):.2f} \n" for key in list( nsel.keys() ) if nsel[key] != 0  ] )
        snoutl = "".join([f"<f_outliers>_{key} = {np.mean(np.divide(noutliers[key],npts[key])):.2f} \n" for key in list( nsel.keys() ) if nsel[key] != 0  ] )
        sfoutl = "".join([f"<noutliers>_{key} = {np.mean(noutliers[key]):.2f} \n" for key in list( nsel.keys() ) if nsel[key] != 0  ] )
        logging.info(sout0+sout1+sninl+sfinl+snoutl+sfoutl)
        # print(sout0+sout1+sninl+sfinl+snoutl+sfoutl)
        print(f"ngold = {self.ngold}")
        nsel, self.nreco, self.nevt_tot = {f'{npanels-1}p':0, f'{npanels}p':0}, {f'{npanels-1}p':0, f'{npanels}p':0}, 0
        
    
            
    def process_inlier(self, reco:ReconstructedParticle):
        """Fill RANSAC inlier dataframe"""
        evt = reco.event
        for i in range(len(reco.inliers)):
            xyz, adc = evt.xyz[i,:], evt.adc[i,:]
            is_inl = reco.inliers[i]
            timestamp_s, timestamp_ns = evt.impacts[adc[2]].timestamp_s, evt.impacts[adc[2]].timestamp_ns
            df_tmp = pd.DataFrame(np.array([ [int(evt.ID), int(timestamp_s), int(timestamp_ns), int(is_inl), int(evt.gold), xyz[0], xyz[1], xyz[2], adc[0], adc[1]] ]), columns=self.col_inlier)
            #self.df_inlier  = self.df_inlier.append(df_tmp)
            self.df_inlier = pd.concat([self.df_inlier, df_tmp])
        
   

if __name__ == '__main__':
    
    
    pass


    
    

  
    
    
  
 