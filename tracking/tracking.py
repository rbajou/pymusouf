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
from dataclasses import dataclass, field
from skimage.measure import ransac, LineModelND
from sklearn.linear_model import RANSACRegressor,LinearRegression
import pandas as pd
from itertools import combinations
from math import floor, log10

#package module(s)
from survey.data import RawData
from telescope import ChannelMap, Telescope
from utils.tools import print_progress


@dataclass
class Hit:
    
    channel_no : int
    bar_no : int
    adc : int
    panelID : int = field(default=0)
    

@dataclass
class Timestamp: 
    
    s : int #sec
    ns : int  #nanosec

    def __post_init__(self):
        prec = floor(log10(self.ns))
        self.res = 10**(-prec) #in sec


@dataclass 
class Impact:
    """XY hits collection on panel"""
    line : str
    zpos : float
    panelID :int= field(default=None)
    evtID: int= field(init=False)
    timestamp : Timestamp = field(init=False)
    nhits: int = field(init=False)
    hits : List[Hit] = field(default_factory=list)
    

    def __post_init__(self):
       
        self._channel_no = []
        self._adc = []
        self.readline()
        

    def readline(self):
        
        l = self.line.split()
        if "\t" in self.line : l= self.line.split("\t")
        try:
            ts_s, self.evtID, ts_ns  = float(l[0]), int(l[1]), float(l[2])
        except:
            raise ValueError(f"{self.line}\n{l}")
        if self.panelID == None: self.panelID = int(l[5])
        self.nhits = int(l[8])
        self.timestamp = Timestamp(ts_s, ts_ns)
        self._channel_no = [int(l[9 + 2 * i]) for i in range(self.nhits)]
        self._adc = [float(l[10 + 2 * i]) for i in range(self.nhits)]
       
        
    def convert_ch_to_bar(self, channelmap):
        
        chmap  = channelmap.dict_ch_to_bar
        keys = list(chmap.keys())
        _bar_no = [ chmap[ch] if ch in keys else float('nan') for ch in self._channel_no ]
        self.hits.extend([Hit(channel_no = int(ch), bar_no= bar, adc=adc) for ch, bar, adc in zip(self._channel_no, _bar_no, self._adc)])
     

@dataclass 
class ImpactPM: 
    """Hit collections on one PMT
    Useful for configuration where one PMT is connected to two detection panels"""
    line : str
    evtID : int = field(init=False)
    timestamp : Timestamp = field(init=False)
    nhits: int = field(init=False)
    impacts : Dict[int, Impact] = field(default_factory=dict)
    

    def __post_init__(self):
       
        self._channel_no = []
        self._adc = []
        self.readline()
        

    def readline(self):

        l = self.line.split()
        if "\t" in self.line : l= self.line.split("\t")
        try:
            ts_s, self.evtID, ts_ns  = float(l[0]), int(l[1]), float(l[2])
        except:
            raise ValueError(f"{self.line}\n{l}")
        self.pmID = int(l[5])
        self.nhits = int(l[8])
        self.timestamp = Timestamp(s=ts_s, ns=ts_ns)
        self._channel_no = [int(l[9 + 2 * i]) for i in range(self.nhits)]  
        self._adc = [float(l[10 + 2 * i]) for i in range(self.nhits)]
       

    def fill_panel_impacts(self,channelmap:ChannelMap, nPM:int, zpos:dict, minPlan:int=6, scint_eff:float=None, langau:dict=None):
       
        chmap  = channelmap.dict_ch_to_bar
        ch_keys = list(chmap.keys())
        l_panelID = []
        for adc, ch in zip(self._adc, self._channel_no):
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
            
            hit = Hit(channel_no=ch, bar_no=chmap[ch], adc=adc, panelID=panID)
            if scint_eff is None: 
                self.impacts[panID].hits.append(hit)
                continue
            else:
                p = np.random.uniform(low=0.0, high=1., size=None)
                if p <= scint_eff : 
                    self.impacts[panID].hits.append(hit)
                    continue
                else: pass
            
    
@dataclass
class Event:

    ID : int 
    timestamp : Timestamp 
    tof : float = field(init=False) #time-of-flight
    xyz : np.ndarray = field(init=False)
    npts : int = 0
    adc : np.ndarray = field(init=False)
    impacts : Dict[int, Impact] = field(default_factory=dict) 
    gold : bool = 0 #if an event forms exactly 1 hit per scintillation layer


    def __post_init__(self):
        
        self.dict_out = {}
        self.dict_out['evtID'] = self.ID
        self.dict_out['timestamp_s'] = self.timestamp.s
        self.dict_out['timestamp_ns'] = self.timestamp.ns
        self.dict_out['tof'] = None
        self.dict_out['gold'] = self.gold
        self.dict_out['npts'] = None
        self.dict_out['nimpacts'] = None


    def _xyz_bar(self) -> np.ndarray:
        """
        Combinations of 1D hit couples to retrieve XY coordinate for each impact

        :return: type: numpy array of XYZ points.
        """
        bar_no = []
        adc = []

        for _, imp in self.impacts.items():  # loop on impacts (=impacted panels)

            z = imp.panelID
            l_hits= [ hit for hit in imp.hits if type(hit.bar_no) == str and hit.adc != 0]
            l_comb = combinations(l_hits, 2)
            for hit1, hit2 in l_comb : 
                if hit1.bar_no[0] == hit2.bar_no[0] : continue    
                bar_no.extend( [( int(hit1.bar_no[1:]), int(hit2.bar_no[1:]), z) if (hit1.bar_no[0] == 'X') else (int(hit2.bar_no[1:]), int(hit1.bar_no[1:]), z) ] )
                adc.extend( [( float(hit1.adc), float(hit2.adc), z )  if (hit1.bar_no[0] == 'X') else (float(hit2.adc), float(hit1.adc), z ) ] ) 
        arr_xyz_bar = np.asarray(bar_no)
        self.adc = np.asarray(adc)
        return arr_xyz_bar


    def _xyz_mm(self, width:dict={}, zpos:dict={}) -> np.ndarray :

        xyz = self._xyz_bar()
        arr_xyz_mm = np.ones(xyz.shape)
        
        for i, xyz in enumerate(xyz):
           
            z = xyz[2]
            scint_width = width[z] #mm
            z_coord = zpos[z] #mm
            x_bar = xyz[0] #bar num
            y_bar = xyz[1]
            
            #scint. center as pt coords
            arr_xyz_mm[i] =  np.array([x_bar-1/2, y_bar-1/2, 1]) * np.append( np.ones(2)*scint_width,  z_coord  )
            #arr_xyz_mm[i] = np.multiply(arr_xyz_mm[i], np.array([-1,-1,-1]))
            #arr_xyz_mm[i] = np.add(arr_xyz_mm[i], np.array([0,0,-900]))
        
        return arr_xyz_mm
        

    def get_xyz(self, in_mm:bool=True, width:dict={}, zpos:dict={}) -> None:
       
        if in_mm: self.xyz = self._xyz_mm(width, zpos)
        else : self.xyz = self._xyz_bar()
        if len(self.xyz) != 0: 
            self.npts = len(np.unique(self.xyz, axis=0))
            self.dict_out['npts'] = self.npts
            self.nimpacts = len(list(set(self.xyz[:,-1]))) 
            self.dict_out['nimpacts'] = self.nimpacts

    def get_time_of_flight(self) -> None:

        l_impacts= list(self.impacts.values())
        l_z = [imp.zpos for imp in l_impacts]
        imp_front, imp_rear =  l_impacts[np.argmin(l_z)], l_impacts[np.argmax(l_z)]
        tns_front, tns_rear = imp_front.timestamp.ns, imp_rear.timestamp.ns
        self.tof = (tns_rear-tns_front) #in 10ns
        self.dict_out['tof'] = self.tof
        

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
        self.xyz = np.zeros(shape=(len(z), 3))
        self.in_panel =  np.zeros(shape=(len(z)), dtype=bool)
        self.get_intersection()
    

    def get_intersection(self):
        self.xyz =  self.model.predict(self.z, axis=self.axis)
        for i, xyz in enumerate(self.xyz): 
            self.in_panel[i] = self.is_point_on_panel(xyz)
            

    def is_point_on_panel(self, xyz:np.ndarray) : 
        xmin, xmax = self.xlim
        ymin, ymax = self.ylim
        if xmin <= xyz[0] and xyz[0] <= xmax and ymin <= xyz[1] and xyz[1] <= ymax: return True
        else: return False
          

class TrackingType(Enum): 
    
    ransac = auto()


class TrackModel:
    
    
    def __init__(self, event:Event):
    

        self.evt = event#self.ID, self.impacts, self.xyz, self.adc = event.ID, event.impacts, event.xyz, event.adc
       
        self.model_robust = Model #LineModelND
        self.quadsumres = float #quadratic sum of residuals to model 
        
        self.dict_out = self.evt.dict_out
        self.dict_out['quadsumres'] = None


    def is_track_avail(self) -> bool : 
        
        is_track = True
        if self.evt.xyz.size == 0 : is_track = False
        if len(self.evt.impacts) < 3 : is_track = False

        return is_track

    def goodness_of_fit(self, xyz, sigma=5):

        self.residuals = self.model_robust.residuals(xyz)
        self.quadsumres = np.around(np.sum(np.power(self.residuals,2)),3)
        self.dict_out['quadsumres'] = np.around(self.quadsumres,1)
        #self.expected_xyz = Intersection(self.model_robust, xyz[:, -1] ).points[0][:,:-1]
        #self.obs_exp = self.xyz[:,:-1] - self.expected_xyz
        #self.chi2 = np.sum(np.divide(np.power(self.obs_exp,2),self.expected_xyz))/sigma**2
        # self.ndof = len(self.residuals) - (len(self.model_robust.params[1])-1)
        #rss = np.sum(np.power(self.residuals, 2))
        #self.rchi2 = self.chi2/self.ndof    
        # self.gof = 0.#np.around(self.rchi2, 3)


class RansacModel(TrackModel):
        
    
    def __init__(self, event:Event):

        TrackModel.__init__(self, event)
        self.inliers, self.outliers = Inliers, Inliers
        
        #RANSAC parameters
        self.residual_threshold = float #in same unit as xyz coordinates
        self.min_samples = int
        self.max_trials = int 
        
        t = 50 #in mm
        p_success = 0.99 #probability of getting a pure inlier sample 
        p_inlier = 0.5 #expected proportion of inliers in data points, assumption
        
        ms = 0
        if len(self.evt.xyz) != 0 : ms = len(set( self.evt.xyz [:,-1] ) ) -1
        N = 100
        #if ms != 0 :  
        #    N = int( np.log(1-p_success)/ np.log(1-p_inlier**ms) ) 
        self.default_ransac_param = {'residual_threshold': t, "min_samples": ms, "max_trials": N,}# "stop_probability":0.99} 
    


    def get(self, **kwargs) -> None:
        
        for key,par in self.default_ransac_param.items():    
            if key not in list(kwargs.keys()): kwargs[key] = par
               
        self.model_robust, self.inliers = None, None
       
        try:
            self.model_robust, self.inliers = ransac(
                self.evt.xyz,
                LineModelND,
                **kwargs
            )
        except:
            print(f"No model found for evt {self.evt.ID}")


    def is_valid(self)->bool:        
        
        model, inliers = self.model_robust, np.asarray(self.inliers)
        
        if model is None or inliers.size ==0:

            return False

        if inliers[inliers==False].size == inliers.size:
            #if 0 inliers
            return False
    
        if model.params[1][2] == 0: 
            #parallel track
            return False
        
        xyz_inliers = np.array([self.evt.xyz[i,:] for i in np.where(inliers == True)])[0,:,:]
        if len(set(xyz_inliers[:,-1]) ) < 3: 
            #if less than 3 impacts containing inliers
            return False

        self.outliers = self.inliers[self.inliers == False] #outlier pts are 'false' inliers
        
        self.goodness_of_fit(self.evt.xyz)
    
        return True
    

    def _get_xyz_track(self, zpan:np.ndarray): 
        
        _xyz = Intersection(self.model_robust, zpan ).xyz
        self.xyz_track = np.around(_xyz,1)
        # self.dict_out['xyz_track'] = self.xyz_track

    def _get_xyz_closest_inliers(self, zpan:np.ndarray) -> np.ndarray: 

        xyz_inliers = self.evt.xyz[self.inliers]
        xyz_inliers_sort = xyz_inliers[xyz_inliers[:,-1].argsort()]
        z_traversed = set(xyz_inliers_sort[:,2])
        ix_near = np.array([np.argmin(np.array([ np.linalg.norm(xyz - self.xyz_track[self.xyz_track[:,2]==z]) for xyz in xyz_inliers_sort ]) ) if z in z_traversed else None  for z in zpan ])
        close_xyz = np.array([ xyz_inliers_sort[ix] if ix is not None else np.zeros(3) for ix in ix_near ] ) 
        xfin, yfin, zfin = close_xyz.T
        xfin[xfin==0.], yfin[yfin==0.] = self.xyz_track[np.where(xfin==0)[0], 0], self.xyz_track[np.where(yfin==0)[0], 1]
        res = np.vstack((xfin, yfin, zfin)).T

        return res


    def get_df_track(self, dict_zloc:dict, **kwargs) -> pd.DataFrame:
        
        loc, z = list(dict_zloc.keys()), list(dict_zloc.values())
        self._get_xyz_track(z)
        shape = len(z)
        
        xfin, yfin = np.ones(shape), np.ones(shape) 

        if 'is_fit_intersect' in kwargs.keys() :
            if kwargs['is_fit_intersect']:
                xfin, yfin, _ = self.xyz_track.T
        else : 
            xfin, yfin, _ = self._get_xyz_closest_inliers(z).T
        #res =  np.concatenate(([evt.ID, evt.gold, evt.timestamp.s, evt.timestamp.ns, evt.tof, evt.npts, evt.nimpacts, self.quadsumres], xfin, yfin, [nin, nout]), axis=0)
        
        for i, k in enumerate(loc) : 
            self.dict_out[f'X_{k}'], self.dict_out[f'Y_{k}']  = np.around(xfin[i],0), np.around(yfin[i],0)

        df = pd.DataFrame(data=[self.dict_out])

        return df
    

    def get_df_model(self, **kwargs) -> pd.DataFrame:
        """Fill RANSAC inlier dataframe"""

        df = pd.DataFrame()

        for i in range(len(self.inliers)):
            _xyz, _adc = np.around(self.evt.xyz[i,:], 0), np.around(self.evt.adc[i,:],0)
            is_inl = int(self.inliers[i])
            timestamp_s, timestamp_ns = int(self.evt.impacts[_adc[2]].timestamp.s), int(self.evt.impacts[_adc[2]].timestamp.ns)
            _dict = {'evtID' : self.evt.ID, 'timestamp_s' : timestamp_s, 'timestamp_ns' :timestamp_ns, 'gold' : self.evt.gold, 'inlier': is_inl}
            _dict['X'], _dict['Y'], _dict['Z'], _dict['ADC_X'], _dict['ADC_Y'] = _xyz[0], _xyz[1], _xyz[2], _adc[0], _adc[1]
            _dftmp = pd.DataFrame(data=[_dict])
            if i == 0 : df = _dftmp
            else: df = pd.concat([df.astype(_dftmp.dtypes), _dftmp.astype(df.dtypes)]) #fix future pandas warning on contatenation

        return df


class OtherModel(TrackModel):


    def __init__(self, event:Event):

        TrackModel.__init__(self, event)


    def get(self, *args, **kwargs) -> None: pass
   
    def is_valid(self, *args, **kwargs) -> bool: pass
   
    def get_df_track(self, *args, **kwargs) -> pd.DataFrame: pass
   
    def get_df_model(self, *args, **kwargs) -> pd.DataFrame: pass


    
class Tracking: 
   

    def __init__(self, telescope:Telescope, data:RawData):
       
        self.tel = telescope
        self.data   = data

        self.PMTs = { pm.ID : pm for pm in self.tel.PMTs}
        self.panels = { p.ID : p for p in self.tel.panels}

        self.nevt_tot, self.ngold = 0, 0
        self.zloc  = { p.position.loc : p.position.z  for p in list(self.panels.values())}
        self.zpos = { id : p.position.z  for id,p in self.panels.items()}
        
        self.df_track, self.df_model = pd.DataFrame(), pd.DataFrame()

    
    def reinit_evt(self, old_evt:Event, impact_pm:ImpactPM) -> Event:
        
        del old_evt
        #new event
        new_evtID = impact_pm.evtID
        evt = Event(ID=new_evtID, timestamp = impact_pm.timestamp)
        for pid, impan in impact_pm.impacts.items() : evt.impacts[pid] = impan
        # evt.timestamp.s, evt.timestamp.ns = impact_pm.timestamp.s, impact_pm.timestamp.ns

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
            tag = "xyz"
            return iscut, tag
       
        #Capture selected events to be reconstructed
        if  evt.nimpacts  < npanels-1 : 
            iscut = True 
            tag = "panels"
            return iscut, tag   
        
        #check hit multiplicity on each impact
        max_multiplicity = max([len(i.hits) for _,i in evt.impacts.items()])
        if max_multiplicity > 10: 
            iscut=True
            tag = "multiplicity"
            return iscut, tag   
          
        nhits = sum([len(i.hits) for _,i in evt.impacts.items()])
        ####is evt gold ?
        if evt.nimpacts == npanels and nhits == 2*npanels : 
            evt.gold = 1 
            evt.dict_out['gold'] = evt.gold
            self.ngold += 1
        return iscut, tag
    


        

class RansacTracking(Tracking): 


    def __init__(self, telescope:Telescope, data:RawData, **kwargs):
       
        Tracking.__init__(self, telescope, data)        

        npanels = len(self.panels)
        self.dict_ninliers  = {f'{npanels-1}p':[], f'{npanels}p':[]}
        self.dict_noutliers = {f'{npanels-1}p':[], f'{npanels}p':[]}


    def __str__(self):

        npanels = len(self.panels)
        arr_nimpacts = self.df_track['nimpacts']
        arr_npts = self.df_track['npts']
        dict_ntrack, dict_arr_npts, dict_finlier , dict_foutlier= {}, {}, {}, {}

        ntrack_tot = len(arr_nimpacts)
        sout = f"RANSAC output:\n\t(ntrack/nevt)_tot = {ntrack_tot}/{self.nevt_tot} = {ntrack_tot/self.nevt_tot:.2f}\n"
        sout += f"\tnevt_gold = {self.ngold}\n"
        
        for n in [npanels-1, npanels]:
            
            m = (arr_nimpacts == n)
            if all(m == False): continue
            sout += f"\t{n}-panel events : \n"
            key = f"{n}p"
            dict_ntrack[key] =  len(arr_nimpacts[m])
            npt = arr_npts[m].values
            dict_arr_npts[key] = npt
            arr_ninl = np.asarray(self.dict_ninliers[key])
            arr_noutl = np.asarray(self.dict_noutliers[key])
            dict_finlier[key] = np.mean(arr_ninl/npt) 
            dict_foutlier[key] = np.mean(arr_noutl/npt)
            sout += "\t  " + "\t".join([f"(ntrack/nevt) = {dict_ntrack[key]}/{self.dict_nsel[key]} = {dict_ntrack[key]/self.dict_nsel[key] :.2f}\n" for key in  list(dict_ntrack.keys()) if self.dict_nsel[key] != 0  ] )  
            sout += "\t  " + "\t".join([f"<f_inliers> = {val:.2f} \n" for key, val in dict_finlier.items()] )
            sout += "\t  " + "\t".join([f"<f_outliers> = {val:.2f} \n" for key, val in dict_foutlier.items()] )

        return sout


    def format_df_cols(self):

        col_trk = self.df_track.columns 
        for col in col_trk:
            self.df_track[col] = np.ndarray.astype(self.df_track[col].values, dtype=int)
        
        if self.df_model is not None:  

            col_mod = self.df_model.columns

            for col in  col_mod :
                dtype=int
                if col == 'ADC_X' or col =='ADC_Y':    dtype=float
                self.df_model[col] = np.ndarray.astype(self.df_model[col].values, dtype=dtype)

   

    def process(self, model_type:Union[RansacModel, OtherModel], progress_bar:bool=False, **kwargs_model)-> None :
        '''
        Process tracking data files event-by-event
        '''
        
        nPM = len(self.tel.PMTs)
        minPlan = np.min([pm.ID for pm in self.tel.PMTs])
        last_evtID = 0
        barwidths = { i :  float(p.matrix.scintillator.width) for i,p  in self.panels.items() }

        #counters
        npanels = len(self.panels)
        self.dict_nsel = {f'{npanels-1}p':0, f'{npanels}p':0}

        nfiles = len(self.data.dataset)
        ntrack = 0

        for nf, file in enumerate(self.data.dataset):
            
            lines = self.data.readfile(file)

            #init : 1st impact on PMT
            impm = ImpactPM(line=lines[0])
            #create pmt impact 
            pmt = self.PMTs[impm.pmID]
            channelmap = pmt.channelmap
            #create panel impacts
            impm.fill_panel_impacts(channelmap, nPM, self.zpos, minPlan)
            evt = Event(ID = impm.evtID, timestamp = impm.timestamp)

            #Add impacts (impacted panels) to evt
            for pid, imp in impm.impacts.items() : evt.impacts[pid] = imp
            
            last_evtID = evt.ID
            
            for nl, line in enumerate(lines[1:]) :
                # print(f'--->EVT{last_evtID}')
                
                impm = ImpactPM(line=line)
                pmt = self.PMTs[impm.pmID]
                channelmap = pmt.channelmap
                impm.fill_panel_impacts(channelmap, nPM, self.zpos, minPlan)
                
                if impm.evtID == last_evtID:
                    for pid, imp in impm.impacts.items() : evt.impacts[pid] = imp
                    if nl == len(lines)-1: pass 
                    else: continue
                #if new evtID, retrieve the last evtID and reconstruct it 
                #get coordinates 
                evt.get_xyz(in_mm=True, width=barwidths, zpos=self.zpos)

                iscut, _ = self.filter(evt)

                key = f'{evt.nimpacts}p'
                if iscut:
                    evt = self.reinit_evt(old_evt=evt, impact_pm=impm)
                    last_evtID = evt.ID
                    if nl != len(lines)-1: self.nevt_tot += 1
                    continue
                else: 
                    self.dict_nsel [key] += 1    

                try:
                    evt.get_time_of_flight()
                except: 
                    l_imp=list(impm.impacts.values())
                    raise ValueError(f"{file}\n{evt.ID}\nError 'evt.get_time_of_flight()'\nl_impacts{l_imp}\nl_z={[imp.zpos for imp in l_imp]}")            

            
                model = object.__new__(model_type)#new track model object instance
                model.__init__(evt)

                if model.is_track_avail(): 

                    model.get(**kwargs_model) 
                    
                    if model.is_valid() :


                        _df_trk = model.get_df_track(dict_zloc=self.zloc)
                        if ntrack == 0 :   self.df_track = _df_trk
                        else : self.df_track = pd.concat([self.df_track.astype(_df_trk.dtypes), _df_trk.astype(self.df_track.dtypes)])
                        

                        if isinstance(model, RansacModel): 
                            _df_mod = model.get_df_model()
                            if ntrack == 0 : self.df_model = _df_mod
                            else : self.df_model = pd.concat([self.df_model.astype(_df_mod.dtypes), _df_mod.astype(self.df_model.dtypes)])
                            ninl, noutl = len(model.inliers) - len(model.outliers), len(model.outliers)
                            self.dict_ninliers[key].append(ninl)
                            self.dict_noutliers[key].append(noutl)

                        ntrack+=1


                evt  = self.reinit_evt(old_evt=evt, impact_pm=impm)
                last_evtID = evt.ID
                if nl != len(lines)-1: self.nevt_tot += 1                
                

        if progress_bar : print_progress(nf+1, nfiles, prefix = '\tFile(s) processed :', suffix = 'completed')

        self.format_df_cols()



   
   
    '''

    def process(self, progress_bar:bool=False, **kwargs)-> None :
        """
        RANSAC tracking of data files event-by-event
        """
        #ransac parameters : 
        lparkeys = ['residual_threshold', 'min_samples', 'max_trials']

        nPM = len(self.tel.PMTs)
        minPlan = np.min([pm.ID for pm in self.tel.PMTs])
        last_evtID = 0
        barwidths = { i :  float(p.matrix.scintillator.width) for i,p  in self.panels.items() }
        headers= list(self.df_track.keys())
        npanels = len(self.panels)
        self.nsel = {f'{npanels-1}p':0, f'{npanels}p':0}
        self.ninl = {f'{npanels-1}p':[], f'{npanels}p':[]}
        self.noutl = {f'{npanels-1}p':[], f'{npanels}p':[]}
        self.npts = {f'{npanels-1}p':[], f'{npanels}p':[]}
        
        n, nfiles = 0, len(self.data.dataset)

        for _, file in enumerate(self.data.dataset):
            
            lines= self.data.readfile(file)
            nlines = len(lines)
            #init : 1st impact on PMT
            impm = ImpactPM(line=lines[0])
            #create pmt impact 
            pmt = self.PMTs[impm.pmID]
            channelmap = pmt.channelmap
            #create panel impacts
            impm.fill_panel_impacts(channelmap, nPM, self.zpos, minPlan)
            evt = Event(ID = impm.evtID, timestamp = impm.timestamp)
            #Add impacts (=traversed scint. panels) to evt
            for pid, imp in impm.impacts.items() : evt.impacts[pid] = imp
            # evt.timestamp.s, evt.timestamp.ns= impm.timestamp.s, impm.timestamp.ns
            
            last_evtID = evt.ID
            out_matrix = np.zeros(shape=(nlines, len(headers)))
            
            for i, l in enumerate(lines[1:]) :
                # print(f'--->EVT{last_evtID}')
                impm = ImpactPM(line=l)
                pmt = self.PMTs[impm.pmID]
                channelmap = pmt.channelmap
                impm.fill_panel_impacts(channelmap, nPM, self.zpos, minPlan)
                if impm.evtID == last_evtID:
                    for pid, imp in impm.impacts.items() : evt.impacts[pid] = imp
                    if i == len(lines)-1: pass #last line of file
                    else: continue
                #if new evtID, retrieve the last evtID and reconstruct it 
                #get coordinates 
                evt.get_xyz(in_mm=True, width=barwidths, zpos=self.zpos)

                iscut, tag = self.filter(evt)
                if iscut:
                    evt = self.reinit_evt(old_evt=evt, impact_pm=impm)
                    last_evtID = evt.ID
                    if i != len(lines)-1: self.nevt_tot += 1
                    continue
                else: 
                    # for _,imp in evt.impacts.items():
                    #     s ="{},{},{}".format(evt.ID,imp.panelID, ','.join(str(h.adc) for h in imp.hits))
                    #     ####Get counts of traversed panels
                    #     self.sel_signal[f'{evt.nimpacts}p'].append(s)
                    self.nsel [f'{evt.nimpacts}p'] += 1    
                try:
                    evt.get_time_of_flight()
                except: 
                    l_imp=list(impm.impacts.values())
                    raise ValueError(f"{file}\n{evt.ID}\nError 'evt.get_time_of_flight()'\nl_impacts{l_imp}\nl_z={[imp.zpos for imp in l_imp]}")            


                model = RansacModel(evt)
                
                if model.is_track_avail(): 

                    kwargs_model = {par : kwargs[par] for par in lparkeys}
                    model.get(**kwargs_model) 
                    
                    if model.is_valid() :
                        
        
                        model.get_df_track( self.zpos)

                        
                        xyz_inter = Intersection(model.model_robust, self.zpan ).xyz
                        xyz_track = np.around(xyz_inter,1)
                        
                        if kwargs['is_fit_intersect'] : 
                            xfin, yfin, _ = xyz_track.T
                        else : 
                            xyz_inliers = evt.xyz[model.inliers]
                            xyz_inliers_sort = xyz_inliers[xyz_inliers[:,-1].argsort()]
                            z_traversed = set(xyz_inliers_sort[:,2])
                            ix_near = np.array([np.argmin(np.array([ np.linalg.norm(xyz - xyz_track[xyz_track[:,2]==z]) for xyz in xyz_inliers_sort ]) ) if z in z_traversed else None  for z in self.zpan ])
                            close_xyz = np.array([ xyz_inliers_sort[ix] if ix is not None else np.zeros(3) for ix in ix_near ] ) 
                            xfin, yfin, _ = close_xyz.T
                            xfin[xfin==0.], yfin[yfin==0.] = xyz_track[np.where(xfin==0)[0], 0], xyz_track[np.where(yfin==0)[0], 1]


                        line = np.concatenate(([evt.ID, evt.gold, evt.timestamp.s, evt.timestamp.ns, evt.tof, evt.npts, evt.nimpacts, model.quadsumres], xfin, yfin, [nin, nout]), axis=0)
                        

                        out_matrix[i, :] = line
                        self.fill_df_inlier(model)
                        self.ntrack[f'{evt.nimpacts}p'] += 1     
                        
                evt = self.reinit_evt(old_evt = evt, impact_pm = impm)
                last_evtID = evt.ID
                if i != len(lines)-1: self.nevt_tot += 1                
            
            out_matrix =  out_matrix[~np.all( (out_matrix == 0.), axis=1)]
            #self.df_track  = self.df_track.append(pd.DataFrame(out_matrix, columns=headers))
            df_file = pd.DataFrame(out_matrix, columns=self.df_track.columns)
            self.df_track = pd.concat([self.df_track.astype(df_file.dtypes), df_file.astype(self.df_track.dtypes)]) #fix future pandas warning on concatenation
            if progress_bar : print_progress(n+1, nfiles, prefix = '\tFile(s) processed :', suffix = 'completed')
            n+=1
         ####format columns 
        self.format_columns()
        #self.df_inlier.set_index(['evtID', 'timestamp_s', 'timestamp_ns'], inplace=True)


    def fill_df_inlier(self, model:RansacModel):
        """Fill RANSAC inlier dataframe"""
        for i in range(len(model.inliers)):
            xyz, adc = model.xyz[i,:], model.adc[i,:]
            is_inl = model.inliers[i]
            evt = model.event
            timestamp_s, timestamp_ns = model.impacts[adc[2]].timestamp.s, evt.impacts[adc[2]].timestamp.ns
            df_tmp = pd.DataFrame(np.array([ [int(evt.ID), int(timestamp_s), int(timestamp_ns), int(is_inl), int(evt.gold), xyz[0], xyz[1], xyz[2], adc[0], adc[1]] ]), columns=self.col_inlier)
            self.df_inlier = pd.concat([self.df_inlier.astype(df_tmp.dtypes), df_tmp.astype(self.df_inlier.dtypes)]) #fix future pandas warning on contatenation
    
    

    '''    



    
    

  
    
    
  
 