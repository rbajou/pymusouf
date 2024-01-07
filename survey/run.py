# -*- coding: utf-8 -*-
#!/usr/bin/env python3

import numpy as np
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Union
from enum import Enum, auto
import yaml
#package module(s)
from .data import RawData
from telescope import Telescope



class RunType(Enum):
    
    calib = auto()
    tomo = auto()


@dataclass
class Run:
   
    name : str
    telescope: Telescope
    rawdata : List[RawData] = field(default_factory=lambda : list())
    
    def __str__(self): 
        sout = f"Run: {self.name}"
        sout += f"\n - {self.telescope}"
        return sout

    def check_rawpath(self, datapath:dict) -> List[Path]:
        
        path = []

        try : 
           
            dpraw = datapath['rawdata']
           
            if dpraw is not None: 
                for p in dpraw : 
                    p = Path(p)
                    if p.exists(): path.append(p)
                    else : print(f"Raw path {p} was not found.")
                
                #if len(path) == 1:  path = path[0]
        
        except KeyError as e: 
            print(f"Path(s) for run '{self.name}' rawdata not provided.")

        return path


    def check_recopath(self, datapath:dict):
        
        path = []

        try : 
           
            dpreco = datapath['recodata']
           
            if dpreco is not None: 
                for p in dpreco : 
                    p = Path(p)
                    if p.exists(): path.append(p)
                    else : print(f"Reco path {p} was not found.")
                
                if len(path) == 1:  path = path[0]
                
            else : print(f'{dpreco} is NoneType')
       
        except KeyError as e: 
            print(f"Path(s) for run '{self.name}' recodata not provided.")
        
        return path

    def edit_datapath(self, ):
        pass


class RunCalib(Run):


    def __init__(self, **kwargs):
        Run.__init__(self, **kwargs)
 

class RunTomo(Run):


    def __init__(self, **kwargs):    
        Run.__init__(self,  **kwargs)
    
    def check_orentation_and_position(self, elevation:float=None, azimuth:float=None, utm:np.ndarray=None):
        if azimuth is not None : 
            self.telescope.azimuth = azimuth 
        if elevation is not None : 
            self.telescope.elevation = elevation 
        if utm is not None :
            self.telescope.utm = utm 
            self.telescope.altitude = utm[-1]


class RunSurvey:


    def __init__(self, telescope:Telescope, file_yaml:Union[str,Path]=None):
        
        self.telescope = telescope
        self.run_tomo = []
        self.run_calib = []
        self.file_yaml = file_yaml
    

    def __setitem__(self, run:Union[Run, RunTomo, RunCalib]):
        
        if isinstance(run, RunTomo): self.run_tomo.append( run)
        elif isinstance(run, RunCalib): self.run_calib.append(run)
        else: raise ValueError('Wrong type of run object')


    def __getitem__(self, name):

        if name in self.run_tomo.keys():
            return self.run_tomo[name]
        elif name in self.run_calib.keys():
            return self.run_calib[name]
        else : raise KeyError('Wrong key for run')
    

    def __str__(self) -> str:
       
        sout = f"Run(s) tomo: \n\n - "+ f"\n - ".join(v.__str__() for _,v in self.run_tomo.items())
        sout += "\n"
        sout += f"Run(s) calib: \n\n - "+ f"\n - ".join(v.__str__() for _,v in self.run_calib.items())
        return sout
   

    def get_runs(self, runtype:List[RunType], **kwargs_tomo) -> dict:
        
        with open( str(self.file_yaml) ) as fyaml:
            try:
                # The FullLoader parameter handles the conversion from YAML
                # scalar values to Python the dictionary format
                content = yaml.load(fyaml, Loader=yaml.SafeLoader)
            except yaml.YAMLError as exc:
                print(exc) 


        self.run_path = {}
        for rt in runtype : 
            
            try :         
                self.run_path[rt._name_] = content['run'][rt._name_]
                
            except KeyError as e: 
                print(f"Path(s) for run type '{rt._name_}' not provided.")
                return

            for name, datapath in self.run_path[rt._name_].items():

                if len(datapath ) == 0 :
                    print(f"Path(s) for run name '{name}' not provided.")
                    return

                else : 

                    kwargs_run = dict(name=name, telescope=self.telescope)

                    if rt == RunType.tomo : 
                        run  = RunTomo(**kwargs_run)
                        run.check_orentation_and_position(**kwargs_tomo)
                    elif rt == RunType.calib: 
                        run  = RunCalib(**kwargs_run)
                    else: 
                        raise ValueError('Wrong type of run object')

                    lpath = run.check_rawpath(datapath)
                    
                    for path in lpath : 
                        raw = RawData(path=path)
                        raw.fill_dataset()
                        run.rawdata.append(raw)


                    self.__setitem__(run)

        return