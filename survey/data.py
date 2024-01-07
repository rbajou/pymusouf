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
import io
#personal modules
from telescope import  Telescope



class DataType(Enum):
    real = auto()
    mc = auto()


class DataSet:
    @abstractmethod
    def dataset(self, dataset):
        pass

    

@dataclass
class BaseData : 
   
    path : Path


class RawData(BaseData):


    def __init__(self, path:Path):

        BaseData.__init__(self, path)      
        self.dataset = DataSet


    def __str__(self):

        return f"RawData {self.path} - nfiles : {len(self.dataset)}"


    def is_gz_file(self, filepath):

        with open(filepath, 'rb') as test_f:
            return test_f.read(2) == b'\x1f\x8b'


    def listdatafiles(self, path, fmt:str="dat", max_nfiles:int=None): 
            
        nfiles = len(os.listdir(path))
        if max_nfiles is not None: nfiles = max_nfiles 
        dataset = [f for n, f in enumerate(glob.glob(str(path) + f'/*{fmt}*')) if n < nfiles ]
       
        return dataset


    def fill_dataset(self, **args):

        if self.path.is_file():
            self.dataset = [self.path]
                   
        elif self.path.is_dir(): 
            self.dataset = self.listdatafiles(self.path, **args)
            
        else: raise Exception("Wrong 'path' object.")

    
    def readfile(self, file):
        
        datalines = list()
        
        try : 
            if self.is_gz_file(file):
                with gzip.open(f"{file}", 'rt') as f:
                    for l in f:
                        if l == "\n": continue
                        datalines.append(l) 
                        
            else : 
                with open(f"{file}", 'rt') as f :
                    for l in f:
                        if l == "\n": continue
                        datalines.append(l) 

        except OSError: 
            raise ValueError("Data files should be either .txt or .dat format, or in compressed gunzip form '.gz' ")

        return datalines
    
    

if __name__ == "__main__":

    path = Path("../rawdata_OM_calib")
    raw_data = RawData(path=path)
    raw_data.fill_dataset()


    path = Path("../data/BR/Calib10/rawdata")
    raw_data = RawData(path=path)
    raw_data.fill_dataset()

    # nlines=0
    # for file in raw_data.dataset:
    #     lines = raw_data.readfile(file)
    #     nlines += len(lines)

    # print(f"nlines = {nlines}")


"""
      
    def listFilesUrl(self, url:str):
        '''
        List files at 'url' to be fetched
        '''
        page = requests.get(url).text
        soup = BeautifulSoup(page, 'html.parser')
        listUrl = [url + '/' + node.get('href') for node in soup.find_all('a') if node.get('href').endswith('')]
        return listUrl


 
    def fetchFiles(self, nfiles:int, save_path:str):
        '''
        Fetch datafiles online at 'url'
        '''
        Path(save_path).mkdir(parents=True, exist_ok=True)   
         #with extension '.dat.gz'
        for i, url in enumerate(self.listFilesUrl()):
            if i > nfiles: break 
            file_basename = os.path.basename(url)
            if not os.path.isfile(os.path.join(save_path, "", file_basename)):
            #check if already existing file 
                file = requests.get(url)
                with open(os.path.join(save_path, "" ,file_basename), 'wb') as f:
                    f.write(file.content)
                f.close()
        else : pass 
   


"""