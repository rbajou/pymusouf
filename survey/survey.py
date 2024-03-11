# -*- coding: utf-8 -*-
#!/usr/bin/env python3

import numpy as np
from pathlib import Path
from typing import List, Union
import glob

#package module(s)
from config import MAIN_PATH, SURVEY_DIR, LIST_AVAIL_SURVEY, CURRENT_SURVEY_NAME
from .run import RunType, RunSurvey
from telescope import DICT_TEL


class Survey: 
   
    def __init__(self, name:str=None, path:Union[str, Path]=None):
        self.name = name
        self.path = path
        self.runs = {}
        self.telescopes = {}
        self.dem_file = Union[Path, str]
        self.surface_grid = np.ndarray #shape : (m, n, 3)

    def __setitem__(self, name:str, run:RunSurvey): 
        self.runs[name] = run
        self.telescopes[name] = run.telescope

    def __getitem__(self,name:str): 
        run = self.runs[name]
        return run

    def __str__(self): 
        sout = f"\nSurvey: {self.name}\n\n - "+ f"\n - ".join(v.__str__() for _,v in self.runs.items())
        return sout

    def set_surface_grid(self, grid:np.ndarray, xy_center:np.ndarray=None): 
        """_summary_

        Args:
            grid (np.ndarray): surface grid shape (3, m, n)
            xy_center (np.ndarray): (2,)
        """
        self.surface_grid = grid 

        if xy_center is None: 
            s = grid.shape
            mx, my = s[0]//2, s[1]//2
            xy_center = np.array([grid[0, mx, my], grid[1, my, mx]])

        self.surface_center = xy_center

DICT_TELNAME_SURVEY = {sur : [t.split('/')[-1] for t in  glob.glob( str(SURVEY_DIR / sur / 'telescope') + '/**' ) ]  
                       for sur in LIST_AVAIL_SURVEY}

DICT_SURVEY = {sur_name : Survey() for sur_name in LIST_AVAIL_SURVEY}



for sur_name, ltel_name  in DICT_TELNAME_SURVEY.items() : 
    
    for tel_name in ltel_name:
        tel = DICT_TEL[tel_name]
        survey_path =  SURVEY_DIR / sur_name
        DICT_SURVEY[sur_name].path = survey_path
        fyaml = survey_path / "telescope" / tel_name / "run.yaml"
        kwargs = {'telescope' : tel, 'file_yaml' : fyaml}
        run_survey = RunSurvey(**kwargs) 
        run_survey.get_runs([RunType.calib, RunType.tomo])
        DICT_SURVEY[sur_name][tel_name] = run_survey

souf_survey = DICT_SURVEY['soufriere']
dem_path = souf_survey.path / "dem"
filename_grid = "soufriereStructure_2.npy" #res 5m
souf_survey.dem_file = dem_path / filename_grid
souf_grid = np.load(dem_path / filename_grid)
souf_xy_center = np.loadtxt(dem_path / "volcanoCenter.txt").T
souf_survey.set_surface_grid(grid = souf_grid, xy_center = souf_xy_center)


cop_survey = DICT_SURVEY['copahue']
dem_path = cop_survey.path / "dem"
filename_grid = "copahueStructure.npy" #res 1m
cop_survey.dem_file = dem_path / filename_grid
cop_grid = np.load(dem_path / filename_grid)
cop_survey.set_surface_grid(grid = cop_grid)

CURRENT_SURVEY = DICT_SURVEY[CURRENT_SURVEY_NAME]

if __name__=="__main__":
    pass
    
    