import numpy as np
from pathlib import Path
from scipy.io import loadmat

from raypath import RayPath
from survey import CURRENT_SURVEY, DICT_SURVEY

survey = CURRENT_SURVEY
# filename_grid = "copahueStructure.npy" #res 5m
#dem_path = sur.path / "dem"
#grid = np.load(dem_path / filename_grid)
#xy_center = np.loadtxt(dem_path / "volcanoCenter.txt").T

fcop = survey.path / "dem" / "copahueStructure.npy"
# cop_grid = loadmat(fcop)
# print(f'cop_grid = {cop_grid}')

import pickle

# fpkl_cop = "/opt/homebrew/lib/python3.11/site-packages/files/survey/copahue/telescope/COP/raypath/az262.0_elev4.0/raypath.pkl"
# with open(fpkl_cop, 'rb') as f: 
#     frp = pickle.load(f)

tel = survey.telescopes['COP']
rp = RayPath(telescope=tel, surface_grid=survey.surface_grid)
fout = survey.path / 'telescope' / tel.name / 'raypath'/ f'az{tel.azimuth:.1f}_elev{tel.elevation:.1f}' / 'raypath'
rp(file=fout, max_range=1500)

souf_survey = DICT_SURVEY['soufriere']
souf_grid = souf_survey.surface_grid 
# print(f'souf_grid = {souf_grid.shape}')

