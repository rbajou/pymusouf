# -*- coding: utf-8 -*-
#!/usr/bin/env python3

import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from mpl_toolkits.mplot3d.axes3d import Axes3D
from matplotlib.colors import Normalize
from scipy.interpolate import griddata,LinearNDInterpolator
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.integrate import dblquad, nquad
import time
from typing import Union
import os
import pickle

#package module(s)
from telescope import Telescope, DICT_TEL
from raypath import RayPath



class RayObject:
    '''
    Identify telescope rays that cross a given voxel ensemble in the scanned object
    '''

    def __init__(self):
        pass
