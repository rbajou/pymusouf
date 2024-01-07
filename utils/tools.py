#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from argparse import ArgumentError
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import SubplotSpec
from mpl_toolkits.mplot3d import axes3d
import sys
import os
import gzip
from scipy.interpolate import griddata
from scipy.interpolate import InterpolatedUnivariateSpline as spline
import glob
import pandas as pd
import argparse
import json


def print_progress(iteration, total, prefix='', suffix='', decimals=1, bar_length=100):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        bar_length  - Optional  : character length of bar (Int)
    """
    str_format = "{0:." + str(decimals) + "f}"
    percents = str_format.format(100 * (iteration / float(total)))
    filled_length = int(round(bar_length * iteration / float(total)))
    bar = '█' * filled_length + '-' * (bar_length - filled_length)

    sys.stdout.write('\r%s |%s| %s%s %s' % (prefix, bar, percents, '%', suffix)),

    if iteration == total:
        sys.stdout.write('\n')
    sys.stdout.flush()
    

def my_import(name):
    components = name.split('.')
    mod = __import__(components[0])
    for comp in components[1:]:
        mod = getattr(mod, comp)
    return mod


def create_subtitle(fig: plt.Figure, grid: SubplotSpec, title: str):
    "Sign sets of subp with title"
    row = fig.add_subplot(grid)
    # the '\n' is important
    row.set_title(f'{title}\n', fontweight='semibold')
    # hide subplot
    row.set_frame_on(False)
    row.axis('off')

def list_files(dir:str, type:str, ext:str, prefix)->list:
    list_files= [f for f in glob.glob(os.path.join(dir, f"*/{prefix}*/*{type}.{ext}")) ]
    if len(list_files)==0: print("No file found.")
    print(f"merge nfiles={len(list_files)}")
    return list_files

def merge_files(recodir:str, filename:str, type:str, ext:str, prefix:str="out")->str:
    chunck_files = list_files(recodir, type, ext, prefix)
    print(f"nfiles_to_be_merged={len(chunck_files)}\nfirst10:\n{chunck_files[:10]}\nlast10:\n{chunck_files[-10:]}")
    outfile = os.path.join(recodir,"", f'{filename}_{type}.{ext}')
    if ext=='csv.gz':
        df = pd.concat([pd.read_csv(f, delimiter="\t", index_col=0) for f in chunck_files])
        df.to_csv(outfile,  sep='\t') #index=False,
    elif ext=='json.gz': 
        merge_dict(files=chunck_files, outfile=outfile, compression=True)            
    else: raise ArgumentError("Unknown file ext (need to modify the merge_files() function).")
    
    return outfile

def merge_dict(files:list, outfile:str, compression:bool=True):
    """with same keys"""
    ds = [ ]
    dout  = {}
    for file in files:
        if compression:
            with gzip.open(file, 'r') as fin:
                d = json.loads(fin.read().decode('utf-8'))
        else: 
            with open(file, 'r') as f: 
                d = json.loads(f.read())
        ds.append( d )
    for k in ds[0].keys():
        dout[k] = np.concatenate(list(d[k] for d in ds)).tolist()#tuple(d[k] for d in ds)
    
    if compression:
        with gzip.open(outfile, 'w') as fout:
            fout.write(json.dumps(dout).encode('utf-8'))                       
    else:
        with open(outfile, 'w') as fout:
            fout.write(json.dumps(dout))
   
    
        
def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def print_progress(iteration, total, prefix='', suffix='', decimals=1, bar_length=100):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        bar_length  - Optional  : character length of bar (Int)
    """
    str_format = "{0:." + str(decimals) + "f}"
    percents = str_format.format(100 * (iteration / float(total)))
    filled_length = int(round(bar_length * iteration / float(total)))
    bar = '█' * filled_length + '-' * (bar_length - filled_length)

    sys.stdout.write('\r%s |%s| %s%s %s' % (prefix, bar, percents, '%', suffix)),

    if iteration == total:
        sys.stdout.write('\n')
    sys.stdout.flush()



def forceAspect(ax,aspect=1):
    """Axis size ratio for pcolor"""
    im = ax.get_images()
    
    xlim , ylim = ax.get_xlim(), ax.get_ylim()
    extent =  [xlim[0],xlim[1],ylim[0],ylim[1] ]#]im[0].get_extent()
  
    ax.set_aspect( (abs( (extent[1]-extent[0])/(extent[3]-extent[2]) )/aspect)*10)


def var_wt(x, sigma_i, thresh=1e-6, iter_max=200): #thresh=1e-6
    sigma_i[sigma_i == 0] =  len(x) #no measurement error
    tol = np.sum(abs(x))*10**-16
    sigma_i[sigma_i <= 0] = np.min( np.concatenate(([tol], sigma_i[sigma_i > 0])) ) / 2
    
    # Initialize
    v = np.std(x)**2
    n = len(x)
    S = np.mean(sigma_i**2) # Avoids recomputing this constant within the loop
    w = 1/np.sum(1/(sigma_i**2+np.sqrt(v)**2))*(1/(sigma_i**2+np.sqrt(v)**2))
    for i in range(iter_max) :
        w = 1/(v + sigma_i**2)
        w = w / np.sum(w)
        m = np.sum(x * w)
        v_0 = max(0, np.sum((x-m)**2 * w) * n / (n-1) - S) 
        if ((v <= v_0 * (1+thresh)) & (v_0 <= v * (1+thresh))) : break
        v = v_0
        w = 1/np.sum(1/(sigma_i**2+np.sqrt(v)**2))*(1/(sigma_i**2+np.sqrt(v)**2))

    return {'variance':v, 'mean':m, 'iterations':i, 'weights':w}


def pretty(d, indent=0):
   for key, value in d.items():
      print('\t' * indent + str(key))
      if isinstance(value, dict):
         pretty(value, indent+1)
      else:
         print('\t' * (indent+1) + str(value))


def find_nearest(array:np.ndarray, value:float):
    """
    find nearest element in 'array' to 'value'
    """
    if np.isnan(value):
        return np.nan, np.nan
    idx = np.nanargmin((np.abs(array - value)))
    return idx, array[idx]

class ErrorPropagationSpline(object):
    """
    Does a spline fit, but returns both the spline value and associated uncertainty.
    """
    def __init__(self, x, y, z,  yerr, N=1000, *args, **kwargs):
        """
        See docstring for InterpolatedUnivariateSpline
        """
        yy = np.vstack([y + np.random.normal(loc=0, scale=yerr) for i in range(N)]).T
        self._splines = [spline(x, yy[:, i], *args, **kwargs) for i in range(N)]

    def __call__(self, x, *args, **kwargs):
        """
        Get the spline value and uncertainty at point(s) x. args and kwargs are passed to spline.__call__
        :param x:
        :return: a tuple with the mean value at x and the standard deviation
        """
        x = np.atleast_1d(x)
        s = np.vstack([curve(x, *args, **kwargs) for curve in self._splines])
        return (np.mean(s, axis=0), np.std(s, axis=0))


# intersection function
def isect_line_plane_v3(p0, p1, p_co, p_no, xyrange=[[0,800], [0,800]], epsilon=1e-6):
    """
    p0, p1: Define the line.
    p_co, p_no: define the plane:
        p_co Is a point on the plane (plane coordinate).
        p_no Is a normal vector defining the plane direction;
             (does not need to be normalized).

    Return a Vector or None (when the intersection can't be found).
    """

    u = sub_v3v3(p1, p0)
    dot = dot_v3v3(p_no, u)

    if abs(dot) > epsilon:
        # The factor of the point between p0 -> p1 (0 - 1)
        # if 'fac' is between (0 - 1) the point intersects with the segment.
        # Otherwise:
        #  < 0.0: behind p0.
        #  > 1.0: infront of p1.
        w = sub_v3v3(p0, p_co)
        fac = -dot_v3v3(p_no, w) / dot
        u = mul_v3_fl(u, fac)
        res = add_v3v3(p0, u)
        in_range = (min(xyrange[0]) < res[0]) & (res[0] < max(xyrange[0])) & (min(xyrange[1]) < res[1]) & (res[1] < max(xyrange[1]))
        if not in_range: return np.zeros(3)
        else : return res

    # The segment is parallel to plane.
    return np.zeros(3)

# ----------------------
# generic math functions

def add_v3v3(v0, v1):
    return np.array([
        v0[0] + v1[0],
        v0[1] + v1[1],
        v0[2] + v1[2]])


def sub_v3v3(v0, v1):
    return (
        v0[0] - v1[0],
        v0[1] - v1[1],
        v0[2] - v1[2],
    )


def dot_v3v3(v0, v1):
    return (
        (v0[0] * v1[0]) +
        (v0[1] * v1[1]) +
        (v0[2] * v1[2])
    )


def len_squared_v3(v0):
    return dot_v3v3(v0, v0)


def mul_v3_fl(v0, f):
    return (
        v0[0] * f,
        v0[1] * f,
        v0[2] * f,
    )


def predict(x, axis=0, params=None):
        """Predict intersection of the estimated line model with a hyperplane
        orthogonal to a given axis.
        Parameters
        ----------
        x : (n, 1) array
            Coordinates along an axis.
        axis : int
            Axis orthogonal to the hyperplane intersecting the line.
        params : (2, ) array, optional
            Optional custom parameter set in the form (`origin`, `direction`).
        Returns
        -------
        data : (n, m) array
            Predicted coordinates.
        Raises
        ------
        ValueError
            If the line is parallel to the given axis.
        """
        if params is None:
            raise ValueError('Parameters cannot be None')
        if len(params) != 2:
            raise ValueError('Parameters are defined by 2 sets.')

        origin, direction = params

        if direction[axis] == 0:
            # line parallel to axis
            raise ValueError('Line parallel to axis %s' % axis)

        l = (x - origin[axis]) / direction[axis]
        data = origin + l[..., np.newaxis] * direction
        return data


def fill_empty_pixels(maps:dict, X:dict, Y:dict, outdir:str, filename:str="", mask:dict=None, *args, **kwargs):
    """Return dict with filled empty pixels with interpolated values"""
    new_maps= maps.copy()
    for k,v in  new_maps.items(): 
        x, y= X[k], Y[k]
        nnan = ((~np.isnan(v)) & (v!=0.))
        points = np.zeros(shape=(len(v[nnan].flatten()),2))
        points[:,0] = x[nnan].flatten()#[:,0]= flux_tomo[~np.isnan(flux_tomo)]
        points[:,1]= y[nnan].flatten()
        values = v[nnan].flatten()
        grid_v = griddata(points, values, (x, y), *args, **kwargs)
        np.savetxt(os.path.join(outdir, f'{filename}_interp2D_{k}.txt'), grid_v, fmt='%.5e')
        v[~nnan] = grid_v[~nnan]
        if mask is not None: v[mask[k]] = np.nan
    return new_maps



def adjust_lightness(color, amount=0.3):
    """
    Lightens the given color by multiplying (1-luminosity) by the given amount.
    Input can be matplotlib color string, hex string, or RGB tuple.

    Examples:
    >> lighten_color('g', 0.3)
    >> lighten_color('#F034A3', 0.6)
    >> lighten_color((.3,.55,.1), 0.5)
    """
    import matplotlib.colors as mc
    import colorsys
    try:
        c = mc.cnames[color]
    except:
        c = color
    c = colorsys.rgb_to_hls(*mc.to_rgb(c))
    return colorsys.hls_to_rgb(c[0], max(0, min(1, amount * c[1])), c[2])

class MyAxes3D(axes3d.Axes3D):
    '''
    https://stackoverflow.com/questions/15042129/changing-position-of-vertical-z-axis-of-3d-plot-matplotlib
    '''
    def __init__(self, baseObject, sides_to_draw):
        self.__class__ = type(baseObject.__class__.__name__,
                              (self.__class__, baseObject.__class__),
                              {})
        self.__dict__ = baseObject.__dict__
        self.sides_to_draw = list(sides_to_draw)
        self.mouse_init()

    def set_some_features_visibility(self, visible):
        for t in self.zaxis.get_ticklines() + self.zaxis.get_ticklabels():
            t.set_visible(visible)
        self.zaxis.line.set_visible(visible)
        self.zaxis.pane.set_visible(visible)
        self.zaxis.label.set_visible(visible)

    def draw(self, renderer):
        # set visibility of some features False 
        self.set_some_features_visibility(False)
        # draw the axes
        super(MyAxes3D, self).draw(renderer)
        # set visibility of some features True. 
        # This could be adapted to set your features to desired visibility, 
        # e.g. storing the previous values and restoring the values
        self.set_some_features_visibility(True)

        zaxis = self.zaxis
        draw_grid_old = zaxis.axes._draw_grid
        # disable draw grid
        zaxis.axes._draw_grid = False

        tmp_planes = zaxis._PLANES

        if 'l' in self.sides_to_draw :
            # draw zaxis on the left side
            zaxis._PLANES = (tmp_planes[2], tmp_planes[3],
                             tmp_planes[0], tmp_planes[1],
                             tmp_planes[4], tmp_planes[5])
            zaxis.draw(renderer)
        if 'r' in self.sides_to_draw :
            # draw zaxis on the right side
            zaxis._PLANES = (tmp_planes[3], tmp_planes[2], 
                             tmp_planes[1], tmp_planes[0], 
                             tmp_planes[4], tmp_planes[5])
            zaxis.draw(renderer)

        zaxis._PLANES = tmp_planes

        # disable draw grid
        zaxis.axes._draw_grid = draw_grid_old

def wrapToPi(x):
    xwrap = np.remainder(x, 2 * np.pi)
    mask = np.abs(xwrap) > np.pi
    xwrap[mask] -= 2 * np.pi * np.sign(xwrap[mask])
    mask1 = x < 0
    mask2 = np.remainder(x, np.pi) == 0
    mask3 = np.remainder(x, 2 * np.pi) != 0
    xwrap[mask1 & mask2 & mask3] -= 2 * np.pi
    return xwrap


if __name__=="__main__":
    
    pass