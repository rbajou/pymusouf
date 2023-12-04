#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 08 2021

@author: Raphaël Bajou
"""

import numpy as np
import matplotlib.pyplot as plt
import os 
from matplotlib.gridspec import GridSpec
import scipy.special as sc #binomial coef
import pylandau
import iminuit



def gaus(x,amp,x0,sigma):
    return amp*np.exp(-(x-x0)**2/(2*sigma**2))


def erf_func(x:np.ndarray, b:float, p:float,t:float,w:float):
    """
    b:lowest efficiency
    p:plateau value at high energy
    t:best estimate threshold position
    w:threshold width
    """
    func = b + p/2*(1+sc.erf( (x-t)/(np.sqrt(2)*w) ))
    return func

def fit_landau_migrad(x, y, p0, limit_mpv, limit_eta, limit_sigma, limit_A):
    def minimizeMe(mpv, eta, sigma, A):
        chi2 = np.sum(np.square(y - pylandau.langau(x, mpv, eta, sigma, A).astype(float)) / np.square(yerr.astype(float)))
        return chi2 / (x.shape[0] - 5)  # devide by NDF

    # Prefit to get correct errors
    yerr = np.sqrt(y)  # Assume error from measured data
    yerr[y < 1] = 1
    m = iminuit.Minuit(minimizeMe,
                       mpv=p0[0],
                       eta=p0[1],
                       sigma=p0[2],
                       A=p0[3],
    )
    m.limits['mpv'] = limit_mpv
    m.errors['mpv'] = 1
    m.limits['eta'] = limit_eta
    m.errors['eta'] = 0.1
    m.limits['sigma'] = limit_sigma
    m.errors['sigma'] = 0.1
    m.limits['A'] = limit_A
    m.errors['A'] = 1
    m.errordef = 1
    m.print_level = 2
    m.migrad()

    if not m._fmin.is_valid:
    #if not m.get_fmin().is_valid:
        raise RuntimeError('Fit did not converge')

    # Main fit with model errors
    yerr = np.sqrt(pylandau.langau(x,
                          mpv=m.values['mpv'],
                          eta=m.values['eta'],
                          sigma=m.values['sigma'],
                          A=m.values['A']))  # Assume error from measured data
    #yerr[y < 1] = 1

    m = iminuit.Minuit(minimizeMe,
                       mpv=m.values['mpv'],
                       eta=m.values['eta'],
                       sigma=m.values['sigma'],
                       A=m.values['A'],
    )
    m.limits['mpv'] = limit_mpv
    m.errors['mpv'] = 1
    m.limits['eta'] = limit_eta
    m.errors['eta'] = 0.1
    m.limits['sigma'] = limit_sigma
    m.errors['sigma'] = 0.1
    m.limits['A'] = limit_A
    m.errors['A'] = 1
    m.errordef = 1
    m.print_level = 2
    m.migrad()

    fit_values = m.values

    values = np.array([fit_values['mpv'],
                       fit_values['eta'],
                       fit_values['sigma'],
                       fit_values['A']])

    m.hesse()

    m.minos()
    minos_errors = m._merrors #m.get_merrors()

    if not minos_errors['mpv'].is_valid:
        print('Warning: MPV error determination with Minos failed! You can still use Hesse errors.')

    errors = np.array([(minos_errors['mpv'].lower, minos_errors['mpv'].upper),
                       (minos_errors['eta'].lower, minos_errors['eta'].upper),
                       (minos_errors['sigma'].lower, minos_errors['sigma'].upper),
                       (minos_errors['A'].lower, minos_errors['A'].upper)])

    return values, errors, m
