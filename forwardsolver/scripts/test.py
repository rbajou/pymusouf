#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from stoppingpower import StoppingPower
from telescope import dict_tel

if __name__ ==  "__main__":

    sp = StoppingPower(I=126)
    print(sp.par)
    print(dict_tel)