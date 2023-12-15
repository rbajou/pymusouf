# -*- coding: utf-8 -*-
#!/usr/bin/env python3

from dataclasses import dataclass, field
from pathlib import Path
from typing import List
from enum import Enum, auto
#package module(s)
from telescope import Telescope, dict_tel

class RunType(Enum):
    TOMO = auto()
    CALIB = auto()

@dataclass
class Run:
    name : str
    telescope: Telescope
    type : RunType 
    start : str = field(default='')
    end : str = field(default='')
    def __str__(self): 
        sout = f"Run: {self.name}"
        if self.start and self.end : sout += f"\t(start {self.start} - end {self.end})" 
        sout += f"\n - {self.telescope}"
        return sout


class Survey: 
    def __init__(self, name:str):
        self.name = name
        self.runs = {}

    def __setitem__(self, name:str, run:Run):
        self.runs[name] = run

    def __getitem__(self,name):
        return self.runs[name]

    def __str__(self): return f"\nSurvey: {self.name}\n\n - "+ f"\n - ".join(v.__str__() for _,v in self.runs.items())

SoufSurvey = Survey(name='Soufrière')
SoufSurvey['SB'] = Run('Tomo-3dat', dict_tel['SB'], RunType.TOMO)
SoufSurvey['SNJ'] = Run('Tomo2', dict_tel['SNJ'], RunType.TOMO)

if __name__=="__main__":
    print(SoufSurvey)