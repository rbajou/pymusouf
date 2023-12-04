#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import numpy as np
from dataclasses import dataclass, field
from typing import List, Union

@dataclass 
class Material:
    name: str 
    A: float 
    Z: float
    rho: float
    I : float
    fraction:float=field(init=False)
    weight:float=field(init=False)
    def __post_init__(self):
        self.Z_A = self.Z/self.A
        self.plasma_energy = 28.816 * np.sqrt(self.rho*self.Z_A)

    def __iter__(self):
        yield 'name', self.name
        yield 'A', self.A
        yield 'Z', self.Z
        yield 'rho', self.rho
        yield 'I', self.I

@dataclass 
class Element:
    name: str 
    symbol: str
    A: float 
    Z: float
    rho: float
    I : float
    fraction:float=field(init=False)
    weight:float=field(init=False)
    def __post_init__(self):
        self.Z_A = self.Z/self.A
        self.plasma_energy = 28.816 * np.sqrt(self.rho*self.Z_A)
        
    def __iter__(self):
        yield 'name', self.name
        yield 'A', self.A
        yield 'Z', self.Z
        yield 'rho', self.rho
        yield 'I', self.I

@dataclass 
class Mineral:
    name : str
    elements: List[Element]=field(default_factory=lambda:None)
    coefs : List[int]=field(default_factory=lambda:None)
    rho: float=field(default_factory=lambda:None)
    A: float =field(init=False)
    Z: float=field(init=False)
    I : float=field(init=False)
    def __post_init__(self):
        if len(self.coefs) != len(self.elements) : raise ValueError
        if self.elements is None: return 
        self.fractions = self.coefs/np.sum(self.coefs)
        self.formula = "".join([f"{e.symbol}{c}" if c!=1 else f"{e.symbol}" for e, c in zip(self.elements, self.coefs)])
        #Atot = np.sum(e.A for e in self.elements)
        self.A = np.sum([ c*e.A for c,e in  zip(self.coefs, self.elements)])
        self.weights = np.array( [c*e.A/ self.A  for c,e in  zip(self.coefs, self.elements)])
        self.Z = np.sum([ w*e.Z for w,e in  zip(self.weights, self.elements)])#average molecular mass
        Z_A = [e.Z/e.A for e in self.elements]
        self.Z_A = np.sum([w*e.Z/e.A for w, e in zip(self.weights, self.elements)])
        #self.Z_A = np.sum([w*e.Z/e.A for w, e in zip(self.weights, self.elements)])
        Na = 6.02*1e23 #Avogadro
        ###crystallographic parameters, look in Strunz and Nickel, 2001 repris dans Lechmann,2018
        a,b,c= 1,1,1#in Angstroms
        alpha,beta,gamma=1,1,1 #in rad internal angles 
        ####
        Vunit_cell = a*b*c*np.sqrt(1  + 2*np.cos(alpha)*np.cos(beta)*np.cos(gamma) - np.cos(alpha)**2-np.cos(beta)**2-np.cos(gamma)**2) 
        Q = 1 #number of formula units per unit cell
        if self.rho is None: self.rho = Q*self.A / (Na*Vunit_cell)#Borchardt-Ott,2009 repris dans Lechmann,2018 #np.sum([ f*e.rho for f,e in zip(self.fractions, self.elements)])
        I =  [ 16*e.Z**0.9 for e in self.elements ]
        #self.I = np.exp(np.sum(self.fractions*self.Z*np.log(I))/np.sum(self.fractions*self.Z)) 
        self.I = 1/np.sum( [ wj * Z_Aj for wj , Z_Aj in zip(self.weights, Z_A) ] ) * np.prod([Ii**(wi*Z_Ai) for Ii, wi, Z_Ai in zip(I, self.weights, Z_A)]) 
        self.plasma_energy = 28.816 * np.sqrt(self.rho*self.Z_A)
@dataclass 
class MixMedium:
    materials : Union[List[Material], List[Element], List[Mineral]]
    fractions : list
    name : str = field(default_factory=lambda:"")
    def __post_init__(self):
        if len(self.fractions) != len(self.materials) : raise ValueError
        for mi,fi in zip(self.materials,self.fractions) : mi.fraction = fi
        if self.name == "" : self.name = "_".join([f"{mi.fraction}{mi.name}" for mi in self.materials]) 
        self.fractions, self.A,self.Z, self.rhom = np.transpose([[mi.fraction, mi.A, mi.Z, mi.rho] for mi in self.materials])
        self.weights= [ mi.fraction*mi.A  for mi in self.materials  ]
        self.weights /= np.sum(self.fractions*self.A)
        for mi,wi in zip(self.materials,self.weights) : mi.weight = wi
        self.Z_A = np.sum([wi*mi.Z_A for wi, mi in zip(self.weights,self.materials)])
        self.rho = np.sum([mi.fraction *mi.rho for mi in self.materials])
        I =  [ 16*Zi**0.9 for Zi in self.Z ]
        norm = np.sum(self.weights*self.Z/self.A)
        self.I = np.exp(np.sum(self.weights * self.Z/self.A * np.log(I))/norm)
        #self.I = np.exp(np.sum([f * m.rho * m.Z_A * np.log(m.I) for f, m in zip(self.fractions, self.materials)])/ np.sum([f * m.rho * m.Z_A  for f, m in zip(self.fractions, self.materials)] )  )
        self.plasma_energy = np.exp( np.sum([wi * Zi/Ai * np.log(28.816*np.sqrt(rhoi*Zi/Ai)) for wi, Zi, Ai, rhoi in zip(self.weights, self.Z,self.A, self.rhom ) ]  / norm) )
        
    def __str__(self):
        return f"{self.name}:\nrho={self.rho}g/cm^3\nI={self.I:.3f}eV\nZ/A={self.Z_A:.3f}\nZ,A={self.Z},{self.A}"
      
    def __iter__(self):
        yield 'name', self.name
        yield 'A', self.A
        yield 'Z', self.Z
        yield 'Z/A', self.Z_A
        yield 'rho', self.rho
        yield 'I', self.I
        
@dataclass 
class RockMedium:
    materials : Union[List[Material], List[Element], List[Mineral]]
    fractions : list #mineral volumetric fractions in rock
    name : str = field(default_factory=lambda:"")
    def __post_init__(self):
        if len(self.fractions) != len(self.materials) : raise ValueError
        for mi,fi in zip(self.materials,self.fractions) : mi.fraction = fi
        if self.name == "" : self.name = "_".join([f"{mi.fraction}{mi.name}" for mi in self.materials]) 
        self.fractions, self.A,self.Z, self.rhom = np.transpose([[mi.fraction, mi.A, mi.Z, mi.rho] for mi in self.materials])
        self.weights= [ mi.fraction*mi.A  for mi in self.materials  ]
        self.weights /= np.sum(self.fractions*self.A)
        for mi,wi in zip(self.materials,self.weights) : mi.weight = wi
        self.Z_A = np.sum([wi*mi.Z_A for wi, mi in zip(self.weights,self.materials)])
        self.rho = np.sum([mi.fraction *mi.rho for mi in self.materials])
        I =  [ 16*Zi**0.9 for Zi in self.Z ]
        norm = np.sum(self.weights*self.Z/self.A)
        self.I = np.exp(np.sum(self.weights * self.Z/self.A * np.log(I))/norm)
        #self.I = np.exp(np.sum([f * m.rho * m.Z_A * np.log(m.I) for f, m in zip(self.fractions, self.materials)])/ np.sum([f * m.rho * m.Z_A  for f, m in zip(self.fractions, self.materials)] )  )
        self.plasma_energy = np.exp( np.sum([wi * Zi/Ai * np.log(28.816*np.sqrt(rhoi*Zi/Ai)) for wi, Zi, Ai, rhoi in zip(self.weights, self.Z,self.A, self.rhom ) ]  / norm) )
        
    def __str__(self):
        return f"{self.name}:\nrho={self.rho}g/cm^3\nI={self.I:.3f}eV\nZ/A={self.Z_A:.3f}\nZ,A={self.Z},{self.A}"
      
    def __iter__(self):
        yield 'name', self.name
        yield 'A', self.A
        yield 'Z', self.Z
        yield 'Z/A', self.Z_A
        yield 'rho', self.rho
        yield 'I', self.I
        
    
#for densities 'rho' [g/cm^3] and mean excitation energies 'I' [eV] : https://www.physics.nist.gov/cgi-bin/Star/compos.pl?matno=001  
#for atomic mass 'A' [amu] and atomic number 'Z' : https://www.periodic-table.org/
Rock = Material(name="rock", A=22,Z=11,rho=2.65,I=138.476)
Water = Material(name="water", A=13.37,Z=7.42,rho=1.0,I=78.4)
H = Element(name="hitrogen", symbol="H",A=2.,Z=1,rho=0.0899,I=19.2)
N=Element(name="nitrogen", symbol="N",A=14.00674,Z=7,rho=1.165e-3,I=82.)
O=Element(name="oxygen", symbol="O",A=16,Z=8,rho=1.332e-3,I=95.0)
Ar=Element(name="argongas", symbol="Ar",A=39.948,Z=18,rho=1.662e-3,I=188.)
C= Element(name="carbon", symbol="C",A=12,Z=6,rho=1.700,I=78.0)
Si = Element(name="silicon", symbol="Si", A=28.0855, Z=14, rho=2.33,I=173) 
Ca = Element(name="calcium", symbol="Ca",A=40.078, Z=20, rho=1.55,I=191) 
Na = Element(name="sodium", symbol="Na",A=22.9897, Z=11, rho=0.968,I=149) 
Al = Element(name="aluminium", symbol="Al", A=26.9815, Z=13, rho=2.7,I=166) 
Mg = Element(name="magnesium", symbol="Mg", A=24.305, Z=12, rho=1.738,I=156) 
K = Element(name="potassium", symbol="K", A=39.0983, Z=19, rho=0.856,I=190) 
Ti = Element(name="titanium", symbol="Ti", A=47.867, Z=	22, rho=4.507,I=233) 
Fe = Element(name="iron", symbol="Fe", A=55.845, Z=26, rho=7.874,I=286) 
F = Element(name="fluorine", symbol="F", A=18.9984, Z=9, rho=1.696,I=115) 

Air = MixMedium([N, O, Ar, C], [0.755267, 0.231781, 0.012827, 0.000124]) #1 atm

#minerals
#andesite
Qtz = Mineral(name="quartz", elements=[Si, O], coefs=[1,2], rho=2.2) #2.2-2.65


Ab =  Mineral(name="albite", elements=[Na, Ca, Al, Si, O], coefs=[0.95, 0.05,1.05,2.95,8], rho=2.63) #http://webmineral.com/data/Albite.shtml
An =  Mineral(name="anorthite", elements=[Na, Ca, Al, Si, O], coefs=[0.05,0.95, 1.95, 2.05, 8], rho=2.75)  #http://webmineral.com/data/Anorthite.shtml
Phl =  Mineral(name="phlogopite", elements=[K,Mg,Al,Si,O,F,H], coefs=[1,3,1,3,11,1,1], rho=2.83) #http://webmineral.com/data/Phlogopite.shtml
Ann =  Mineral(name="annite", elements=[K,Fe,Al,Si,O,H,F], coefs=[1,3,1,3,11.5,1.5,0.5], rho=3.34) #http://webmineral.com/data/Annite.shtml
Mg_Hbl =  Mineral(name="magnesium hornblende", elements=[Ca,Mg,Al,Fe,Si,O,H], coefs=[2,4,1.75,0.25,7,24,2], rho=2.96) #http://webmineral.com/data/Magnesiohornblende.shtml
Fe_Hbl =  Mineral(name="iron hornblende", elements=[Ca,Fe,Al,Si,O,H], coefs=[2,4.25,1.75,7,24,2], rho=3.38) #http://webmineral.com/data/Ferrohornblende.shtml
Aug =  Mineral(name="augite", elements=[Na, C, Mg, Ti, Al, Fe], coefs=[1,2, 1, 1, 1,1], rho=3.51) #http://webmineral.com/data/Augite.shtml
Or =  Mineral(name="orthoclase", elements=[K, Al, Si, O], coefs=[1,1,3,8], rho=2.55) 
#igneous rocks
Andesite = RockMedium(name="andesite", materials=[Qtz,Ab,An,Phl,Ann,Mg_Hbl,Fe_Hbl,Aug], fractions=np.array([11.7,37.7,25.3,4.5,2.1,4.2,6.4,8.1])*0.01) #Lechmann2018
Andesite.rho = 2.812 #Lechmann 2018
Andesite.Z_A = 0.4960 #Lechmann 2018
Andesite.I = 147.77 #Lechmann 2018
Granite = RockMedium(name="granite", materials=[Qtz,Or,Ab,Phl,Ann,Mg_Hbl,Fe_Hbl], fractions=np.array([36.1, 28.1,27.3,2.95,2.95,2.25,2.25])*0.01) #Lechmann2018


Rock.I = 136.40 #Lechmann 2018

#sedimentary rocks
Cal = Mineral(name="calcite", elements=[Ca,C,O], coefs=[1,1,3], rho=2.71)
Limestone = Cal

fout="/Users/raphael/simu/analysis/materials.json"
dict_materials = {Rock.name:Rock,Water.name:Water}

if __name__=="__main__":
    #print(Air)
    #mix_rock_water = MixMedium(name="mix", materials=[Rock, Water], fractions=[0.5, 0.5])
    #print(mix_rock_water.name)
    print(Qtz.name, Qtz.formula, Qtz.A)
    print(Ab.name, Ab.formula, Ab.A)
    print(An.name, An.formula, An.A)
    print(Phl.name, Phl.formula, Phl.A)
    print(Ann.name, Ann.formula, Ann.A)
    print(Mg_Hbl.name, Mg_Hbl.formula, Mg_Hbl.A)
    print(Fe_Hbl.name, Fe_Hbl.formula, Fe_Hbl.A)
    print(Aug.name, Aug.formula, Aug.A)
    print(Or.name, Or.formula, Or.A)
    print(Cal.name, Cal.formula, Cal.A)
    
    print("\n", Rock.name, f"{Rock.rho:.3f}", "g/cm^3", f"{Rock.Z_A:.3f}", f"{Rock.I:.3f}eV", f"{Rock.plasma_energy:.3f}eV ")
    print("\n", Andesite.name, f"{Andesite.rho:.3f}", "g/cm^3", f"{Andesite.Z_A:.3f}", f"{Andesite.I:.3f}",  f"{Andesite.plasma_energy:.3f}eV")
    print("\n", Granite.name, f"{Granite.rho:.3f}", "g/cm^3", f"{Granite.Z_A:.3f}",  f"{Granite.I:.3f}",  f"{Granite.plasma_energy:.3f}eV")
    print("\n", Limestone.name, f"{Limestone.rho:.3f}", "g/cm^3",f"{ Limestone.Z_A:.3f}", f"{Limestone.I:.3f}",  f"{Limestone.plasma_energy:.3f}eV")