#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from re import A
import numpy as np
from scipy.integrate import quad, solve_ivp
import scipy.io as sio
import matplotlib.pyplot as plt
import matplotlib.lines as mlines  #use for legend settings
import matplotlib.patches as mpatch  #use for legend settings
import os
from random import choice, seed
from pathlib import Path

#https://pdg.lbl.gov/2021/AtomicNuclearProperties/adndt.pdf

class StoppingPower:
    def __init__(self, I:float=None, plasma_energy:float=None):
        """
        I : Mean excitation energy in eV
        plasma_energy 'hbar*omega' : in eV, typically 'hbar*omega = 28.816*np.sqrt(rho*Z/A)'
        """
        self.par = {
                    'me':0.511,#MeV
                    'mu': 105.658,#MeV
                    'Na' : 6.022e23, #mol^-1
                    'e': 1e-6, #MeV elementary charge 
                    'z': -1, #charge multiples
                    'eps0': 55.26349406 * 1e-6**2 * 1e-3 * 1e13, # e^2.GeV^−1.fm−1 vaccum permittivity
                    'c':1, #natural units
                    'I':I, #Mean excitation energy in eV (I=136.4 eV for Z=11)        
                    'alpha':1/137, #dimensionless, fine structure constant
                    're':2.817940, #fm  classical electron radius
                    'plasma_energy':plasma_energy,  #  #eV  
                    }

    
    def delta(self, x, Z_A, rho, stern_coef=None, rho0=None):
        """
        density-effect corrections
        x= log10(p/M)=log10(beta*gamma)
        """
        if stern_coef is None:
            I = self.par["I"] #eV
            plasma_energy = self.par["plasma_energy"]
            if plasma_energy is None:  plasma_energy = 28.816*np.sqrt(rho*Z_A)  #eV  
            Cbar = 2*np.log(I/(plasma_energy)) + 1 ##Cbar=3.886
            if I >= 100: 
                x1 = 3.0  #I > 100eV , 2.0 else
                if Cbar <  5.215:
                    x0 = 0.2
                else:
                    x0 = 0.326*Cbar - 1.5
            else:  
                x1 = 2.0
                if Cbar <  3.681:
                    x0 = 0.2
                else: 
                    x0 = 0.326*Cbar - 1.0
            k = 3.0
            a = (Cbar - 2*np.log(10)*x0)/(x1-x0)**3
        else : 
            a, k, x0, x1, I, Cbar, delta0 = stern_coef.T 
        # print("Sternheimer coef (a, k, x0, x1, I, Cbar)=", a, k, x0, x1, I, Cbar)
        # if rho0 is not None:
        #     r = rho/rho0
        #     Cbar = Cbar - np.log(r)
        #     x0 = x0 - 0.5*np.log10(r)
        #     x1 = x1 - 0.5*np.log10(r)
            
        if isinstance(x, float):
            if x >= x1 : delta = 2*np.log(10)*x[(x>=x1)] - Cbar 
            elif (x0 <= x) & (x < x1) : delta =2*np.log(10)*x[((x0 <= x) & (x < x1))] - Cbar + a*(x1-x[((x0 <= x) & (x < x1))])**k
            else : delta = 0
            return delta
        #print(x)
        delta = np.ones(len(x))
        delta[(x>=x1)] = 2*np.log(10)*x[(x>=x1)] - Cbar 
        delta[((x0 <= x) & (x < x1))]= 2*np.log(10)*x[((x0 <= x) & (x < x1))] - Cbar + a*(x1-x[((x0 <= x) & (x < x1))])**k
        delta[(x < x0)] = 0 #for non conductors 
        #delta[(x < x0)] =  delta0*10**(2*(x[(x < x0)]-x0)) #for conductors
        return delta

    def bethe_bloch(self, T, Z_A, rho, brem_corr=True, dens_effect=True, stern_coef=None, rho0=None):
        """
        Groom et al. 2001
        Mass stopping power [MeV/(g.cm^-2)] as a function of kinetic energy T [MeV].
        Inputs: 
            - T: kinetic energy in MeV
            - Z_A: Z/A ratio with Z=atomic number A=atomic mass [g/mol]
            - rho: medium density in g/cm^3
            - brem_corr: high-energy correction term 'Delta(Z,A)' 
        Returns: 
            - dE/dx mass stopping power in MeV.cm^2/g
        """
        me, mu = self.par['me'], self.par['mu'] ##MeV
        I = self.par["I"] *1e-6  #MeV
        alpha = self.par['alpha'] 
        c  = 1 #natural units
        K  = 0.307#MeV/g/cm^-2 #4*np.pi*Na*re**2*me*c**2
        gamma = T/(mu*c**2) +1 
        beta  = np.sqrt(1-1/gamma**2)
        Qmax  = 2*me * c**2 * beta**2 * gamma**2 / ( 1 + 2*gamma*me/mu + (me/mu)**2 ) #in MeV
        C1 =  K * Z_A
        E = gamma*mu*c**2
        Delta = 0
        ###Bremsstrahlung  correction from atomic electrons
        if brem_corr : 
            Delta = C1 /(4*np.pi)*alpha*( np.log(2*E/(mu*c**2)) - 1/3 * np.log(2*Qmax/(me*c**2))  ) * np.log(2*Qmax/(me*c**2))**2   
        ####
        x = np.log10(beta*gamma)
        if rho0 is not None: 
            """correction when modified density rho from reference material (rho0,Z,A)  (Sternheimer 1971)"""
            r = rho/rho0
            x += np.log10(np.sqrt(r))
         
        delta = 0
        if dens_effect : 
            delta = self.delta(x, Z_A, rho, stern_coef)
    
        dEdx  = C1/beta**2 * ( 0.5*np.log( (2*me*c**2 * beta**2 * gamma**2 * Qmax)/I**2) - beta**2 - delta/2  + 1/8 * Qmax**2/(gamma*mu*c**2)**2 ) + Delta #
        if not isinstance(dEdx,np.float64):
            dEdx[np.isnan(dEdx)] = 0
        return dEdx

    def valentin(self, T, Z_A, Z, rho):
        """
        Luc Valentin 1995
        """
        me, mu = self.par['me'], self.par['mu'] #MeV
        I = self.par["I"]*1e-6 #MeV
        Na = self.par["Na"]
        e = 1#self.par["e"]*1e6# *1.6e-19 #MeV
        
        c  = 1 #natural units
        gamma = T/(mu*c**2) +1 #
        beta  = np.sqrt(1-1/gamma**2)
        N =  rho * Z_A * Na #cm^-2
        eps0 = 55.263 *1e-3*1e13 # e2⋅GeV−1⋅fm−1 -> e2⋅MeV−1⋅cm−1
        Q = (e**2/(4*np.pi*eps0)) **2
        C = (4*np.pi * N) / (me*beta**2)  * Q
        dEdx = C * (np.log( (2*me * beta**2 * gamma**2 )/(I) ) - beta**2 )  / rho
        return dEdx
        
    def integrand_cs_pair_nuc(self, rho, nu, E, Z):
        """
        G(Z,E,nu, rho) in sec.10.3 in https://twiki.cern.ch/twiki/pub/LHCb/DocSimDigi/G4_PhysicsReferenceManual.pdf
        """
        alpha, re, me, mu = self.par["alpha"], self.par["re"]*1e-13, self.par["me"]*1e-3, self.par["mu"]*1e-3
        #####ISSUE WITH KSI < -1 ###nu < 1
        
        ksi = (mu*nu)**2/(2*me)**2 * (1 - rho**2) / (1-nu) #ok
        beta = nu**2/(2*(1-nu)) #ok
        Astar = 183 #189??? #https://articles.adsabs.harvard.edu//full/1970ICRC....4..277K/D000278.000.html
        B_e = ((2+rho**2)*(1+beta) + ksi*(3+rho**2)) * np.log( 1 + 1/ksi ) + (1 - rho**2 - beta)/(1+ksi) - (3+rho**2)    
        if ksi >= 1e3:  B_e = 1/(2*ksi) * ((3-rho**2) + 2*beta*(1+rho**2)) #ok
        Y_e = (5 - rho**2 + 4*beta*(1+rho**2))/ ( 2*(1+3*beta)*np.log(3+1/ksi) - rho**2 - 2*beta*(2-rho**2) )
        ####
       # if ksi < -1 : print(rho,nu,f"ksi={ksi}" )
        #q= (1+ksi)*(1+Y_e)
        #if q < 0 : print(rho,nu,q )
        #r = Astar*Z**(-1/3) * np.sqrt((1+ksi)*(1+Y_e))  / ( 1 + (2*me*np.sqrt(np.exp(1)) * Astar * Z**(-1/3) * (1+ksi)*(1+Y_e) ) / (E*nu*(1-rho**2)) )
        ####
        
        Lprim_e = np.log( Astar*Z**(-1/3) * np.sqrt((1+ksi)*(1+Y_e))  / ( 1 + (2*me*np.sqrt(np.exp(1)) * Astar * Z**(-1/3) * (1+ksi)*(1+Y_e) ) / (E*nu*(1-rho**2)) ) ) - 0.5*np.log( 1 + ((3*me*Z**(1/3))/(2*mu) )**2 * (1+ksi)*(1+Y_e) ) #ok
        B_mu = ((1+rho**2)*(1+3*beta/2) - 1/ksi * (1+2*beta)*(1-rho**2) ) * np.log(1+ksi) #ok
        B_mu += (ksi*(1-rho**2-beta**2))/(1+ksi) + (1+2*beta)*(1-rho**2)  #ok
        if ksi <= 1e-3:  B_mu = ksi/2 * ((5-rho**2) + beta*(3+rho**2)) #ok
        Y_mu = (4 + rho**2+ 3*beta*(1+rho**2) )/( (1+rho**2) * (3/2 + 2*beta) * np.log(3+ksi) + 1 - 3/2*rho**2 ) #ok
        Lprim_mu = np.log( (mu/me)*Astar*Z**(-1/3) * np.sqrt((1+1/ksi) * (1 + Y_mu)) / ( 1 + (2*me*np.sqrt(np.exp(1)) * Astar * Z**(-1/3) * (1+ksi)*(1+Y_mu))/(E*nu*(1-rho**2)) ) ) #ok
        Lprim_mu -= np.log(3/2 * Z**(1/3) * np.sqrt((1+(1/ksi)) * (1+Y_mu))) #ok
        phi_e = B_e * Lprim_e
        if phi_e < 0 : phi_e = 0
        phi_mu = B_mu * Lprim_mu
        if phi_mu < 0 : phi_mu = 0
        integrand = phi_e + (me/mu)**2*phi_mu
        return integrand
        
    def integrand_cs_pair_nuc_bis(self, t, nu, E, Z):
        """
        G(Z,E,nu, rho) in sec.10.3 in https://twiki.cern.ch/twiki/pub/LHCb/DocSimDigi/G4_PhysicsReferenceManual.pdf
        switch variable: 
        t = ln(1-rho)
        rho**2 = (1-exp(t))**2
        """
        alpha, re, me, mu = self.par["alpha"], self.par["re"]*1e-13, self.par["me"]*1e-3, self.par["mu"]*1e-3
        ksi = (mu*nu)**2/(4*me)**2 * np.exp(t)*(2-np.exp(t)) / (1-nu) #ok
        #if ksi < 0: print("ksi neg")
        beta = nu**2/(2*(1-nu)) #ok
        Astar = 183
        B_e = ((2+(1-np.exp(t))**2)*(1+beta) + ksi*((np.exp(t)-2)*np.exp(t)+4)) * np.log( 1 + 1/ksi ) + (np.exp(t)*(2-np.exp(t)) - beta)/(1+ksi) - (3+(1-np.exp(t))**2)    
        if ksi >= 1e-3:  B_e = 1/(2*ksi) * ((3-(1-np.exp(t))**2) + 2*beta*(1+(1-np.exp(t))**2))
        Y_e = (5 - (1-np.exp(t))**2 + 4*beta*(1+(1-np.exp(t))**2))/ (2*(1+3*beta)*np.log(3+1/ksi) - (1-np.exp(t))**2 - 2*beta*(2-(1-np.exp(t))**2))
        if ((1-(1-np.exp(t))**2))  <= 0 : print((1-np.exp(t))**2)
        Lprim_e = np.log( Astar*Z**(-1/3) * np.sqrt((1+ksi)*(1+Y_e))  / ( 1 + (2*me*np.sqrt(np.exp(1)) * Astar * Z**(-1/3) * (1+ksi)*(1+Y_e) ) / (E*nu*(1-(1-np.exp(t))**2)) ) ) #ok
        Lprim_e -= 0.5*np.log( 1 + ((3*me*Z**(1/3))/(2*mu) )**2 * (1+ksi)*(1+Y_e) ) #ok
        B_mu = ((1+(1-np.exp(t))**2)*(1+3*beta/2) - 1/ksi * (1+2*beta)*(1-(1-np.exp(t))**2) ) * np.log(1+ksi) #ok
        B_mu += (ksi*(1-(1-np.exp(t))**2-beta**2))/(1+ksi) + (1+2*beta)*(1-(1-np.exp(t))**2)  #ok
        if ksi <= 1e-3:  B_mu = ksi/2 * ((5-(1-np.exp(t))**2) + beta*(3+(1-np.exp(t))**2))
        Y_mu = (4+(1-np.exp(t))**2+3*beta*(1+(1-np.exp(t))**2))/( (1+(1-np.exp(t))**2) * (3/2 + 2*beta) * np.log(3+ksi) + 1 - 3/2*(1-np.exp(t))**2 )
        Lprim_mu = np.log( (mu/me)*Astar*Z**(-1/3) * np.sqrt((1+1/ksi) * (1 + Y_mu)) / ( 1 + (2*me*np.sqrt(np.exp(1)) * Astar * Z**(-1/3) * (1+ksi)*(1+Y_mu))/(E*nu*(1-(1-np.exp(t))**2)) ) ) #ok
        Lprim_mu -= np.log(3/2 * Z**(1/3) * np.sqrt((1+(1/ksi)) * (1+Y_mu))) #ok
        phi_e = B_e * Lprim_e
        if phi_e < 0 : phi_e = 0
        phi_mu = B_mu * Lprim_mu
        if phi_mu < 0 : phi_mu = 0
        integrand = phi_e + (me/mu)**2*phi_mu
        return integrand * np.exp(t)
    
    def cs_pair_nuc(self, eps, E, A, Z):
        """
        differential cross section per atom for (e+,e-) pair creation by muon
        cf. sec.10.3 in https://twiki.cern.ch/twiki/pub/LHCb/DocSimDigi/G4_PhysicsReferenceManual.pdf
        """
        alpha, re, me, mu = self.par["alpha"], self.par["re"]*1e-13, self.par["me"]*1e-3, self.par["mu"]*1e-3
        nu = eps/E
        #print(eps)
        eps_min = 4*me
        # if  eps_min > eps or eps > eps_max: 
        #     return 0 
        Eprim = E-eps
        gamma1, gamma2 = 1.95e-5, 5.30e-5 #for all elements except hydrogen
        zeta_num  =  0.073 * np.log((E/mu)/(1+gamma1*Z**(2/3)*E/mu)) - 0.26  #ok
        zeta_den  =  0.058 * np.log((E/mu)/(1+gamma2*Z**(1/3)*E/mu)) - 0.14  #ok
        if zeta_num < 0 : zeta = 0
        elif E <= 35*mu : zeta = 0
        else: zeta = zeta_num/zeta_den
        rho_max = (1-6*mu**2/(E*Eprim))*np.sqrt( 1 - eps_min/eps ) #ok
        #print(f"rho_max={rho_max}")
        tmin = np.log((4*me/eps + 12*mu**2/(E*Eprim) *(1- eps_min/eps) )/(1+ (1-6*mu**2/(E*Eprim)) * np.sqrt(1-4*me/eps))  )
        factor_term = 4/(3*np.pi) * Z*(Z+zeta) *(alpha*re)**2 * (1-nu)/eps
        ###G(Z,E,nu,rho)
        integral_term =  quad(self.integrand_cs_pair_nuc, 0, rho_max, args=(nu, E, Z))[0]
#        integral_term =  quad(self.integrand_cs_pair_nuc_bis, tmin, 0, args=(nu, E, Z))[0]
        cs = factor_term * integral_term
        return cs        
              
    def b_pair(self, E, A, Z):
        me, mu, Na = self.par["me"]*1e-3, self.par["mu"]*1e-3, self.par["Na"]
        integrand_pair_nuc = lambda eps, E, A, Z : Na/A*eps/E*self.cs_pair_nuc(eps, E, A, Z)
        # integrate between eps_min and eps_max (cf. https://lappweb.in2p3.fr/~maire/tutorials/directpair.pdf ) 
        eps_min = 4*me
        res=  quad(integrand_pair_nuc, eps_min , (E - 3*np.sqrt(np.exp(1))/4 * mu * Z**(1/3)), args=(E, A, Z), limit=100)[0] 
        #b_pair_elec =  self.b_pair_elec(E, A,Z)
        #res += b_pair_elec
        return res
    
    def cs_brems(self, nu, E, A, Z):
        '''
        cf. sec10.2 G4_PhysicsReferenceManual
        Inputs:
            nu: fractional energy transfer [0,1] dimensionless
            E: kinetic energy in GeV
        Return: 
            Bremsstrahlung differential cross section in cm^2/g
        '''
        alpha, me, mu, re, Na = self.par['alpha'], self.par['me']*1e-3, self.par['mu']*1e-3, self.par['re']*1e-13, self.par['Na']
        Dn = 1.54*A**0.27 #ok
        Dnprim = Dn**(1-1/Z)
        B = 182.7  # for all elements except hydrogen B = 202.4
        Bprim = 1429 # for all elements except hydrogen B = 446
        delta = mu ** 2 * nu / (2*E*(1-nu)) #ok
        e = np.exp(1)
        #DeltaN = lambda d: np.log(Dn / (1+ d * (Dn*np.sqrt(e) - 2)/mu)) #ok
        #Phi = lambda d: np.log( (B*mu*Z**-1/3 / me) / (1+d*np.sqrt(e) * B * Z **(-1/3) / me) ) - DeltaN(d) #ok
        Phin = np.log( ( B*Z**(-1/3)*(mu+delta*(Dnprim*np.sqrt(e)-2)) ) / (Dnprim*(me+delta*np.sqrt(e)*B*Z**(-1/3))   ) )
        if Phin < 0: Phin = 0
        Phie = np.log( ( Bprim*Z**(-2/3)*mu ) / ( (1+delta*mu/(me**2*np.sqrt(e))) * (me + delta*np.sqrt(e)*Bprim*Z**(-2/3)) ) )
        eps_max = E/(1+mu**2/(2*me*E))
        if nu > eps_max/E : Phie = 0
        elif Phie < 0: Phie = 0
        else:pass
        ###differential cross section
        #cs_nuc = alpha * (2*Z*me/mu*re)**2 * (4/3 - 4/3*nu+ nu**2) * Phi(delta)/nu #ok
        cs = 16/3 * alpha * (me/mu * re)**2 * 1/nu * Z*(Z*Phin + Phie) * (1-nu+3/4*nu**2) # ok #per atom
        if nu*E >= E-mu : cs = 0   
        #Phi_in = lambda d: np.log(mu/d/(mu*d/me**2 + np.sqrt(e))) - np.log(1 + me/(d*B_bis*Z**(-2/3)*np.sqrt(e)))
        #cs_elec = alpha * Z * (2*me/mu * re)**2 * (4/3 - 4/3*nu + nu**2) * Phi_in(delta)/nu
        #cs  = cs_nuc + cs_elec 
        
        return cs

    def b_brems(self, E, A, Z):
        Na = self.par['Na']
        integrand_brems = lambda nu, E, A, Z : Na/A*nu*self.cs_brems(nu, E, A, Z)
        res= quad(integrand_brems, 0, 1, args=(E, A, Z) )[0] 
        return res 

    def cs_nuc(self, nu, E, A):
       
        '''
        cf. Groom et al. 2001
        Inputs:
            nu: fractional energy transfer [0,1] dimensionless
            E: kinetic energy in GeV
        Return: 
            Bremsstrahlung differential cross section in cm^2/g
        '''
        alpha, mu = self.par["alpha"], self.par["mu"]*1e-3 #GeV
        t= mu**2*nu**2 / (1-nu) #ok
        kappa = 1 - 2/nu + 2/nu**2 #ok
        eps = nu*E 
        ###photoabsorption cross section 1b = 1e-24 cm^2
        sig_gamN = ( 114.3 + 1.647*np.log(0.0213*eps)**2 )* 1e-30#in micro-barn -> cm^2 #ok
        x = 0.00282 * A**(1/3) * sig_gamN #ok 
        G = 3/x**3 * (x**2/2 - 1 + np.exp(-x)*(1+x))
        m1 = np.sqrt(0.54) #GeV 
        m2 = np.sqrt(1.8) #GeV
        term1 = 0.75 * G * ( kappa * np.log(1+m1**2/t) - kappa * m1**2/(m1**2 + t) - 2*mu**2/t) #ok
        term2 = 0.25 * ( kappa * np.log(1 + m2**2 / t) - 2*mu**2/t ) #ok
        term3 = mu**2 / (2*t) * ( 0.75 * G * m1**2/(m1**2 + t) + 0.25*m2**2/t * np.log(1+t/m2**2) ) #ok
        cs = alpha/(2*np.pi) * A * sig_gamN * nu * ( term1 + term2 + term3 )
        return cs # in cm2/g
        
    def b_nuc(self, E, A, Z=None):
        """
        cf. Groom et al. 2001
        Inputs:
            E: float (energy in GeV)
            A: float (atomic mass in g/mol)
        Return: 
            b: float (photo-nuclear effect in cm^2/g)
        """
        Na = self.par["Na"]
        integrand_nuc = lambda nu, E, A : Na/A*nu*self.cs_nuc(nu, E, A)
        res= quad(integrand_nuc, 0., 1., args=(E, A) )[0]          
        return res


    def get_min_energy(self, func, x:float, E0:float, Emax:float=1e8, **kwargs): 
        #try:
        if x == 0: emin = E0
        else: 
            emin =  solve_ivp(fun=func, t_span=[0, x], y0=[E0], t_eval=[x], **kwargs).y[0][-1]  
        return emin
        #except:
        #    raise ValueError
    
    def minimum_energy(self, func, opacity:np.ndarray, E0:float, **kwargs):
        """
        To cross distance 'L'[m] in medium with density 'rho'[g.cm^-3] 
        """
        out_arr = np.zeros(opacity.shape)
        for i in range(opacity.shape[0]):
            for j in range(opacity.shape[1]):
                x = opacity[i,j]
                out_arr[i,j]= self.get_min_energy(func, x, E0=E0, **kwargs)
        return out_arr
    


'''
import sympy as sy

def cs_pair_nuc_nikishov(self, nu, E, Z):
        alpha, re, me, mu = self.par["alpha"], self.par["re"]*1e-13, self.par["me"]*1e-3, self.par["mu"]*1e-3
        factor = (2*alpha*re*Z)**2/np.pi * (1-nu)/nu 
        eps = nu*E
        theta= (me/mu)**2
        z = nu**2/(theta*(1-nu))
        B = lambda v : np.sqrt(1/4 + 1/v) 
        z1, z2 = B(z)-0.5, B(z)+0.5
        y = (z1+z2)/z2**2
        integrand = lambda t : -np.log(1-t)/t 
        Li2 = lambda x : quad(integrand, 0, x)[0]
        ###f1 + theta * f3 
        term1 = 44/(45*z) - 16/45 - 4/9*theta - (7/9 + 8/45 * z + 7/18*z*theta)*np.log(z) #ok
        term1 += (16/45*z+38/45-44/(45*z) + 4/(3*(z+4)) + (7/9*z - 2/9 + 8/(3*(z+4)))*theta)*B(z)*np.log(z2/z1) #ok
    
        ###phi2 + theta*phi4
        term2 = (7/36 + 2/45*z +7/72*z*theta) * (np.log(z2/z1)**2+np.pi**2+2*np.log(z)**2)#ok
        term2 += (7/18 + 3/20*z + 7/36*z*theta)*np.log(z) + 653/270 - 28/(9*z) + 2/3*theta #ok
        term2 += (-3/10*z - 92/45 + 52/(45*z) - (2/9 - 7/18*z)*theta)*B(z)*np.log(z2/z1) #ok
        term2 += B(z)*(-8/45*z - 19/45 - 8/(45*z) - (2/9 + 7/18*z)*theta)*(Li2(y) + 2*Li2(1/z2) + 3/2*np.log(z2/z1)**2) #ok
        term2 += (8/z + z*theta)*B(z)/(3*(z+4))*(6*Li2(1/z2)-Li2(y)+0.5*np.log(z2/z1)**2) #ok
    
        ###I         
        ene = sy.Symbol('ene')
        gamma = ene/mu #P0 / mu
        gammaprime = gamma.diff(ene)
        Dgamma = sy.lambdify(ene, gammaprime, 'numpy')
        s = (2*np.sqrt(E/mu*Dgamma(E))*Z**(1/3))/(183*np.sqrt(np.exp(1)))
        w = s*np.sqrt(z)
        u = w + z
        u1 = B(u)-0.5
        u2 = B(u)+0.5
        w1 = B(w)-0.5
        w2 = B(w)+0.5
        H = Li2(z/u+4) - Li2((z+4)/(u+4)) + Li2(z/(z+4)) - 2*Li2(u/(u+4)) #ok
        H += Li2(4*w/(u*(z+4))) + Li2(4*z/(u*(w+4))) - Li2(4/(w+4)) * np.pi**2/6 #ok
        H += 2*np.log(z1)*np.log(z2) - 4*np.log(u1)*np.log(u2) - np.log(z)**2 + np.log(z+4)**2 - np.log(1+4/w)*np.log(u+4) #ok
        H += - np.log(4*w)*np.log(z+4) + np.log(16)*np.log(u+4)-np.log(u+4)**2 + 2*np.log(u)**2 #ok
        H += np.log(u)*np.log((z+4)/4 * (w+4)/(4*w)) - np.log(z)*np.log((z+4)/4 * u/w) #ok

        Jplus1 = 2*Li2(1/z2) - Li2(y) + np.log(z1) * np.log(z2/z1) #ok
        Jplus2 = Li2(u1/z1) - Li2(u2/z2) + Li2(z1/(z1+u2)) - Li2(z2/(z2+u1)) + np.log(u1/z1) * np.log(1-u1/z1) #ok
        Jplus2 += -np.log(u2/z2)*np.log(1-u2/z2) + np.log(z2/z1) * np.log(u*(z1+u2)) #ok
        Jplus = Jplus1 + Jplus2
        Iplus = Li2(u1/w1) - Li2(u2/w2) - 2*Li2(w1/w2) + Li2(w1/(w1+u2)) #ok
        Iplus += -Li2(w2/(w2+u1)) + np.pi**2 / 3 + np.log(w2/w1)*np.log((w1+u2)/w2 * u/z) #ok
        Iplus += np.log(u1/w1)*np.log(1-u1/w1) - np.log(u2/w2)*np.log(1-u2/w2) #ok
        #####
        term3 = (7/9 + 8/45*z + 7/18*z*theta)*H - (16/45*z + 38/45 + 16/(45*z) + (7/9*z+4/9)*theta)*B(z)*Jplus #ok
        term3 += (-16/45*z - 14/9 - 8/(9*w) + 2/45*z/w - 4/5*z/w**2 + 2*z/(3*(w+4)) - (7/9*z+4/9*z/w)*theta)*B(w)*Iplus #ok
        term3 += (32/45*u/w - 88/(45*z) - 16/(45*w) + 8/5*z/w**2 + 8/9*u/w*theta)*B(u)*np.log(u2/u1) #ok 
        term3 += (68/45 - 16/(45*z) + 8/(3*w) - 2/3*z/w - 8/9*theta)*B(z)*np.log(z2/z1) + 104/(45*z) #ok
        term3 += -8/(15*w) - 62/27 - (8/(9*w) +1/45*z/w + 4/5*z/w**2 + 4/9*z/w*theta)*np.log(z)#ok
        term3 += (1 + 0.5*z*theta)*1/(3*w)*(np.log(u2/u1)**2 - np.log(z2/z1)**2) #ok
        term3 += (8/z + z*theta) * B(z)/(3*(z+4)) - (2*Jplus2 + np.log(z2)**2 - np.log(z1)**2) #ok
        cs = factor*(term1 * np.log(2*eps/me) + term2 + term3)
        return cs
'''


if __name__ == "__main__":
    
    pass