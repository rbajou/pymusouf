# -*- coding: utf-8 -*-
#!/usr/bin/env python3

import numpy as np
from pathlib import Path
from scipy.interpolate import LinearNDInterpolator
from scipy import optimize 
import time
from typing import Union
import os 
import pickle
#personal modules
from config import MAIN_PATH
from survey import DICT_SURVEY
from telescope import Telescope

class RayPath:
    '''
    Compute apparent rock thickness along telescope ray paths with a given digital terrain model in 'surface_grid' object
    '''
    def __init__(self, telescope:Telescope, surface_grid:np.ndarray):
        self.tel = telescope
        self.tx, self.ty, self.tz = telescope.utm
        self.tel_azimuth, self.tel_zenith = telescope.azimuth*np.pi/180, telescope.zenith*np.pi/180
        self.SX, self.SY, self.SZ = surface_grid 
        self.raypath = None
    

    def __call__(self, file:Union[str, Path], max_range:float):
        self.tel.compute_angle_matrix()
        self.raypath = {}
        if isinstance(file, str): file = Path(file)
        file.parents[0].mkdir(parents=True, exist_ok=True)
        pkl_file = str(file)+'.pkl'
        if not os.path.exists(pkl_file): 
            for conf,_ in self.tel.configurations.items():
                print(f"Compute ray paths apparent thickness for {self.tel.name} ({conf})...")
                self.azimuthMatrix, self.zenithMatrix = self.tel.azimuthMatrix[conf], self.tel.zenithMatrix[conf]
                self.raypath[conf] =  self.compute_apparent_thickness(max_range=max_range)
            with open(pkl_file, 'wb') as f : 
                pickle.dump(self.raypath, f, pickle.HIGHEST_PROTOCOL)
        else : 
            print(f"Load {file}.pkl ")
            with open(pkl_file, 'rb') as f : 
                self.raypath = pickle.load(f)

    def AllZeros(self, func, xmin:float,xmax:float,N:int=100, **kwargs):
        '''
        Parameters:
            func : lambda function of one variable
            [xmin - xmax] : range where f is continuous containing zeros
            N (int) : npts, control minimum distance (xmax-xmin)/N between two zeros
        Returns: 
            roots (np.ndarray) : roots of given func
        '''
        tf_0 = time.time()
        dx=(xmax-xmin)/N
        x2=xmin
        y2=func(x2)
        roots=[]
        x1 = np.ones(N)*xmin 
        x1[1:] += np.arange(0,N-1)*dx
        x2 = np.ones(N)*xmin + np.arange(0,N)*dx
        y1, y2 = func(x1), func(x2)
        y3 = np.multiply(y1, y2)
        n = np.where(y3<0)
        if len(n[0])>0 : 
            x1n, x2n, y1n, y2n =  x1[n], x2[n], y1[n], y2[n]
            x3 = (x2n*y1n-x1n*y2n)/(y1n-y2n) 
            root = optimize.fsolve(func, x0=x3, **kwargs)
            #root = optimize.root_scalar(f, x0=(x2*y1-x1*y2)/(y1-y2),bracket=[x1,x2], **kwargs).root
            #root = optimize.root(f, x0=(x2*y1-x1*y2)/(y1-y2)).x
            #root = optimize.newton(f, x0=(x2*y1-x1*y2)/(y1-y2))
            #root = optimize.brentq(f, a=x1, b=x2) #super slow
            roots.append(root)
        roots = np.unique(roots)
        #print(f"AllZeros() -- {time.time()-tf_0:.3f} s")
        return roots



    def compute_apparent_thickness(self, max_range:float=1500., **kwargs) -> dict:
        '''
        Parameters: 
            max_range (int) : maximal distance in meter to seek for ray path / object intersection
        Returns:
            dict_thick (dict) :  {'distance' : telescope-1st surface interception matrix [m], 'thickness' : rock thickness (or travel length) matrix [m], 'xyz_in': xyz coordinate matrix of ray path entry interception points, 'xyz_out': xyz coordinate matrix of ray path exit interception points}
        '''
        t0  = time.time()
        SX,SY,SZ = self.SX, self.SY, self.SZ 
        tx, ty, tz = self.tx, self.ty, self.tz
        points, values = (SX.flatten(),SY.flatten()), SZ.flatten()
        finterp = LinearNDInterpolator(points, values)
        topo_alt = finterp((tx, ty ))
        isTelescopeUnderground = False

        if tz <= topo_alt: 
            # weird behaviour if isTelescopeUnderground
            isTelescopeUnderground = True
            msg = 'Telescope undergound -> increase altitude telescope'
            raise ValueError(msg)
        else : pass 
        

        azimuths = self.azimuthMatrix * 180/np.pi
        zeniths = self.zenithMatrix * 180/np.pi
        elevations = 90 - zeniths

        shape = azimuths.shape
        thickness = np.ones(shape)*np.nan
        xyz_in = np.ones(shape=(thickness.shape[0], thickness.shape[1], 3))*np.nan
        xyz_out = np.ones(shape=(thickness.shape[0], thickness.shape[1], 3))*np.nan
        distance = np.ones(shape=thickness.shape)*np.nan
        profile_topo = np.ones(shape=(shape[0], 2))*np.nan

        allZeros_N = int(max_range/10) #  manually adjust for each case

        k = 0 
        for i  in range(shape[0]) :
            for j in range(shape[1]):              
    
                x_line = lambda r : tx + r * np.sin(azimuths[i,j]*np.pi/180) * np.cos(elevations[i,j]*np.pi/180) 
                y_line = lambda r : ty + r * np.cos(azimuths[i,j]*np.pi/180) * np.cos(elevations[i,j]*np.pi/180) 
                z_line = lambda r : tz + r * np.sin(elevations[i,j]*np.pi/180)  
                # height above topography function 
                height_above_topo = lambda r : z_line(r) -  finterp((x_line(r),y_line(r)))
                # r_arr = np.linspace(0,max_range, 100)
                # arg_nnan = np.where(np.isnan(height_above_topo(r_arr)))                
                # max_range_tmp = r_arr[np.nanmin(arg_nnan)]
                # we want to know at which r the function is zero
                r_intercept = self.AllZeros(func=height_above_topo, xmin=0, xmax=max_range, N=allZeros_N, **kwargs)
                if len(r_intercept) != 0 and isTelescopeUnderground:
                    #  save interception with topography
                    xyz_in[i,j,0] = x_line(r_intercept[0]) 
                    xyz_in[i,j,1] = y_line(r_intercept[0])  
                    xyz_in[i,j,2] = z_line(r_intercept[0]) 
                    xyz_out[i,j,0] = x_line(r_intercept[-1])  
                    xyz_out[i,j,1] = y_line(r_intercept[-1]) 
                    xyz_out[i,j,2] = z_line(r_intercept[-1]) 
                    #  travel length     
                    xt = x_line(r_intercept[0])  
                    yt = y_line(r_intercept[0]) 
                    zt = z_line(r_intercept[0]) 
                    travel_len = np.sqrt((xt - tx)**2 + (yt - ty)**2 + (zt - tz)**2)
                    if len(r_intercept) >= 3 and len(r_intercept) % 2 != 0:
                        for kk in range(2,len(r_intercept), 2): #  odd numbers correspond to ray paths going from the inside to the outside of the dome
                            xt = x_line(r_intercept[kk])
                            yt = y_line(r_intercept[kk])
                            zt = z_line(r_intercept[kk])  #  we are interested in travel lenghts from even to odd index number (of r_intercept vector)
                            xtt = x_line(r_intercept[kk-1])
                            ytt = y_line(r_intercept[kk-1])
                            ztt = z_line(r_intercept[kk-1])
                            travel_len = travel_len + np.sqrt((xt - xtt)**2 + (yt - ytt)**2 + (zt - ztt)**2)
                        
                        d = 0
                        for kk in range(1,len(r_intercept), 2):
                            xt = x_line(r_intercept[kk])
                            yt = y_line(r_intercept[kk])
                            zt = z_line(r_intercept[kk])
                            xtt = x_line(r_intercept[kk-1])
                            ytt = y_line(r_intercept[kk-1])
                            ztt = z_line(r_intercept[kk-1])
                            d = d + np.sqrt((xt - xtt)**2 + (yt - ytt)**2 + (zt - ztt)**2)
                        
                    elif np.mod(len(r_intercept),2) == 0 : 
                        travel_len = np.nan  # line entering topography but not leaving
                        d = np.nan
                    else :
                        d = np.nan
                    
                elif len(r_intercept) != 0 and ~isTelescopeUnderground:
                    #  travel length
                    travel_len = 0
                    
                    if len(r_intercept) >= 2 and np.mod(len(r_intercept),2) == 0:

                        for kk in range(1,len(r_intercept),2): #  even numbers correspond to ray paths going from the inside to the outside of the dome
                            xt = x_line(r_intercept[kk])
                            yt = y_line(r_intercept[kk])
                            zt = z_line(r_intercept[kk])  #  we are interested in travel lenghts from odd to even index number (of r_intercept vector)
                            xtt = x_line(r_intercept[kk-1])
                            ytt = y_line(r_intercept[kk-1])
                            ztt = z_line(r_intercept[kk-1])
                            travel_len = travel_len + np.sqrt((xt - xtt)**2 + (yt - ytt)**2 + (zt - ztt)**2)
                        
                        #  save interception with topography
                        xyz_in[i,j,0] = x_line(r_intercept[0])
                        xyz_in[i,j,1] = y_line(r_intercept[0])
                        xyz_in[i,j,2] = z_line(r_intercept[0])
                        xyz_out[i,j,0] = x_line(r_intercept[-1])
                        xyz_out[i,j,1] = y_line(r_intercept[-1])
                        xyz_out[i,j,2] = z_line(r_intercept[-1])

                        xt = x_line(r_intercept[0])
                        yt = y_line(r_intercept[0])
                        zt = z_line(r_intercept[0])

        

                        d = np.sqrt((xt - tx)**2 + (yt - ty)**2 + (zt - tz)**2)

                        for kk in range(2,len(r_intercept),2):
                            xt = x_line(r_intercept[kk])
                            yt = y_line(r_intercept[kk])
                            zt = z_line(r_intercept[kk]) 
                            xtt = x_line(r_intercept[kk-1])
                            ytt = y_line(r_intercept[kk-1])
                            ztt = z_line(r_intercept[kk-1])
                            d = d + np.sqrt((xt - xtt)**2 + (yt - ytt)**2 + (zt - ztt)**2)
                        

                    elif len(r_intercept)%2 != 0:
                        travel_len = np.nan  # line entering topography but not leaving
                        d = np.nan
                    else :
                        d=np.nan
                    
                else : 
                    travel_len = np.nan
                    d = np.nan
                print(f"k={k}")
                thickness[i,j] = travel_len
                distance[i,j] = d
                k = k + 1 
              

        ##Compute 'profile_topo' angular coord
        for i  in range(shape[0]) :
            j = 0
            done = 0
            while (j < (shape[1]-2))  & (done == 0):
                j = j+1
                th1 = thickness[j,i]
                th2 = thickness[j+1,i]
                if (np.isnan(th1)) & (~np.isnan(th2)) : 
                    profile_topo[i, : ] = np.array([azimuths[j,i], 0.5*(zeniths[j,i]+zeniths[j+1, i])])
                    done = 1




        print(f"compute_apparent_thickness() end --- {time.time()-t0:.2f} s" )

        dict_thick = {"distance" : distance, "thickness" : thickness, "xyz_in" : xyz_in, "xyz_out" : xyz_out, "profile_topo" : profile_topo}

        return  dict_thick


    def save(self, file:Union[str, Path]): 
        '''
        Save collection of arrays as .npz file (numpy archive)
        '''        
        if isinstance(file, str): file = Path(file)
        file.parents[0].mkdir(parents=True, exist_ok=True)
        np.savez(file, 
            distance = self.distance,
            thickness = self.thickness, #Provide arrays as keyword arguments to store them under the corresponding name in the output file
            xyz_in =  self.xyz_in,
            xyz_out = self.xyz_out,
            profile_topo = self.profile_topo
        ) 


RayPathSoufriere = { }
survey = DICT_SURVEY['soufriere']
dem_filename = "soufriereStructure_2.npy" #5m resolution 
dem_path = survey.path / "dem"
main_tel_path = survey.path / "telescope" 
surface_grid = np.load(dem_path / dem_filename)

for _, run in survey.runs.items(): 

    tel = run.telescope
    tel_path = main_tel_path / tel.name
    raypath= RayPath(telescope=tel, surface_grid=surface_grid)
    fout = tel_path / 'raypath'/ f'az{tel.azimuth:.1f}_elev{tel.elevation:.1f}' / 'raypath'
    raypath(file=fout, max_range=1500)
    RayPathSoufriere[tel.name] = raypath


RayPathCopahue = { }
survey = DICT_SURVEY['copahue']
dem_filename = "copahueStructure.npy" #5m resolution 
dem_path = survey.path / "dem"
main_tel_path = survey.path / "telescope" 
surface_grid = np.load(dem_path / dem_filename)

for _, run in survey.runs.items(): 

    tel = run.telescope
    tel_path = main_tel_path / tel.name
    raypath= RayPath(telescope=tel, surface_grid=surface_grid)
    fout = tel_path / 'raypath'/ f'az{tel.azimuth:.1f}_elev{tel.elevation:.1f}' / 'raypath'
    raypath(file=fout, max_range=1500)
    RayPathCopahue[tel.name] = raypath

if __name__ == "__main__":
   pass

   

'''
    
    def getApparentOpenSkyThickness(self):
        ndx, ndy = self.azimuthMatrix.shape
        apparentThicknessFront, apparentThicknessRear = np.zeros(shape=(ndx, ndy)), np.zeros(shape=(ndx, ndy))
        d = 2000 # m
        NN = 2000
        k = 0
        #eps = 0.02
        #az0, ze0 = 5.181999e+01, 7.124958e+01
        for i in range(ndx): 
            for j in range(ndy):
                az, ze = self.azimuthMatrix[i,j], self.zenithMatrix[i,j]
                # if ((1-eps)*az0 <= az*180/np.pi) and (az*180/np.pi <= (1+eps)*az0) :
                #     if ((1-eps)*ze0 <= ze*180/np.pi) and (ze*180/np.pi <= (1+eps)*ze0):
                #         pass
                #     else : continue
                # else: 
                #     continue
                #if k > 1 : break
                #if ze*180/np.pi < 70 : continue
                #print(f"az, ze = {az*180/np.pi:.2f}, {ze*180/np.pi:.2f}")
                #print(k)
                #k+=1
                
                uxF = np.linspace(self.tx,(self.tx + d*np.cos(np.pi/2-az)*np.sin(ze)), NN)
                uyF = np.linspace(self.ty,(self.ty + d*np.sin(np.pi/2-az)*np.sin(ze)), NN)
                uzF = np.linspace(self.tz,(self.tz + d*np.cos(ze)),NN)

                uxR = np.linspace(self.tx,(self.tx - d*np.cos(np.pi/2-az)*np.sin(ze)), NN)
                uyR = np.linspace(self.ty,(self.ty - d*np.sin(np.pi/2-az)*np.sin(ze)), NN)
                uzR = np.linspace(self.tz,(self.tz - d*np.cos(ze)), NN)
                
                #replace with griddata
                points, values = np.array([self.SX.flatten(), self.SY.flatten()]).T, self.SZ.flatten()
                ZIF, ZIR = griddata(points, values, (uxF, uyF) ), griddata(points, values, (uxR, uyR) )
                #print(f"ZIF = {ZIF}, {np.all(np.isnan(ZIF))}, {ZIF.shape}")
                intersecF = np.argwhere(abs(np.diff((ZIF-uzF)>0)) == 1).flatten()  #find(abs(diff((ZIF-uzF)>0)) == 1)
                #print(len(intersecF), intersecF[:10])
                intersecR = np.argwhere(abs(np.diff((ZIR-uzR)>0)) == 1).flatten()

                #print(intersecF,intersecF.shape)

                nF, nR = len(intersecF), len(intersecR)
                #print(f"nF, nR = {nF}, {nR}")
                #continue
                minIF, minIR = np.nanmin((len(uzF), np.sum(~np.isnan(ZIF))))-1, np.nanmin((len(uzR)-1, np.sum(~np.isnan(ZIR))))-1
                #print(f"len(uzF), np.sum(~np.isnan(ZIF) = {len(uzF)}, {np.sum(~np.isnan(ZIF))}")
                #print(f"minIF, minIR = {minIF}, {minIR}")
                #print(f"ZIF[minIF], uzF[minIF] = {ZIF[minIF]}, {uzF[minIF]}")
                #continue
                if (ZIF[minIF] > uzF[minIF]): # apparent thickness too important
                    #print("too thick")
                    apparentThicknessFront[i,j] = np.nan
                
                elif (nF == 0): # pointing the sky
                    #print("sky")
                    apparentThicknessFront[i,j] = np.nan
                
                elif (ZIF[minIF] < uzF[minIF]):
                    #print("ZIF[minIF] < uzF[minIF]")
                    if (nF > 1):
                        if nF% 2 == 0: # or (np.mod(nF,2) == 0): 
                        
                            distanceInMatter = 0        
                            for k in range(0, nF-1, 2): #increment of 2
                                distanceInMatter = distanceInMatter + np.sqrt((uxF[intersecF[k]]-uxF[intersecF[k+1]])**2+(uyF[intersecF[k]]-uyF[intersecF[k+1]])**2+(ZIF[intersecF[k]]-ZIF[intersecF[k+1]])**2)
                            apparentThicknessFront[i,j] = distanceInMatter
                        
                        else:               
                            apparentThicknessFront[i,j] = np.nan             
                else: pass
                
                if (ZIR[minIR] > uzR[minIR]): # apparent thickness too important
                    apparentThicknessRear[i,j] = np.nan
                
                elif (nR == 0): # pointing the sky
                    apparentThicknessRear[i,j] = np.nan
                
                elif (ZIR[minIR] < uzR[minIR]):
                    if (nR > 1):
                        if nR% 2 == 0: # or np.mod(nR,2) == 0):

                            distanceInMatter = 0         
                            for k in range(0, nR-1, 2): #increment of 2
                                distanceInMatter = distanceInMatter + np.sqrt((uxR[intersecR[k]]-uxR[intersecR[k+1]])**2+(uyR[intersecR[k]]-uyR[intersecR[k+1]])**2+(ZIR[intersecR[k]]-ZIR[intersecR[k+1]])**2)
                            

                            apparentThicknessRear[i,j] = distanceInMatter

                        else:          
                            apparentThicknessRear[i,j] = np.nan                
                else: pass
                
                #print(f"(tF, tR) = {apparentThicknessFront[i,j]:.3f}, {apparentThicknessRear[i,j]:.3f}")

            
        self.apparentThicknessFront = apparentThicknessFront
        self.apparentThicknessRear  = apparentThicknessRear



'''