### Context  
The repository contains Python packages to reconstruct and analyze muography data recorded during two surveys at :
- La Soufrière de Guadeloupe, located in the Lesser Antilles. The volcano is monitored by the Volcanological and Seismological Observatory of Guadeloupe (OVSG), under the responsibility of IPGP. 
- the Copahue volcano, on the border between Argentina and Chile.

The detectors used in this study are scintillator-based hodoscopes named "telescopes" developed in IP2I, Lyon. 
The data processed by this package were acquired in the framework of the ANR DIAPHANE and ANR MEGAMU projects.  
  
### Data processing steps  
First the ```CURRENT_SURVEY_NAME``` variable needs to be edited in the ```config/config.py``` module, either with ```soufriere``` or ```copahue```. Then, the package needs to be compiled, check ```INSTALL``` file.

## 1. Tracking:  
To process raw telescope data and reconstruct particle track event-by-event, the user needs to run ```python3 tracking/main.py``` with the following arguments:    
INPUTS:  
    - ```--telescope [-tel]``` (str2telescope) the telescope name (required): Check the available telescope configurations in  dictionary ```dict_tel``` in ```telescope/telescope.py```.  
    - ```--input_dat [-i]``` (str or List[str]) :  /path/to/datafile/  One can input a data directory, one or several .dat file e.g ```--input_data <file1.dat> <file2.dat>```.  
    - ```--out_dir [-o]``` (str) : /path/to/outdir/ where the output directory will be created.  
    - (optional)```--input_type [-it]``` (str, ```real``` or ```mc```) either real data or monte-carlo data  
    - (optional) ```--max_nfiles [-max]```  (int, default is ```1```) the maximum number of data files to process.  
    - (optional) ```--is_ransac```  (bool, default is ```True```)  
    RANSAC parameters:  
    - (optional) ```--residual_threshold [-rt]```  (float, default is ```50```mm, i.e size of detector pixel)  
    - (optional) ```--min_samples [-ms]```  (float, default is ```2```) the size of the inital data sample  
    - (optional) ```--max_trials [-mt]```  (float, default is ```100```) maximum number of iterations to find best trajectory model  
    - ...  

OUTPUTS:  
    Two dataframes with event id index:    
        - ```df_track.csv.gz``` : Tracking output (intersection points XY coordinates between fitted trajectories and each telescope panel) for each filtered track.  
        - ```df_inlier.csv.gz``` : Ransac inlier and outlier XYZ points coordinates and their associated charge content (in ADC)  

## 2. Reco  
Once the processing output is here, you can run the ```reco/main.py``` script to get panel hit XY maps and DXDY maps with the following arguments:  
INPUTS:  
    - ```--telescope [-tel]``` (str2telescope)  
    - ```--in_dir [-i]``` (str) : /path/to/output/tracking/  
    - ```--out_dir [-o]``` (str) : /path/to/output/dir/  

## 3. Muography (2D)     
For a given telescope edit ```files/telescopes/<tel>/run.yaml``` file with track reco data paths for calib (open-sky run) and tomo datasets (see step 1.); then run ```muo2d/main.py``` to estimate acceptance, flux, opacity, mean density.
INPUTS:  
    - ```--telescope [-tel]``` (str2telescope)  

OUTPUTS: 
    - Acceptance estimate : 2-d NumPy array(s) saved as binary pickle file format ```acceptance.pkl```
    - Flux, opacity, mean density estimates : 2-d NumPy array(s) saved as binary pickle file format ```flux.pkl```, ```opacity.pkl```, and ```mean_density.pkl```


