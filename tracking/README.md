## Tracking:  
To process raw telescope data and reconstruct particle track event-by-event, the user needs to run ```python3 tracking/main.py``` with the following arguments:    
INPUTS:  
    - ```--telescope [-tel]``` (str2telescope) the telescope name (required): Check the available telescope configurations in  dictionary ```dict_tel``` in ```telescope/telescope.py```.  
    - ```--input_dat [-i]``` (str or List[str] or ```.txt``` file) :  /path/to/datafile/  One can input a data directory, a single .datf file, or a list of data files e.g ```--input_data <file1.dat> <file2.dat>```.  
    - ```--out_dir [-o]``` (str) : /path/to/outdir/ where the output directory will be created.  
    - (optional)```--input_type [-it]``` (str, ```DATA``` or ```MC```) either real data or monte-carlo data  
    - (optional) ```--max_nfiles [-max]```  (int, default is ```1```) the maximum number of data files to process.  
    - (optional) ```--is_ransac```  (bool, default is ```True```)  
    RANSAC parameters:  
    - (optional) ```--residual_threshold [-rt]```  (float, default is ```50```mm, i.e size of detector pixel)  
    - (optional) ```--min_samples [-ms]```  (float, default is ```2```) the size of the inital data sample  
    - (optional) ```--max_trials [-mt]```  (float, default is ```100```) maximum number of iterations to find best trajectory model  
    - ...  

OUTPUTS:  
    Two dataframes with event id index:    
        - ```reco.csv.gz``` : RANSAC output (intersection points XY coordinates between fitted trajectories and each telescope panel) for each filtered track.  
        - ```inlier.csv.gz``` : inlier and outlier XYZ coordinates and their associated charge content in X and Y (in ADC)  

