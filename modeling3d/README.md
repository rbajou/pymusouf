# 1. voxel:  
(Adapted from Marina Rosas-Carbajal's MatLab code)  
This sub-package contains a module ```voxel.py``` to voxelize the volume of a given object (here La Soufrière lava dome) from its digital elevation model (DEM) and a given voxel resolution in meter.   
To compute a voxel model, run ```main.py``` with:  
INPUTS:  
    - ```--res_vox [-rv]``` (int): voxel resolution in meter.  
    - ```--max_range [-mr]``` (int): Maximal distance range to seek for ray path/dem interceptions.  

OUTPUTS (default):  
    - ```vox_matrix_res<res_vox>m.npy``` (binary file in NumPy .npy format) : matrix (nvox, 31) describing cubic voxel faces geometry, summits xyz position, barycenter, position on surface or not. By default, the matrix is saved in ```files/voxel``` folder.  
    For each Soufrière telescope in ```files/telescopes/*/raypath``` :  
    - ```raypath.pkl``` : pickle binary file containing ray path apparent thickness in structure.
    - ```voxel/voxray_res<res_vox>m.pkl``` : matrix (nvox, nrays) containg the distance crossed by ray paths in each voxel.  
    - ```voxel/voxray_volume_res<res_vox>m.txt```: volume in cubic meter encompassed by telescopes sight cone (3 configurations if 4-panels telescope).  


# 2. inversion (work in progress):  
Based on the voxel model, conduct 3d inversion of synthetic density data and/or real mean density (opacity) data estimated from 2d muographs.  