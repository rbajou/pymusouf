# -*- coding: utf-8 -*-
#!/usr/bin/env python3

import numpy as np
from pathlib import Path
import time
import argparse
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import Normalize, to_rgba
#personal modules
from raypath import RayPath
from telescope import dict_tel
from voxel import Voxel, DirectProblem
from utils import rotanimate


if __name__ == "__main__":
    
    t0 = time.time()
    print("Start: ", time.strftime("%H:%M:%S", time.localtime()))#start time
   
    main_path =  Path(__file__).parents[2]
    files_path = main_path/ "files"
    dem_path = files_path / "dem"
    filename2 = "soufriereStructure_2.npy" #res 5m
    surface_grid = np.load(dem_path / filename2)
    surface_center = np.loadtxt(dem_path / "volcanoCenter.txt").T
    
    parser=argparse.ArgumentParser(
    description='''La Soufrière dome voxelization and computation of kernel matrices (inverse problem solving) for a given telescope list''', epilog="""All is well that ends well.""")
    parser.add_argument('--res_vox', '-rv', default=64, help='Voxel resolution in meter',  type=int)
    parser.add_argument('--max_range', '-mr', default=1500, help='Maximal distance range to seek for ray path-surface interceptions', type=int)
    #parser.add_argument('--out_dir', '-o', default=str(main_path/"out"), help='/path/to/output/dir/',  type=str)
    args=parser.parse_args()
    
    res_vox = args.res_vox #m
    max_range = args.max_range
    voxel = Voxel(    surface_grid=surface_grid,
                      surface_center=surface_center, 
                      res_vox=res_vox)
    
    dout_vox_struct = files_path / "voxel"
    dout_vox_struct.mkdir(parents=True, exist_ok=True)
    fout_vox_struct = dout_vox_struct / f"voxMatrix_res{res_vox}m.npy"
    if fout_vox_struct.exists(): 
        print(f"Load {fout_vox_struct.relative_to(main_path)}")
        vox_matrix = np.load(fout_vox_struct)
        voxel.vox_matrix = vox_matrix
    else : 
        print(f"generateMesh() start")
        voxel.generateMesh()
        vox_matrix = voxel.vox_matrix
        np.save(fout_vox_struct, vox_matrix)
        print(f"generateMesh() end --- {time.time() - t0:.1f} s")

    voxel.getVoxels()

    lname = ['SB', 'SNJ', 'BR', 'OM']
    ltel = [dict_tel[n] for n in lname]

    for tel in ltel:
        tel_files_path = files_path / "telescopes"  / tel.name
        dout_ray = tel_files_path / "raypath" / f"az{tel.azimuth:.1f}_elev{tel.elevation:.1f}" 
        rp = RayPath(telescope=tel,
                        surface_grid=surface_grid,)
        rp(file=dout_ray / "raypath", max_range=max_range)

        dirpb = DirectProblem(telescope=tel, 
                                    vox_matrix=voxel.vox_matrix, 
                                    res_vox=res_vox)
        fout_voxray = dout_ray / "voxel" / f"voxray_res{res_vox}m"
        fout_voxray.parent.mkdir(parents=True, exist_ok=True)
        dirpb(file=fout_voxray, raypath=rp.raypath)
        
        out_file = fout_voxray.parent / f"voxray_volume_res{res_vox}m.txt"
        
        with open(str(out_file), 'w') as f: 
            for c,_ in tel.configurations.items():
                voxrayMatrix = dirpb.voxray[c]
                voxrayMatrix[np.isnan(voxrayMatrix)] = 0
                mvox = np.any(voxrayMatrix>0, axis=0)
                sv_vox = np.where(mvox==True)[0]
                vol = voxel.getVolumeRegion(sv_vox=sv_vox)
                print(f"Volume covered by {tel.name} ({c}) : {vol:.5e} m^3")
                f.write(f"{c}\t{vol:.5e}\n")
        print(f"Saved in {str(out_file.relative_to(main_path))}")
        
    ###PLOTS

    '''
    fig = plt.figure(figsize=(12,7))
    ax = fig.add_subplot(111, projection='3d')    
    color_voxdome = np.array([[0.98039216, 0.8,  0.98039216, 1.        ]])
    kwargs_voxdome =  dict(alpha=0.1,
                                facecolors=np.repeat(color_voxdome,6,axis=0),
                                edgecolors=np.clip(color_voxdome - 0.5, 0, 1),  # brighter 
                                norm=Normalize(clip=True), 
                                linewidth=0.3)
    mask_notsurf = (vox_matrix[:,0] == 1)
    voxdome_xyz = np.copy(voxel.vox_xyz)  
    voxel.plot3Dmesh(ax=ax, vox_xyz=voxdome_xyz[mask_notsurf], **kwargs_voxdome)
    kwargs_topo = dict ( alpha=0.2, color='lightgrey', edgecolor='grey' )
    voxel.plotTopography(ax, **kwargs_topo)
    for tel in ltel: 
        col_voxtel = np.array([[i for i in to_rgba(tel.color)]])
        kwargs_voxtel =  dict(alpha=0.5,
                            facecolors=np.repeat(col_voxtel,6,axis=0),
                            edgecolors=np.clip(col_voxtel - 0.5, 0, 1),  # brighter 
                            norm=Normalize(clip=True), 
                            linewidth=0.3)
        for conf, panels in tel.configurations.items():
            print(f"Plot voxel volume covered by {tel.name} ({c})")
            voxrayMatrix = dirpb.voxray[conf]
            voxrayMatrix[np.isnan(voxrayMatrix)] = 0
            mask_tel = np.any(voxrayMatrix>0, axis=0)
            voxtel_xyz = voxel.vox_xyz[mask_tel]
            voxel.plot3Dmesh(ax,vox_xyz=voxtel_xyz, **kwargs_voxtel)

            thick = rp.raypath[conf]['thickness'] 
            mask = np.isnan(thick).flatten()
            nrays = len(mask[False])
            front, rear = panels[0], panels[-1]
            rmax = 1000
            tel.plot_ray_paths(ax, 
                                front_panel=front, 
                                rear_panel=rear, 
                                mask=mask, 
                                rmax=rmax, 
                                c="grey",
                                linewidth=0.5 )

    ltel_coord = np.array([ tel.utm for tel in ltel])
    ltel_color = np.array([ tel.color for tel in ltel])
    ax.scatter(ltel_coord[:,0], ltel_coord[:,1], ltel_coord[:,-1], c=ltel_color, s=40,marker='*',edgecolor='black',)
    dx, dy = 800, 1100
    xrange = [surface_center[0]-dx, surface_center[0]+dx]
    yrange = [surface_center[1]-dy, surface_center[1]+dy]
    zrange = [1.0494e+03, 1.4658e+03 + 50]
    ax.set_xlim(xrange)
    ax.set_ylim(yrange)
    ax.set_zlim(zrange)
    ax.grid()
    #ax.dist = 8    # define perspective
    xstart, xend = ax.get_xlim()
    ax.xaxis.set_ticks(np.arange(xstart, xend, 5e2))
    ystart, yend = ax.get_ylim()
    ax.yaxis.set_ticks(np.arange(ystart, yend, 5e2))
    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")

    plt.show()
    '''

    # elev = 30
    # langle_az = np.linspace(-180, 180, 50)
  
    # def init():
    #     ax.view_init(elev=30, azim=-60)
    #     return fig,

    # def animate(az):
    #     ax.view_init(elev=elev, azim=az)
    #     return fig,
    
    # # Animate
    # anim = animation.FuncAnimation(fig, animate, init_func=init,
    #                            frames=10, interval=10, blit=True)

    # create an animated gif (20ms between frames)
    #fout = 'out/voxdome.html'
    #angles = np.linspace(0,360,21)[:-1] # Take 20 angles between 0 and 360
    #rotanimate(ax, angles,fout,delay=3) 


    # Save
    #anim.save(fout, writer="html", fps=1)#, extra_args=['-vcodec', 'libx264'])
    #fig.savefig(, orientation='landscape', transparent=True)


   
    print(f"End --- {time.time() - t0:.3f} s")
