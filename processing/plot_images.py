import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, Normalize
from mpl_toolkits.axes_grid1 import make_axes_locatable, inset_locator
import numpy as np 
import pandas as pd
import pickle
from scipy.interpolate import griddata
from scipy.io import loadmat
#package modules
from cli import get_common_args
from config import STRUCT_DIR
from survey import CURRENT_SURVEY
from eventrate import EventRate
from utils.common import Common
from utils.tools import print_file_datetime

params = {'legend.fontsize': 'medium',
          'legend.title_fontsize' : 'x-large',
         'axes.labelsize': 'xx-large',
         'axes.titlesize':'xx-large',
         'xtick.labelsize':'x-large',
         'ytick.labelsize':'x-large',
         'axes.labelpad':1,
         'mathtext.fontset': 'stix',
         'font.family': 'STIXGeneral',
         'axes.grid': False,
         'figure.figsize': (8,8),
          'savefig.bbox': "tight",   
        'savefig.dpi':200    }
props = dict(boxstyle='round', facecolor='white', alpha=0.5)
plt.rcParams.update(params)

if __name__ == "__main__":

    args = get_common_args(save=False)
    cmn = Common(args)
    survey = cmn.survey
    survey_name = survey.name   
    dir_survey = STRUCT_DIR / survey_name
    dir_flux = dir_survey / "flux"
    tel = cmn.telescope
    runs = survey.runs[tel.name]
    dict_conf = tel.configurations
    
    tel.compute_angular_coordinates()

    file_track = cmn.run.dirs["reco"] / "df_track.csv.gz"
    print_file_datetime(file_track)
    dir_out = cmn.run.dirs["png"] 
    df = pd.read_csv(file_track)
    df.set_index("event_id", inplace=True)
    
    grp_evt_id = df.groupby('event_id')
    t = grp_evt_id['timestamp'].first().to_numpy()
    er = EventRate(time=t, t0=0)
    
    fig, ax = plt.subplots(figsize=(12,9))  
    window = 12 if "tomo" in args.run else 3
    _, rate = er.time_series(ax, width=3600, window=window, label="", **{"alpha":1., "linewidth":1, "color":tel.color})
    ax.set_xlim(0, ax.get_xticks()[-1])
    ax.set_ylim(0, ax.get_yticks()[-1])
    nev = len(t)
    ax.text( 0.82,0.95,
            f"nev={nev:.2e}", 
            fontsize="xx-large", ha="left",va="bottom", transform=ax.transAxes, # fontweight="bold",
        )
    fout = dir_out / f"eventrate.png"
    ax.grid(alpha=0.2)
    fig.savefig(fout)
    print(f"Save {fout}")

    ncols,nrows = len(dict_conf),2
    fig, axs = plt.subplots(ncols=ncols, nrows=nrows, figsize = (6*ncols, 6*nrows), constrained_layout=True,  sharex=True, sharey=True) #gridspec_kw=kwargs_size)
    if not isinstance(axs, np.ndarray): axs = np.array([axs])
    if axs.ndim ==1 : axs=axs[:,np.newaxis]
    vmax = 0
    dict_out = {}
    l_h2d = []
    gold = (df['nhits_0']==1) & (df['nhits_1']==1) & (df['nhits_2']==1) 
    mip_like = (df['nhits_0']<=10) & (df['nhits_2']<=10) 
    gold = gold & (df['nhits_3']==1) if len(tel.panels) > 3 else gold
    rms =  df["rms"]
    for i in range(2):
        fin = i 
        for j,(c, conf) in enumerate(dict_conf.items()):
            mask_conf = (df['config'] == c) 
            mask = mask_conf
            if fin == 1 : 
                p90 = np.percentile(rms, 90)
                mask_rms = (rms < p90).astype(bool)
                mask = mask_conf & mip_like & mask_rms
            x, y = df["dx_dz"][mask], df["dy_dz"][mask]
            range_xy = conf.range_uv
            shp = tel.azimuth_matrix[c].shape
            bins = shp #
            h, binx, biny = np.histogram2d(x, y, bins=bins, range=range_xy)
            if i == 0 : dict_out[c] = {"h":h,"binx":binx,"biny":biny}
            _max = np.max(h) 
            vmax= _max if vmax < _max else vmax
            if i == 0 :axs[i,j].set_title(c + " config")
            axs[i,j].text(0.05, 0.92, f"total: {np.sum(h):.2e}", fontsize="x-large", color="black", 
                    transform=axs[i,j].transAxes, 
                    bbox= props,
                    ha="left",va="bottom",**{"fontweight":"bold"})  
            l_h2d.append((h, binx, biny))

    for i, ax in enumerate(axs.ravel()):
        h, bx, by = l_h2d[i]
        h = np.flipud(h) if (tel.name == "SXF") or (tel.name == "OM") else h
        # im = ax.imshow(h, norm=LogNorm(1, vmax))
        im = ax.pcolormesh(bx, by, h, norm=LogNorm(1, vmax))
        if i == len(axs.ravel())-1:
            # divider = make_axes_locatable(ax)
            # cax = divider.append_axes("right", size="5%", pad=0.1)
            cax = inset_locator.inset_axes(ax, width="4%",  height="100%", borderpad=-3,loc = 'right')
            cb= fig.colorbar(im, cax=cax, extend='max')
            cb.set_label(f'Entries', labelpad=1)
            cb.ax.tick_params(which="both", labelsize="x-large",pad=1)    
        # ax.set_aspect("equal")
        ax.set_xlabel("tan($\\theta_x$)")
        ax.set_ylabel("tan($\\theta_y$)")
        ax.label_outer()
    # fig.tight_layout()
    file_out = dir_out / "images_brut.png"
    fig.savefig(file_out, bbox_inches="tight", dpi=200)
    print(f"Save {file_out}")
    # '''

    ncols,nrows=len(tel.configurations),1
    fig, axs = plt.subplots(ncols=ncols, nrows=nrows, figsize = (6*ncols, 6*nrows), constrained_layout=True,  sharex=True, sharey=True) #gridspec_kw=kwargs_size
    if not isinstance(axs, np.ndarray): axs = np.array([axs])
    for j,(c, conf) in enumerate(dict_conf.items()):
        ax = axs[j]
        ax.set_title( tel.name + "-" + c)
        mask_conf = (df['config'] == c) 
        rms =  df["rms"][mask_conf]
        vmin, vmax = np.min(rms), np.max(rms)
        h, edges = np.histogram(rms, bins=50, range=(vmin, vmax))
        ax.bar(edges[:-1], h, width=np.diff(edges), edgecolor='darkgrey', align="edge", alpha=1, label="All") 
        # mask_det = mask_conf & (df['ransac'] == 0)
        # rms_det =  df["rms"][mask_det]
        # h, edges = np.histogram(rms_det, bins=50, range=(vmin, vmax))
        # ax.bar(edges[:-1], h, width=np.diff(edges), edgecolor='darkgrey', align="edge", alpha=.7, label="Deterministic")   
        # mask_ransac = mask_conf & (df['ransac'] == 1)
        # rms_ransac =  df["rms"][mask_ransac]
        # h, edges = np.histogram(rms_ransac, bins=50, range=(vmin, vmax))
        # ax.bar(edges[:-1], h, width=np.diff(edges), edgecolor='darkgrey', align="edge", alpha=.5, label="RANSAC")   
        # ax.set_title(c + " config")
        ax.set_xlabel("RMS [mm]")
        ax.set_ylabel("Entries")
        p90 = np.percentile(rms, 90)
        ax.axvline(p90, color='orange', linestyle='--', label=f'90th perc: {p90:.2f} mm')
        p95 = np.percentile(rms, 95)
        ax.axvline(p95, color='red', linestyle='--', label=f'95th perc: {p95:.2f} mm')
        # ax.set_yscale("log")
        ax.legend() 
    file_out = dir_out / "rms.png"
    fig.savefig(file_out, bbox_inches="tight", dpi=200)
    print(f"Save {file_out}")


    ncols,nrows=len(tel.configurations),2
    fig, axs = plt.subplots(ncols=ncols, nrows=nrows, figsize = (6*ncols, 6*nrows), constrained_layout=True,  sharex=True, sharey=True) #gridspec_kw=kwargs_size
    if not isinstance(axs, np.ndarray): axs = np.array([axs])
    if axs.ndim ==1 : axs=axs[:,np.newaxis]    
    for j,(c, conf) in enumerate(dict_conf.items()):
        mask_conf = (df['config'] == c) 
        for i, coord in enumerate(["x","y"]):
            ax = axs[i,j]
            if i==0:
                ax.set_title( tel.name + "-" + c)
            k = f"mip_score_{coord}"
            score_x =  df[k][mask_conf]
            vmin, vmax = np.min(score_x), np.max(score_x)
            h, edges = np.histogram(score_x, bins=50, range=(vmin, vmax))
            ax.bar(edges[:-1], h, width=np.diff(edges), align="edge", alpha=1, label="All")     
            # ax.set_title(c + " config")
            ax.set_xlabel(k)
            ax.set_ylabel("Entries")
            # ax.set_yscale("log")
            ax.legend() 
    file_out = dir_out / "mip_scores.png"
    fig.savefig(file_out, bbox_inches="tight", dpi=200)
    print(f"Save {file_out}")


    ncols,nrows=len(tel.configurations),2
    fig, axs = plt.subplots(ncols=ncols, nrows=nrows, figsize = (6*ncols, 6*nrows), constrained_layout=True,  sharex=True, sharey=True) #gridspec_kw=kwargs_size
    if not isinstance(axs, np.ndarray): axs = np.array([axs])
    if axs.ndim ==1 : axs=axs[:,np.newaxis]    
    for j,(c, conf) in enumerate(dict_conf.items()):
        mask_conf = (df['config'] == c) 
        for i, coord in enumerate(["x","y"]):
            if i==0: ax.set_title( tel.name + "-" + c)
            ax = axs[i,j]
            h2d, ex, ey = np.histogram2d(df[f"mip_score_{coord}"][mask_conf], df["rms"][mask_conf], bins=50)
            bx, by = (ex[:-1] + ex[1:]) / 2, (ey[:-1] + ey[1:]) / 2
            ax.pcolormesh(bx, by, h2d.T, norm=LogNorm(1, np.max(h2d)))
            ax.set_xlabel("mip_score_" + coord)
            ax.set_ylabel("RMS [mm]") 
    file_out = dir_out / "rms_vs_mip_scores.png"
    fig.savefig(file_out, bbox_inches="tight", dpi=200)
    print(f"Save {file_out}")

    ncols,nrows=len(tel.panels),1
    fig, axs = plt.subplots(ncols=ncols, nrows=nrows, figsize = (6*ncols, 6*nrows), constrained_layout=True,  sharex=True, sharey=True) #gridspec_kw=kwargs_size
    if not isinstance(axs, np.ndarray): axs = np.array([axs])
    # if axs.ndim ==1 : axs=axs[:,np.newaxis]    
    for j, panel in enumerate(tel.panels):
        ax = axs[j]
        v = df[f"nhits_{j}"]
        v = v[v>0]
        vmin, vmax = np.min(v), np.max(v)
        h, edges = np.histogram(v, bins=range(1, int(vmax)+2))
        ax.bar(edges[:-1], h, width=np.diff(edges), align="edge", alpha=1, label="All")     
        ax.set_title(f"Panel {panel.id} - nhits")
        ax.set_xlabel("Number of hits")
        ax.set_ylabel("Entries")    
    file_out = dir_out / "hit_multiplicity.png"
    fig.savefig(file_out, bbox_inches="tight", dpi=200)
    print(f"Save {file_out}")