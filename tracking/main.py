#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import os
from pathlib import Path
import argparse
import time
import logging
import pandas as pd

#personal modules
from survey import CURRENT_SURVEY
from survey.data import RawData
from survey.run import Run
from telescope import DICT_TEL,  str2telescope
from tracking import RansacModel, RansacTracking
from utils.tools import str2bool, print_progress


start_time = time.time()
t0 = time.strftime("%H:%M:%S", time.localtime())
print("Start: ", t0)#start time
t_start = time.perf_counter()
home_path = os.environ["HOME"]
parser=argparse.ArgumentParser(
description='''For a given muon telescope configuration, this script allows to perform RANSAC tracking and outputs trajectrory-panel crossing XY coordinates''', epilog="""All is well that ends well.""")
parser.add_argument('--telescope', '-tel', default=DICT_TEL["SNJ"], help='Input telescope name. It provides the associated configuration.', type=str2telescope)
parser.add_argument('--input_data', '-i', default=[], nargs="*", help='/path/to/datafile/  One can input a data directory, a single datfile, or a list of data files e.g "--input_data <file1.dat> <file2.dat>"', type=str)
# parser.add_argument('--input_data', '-i', default=None, help='/path/to/datafile/  One can input a data directory, a single datfile, or a list of data files e.g "--input_data <file1.dat> <file2.dat>"', type=str)
parser.add_argument('--out_dir', '-o', default='out', help='Path to processing output', type=str) 
parser.add_argument('--input_type', '-it', default='real',  help="'real' or 'mc'", type=str)
parser.add_argument('--max_nfiles', '-max', default=1, help='Maximum number of dataset files to process.', type=int)
parser.add_argument('--residual_threshold', '-rt', default=50, help="RANSAC 'distance-to-model' parameter in mm",type=float)
parser.add_argument('--min_samples', '-ms', default=2, help='RANSAC size of the initial sample',type=int)
parser.add_argument('--max_trials', '-mt', default=100, help='RANSAC number of iterations',type=int)
parser.add_argument('--fit_intersect', '-intersect', default=False, help='if true record line model intersection points on panel; else record closest XY inlier points to model',type=str2bool)
parser.add_argument('--info', '-info', default=None, help='Additional info',type=str)
parser.add_argument('--progress_bar', '-bar', default=False, help='Display progress bar',type=str2bool)
args=parser.parse_args()


survey = CURRENT_SURVEY[args.telescope.name]

out_dir = Path(args.out_dir)
out_dir.mkdir(parents=True, exist_ok=True)

start_time = time.time()
reco_dir = out_dir 
reco_dir.mkdir(parents=True, exist_ok=True)

strdate = time.strftime("%d%m%Y_%H%M")
flog =str(out_dir/f'{strdate}.log')
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s",
                    filemode='w',
                    #filename=flog,)
                    stream=sys.stdout,) #either set 'filename' to save info in log file or 'stream' to print out on console
logging.info(sys.argv)
logging.info(f"Start -- {t0}")
if args.info : logging.info(args.info) #additional info

kwargs_ransac = dict(residual_threshold=args.residual_threshold, 
            min_samples=args.min_samples, 
            max_trials=args.max_trials,  
) 
            #is_fit_intersect=args.fit_intersect)

logging.info('\nRansac Tracking...\n')
rawdata_path = [ Path(p) for p in args.input_data ]

runs = []
if len(rawdata_path) == 0 : 
    runs = survey.run_tomo
else : 
    for praw in rawdata_path: 
        raw = RawData(path=praw)
        run = Run(name = praw, telescope = args.telescope, rawdata = [raw])
        runs.append(run)

n, nruns = 0, len(runs)
for run in runs:
    logging.info(run)
    for raw in run.rawdata:
        # print(raw)
        raw.fill_dataset(max_nfiles = args.max_nfiles)

        tracking = RansacTracking(telescope = args.telescope, data = raw )
        tracking.process(model_type = RansacModel, progress_bar = args.progress_bar, **kwargs_ransac)
        
        logging.info(tracking)

        print(f"df_track.head = {tracking.df_track.head}")
        ftrack = out_dir / 'df_track.csv.gz'
        tracking.df_track.to_csv(ftrack, compression='gzip', index=False, sep='\t')
        logging.info(f"Save dataframe {ftrack}")

        logging.info(f"df_model.head = {tracking.df_model.head}")
        fmodel = out_dir / 'df_inlier.csv.gz' #ransac inlier pt-tagging output for all reco events
        tracking.df_model.to_csv(fmodel, compression='gzip', index=False, sep='\t')
        logging.info(f"Save dataframe {fmodel}")

    print_progress(n+1, nruns, prefix = 'Run(s) processed :', suffix = 'completed')
    n += 1

t_sec = round(time.time() - start_time)
(t_min, t_sec) = divmod(t_sec,60)
(t_hour,t_min) = divmod(t_min,60)
t_end = 'Duration : {}hour:{}min:{}sec'.format(t_hour,t_min,t_sec)
logging.info(t_end)

logging.info(f"Output directory : {reco_dir}")
