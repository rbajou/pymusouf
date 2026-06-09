#!/usr/bin/python3
# -*- coding: utf-8 -*-
import argparse
import json
from pathlib import Path

from telescope import str2telescope

CACHE_PATH = Path(".args_cache.json")

def load_args_cache(path=CACHE_PATH):
    if path.exists():
        try:
            with open(path, "r") as f:
                return json.load(f)
        except Exception as e:
            print(f"Warning: could not load args cache: {e}")
    return {}

def save_args_cache(args_dict, path=CACHE_PATH):
    try:
        with open(path, "w") as f:
            json.dump(args_dict, f, indent=2)
    except Exception as e:
        print(f"Warning: could not save args cache: {e}")

def get_pre_parser(saved_args={}):
    pre_parser = argparse.ArgumentParser(add_help=False)
    #pre_parser.add_argument(
        # '--tracking_type', 
        # default=saved_args.get("tracking_type", "ransac"),
        # choices=['ransac', 'hough', 'other'],
        # help="Tracking type: ransac, hough, or other"
    # )
    return pre_parser

def set_common_args(parser, saved_args = {}): #simu:bool
    parser.add_argument('--survey', '-s', default=saved_args.get("survey", "soufriere"), help="Survey object. See 'survey/survey.yaml'", type=str)
    parser.add_argument('--telescope', '-t', default=saved_args.get("telescope", "SNJ"), help="Telescope configuration. See 'survey/survey.yaml'", type=str)
    parser.add_argument('--run', '-r', default=saved_args.get("run", "tomo"), help="Run label", type=str)
 
def set_common_parser():
    saved_args = load_args_cache()
    pre_parser = get_pre_parser(saved_args) 
    parser = argparse.ArgumentParser(
        parents=[pre_parser],
        description="Common parser"
    )
    # args_pre = pre_parser.parse_known_args()
    set_common_args(parser, saved_args)   #args_pre.simu,
    return parser

def get_common_args(save:bool=True) :
    parser = set_common_parser()
    args = parser.parse_args()
    if save: save_args_cache(vars(args))
    return args
