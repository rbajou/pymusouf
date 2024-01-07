# -*- coding: utf-8 -*-
#!/usr/bin/env python3

from pathlib import Path
import glob

MAIN_PATH = Path(__file__).parents[1]
FILES_DIR = MAIN_PATH / "files"
SURVEY_DIR = FILES_DIR / "survey"
LIST_AVAIL_SURVEY = [name.split('/')[-1] for name in glob.glob(str(SURVEY_DIR) + "/*")]

CURRENT_SURVEY_NAME = "soufriere"

if CURRENT_SURVEY_NAME not in LIST_AVAIL_SURVEY: raise FileExistsError(f"CURRENT_SURVEY_NAME '{CURRENT_SURVEY_NAME}' not available. \nChoose among {LIST_AVAIL_SURVEY}")


def use_paths():
    print(f"Main Path: {MAIN_PATH}")
    print(f"Files Path: {FILES_DIR}")


