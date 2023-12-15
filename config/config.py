# -*- coding: utf-8 -*-
#!/usr/bin/env python3

from pathlib import Path
import os

MAIN_PATH = Path(__file__).parents[1]
FILES_PATH = MAIN_PATH / "files"

def use_paths():
    print(f"Main Path: {MAIN_PATH}")
    print(f"Files Path: {FILES_PATH}")

