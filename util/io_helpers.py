# tools/io_helpers.py

import os
import numpy as np
from datetime import datetime

# Base directories relative to this file
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
LOG_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../log"))
OUT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../out"))

# Core utility functions
def log(message: str, file_name: str = None):
    """
    Append a timestamped message to a log file in the log folder.
    If file_name is provided, use it; otherwise use a generic default.
    """
    if file_name is None:
        file_name = "log.log"
    file_path = os.path.join(LOG_DIR, file_name)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(file_path, "a") as f:
        f.write(f"[{timestamp}] {message}\n")
    return file_name

def out(message: str, file_name: str = None):
    """
    Overwrite a file in the out folder.
    If file_name is provided, use it; otherwise use a generic default.
    """
    if file_name is None:
        file_name = "out.txt"
    file_path = os.path.join(OUT_DIR, file_name)
    with open(file_path, "a") as f:
        f.write(message + "\n")
    return file_name

def _ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path

