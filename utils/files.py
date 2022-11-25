from typing import Any, List, Dict
import os
import os.path as osp

def check_file_exist(filename: str):
    if not osp.isfile(filename):
        raise FileNotFoundError(f"{filename} does not exist")

def check_file_type(filename: str, types: List):
    fileext = osp.splitext(filename)[1]
    if fileext not in types:
        raise IOError(f"file type {fileext} is not supported")

def makedir(path: str, exist_ok: bool=True):
    if path == '':
        return
    os.makedirs(path, exist_ok=exist_ok)

        
