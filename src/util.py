# -*- coding: utf-8 -*-

# python core libraries
import os

def ensure_dir(file_path:str) -> str:
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)
    return file_path

def path(*structure) -> str:
    return ensure_dir(os.path.join(*structure))

def list_files(directory:str):
    return [os.path.join(directory, entry) for entry in os.scandir(directory) if entry.is_file()]
