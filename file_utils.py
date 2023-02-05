import os
import glob

def create_directory(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)

def create_directories(dirs : dict):
    for key in dirs.keys():
        create_directory(dirs[key])

def get_files(path : str, extension : str = "*") -> list:
    print(path)
    path = os.path.join(path, "*." + extension)

    entries = glob.glob(path)
    files = []
    for entry in entries:
        if not os.path.isdir(entry):
            files.append(entry)
    return files

def get_directories(path : str) -> list:
    entries = os.listdir(path)
    directories = []
    for entry in entries:
        if os.path.isdir(entry):
            directories.append(entry)
    return directories
