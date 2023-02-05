import os

def create_directories(dirs : dict):
    for key in dirs.keys():
        dir = dirs[key]
        if not os.path.exists(dir):
            os.makedirs(dir)
