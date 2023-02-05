import os
from file_utils import get_files, get_directories

class AutoIncrementFilename():
    def __init__(self, path : str, extension : str | None):
        self.path = path
        self.running_file_count = len(get_files(path, extension))

    def next_filename(self, basename : str, extension : str) -> tuple[str, int]:
        filename = os.path.join(self.path, f"{basename}{self.running_file_count}.{extension}")
        this_index = self.running_file_count
        self.running_file_count += 1
        return filename, this_index

class AutoIncrementDirectory():
    def __init__(self, path : str):
        self.path = path
        self.running_dir_count = len(get_directories(path))

    def next_directory(self, basename : str, auto_create=True) -> tuple[str, int]:
        dirname = os.path.join(self.path, f"{basename}{self.running_dir_count}")
        if auto_create:
            if not os.path.exists(dirname):
                os.makedirs(dirname)
        this_index = self.running_dir_count
        self.running_dir_count += 1
        return dirname, this_index
