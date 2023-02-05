import os

class AutoIncrementFilename():
    def __init__(self, path : str):
        self.path = path
        self.num_files = len(os.listdir(path))

    def next_filename(self, basename : str, extension : str) -> str:
        filename = os.path.join(self.path, f"{basename}{self.num_files}.{extension}")
        self.num_files += 1
        return filename
