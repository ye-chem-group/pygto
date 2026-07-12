import os


def mkdir(path, recursive=False):
    if recursive:
        os.makedirs(path, exist_ok=True)
    else:
        if not os.path.isdir(path):
            os.mkdir(path)
