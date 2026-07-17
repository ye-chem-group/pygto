import os


def mkdir(path, recursive=False):
    ''' Create a directory if it does not already exist.

        Args:
            path (str or path-like):
                Directory path.
            recursive (bool):
                Whether to create missing parent directories. Default is False.
    '''
    if recursive:
        os.makedirs(path, exist_ok=True)
    else:
        if not os.path.isdir(path):
            os.mkdir(path)
