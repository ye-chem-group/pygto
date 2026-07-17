import h5py


def dump(chkfile, key, val):
    ''' Save a value under a key in an HDF5 checkpoint file.

        Args:
            chkfile (str):
                Path to checkpoint file.
            key (str):
                HDF5 key. An existing value at this key is replaced.
            val (object):
                Value accepted by h5py.
    '''
    with h5py.File(chkfile, 'a') as f:
        if key in f: del f[key]
        f[key] = val


def load(chkfile, key):
    ''' Load a value from an HDF5 checkpoint file.

        Args:
            chkfile (str):
                Path to checkpoint file.
            key (str):
                HDF5 key.

        Return:
            value (object):
                Stored value. Byte strings are decoded to text.
    '''
    with h5py.File(chkfile, 'r') as f:
        if key in f:
            val = f[key][()]
            if isinstance(val, bytes):
                val = val.decode()
            return val
        else:
            raise KeyError('key "%s" not found' % (str(key)))
