import h5py


def dump(chkfile, key, val):
    with h5py.File(chkfile, 'a') as f:
        if key in f: del f[key]
        f[key] = val


def load(chkfile, key):
    with h5py.File(chkfile, 'r') as f:
        if key in f:
            val = f[key][()]
            if isinstance(val, bytes):
                val = val.decode()
            return val
        else:
            raise KeyError('key "%s" not found' % (str(key)))
