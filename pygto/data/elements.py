import numpy as np


ATOM_SYMBOLS = [
    'X',                                                            # Ghost
    'H', 'He',                                                      # 1--2
    'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne',                      # 3--10
    'Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 'Ar',                   # 11--18
    'K', 'Ca',                                                      # 19--20
    'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn',      # 21--30
    'Ga', 'Ge', 'As', 'Se', 'Br', 'Kr',                             # 31--36
    'Rb', 'Sr',                                                     # 37--38
    'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd',      # 39--48
    'In', 'Sn', 'Sb', 'Te', 'I', 'Xe',                              # 49--54
    'Cs', 'Ba',                                                     # 55--56
        'La', 'Ce', 'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb',
        'Dy', 'Ho', 'Er', 'Tm', 'Yb',                               # 57--70 (La series)
    'Lu', 'Hf', 'Ta', 'W' , 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg',     # 71--80
    'Tl', 'Pb', 'Bi', 'Po', 'At', 'Rn',                             # 81--86
    'Fr', 'Ra',                                                     # 87--88
        'Ac', 'Th', 'Pa', 'U', 'Np', 'Pu', 'Am', 'Cm', 'Bk',
        'Cf', 'Es', 'Fm', 'Md', 'No',                               # 89--102 (Ac series)
    'Lr', 'Rf', 'Db', 'Sg', 'Bh', 'Hs', 'Mt', 'Ds', 'Rg', 'Cn',     # 103--112
    'Nh', 'Fl', 'Mc', 'Lv', 'Ts', 'Og',                             # 113--118
]

# ground-state spin
SPIN = [
    0,
    1, 0,   # H, He
    1, 0, 1, 2, 3, 2, 1, 0, # Li--Ne
    1, 0, 1, 2, 3, 2, 1, 0, # Na--Ar
    1, 0, 1, 2, 3, 6, 5, 4, 3, 2, 1, 0, 1, 2, 3, 2, 1, 0, # K--Kr
]

# number of electrons for valence correlation
NEVAL = [
    0,
    1, 2,   # H, He
    1, 2, 3, 4, 5, 6, 7, 8, # Li--Ne
    1, 2, 3, 4, 5, 6, 7, 8, # Na--Ar
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 3, 4, 5, 6, 7, 8, # K--Kr
]


# Electronic configurations of the elements
# H--U are taken from https://math.nist.gov/DFTdata/atomdata/configuration.html
# Np--Lr are taken from the NIST Elemental Data Index.
# Elements after Lr use predicted configurations.
CONFIGURATION = [
    [ 0,  0,  0,  0],   # 0: X
    [ 1,  0,  0,  0],   # 1: H
    [ 2,  0,  0,  0],   # 2: He
    [ 3,  0,  0,  0],   # 3: Li
    [ 4,  0,  0,  0],   # 4: Be
    [ 4,  1,  0,  0],   # 5: B
    [ 4,  2,  0,  0],   # 6: C
    [ 4,  3,  0,  0],   # 7: N
    [ 4,  4,  0,  0],   # 8: O
    [ 4,  5,  0,  0],   # 9: F
    [ 4,  6,  0,  0],   # 10: Ne
    [ 5,  6,  0,  0],   # 11: Na
    [ 6,  6,  0,  0],   # 12: Mg
    [ 6,  7,  0,  0],   # 13: Al
    [ 6,  8,  0,  0],   # 14: Si
    [ 6,  9,  0,  0],   # 15: P
    [ 6, 10,  0,  0],   # 16: S
    [ 6, 11,  0,  0],   # 17: Cl
    [ 6, 12,  0,  0],   # 18: Ar
    [ 7, 12,  0,  0],   # 19: K
    [ 8, 12,  0,  0],   # 20: Ca
    [ 8, 12,  1,  0],   # 21: Sc
    [ 8, 12,  2,  0],   # 22: Ti
    [ 8, 12,  3,  0],   # 23: V
    [ 7, 12,  5,  0],   # 24: Cr
    [ 8, 12,  5,  0],   # 25: Mn
    [ 8, 12,  6,  0],   # 26: Fe
    [ 8, 12,  7,  0],   # 27: Co
    [ 8, 12,  8,  0],   # 28: Ni
    [ 7, 12, 10,  0],   # 29: Cu
    [ 8, 12, 10,  0],   # 30: Zn
    [ 8, 13, 10,  0],   # 31: Ga
    [ 8, 14, 10,  0],   # 32: Ge
    [ 8, 15, 10,  0],   # 33: As
    [ 8, 16, 10,  0],   # 34: Se
    [ 8, 17, 10,  0],   # 35: Br
    [ 8, 18, 10,  0],   # 36: Kr
    [ 9, 18, 10,  0],   # 37: Rb
    [10, 18, 10,  0],   # 38: Sr
    [10, 18, 11,  0],   # 39: Y
    [10, 18, 12,  0],   # 40: Zr
    [ 9, 18, 14,  0],   # 41: Nb
    [ 9, 18, 15,  0],   # 42: Mo
    [10, 18, 15,  0],   # 43: Tc
    [ 9, 18, 17,  0],   # 44: Ru
    [ 9, 18, 18,  0],   # 45: Rh
    [ 8, 18, 20,  0],   # 46: Pd
    [ 9, 18, 20,  0],   # 47: Ag
    [10, 18, 20,  0],   # 48: Cd
    [10, 19, 20,  0],   # 49: In
    [10, 20, 20,  0],   # 50: Sn
    [10, 21, 20,  0],   # 51: Sb
    [10, 22, 20,  0],   # 52: Te
    [10, 23, 20,  0],   # 53: I
    [10, 24, 20,  0],   # 54: Xe
    [11, 24, 20,  0],   # 55: Cs
    [12, 24, 20,  0],   # 56: Ba
    [12, 24, 21,  0],   # 57: La
    [12, 24, 21,  1],   # 58: Ce
    [12, 24, 20,  3],   # 59: Pr
    [12, 24, 20,  4],   # 60: Nd
    [12, 24, 20,  5],   # 61: Pm
    [12, 24, 20,  6],   # 62: Sm
    [12, 24, 20,  7],   # 63: Eu
    [12, 24, 21,  7],   # 64: Gd
    [12, 24, 20,  9],   # 65: Tb
    [12, 24, 20, 10],   # 66: Dy
    [12, 24, 20, 11],   # 67: Ho
    [12, 24, 20, 12],   # 68: Er
    [12, 24, 20, 13],   # 69: Tm
    [12, 24, 20, 14],   # 70: Yb
    [12, 24, 21, 14],   # 71: Lu
    [12, 24, 22, 14],   # 72: Hf
    [12, 24, 23, 14],   # 73: Ta
    [12, 24, 24, 14],   # 74: W
    [12, 24, 25, 14],   # 75: Re
    [12, 24, 26, 14],   # 76: Os
    [12, 24, 27, 14],   # 77: Ir
    [11, 24, 29, 14],   # 78: Pt
    [11, 24, 30, 14],   # 79: Au
    [12, 24, 30, 14],   # 80: Hg
    [12, 25, 30, 14],   # 81: Tl
    [12, 26, 30, 14],   # 82: Pb
    [12, 27, 30, 14],   # 83: Bi
    [12, 28, 30, 14],   # 84: Po
    [12, 29, 30, 14],   # 85: At
    [12, 30, 30, 14],   # 86: Rn
    [13, 30, 30, 14],   # 87: Fr
    [14, 30, 30, 14],   # 88: Ra
    [14, 30, 31, 14],   # 89: Ac
    [14, 30, 32, 14],   # 90: Th
    [14, 30, 31, 16],   # 91: Pa
    [14, 30, 31, 17],   # 92: U
    [14, 30, 31, 18],   # 93: Np
    [14, 30, 30, 20],   # 94: Pu
    [14, 30, 30, 21],   # 95: Am
    [14, 30, 31, 21],   # 96: Cm
    [14, 30, 30, 23],   # 97: Bk
    [14, 30, 30, 24],   # 98: Cf
    [14, 30, 30, 25],   # 99: Es
    [14, 30, 30, 26],   # 100: Fm
    [14, 30, 30, 27],   # 101: Md
    [14, 30, 30, 28],   # 102: No
    [14, 31, 30, 28],   # 103: Lr
    [14, 30, 32, 28],   # 104: Rf
    [14, 30, 33, 28],   # 105: Db
    [14, 30, 34, 28],   # 106: Sg
    [14, 30, 35, 28],   # 107: Bh
    [14, 30, 36, 28],   # 108: Hs
    [14, 30, 37, 28],   # 109: Mt
    [14, 30, 38, 28],   # 110: Ds
    [14, 30, 39, 28],   # 111: Rg
    [14, 30, 40, 28],   # 112: Cn
    [14, 31, 40, 28],   # 113: Nh
    [14, 32, 40, 28],   # 114: Fl
    [14, 33, 40, 28],   # 115: Mc
    [14, 34, 40, 28],   # 116: Lv
    [14, 35, 40, 28],   # 117: Ts
    [14, 36, 40, 28],   # 118: Og
]


def get_atom_symbol(Z):
    if Z <= 0 or Z > len(ATOM_SYMBOLS):
        raise ValueError('Z must be in [0, %d]' % (len(ATOM_SYMBOLS)))
    return ATOM_SYMBOLS[Z]

def get_Z(atm):
    if atm in ATOM_SYMBOLS:
        return ATOM_SYMBOLS.index(atm)
    else:
        raise ValueError('Invalid atm "%s"' % (str(atm)))
get_charge = get_Z

def get_spin(atm):
    Z = get_Z(atm)
    if Z > 36:
        raise NotImplementedError('Default spin is not available for Z > 36.')
    return SPIN[Z]

def get_neval(atm):
    Z = get_Z(atm)
    if Z > 36:
        raise NotImplementedError('Default neval is not available for Z > 36.')
    return NEVAL[Z]

def get_all_configs(atm):
    atm_cores = [
        'He',               # 1s
        'Be', 'Ne',         # 2s 2p
        'Mg', 'Ar',         # 3s 3p
        'Ca', 'Zn', 'Kr',   # 4s 3d 4p
        'Sr', 'Cd', 'Xe',   # 5s 4d 5p
        'Ba', 'Hg', 'Rn',   # 6s 4f+5d 6p
        'Ra', 'Cn'          # 7s 5f+6d
    ]
    config = get_config(atm)
    ne = sum(config)
    configs = [config]
    for atm_core in atm_cores:
        config_core = get_config(atm_core)
        ne_core = sum(config_core)
        if ne > ne_core:
            config_diff = (np.asarray(config) - np.asarray(config_core)).astype(int)
            if np.all(config_diff >= 0):
                configs.append(config_diff.tolist())
    configs = sorted(configs, key=sum)
    return configs

def get_config(atm, nelectron=None):
    config_alle = CONFIGURATION[get_Z(atm)].copy()
    if nelectron is None:
        return config_alle
    elif isinstance(nelectron, int):
        nelectron_alle = sum(config_alle)
        if nelectron == nelectron_alle:
            return config_alle
        configs = get_all_configs(atm)
        allowed_nelectron = [sum(config) for config in configs]
        if nelectron <= nelectron_alle:
            found = False
            for config in configs:
                if sum(config) == nelectron:
                    found = True
                    break
            if found:
                return config
        raise ValueError('Invalid nelectron %s for atm %s. Allowed values are %s' % (
            str(nelectron), atm, ', '.join([f'{x}' for x in allowed_nelectron])
        ))
    else:
        raise TypeError('nelectron must be positive integer.')


if __name__ == '__main__':
    print(get_config('C', 3))
    print(get_config('Cu', 29))
