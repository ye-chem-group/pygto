import sys
import numpy as np

BOHR = 0.52917721092    # Angstrom

from .flow_helper import StreamObject


class Lattice(StreamObject):

    def __init__(self, atom, a):
        ''' `atom` and `a` are PySCF format in Angstrom
        '''
        self.atom = atom
        self.a = a

    @classmethod
    def init_from_vasp_poscar(cls, fvasp):
        ''' Initialize Lattice from a VASP POSCAR file
        '''
        atom, a = read_vasp_poscar(fvasp)
        return cls(atom, a)

    @classmethod
    def init_from_xyz_alat(cls, fxyz, falat):
        ''' Initialize Lattice from a XYZ file and a ALAT file
        '''
        atom = read_xyz(fxyz)
        a = np.loadtxt(falat)
        return cls(atom, a)

    @classmethod
    def init_from_pyscf_cell(cls, cell):
        ''' Initialize Lattice from a pyscf.gto.Cell object.
        '''
        atom = [(atm,np.asarray(r)*BOHR) for atm,r in cell._atom]
        a = cell.lattice_vectors() * BOHR
        return cls(atom, a)

    @property
    def atms(self):
        return [x[0] for x in self.atom]

    @property
    def rs(self):
        return np.asarray([x[1] for x in self.atom])

    def get_scaled_atom(self, scale=None):
        if scale is None:
            return self.atom
        return [(atm,np.asarray(r)*scale) for atm,r in self.atom]

    def get_scaled_a(self, scale=None):
        if scale is None:
            return self.a
        return self.a * scale

    def get_scaled_lattice(self, scale=None):
        ''' Return PySCF-format `atom` and `a` with the `scale` factor applied.
        '''
        atom = self.get_scaled_atom(scale)
        a = self.get_scaled_a(scale)
        return atom, a

    def get_pyscf_cell(self, basis=None, scale=None, symmetry=False, cell_settings=None):
        ''' Return a PySCF pbc.gto.Cell object.
        '''
        from pyscf.pbc import gto
        if basis is None: basis = 'def2-svp'
        atom, a = self.get_scaled_lattice(scale)
        cell = gto.Cell(atom=atom, basis=basis, a=a, spin=None).set(verbose=0)
        if symmetry:
            cell.space_group_symmetry = True
            cell.symmorphic = True
        if cell_settings is not None:
            cell.set(**cell_settings)
        cell.build()
        return cell

    def dump_xyz(self, stdout=None, scale=None, comment=None):
        if stdout is None: stdout = self.stdout
        atom = self.get_scaled_atom(scale)
        natm = len(atom)
        sout = [f'{natm}']
        if comment is None:
            sout.append('')
        elif isinstance(comment, str):
            if len(comment.splitlines()) > 1:
                raise ValueError(r'comment must be single-line.')
            sout.append(comment)
        else:
            raise TypeError('comment must be str.')

        for atm,r in atom:
            sout.append(f'{atm:2s}  {r[0]: .15e}  {r[1]: .15e}  {r[2]: .15e}')

        sout = '\n'.join(sout)
        stdout.write(sout + '\n')

    def dump_a(self, stdout=None, scale=None):
        if stdout is None: stdout = self.stdout
        a = self.get_scaled_a(scale)
        sout = '\n'.join([' '.join([f'{y: .15e}' for y in x]) for x in a])
        stdout.write(sout + '\n')

    def dump_vasp_poscar(self, stdout=None, scale=None, comment=None):
        if stdout is None: stdout = self.stdout

        if isinstance(comment, str) and len(comment.splitlines()) > 1:
            raise ValueError(r'comment must be single-line.')
        elif comment is not None:
            raise TypeError('comment must be str.')

        atom, a = self.get_scaled_lattice(scale)
        sout = write_vasp_poscar(atom, a, comment)
        stdout.write(sout + '\n')


def read_vasp_poscar(fvasp):
    ''' Parse a VASP POSCAR into PySCF `atom` and `a`.
    '''
    fdata = open(fvasp, 'r').read().rstrip('\n').split('\n')
    comment = fdata[0]
    def read_line(l, dtype=float, nelem=None):
        spl = l.split()
        if nelem is not None: spl = spl[:nelem]
        return list(map(dtype,spl))
# alat
    alat_scale = read_line(fdata[1])[0]
    if alat_scale < 0:
        raise NotImplementedError('Negative lattice scale is not implemented.')
    alat_frac = np.array([read_line(line) for line in fdata[2:5]])
    alat = alat_scale * alat_frac
# atms
    atms_uniq = fdata[5].split()
    natms_uniq = read_line(fdata[6], int)
    atms = [atm for atm,natm_uniq in zip(atms_uniq, natms_uniq)
            for i in range(natm_uniq)]
    natm = len(atms)
# rs
    found = False
    for i0 in range(7,len(fdata)):
        x = fdata[i0].lower()
        if x.startswith('cart') or x.startswith('dir'):
            found = True
            break
    if not found:
        raise RuntimeError('Coordinates not found!')
    rstype = fdata[i0].lstrip().lower()
    rs = np.asarray([read_line(fdata[i],nelem=3) for i in range(i0+1,i0+1+natm)])
    if rstype.startswith('cart'):
        rs_scale = None
    elif rstype.startswith('dir'):
        rs_scale = alat
    else:
        raise NotImplementedError
    if rs_scale is not None:
        rs = np.dot(rs, rs_scale)
# atms + rs --> atom
    atom = [(atm, np.asarray(r)) for atm,r in zip(atms,rs)]

    return atom, alat


def write_vasp_poscar(atom, alat, header=None):
    atms = np.asarray([x[0] for x in atom])
    rs = np.asarray([x[1] for x in atom]).reshape(-1,3)

    # unique but preserving order
    atms_uniq = []
    for atm in atms:
        if atm not in atms_uniq:
            atms_uniq.append(atm)

    atmid_map = [np.where(atms==atm)[0] for atm in atms_uniq]
    natmuniqs = np.array([len(atmid) for atmid in atmid_map])
    if header is None:
        header = ' '.join(['%s%d'%(atm,natmuniq)
                           for atm,natmuniq in zip(atms_uniq,natmuniqs)])
    sout = [header]
    sout += ['1.0']
    for i in range(3):
        sout += [' '.join(['%.10f'%(alat[i,j]) for j in range(3)])]
    sout += [' '.join(atms_uniq)]
    sout += [' '.join(['%d'%n for n in natmuniqs])]
    sout += ['Cartesian']
    for idx in atmid_map:
        sout += ['\n'.join([' '.join([f'{rs[i,j]: .15e}' for j in range(3)]) for i in idx])]
    sout = '\n'.join(sout)
    return sout


def read_xyz(fxyz, kind='list', parse_alat=False):
    r''' Parse xyz file to PySCF atom.

    Args:
        fxyz (str):
            Path to a xyz file. The first two lines in the file are ignored.
        kind (str):
            Determine return type.
            'str'  : return string, e.g., 'H 0 0 0 \n H 1 0 0'
            'list' : return list, e.g., [['H', [0,0,0]], ["H", [1,0,0]]]
        parse_alat (bool):
            If True, the comment line (i.e., second line in the xyz file) will
            be parsed as lattice constant and returned as the second entry.
            The comment line must be nine numbers separated by space(s) which
            correspond to cell.a.reshape(-1).
    '''
    fdata = open(fxyz,'r').read().rstrip('\n').split('\n')
    if kind.startswith('str'):
        atom = '\n'.join(fdata[2:])
    elif kind in ['list','lst']:
        atom = []
        for dat in fdata[2:]:
            spl = dat.split()
            atom.append([spl[0], list(map(float,spl[1:4]))])
    else:
        raise ValueError('Unknown kind = "%s"' % (str(kind)))
    if parse_alat:
        alat = np.array(list(map(float,fdata[1].split()))).reshape(3,3)
        return atom, alat
    else:
        return atom
