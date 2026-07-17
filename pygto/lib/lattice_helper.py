import sys
import numpy as np

BOHR = 0.52917721092    # Angstrom

from .flow_helper import StreamObject


class Lattice(StreamObject):
    ''' Atomic coordinates and lattice vectors in Angstrom.

        Args:
            atom (list):
                Atomic symbols and Cartesian coordinates in PySCF format.
            a (array_like):
                Lattice vectors with shape `(3, 3)`.
    '''

    def __init__(self, atom, a):
        self.atom = atom
        self.a = a

    @classmethod
    def init_from_vasp_poscar(cls, fvasp):
        ''' Initialize a lattice from a VASP POSCAR file.

            Args:
                fvasp (str):
                    Path to POSCAR file.

            Return:
                lattice (Lattice):
                    Parsed lattice.
        '''
        atom, a = read_vasp_poscar(fvasp)
        return cls(atom, a)

    @classmethod
    def init_from_xyz_alat(cls, fxyz, falat):
        ''' Initialize a lattice from XYZ coordinates and lattice vectors.

            Args:
                fxyz (str):
                    Path to XYZ file.
                falat (str):
                    Path to text file containing the lattice vectors.

            Return:
                lattice (Lattice):
                    Parsed lattice.
        '''
        atom = read_xyz(fxyz)
        a = np.loadtxt(falat)
        return cls(atom, a)

    @classmethod
    def init_from_pyscf_cell(cls, cell):
        ''' Initialize a lattice from a PySCF periodic cell.

            Args:
                cell (pyscf.pbc.gto.Cell):
                    Built PySCF periodic cell.

            Return:
                lattice (Lattice):
                    Lattice converted to Angstrom.
        '''
        atom = [(atm,np.asarray(r)*BOHR) for atm,r in cell._atom]
        a = cell.lattice_vectors() * BOHR
        return cls(atom, a)

    @property
    def atms(self):
        ''' Return atomic symbols in input order.

            Return:
                atms (list of str):
                    Atomic symbols.
        '''
        return [x[0] for x in self.atom]

    @property
    def rs(self):
        ''' Return Cartesian atomic coordinates.

            Return:
                coordinates (ndarray):
                    Coordinates in Angstrom.
        '''
        return np.asarray([x[1] for x in self.atom])

    def get_scaled_atom(self, scale=None):
        ''' Return atomic coordinates with an optional scale applied.

            Args:
                scale (float):
                    Coordinate scale factor. Default is None, which leaves coordinates
                    unchanged.

            Return:
                atom (list):
                    Atomic symbols and Cartesian coordinates in PySCF format.
        '''
        if scale is None:
            return self.atom
        return [(atm,np.asarray(r)*scale) for atm,r in self.atom]

    def get_scaled_a(self, scale=None):
        ''' Return lattice vectors with an optional scale applied.

            Args:
                scale (float):
                    Lattice scale factor. Default is None, which leaves vectors
                    unchanged.

            Return:
                a (array_like):
                    Lattice vectors in Angstrom.
        '''
        if scale is None:
            return self.a
        return self.a * scale

    def get_scaled_lattice(self, scale=None):
        ''' Return coordinates and lattice vectors with an optional scale applied.

            Args:
                scale (float):
                    Uniform scale factor. Default is None.

            Return:
                atom (list):
                    Scaled atomic coordinates in PySCF format.
                a (array_like):
                    Scaled lattice vectors in Angstrom.
        '''
        atom = self.get_scaled_atom(scale)
        a = self.get_scaled_a(scale)
        return atom, a

    def get_pyscf_cell(self, basis=None, scale=None, symmetry=False, cell_settings=None):
        ''' Build and return a PySCF periodic cell.

            Args:
                basis (str, dict, or list):
                    Basis accepted by PySCF. Default is None, which uses "def2-svp".
                scale (float):
                    Uniform lattice scale factor. Default is None.
                symmetry (bool):
                    Whether to enable space-group symmetry. Default is False.
                cell_settings (dict):
                    Additional Cell attributes set before building. Default is None.

            Return:
                cell (pyscf.pbc.gto.Cell):
                    Built periodic cell.
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
        ''' Write atomic coordinates in XYZ format.

            Args:
                stdout (file-like object):
                    Destination. Default is None, which uses `self.stdout`.
                scale (float):
                    Coordinate scale factor. Default is None.
                comment (str):
                    Single-line XYZ comment. Default is None, which writes an empty
                    comment line.
        '''
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
        ''' Write lattice vectors as a numeric text block.

            Args:
                stdout (file-like object):
                    Destination. Default is None, which uses `self.stdout`.
                scale (float):
                    Lattice scale factor. Default is None.
        '''
        if stdout is None: stdout = self.stdout
        a = self.get_scaled_a(scale)
        sout = '\n'.join([' '.join([f'{y: .15e}' for y in x]) for x in a])
        stdout.write(sout + '\n')

    def dump_vasp_poscar(self, stdout=None, scale=None, comment=None):
        ''' Write the lattice in VASP POSCAR format.

            Args:
                stdout (file-like object):
                    Destination. Default is None, which uses `self.stdout`.
                scale (float):
                    Uniform lattice scale factor. Default is None.
                comment (str):
                    Single-line POSCAR header. Default is None, which generates a
                    composition header.
        '''
        if stdout is None: stdout = self.stdout

        if isinstance(comment, str) and len(comment.splitlines()) > 1:
            raise ValueError(r'comment must be single-line.')
        elif comment is not None:
            raise TypeError('comment must be str.')

        atom, a = self.get_scaled_lattice(scale)
        sout = write_vasp_poscar(atom, a, comment)
        stdout.write(sout + '\n')


def read_vasp_poscar(fvasp):
    ''' Parse a VASP POSCAR file.

        Args:
            fvasp (str):
                Path to POSCAR file.

        Return:
            atom (list):
                Atomic symbols and Cartesian coordinates in PySCF format.
            a (ndarray):
                Lattice vectors in Angstrom.
    '''
    fdata = open(fvasp, 'r').read().rstrip('\n').split('\n')
    comment = fdata[0]
    def read_line(l, dtype=float, nelem=None):
        ''' Parse selected whitespace-separated fields from one line. '''
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
    ''' Convert a lattice to a VASP POSCAR string.

        Args:
            atom (list):
                Atomic symbols and Cartesian coordinates in PySCF format.
            alat (array_like):
                Lattice vectors with shape `(3, 3)`.
            header (str):
                POSCAR header. Default is None, which generates a composition header.

        Return:
            poscar (str):
                POSCAR-formatted text.
    '''
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
    r''' Parse an XYZ file into PySCF-format atoms.

        Args:
            fxyz (str):
                Path to XYZ file.
            kind (str):
                Return atoms as a string for "str" or a list for "list"/"lst".
                Default is "list".
            parse_alat (bool):
                Whether to parse nine lattice-vector values from the comment line.
                Default is False.

        Return:
            atom (str or list):
                Atomic coordinates in the requested representation.
            alat (ndarray, optional):
                Lattice vectors returned only when `parse_alat` is True.
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
