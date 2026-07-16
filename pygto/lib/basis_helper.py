import os
import sys
import textwrap
import numpy as np


lnames = 'spdfghikl'
LNAMEs = lnames.upper()
LMAX = len(lnames)-1


def load_basis_nwchem(basis_str_or_file, atm):
    ''' Load an atomic basis from an NWChem string or file.

        Args:
            basis_str_or_file (str):
                NWChem basis data or the path to an NWChem basis file.
            atm (str):
                Atomic symbol of the basis to load.

        Return:
            Basis data in PySCF format.
    '''
    if not isinstance(basis_str_or_file, str):
        raise TypeError('basis_str_or_file must be str.')

    fbas = basis_str_or_file.strip()
    if os.path.isfile(fbas):
        with open(fbas, 'r') as f:
            basis_text = f.read()
        source = fbas
    else:
        basis_text = textwrap.dedent(basis_str_or_file)
        source = '<string>'

    # Stripping each line supports indented triple-quoted strings and prevents
    # whitespace-only lines from being interpreted as empty primitives.
    lines = [line.strip() for line in basis_text.strip().splitlines()]
    loc = [i for i,line in enumerate(lines) if line.startswith('#B')]
    if not loc:
        raise ValueError('Invalid basis_str_or_file: no #BASIS SET header found.')

    # A basis block ends at the next header or at EOF.  The latter is required
    # for a standalone basis string and for the last element in a basis file.
    bounds = loc + [len(lines)]
    hit = []
    for ib, (i0, i1) in enumerate(zip(bounds[:-1], bounds[1:])):
        for line in lines[i0+1:i1]:
            fields = line.split()
            if fields and fields[0].lower() == atm.lower():
                hit.append(ib)
                break

    if len(hit) == 0:
        raise RuntimeError('Basis data for atm %s not found in %s' % (atm, source))
    if len(hit) > 1:
        raise RuntimeError('Multiple basis data for atm %s are found in %s' % (atm, source))
    hit = hit[0]
    i0, i1 = bounds[hit:hit+2]

    lines = [
        line for line in lines[i0+1:i1]
        if line and not line.startswith('#') and line.upper() != 'END'
    ]
    loc = [
        i for i,line in enumerate(lines)
        if line.split()[0].lower() == atm.lower()
    ]
    loc += [len(lines)]
    nblock = len(loc)-1

    basis = []
    for i in range(nblock):
        i0,i1 = loc[i:i+2]
        lname = lines[i0].split()[1]
        if len(lname) != 1 or lname.lower() not in lnames:
            raise ValueError('Invalid angular momentum symbol %s' % (lname))
        l = lnames.index(lname.lower())
        primitives = []
        for line in lines[i0+1:i1]:
            try:
                primitive = [float(x.replace('D', 'E').replace('d', 'e'))
                             for x in line.split()]
            except ValueError as err:
                raise ValueError('Invalid primitive data: %s' % line) from err
            if len(primitive) < 2:
                raise ValueError('Invalid primitive data: %s' % line)
            primitives.append(primitive)
        b = [l] + primitives
        basis.append(b)

    return basis


def get_header_nwchem(basis):
    ''' Generate an NWChem basis header.

        Args:
            basis (list):
                Basis data in PySCF format.

        Return:
            NWChem basis header string.
    '''
    nprims = np.zeros(LMAX+1, dtype=int)
    nctrs = np.zeros(LMAX+1, dtype=int)
    for b in basis:
        l = int(b[0])
        ecs = np.asarray(b[1:])
        if ecs.ndim == 1: ecs.reshape(1,-1)
        nprims[l] += ecs.shape[0]
        nctrs[l] += ecs.shape[1]-1
    prim = ','.join([f'{nprim}{lnames[l]}' for l,nprim in enumerate(nprims) if nprim > 0])
    ctr = ','.join([f'{nctr}{lnames[l]}' for l,nctr in enumerate(nctrs) if nctr > 0])
    header = '#BASIS SET: (%s) -> [%s]' % (prim, ctr)
    return header


def dump_basis_nwchem(basis, stdout=None, atm=None, header=True, sort=True):
    ''' Write basis data in NWChem format.

        Args:
            basis (list):
                Basis data in PySCF format.
            stdout (file-like object):
                Destination for the basis data. Default is None, which uses sys.stdout.
            atm (str):
                Atomic symbol. Default is None, which uses X.
            header (bool):
                Whether to include the basis header. Default is True.
            sort (bool):
                Whether to sort exponents in descending order. Default is True.

        Return:
            None.
    '''
    if stdout is None: stdout = sys.stdout
    s = get_basis_str_nwchem(basis, atm, header, sort)
    stdout.write(s+'\n')


def get_basis_str_nwchem(basis, atm=None, header=True, sort=True):
    ''' Convert basis data to an NWChem-format string.

        Args:
            basis (list):
                Basis data in PySCF format.
            atm (str):
                Atomic symbol. Default is None, which uses X.
            header (bool):
                Whether to include the basis header. Default is True.
            sort (bool):
                Whether to sort exponents in descending order. Default is True.

        Return:
            NWChem-format basis string.
    '''
    if atm is None: atm = 'X'
    if header:
        s = [get_header_nwchem(basis)]
    else:
        s = []

    basis_data = []
    for b in basis:
        l = int(b[0])
        ecs = np.asarray(b[1:])
        if ecs.ndim == 1: ecs = ecs.reshape(1,-1)
        if sort:
            order = np.argsort(ecs[:,0])[::-1]
            ecs = ecs[order]

        basis_data.append((l, ecs))

    if sort:
        basis_data.sort(key=lambda item: (item[0], -item[1][0,0]))

    for l, ecs in basis_data:
        s.append( f'{atm}    {LNAMEs[l]}' )
        for ec in ecs:
            e = ec[0]
            cs = ec[1:]
            s.append( '    '.join([f'{e:15.7f}'] + [f'{c: 9.7f}' for c in cs]) )
    s = '\n'.join(s)
    return s


if __name__ == '__main__':
    atm = 'C'
    fbas = '/Users/hzye/local/opt/pyscf/pyscf/gto/basis/cc-pvtz.dat'
    basis = load_basis_nwchem(fbas, atm)

    dump_basis_nwchem(basis, atm)

    from pyscf import gto, scf
    mol = gto.M(atom=atm, basis=basis, spin=2)
    mf = scf.RHF(mol).run()

    mol = gto.M(atom=atm, basis=fbas, spin=2)
    mf = scf.RHF(mol).run()
