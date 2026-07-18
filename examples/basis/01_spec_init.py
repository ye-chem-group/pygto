''' In pygto, every basis-set workflow starts from a BasisSpec, which represents
    primitive GTO exponents grouped into angular-momentum channels.

    This example demonstrates five ways to initialize a BasisSpec:
        - from a named basis, e.g., "cc-pvdz"
        - from an NWChem-format basis string
        - from an NWChem-format basis file
        - from PySCF-format basis data
        - from even-tempered basis (ETB) parameters.

    The first four representations are accepted by the unified constructor

        spec = BasisSpec.init_from_basis(basis, atm)

    where `basis` can be any of those four input types. To construct an ETB directly, use

        spec = BasisSpec.init_from_etb_params(etb_params, atm)

    with parameters of the form

        etb_params = [
            (l1, nprim1, amin1, beta1),
            (l2, nprim2, amin2, beta2),
            ...
        ]

    Here, `l` is the angular momentum, `nprim` is the number of primitives in that
    channel, and `amin` and `beta` are the smallest exponent and geometric progression
    factor, respectively.
'''

from pathlib import Path

from pygto import lib
from pygto.basis import BasisSpec


DATA_DIR = Path(__file__).resolve().parent / 'data'


if __name__ == '__main__':
    ''' A BasisSpec can be initialized directly from a standard named basis.

        pygto will use the following backends to load a named basis:
        1. If PySCF is available, `:func:pyscf.gto.basis.load` will be used.
        2. If PySCF is not found but the Basis Set Exchange (BSE) package is available,
           pygto will use `:func:basis_set_exchange.get_basis`.
        3. If neither PySCF nor BSE is found, pygto will download the basis data through
           the BSE REST API. This step requires internet access.
        4. If all resorts above fail, a RuntimeError will be raised.
    '''
    try:
        atm = 'C'
        basis = 'cc-pvdz'
        spec = BasisSpec.init_from_basis(basis, atm)
        # spec = BasisSpec.init_from_named_basis(basis, atm)    # Equivalent.
    except:
        spec = None

    if spec is not None:
        spec.dump_basis()
        spec.log_note('')
    else:
        print('')
        print('WARN: Getting named basis %s fails for atm %s.' % (basis, atm))
        print('')

    ''' A BasisSpec can also be initialized from an NWChem-format basis string.
    '''
    atm = 'N'
    basis = '''
#BASIS SET: (10s,5p,2d,1f) -> [4s,3p,2d,1f]
N    S
11420.0000000              0.0005230             -0.0001150
1712.0000000              0.0040450             -0.0008950
389.3000000              0.0207750             -0.0046240
110.0000000              0.0807270             -0.0185280
 35.5700000              0.2330740             -0.0573390
 12.5400000              0.4335010             -0.1320760
  4.6440000              0.3474720             -0.1725100
  0.5118000             -0.0085080              0.5999440
N    S
  1.2930000              1.0000000
N    S
  0.1787000              1.0000000
N    P
 26.6300000              0.0146700
  5.9480000              0.0917640
  1.7420000              0.2986830
N    P
  0.5550000              1.0000000
N    P
  0.1725000              1.0000000
N    D
  1.6540000              1.0000000
N    D
  0.4690000              1.0000000
N    F
  1.0930000              1.0000000
    '''
    spec = BasisSpec.init_from_basis(basis, atm)
    # spec = BasisSpec.init_from_nwchem_basis(basis, atm)   # Equivalent.
    spec.dump_basis()
    spec.log_note('')

    ''' The same constructor accepts a file containing NWChem-format basis data.
    '''
    atm = 'Be'
    # Source: https://github.com/hongzhouye/ccgto/blob/main/basis/gth-hf-rev/cc-pvdz-lc.dat#L95
    basis = str(DATA_DIR / 'be_gth-cc-pvdz.dat')
    spec = BasisSpec.init_from_basis(basis, atm)
    # spec = BasisSpec.init_from_nwchem_basis(basis, atm)   # Equivalent.
    spec.dump_basis()
    spec.log_note('')

    ''' PySCF users can pass basis data returned by PySCF directly.
    '''
    if lib.has_pyscf():
        from pyscf import gto
        atm = 'Si'
        basis = 'def2-svp'
        basis = gto.basis.load(basis, atm)
        spec = BasisSpec.init_from_basis(basis, atm)
        # spec = BasisSpec.init_from_pyscf_basis(basis).set(atm=atm)    # Equivalent.
        spec.dump_basis()
        spec.log_note('')
    else:
        print('')
        print('WARN: Importing PySCF fails.')
        print('')

    ''' Construct an even-tempered basis (ETB) directly from its parameters.

        Here we generate a 4s4p3d2f1g basis for carbon.
    '''
    atm = 'C'
    etb_params = [
        (0, 4, 0.1, 3.5),
        (1, 4, 0.1, 3.5),
        (2, 3, 0.2, 3.5),
        (3, 2, 0.4, 3.5),
        (4, 1, 0.6, 3.5),
    ]
    spec = BasisSpec.init_from_etb_params(etb_params, atm)
    spec.dump_basis()
