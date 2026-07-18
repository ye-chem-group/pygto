''' Unlike BasisSpec, which decontracts the input basis, ContractedBasis preserves
    contraction coefficients and stores contracted basis data.

    This example demonstrates four ways to initialize a ContractedBasis:
        - from a named basis, e.g., "cc-pvdz"
        - from an NWChem-format basis string
        - from an NWChem-format basis file
        - from PySCF-format basis data
'''

from pathlib import Path

from pygto import lib
from pygto.basis import ContractedBasis


DATA_DIR = Path(__file__).resolve().parent / 'data'


if __name__ == '__main__':
    ''' A ContractedBasis can be initialized directly from a standard named basis.

        pygto will use the following backends to load a named basis:
        1. If PySCF is available, pygto uses `pyscf.gto.basis.load`.
        2. Otherwise, if the Basis Set Exchange (BSE) package is available, pygto uses
           `basis_set_exchange.get_basis`.
        3. If neither package is available, pygto requests the basis through the BSE
           REST API. This fallback requires internet access.

        If the selected backend cannot load the requested basis, its exception is
        propagated to the caller.
    '''
    load_error = None
    try:
        atm = 'C'
        basis = 'cc-pvdz'
        cgto = ContractedBasis.init_from_basis(basis, atm)
        # cgto = ContractedBasis.init_from_named_basis(basis, atm)    # Equivalent.
    except Exception as err:
        cgto = None
        load_error = err

    if cgto is not None:
        cgto.dump_basis()
        cgto.log_note('')
    else:
        print('')
        print('WARN: Failed to load named basis %s for atom %s: %s' % (
            basis, atm, load_error))
        print('')

    ''' A ContractedBasis can also be initialized from an NWChem-format basis string.
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
    cgto = ContractedBasis.init_from_basis(basis, atm)
    # cgto = ContractedBasis.init_from_nwchem_basis(basis, atm)   # Equivalent.
    cgto.dump_basis()
    cgto.log_note('')

    ''' The same constructor accepts a file containing NWChem-format basis data.
    '''
    atm = 'Be'
    # Source: https://github.com/hongzhouye/ccgto/blob/main/basis/gth-hf-rev/cc-pvdz-lc.dat#L95
    basis = str(DATA_DIR / 'be_gth-cc-pvdz.dat')
    cgto = ContractedBasis.init_from_basis(basis, atm)
    # cgto = ContractedBasis.init_from_nwchem_basis(basis, atm)   # Equivalent.
    cgto.dump_basis()
    cgto.log_note('')

    ''' PySCF users can pass basis data returned by PySCF directly.
    '''
    if lib.has_pyscf():
        from pyscf import gto
        atm = 'Si'
        basis = 'def2-svp'
        basis = gto.basis.load(basis, atm)
        cgto = ContractedBasis.init_from_basis(basis, atm)
        # cgto = ContractedBasis.init_from_pyscf_basis(basis).set(atm=atm)    # Equivalent.
        cgto.dump_basis()
        cgto.log_note('')
    else:
        print('')
        print('WARN: Importing PySCF fails.')
        print('')
