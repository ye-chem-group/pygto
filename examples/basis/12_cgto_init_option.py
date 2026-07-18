''' This example demonstrates how to select angular-momentum channels and filter
    primitive exponents while initializing a ContractedBasis.
'''

from pathlib import Path

from pygto.basis import ContractedBasis


DATA_DIR = Path(__file__).resolve().parent / 'data'


if __name__ == '__main__':
    # Use `keep_l` to retain only selected angular-momentum channels.
    atm = 'Be'
    fbas = str(DATA_DIR / 'be_gth-cc-pvdz.dat')
    keep_l = [0, 2]
    cgto = ContractedBasis.init_from_basis(fbas, atm, keep_l=keep_l)
    cgto.log_note('Keeping only l= %s:' % (str(keep_l)))
    cgto.dump_basis()
    cgto.log_note('')

    # Use `emin` and `emax` to retain exponents in the closed interval [emin, emax].
    atm = 'Al'
    fbas = 'cc-pVDZ'
    cgto = ContractedBasis.init_from_basis(fbas, atm)
    cgto.log_note('Raw %s basis set of %s' % (fbas, atm))
    cgto.log_note('contraction= %s' % (cgto.contraction_summary), indent=1)
    cgto.log_note('nbas= %d  nao= %d' % (cgto.nbas, cgto.nao), indent=1)

    emin = 0.1  # Remove exponents below 0.1.
    emax = 1e3  # Remove exponents above 1e3.
    cgto = ContractedBasis.init_from_basis(fbas, atm, emin=emin, emax=emax)
    cgto.log_note('Filtered with emin= %.3g  emax= %.3g' % (emin, emax))
    cgto.log_note('contraction= %s' % (cgto.contraction_summary), indent=1)
    cgto.log_note('nbas= %d  nao= %d' % (cgto.nbas, cgto.nao), indent=1)
    cgto.log_note('')
