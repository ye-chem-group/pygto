''' This example demonstrates options for selecting and parameterizing channels while
    initializing a BasisSpec.
'''

from pathlib import Path

from pygto.basis import BasisSpec


DATA_DIR = Path(__file__).resolve().parent / 'data'


if __name__ == '__main__':
    # Use `keep_l` to retain only selected angular-momentum channels.
    atm = 'Be'
    fbas = str(DATA_DIR / 'be_gth-cc-pvdz.dat')
    keep_l = [0,2]
    spec = BasisSpec.init_from_basis(fbas, atm, keep_l=keep_l)
    spec.log_note('Keeping only l= %s:' % (str(keep_l)))
    spec.dump_basis()
    spec.log_note('')

    # Use `emin` and `emax` to retain exponents in the closed interval [emin, emax].
    atm = 'Al'
    fbas = 'cc-pVDZ'
    spec = BasisSpec.init_from_basis(fbas, atm)
    spec.log_note('Raw %s basis set of %s' % (fbas, atm))
    spec.log_note('structure= %s' % (spec.structure), indent=1)
    spec.log_note('nbas= %d  nao= %d' % (spec.nbas, spec.nao), indent=1)

    emin = 0.1  # Remove exponents below 0.1.
    emax = 1e3  # Remove exponents above 1e3.
    spec = BasisSpec.init_from_basis(fbas, atm, emin=emin, emax=emax)
    spec.log_note('Filtered with emin= %.3g  emax= %.3g' % (emin, emax))
    spec.log_note('structure= %s' % (spec.structure), indent=1)
    spec.log_note('nbas= %d  nao= %d' % (spec.nbas, spec.nao), indent=1)
    spec.log_note('')

    # Convert each input channel to its best-fit even-tempered representation.
    atm = 'Be'
    fbas = str(DATA_DIR / 'be_gth-cc-pvdz.dat')
    spec = BasisSpec.init_from_basis(fbas, atm)
    spec.log_note('Original basis:')
    spec.dump_basis()
    spec.log_note('')

    spec = BasisSpec.init_from_basis(fbas, atm, channel_type='etb')
    spec.log_note('Best ETB fit:')
    spec.dump_basis()
    spec.log_note('')
