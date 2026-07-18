''' This example demonstrates commonly used BasisSpec attributes and representations.
'''

from pathlib import Path

from pygto.basis import BasisSpec


DATA_DIR = Path(__file__).resolve().parent / 'data'


if __name__ == '__main__':
    atm = 'Be'
    fbas = str(DATA_DIR / 'be_gth-cc-pvdz.dat')
    spec = BasisSpec.init_from_basis(fbas, atm)
    spec.dump_basis()
    spec.log_note('')

    # `structure` summarizes the number of radial functions in each channel. `nbas`
    # counts radial functions, whereas `nao` includes their magnetic components.
    spec.log_note('structure= %s' % (spec.structure))
    spec.log_note('nchannel= %d' % (spec.nchannel))
    spec.log_note('nbas= %d' % (spec.nbas))
    spec.log_note('nao= %d' % (spec.nao))

    # Angular momenta are reported in channel order and may repeat when multiple
    # independently parameterized channels have the same angular momentum.
    spec.log_note('angular_momenta= %s' % (str(spec.angular_momenta)))
    for i,c in enumerate(spec.channels):
        spec.log_note('Channel= %d  l= %d  Exponents= %s' % (
            i, c.l, ', '.join([f'{x:.4g}' for x in c.exponents])
        ))

    # `pyscf_basis` materializes the current specification in PySCF basis format.
    spec.log_note('pyscf_basis= %s' % (str(spec.pyscf_basis)))
