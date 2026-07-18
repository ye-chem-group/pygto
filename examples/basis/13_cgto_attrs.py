''' This example demonstrates commonly used ContractedBasis attributes and representations.
'''

from pathlib import Path

from pygto.basis import ContractedBasis


DATA_DIR = Path(__file__).resolve().parent / 'data'


if __name__ == '__main__':
    atm = 'Be'
    fbas = str(DATA_DIR / 'be_gth-cc-pvdz.dat')
    cgto = ContractedBasis.init_from_basis(fbas, atm)
    cgto.dump_basis()
    cgto.log_note('')

    # `structure_primitive` and `structure_contracted` summarize the numbers of
    # primitive and contracted radial functions, respectively, in each channel.
    # `nbas` counts contracted radial functions, whereas `nao` also includes their
    # magnetic components.
    cgto.log_note('structure_primitive= %s' % (cgto.structure_primitive))
    cgto.log_note('structure_contracted= %s' % (cgto.structure_contracted))
    cgto.log_note('contraction_summary= %s' % (cgto.contraction_summary))
    cgto.log_note('nchannel= %d' % (cgto.nchannel))
    cgto.log_note('nbas= %d' % (cgto.nbas))
    cgto.log_note('nao= %d' % (cgto.nao))

    # Angular momenta are reported in channel order and may repeat because distinct
    # contracted channels can have the same angular momentum.
    cgto.log_note('angular_momenta= %s' % (str(cgto.angular_momenta)))
    for i, c in enumerate(cgto.channels):
        cgto.log_note('Channel= %d  l= %d  Exponents= %s' % (
            i, c.l, ', '.join([f'{x:.4g}' for x in c.exponents])
        ))

    # `pyscf_basis` materializes the current contracted basis in PySCF basis format.
    cgto.log_note('pyscf_basis= %s' % (str(cgto.pyscf_basis)))
