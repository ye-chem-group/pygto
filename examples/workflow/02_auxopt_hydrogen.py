''' This example demonstrates how to use the AuxOpt workflow to optimize an auxiliary
    basis for a specified orbital basis.

    Specifically, we optimize an auxiliary basis for hydrogen/cc-pVTZ and compare its
    errors and size with those of cc-pVTZ-JKFIT. An isolated hydrogen atom has only one
    electron and therefore has zero correlation energy, so its correlation metric cannot
    constrain auxiliary functions for the polarization channels. We therefore optimize
    the hydrogen auxiliary basis using H2 instead.

    NOTE: This example relies on PySCF for HF and MP2 calculations, for generating the
    initial AutoAux basis, and for providing the cc-pVTZ-JKFIT reference basis.
'''

from pygto import lib
from pygto.basis import BasisSpec
from pygto.workflow import AuxOpt
from pygto.data.elements import get_spin

from pyscf import gto, scf, df


if __name__ == '__main__':
    atm = 'H'
    atom = 'H 0 0 0; H 0 0 0.7414'
    spin = 0
    aobasis = 'cc-pvtz'
    val_l = [0]
    pol_l = [1,2]

    ''' Construct the auxiliary-basis cost function for H2.

        cost = max(
            HF Coulomb matrix error * gamma_vjk,
            HF exchange matrix error * gamma_vjk,
            HF Coulomb energy error,
            HF exchange energy error,
            MP2 correlation energy error,
            MP2 T2 amplitudes error,
        )

        The matrix-error terms have different units and scales from the energy-error
        terms, so they are multiplied by `gamma_vjk`, whose default value is 0.1.
    '''
    cost_func = lib.pyscf_helper.get_cost_func_auxopt(
        atom,   # Use the H2 geometry rather than an isolated H atom.
        aobasis, scf.ROHF, mol_settings={'spin':spin}
    )

    cost = {}
    ''' Generate an initial atomic auxiliary basis with PySCF AutoAux and convert each
        channel to an even-tempered representation.

        AutoAux supplies broad initial exponent ranges and angular-momentum coverage.
        The ETB conversion substantially reduces the number of optimization parameters.
        Although AuxOpt also supports fully independent exponents, the ETB form is much
        faster to optimize and gives comparable accuracy.
    '''
    auxbasis = df.autoaux(gto.M(atom=atm, basis=aobasis, spin=None))[atm]
    spec_init = BasisSpec.init_from_basis(auxbasis, atm,
        channel_type='etb', # Convert each channel to an ETB representation.
    )
    cost['init'] = cost_func(spec_init, True)

    ''' Optimize and reduce the initial auxiliary basis to the default target error
        of `1e-5`.
    '''
    spec = spec_init.copy()
    opt = AuxOpt(spec, cost_func).set(verbose=5)
    opt.kernel()
    cost['opt'] = (opt.cost, opt.cost_vec)

    ''' Compare with the reference cc-pVTZ-JKFIT auxiliary basis.

        Reference output:

            **** H2 Accuracy ****
            Init    AutoAux cost= 4.164e-06  cost_vec= 2.002e-07, 2.838e-06, 9.757e-08, 4.878e-08, 2.579e-06, 4.164e-06
            Optimized   ETB cost= 9.959e-06  cost_vec= 3.019e-06, 4.117e-06, 1.612e-06, 8.059e-07, 9.959e-06, 9.959e-06
            Reference JKFIT cost= 8.879e-05  cost_vec= 6.614e-07, 6.499e-05, 1.092e-06, 5.461e-07, 5.486e-05, 8.879e-05

            **** AuxBasis Size ****
            Init    AutoAux nauxao=  52  structure= 11s,4p,3d,2f
            Optimized   ETB nauxao=  31  structure= 8s,2p,2d,1f
            Reference JKFIT nauxao=  30  structure= 4s,3p,2d,1f

        The optimized ETB and reference cc-pVTZ-JKFIT bases have comparable sizes, while
        the optimized ETB gives a smaller fitting error for H2. The initial AutoAux basis
        gives a still smaller error but uses substantially more auxiliary functions.
    '''
    spec_ref = BasisSpec.init_from_basis(f'{aobasis}-jkfit', atm)
    cost['ref'] = cost_func(spec_ref, True)

    spec.log_note('**** H2 Accuracy ****')
    spec.log_note('Init    AutoAux cost= %.3e  cost_vec= %s' % (
        cost['init'][0], ', '.join([f'{x:.3e}' for x in cost['init'][1]])
    ))
    spec.log_note('Optimized   ETB cost= %.3e  cost_vec= %s' % (
        cost['opt'][0], ', '.join([f'{x:.3e}' for x in cost['opt'][1]])
    ))
    spec.log_note('Reference JKFIT cost= %.3e  cost_vec= %s' % (
        cost['ref'][0], ', '.join([f'{x:.3e}' for x in cost['ref'][1]])
    ))
    spec.log_note('')

    spec.log_note('**** AuxBasis Size ****')
    spec.log_note('Init    AutoAux nauxao= %3d  structure= %s' % (
        spec_init.nao, spec_init.structure
    ))
    spec.log_note('Optimized   ETB nauxao= %3d  structure= %s' % (
        spec.nao, spec.structure
    ))
    spec.log_note('Reference JKFIT nauxao= %3d  structure= %s' % (
        spec_ref.nao, spec_ref.structure
    ))
    spec.log_note('')
