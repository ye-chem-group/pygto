''' This example demonstrates how to use the AuxOpt workflow to optimize an auxiliary
    basis for a specified orbital basis.

    Specifically, we optimize an auxiliary basis for nitrogen/cc-pVDZ, compare its atomic
    errors and size with cc-pVDZ-JKFIT, and then assess its transferability to N2.

    NOTE: This example relies on PySCF for HF and MP2 calculations, for generating the
    initial AutoAux basis, and for providing the cc-pVDZ-JKFIT reference basis.
'''

from pygto import lib
from pygto.basis import BasisSpec
from pygto.workflow import AuxOpt
from pygto.data.elements import get_spin

from pyscf import gto, scf, df


if __name__ == '__main__':
    atm = 'N'
    spin = get_spin(atm)
    aobasis = 'cc-pvdz'
    val_l = [0,1]
    pol_l = [2]
    frozen = 1

    ''' Construct the auxiliary-basis cost function.

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
        atm, aobasis, scf.ROHF, mol_settings={'spin':spin},
        corr_settings={'frozen':frozen},
    )

    cost = {}
    ''' Generate an initial auxiliary basis with PySCF AutoAux and convert each channel
        to an even-tempered representation.

        AutoAux supplies broad initial exponent ranges and angular-momentum coverage.
        The ETB conversion substantially reduces the number of optimization parameters.
        Although AuxOpt also supports fully independent exponents, the ETB form is much
        faster to optimize and gives comparable accuracy.
    '''
    auxbasis = df.autoaux(gto.M(atom=atm, basis=aobasis, spin=None))[atm]
    spec_init = BasisSpec.init_from_basis(auxbasis, atm,
        channel_type='etb', # CONVERT TO ETB!!!
    )
    cost['init'] = cost_func(spec_init, True)

    ''' Optimize and reduce the initial auxiliary basis to the default target error
        of `1e-5`.
    '''
    spec = spec_init.copy()
    opt = AuxOpt(spec, cost_func).set(verbose=4)
    opt.kernel()
    cost['opt'] = (opt.cost, opt.cost_vec)

    ''' Compare with the reference cc-pVDZ-JKFIT auxiliary basis.

        Reference output:

            **** Atomic Accuracy ****
            Init    AutoAux cost= 4.740e-05  cost_vec= 5.612e-06, 7.021e-06, 4.740e-05, 9.296e-06, 4.918e-06, 7.380e-06
            Optimized   ETB cost= 7.900e-06  cost_vec= 7.308e-07, 7.726e-06, 7.900e-06, 7.038e-06, 5.431e-09, 3.967e-06
            Reference JKFIT cost= 4.680e-05  cost_vec= 2.571e-06, 4.680e-05, 1.451e-05, 1.443e-05, 2.642e-06, 2.282e-05

            **** AuxBasis Size ****
            Init    AutoAux nauxao= 110  structure= 13s,11p,10d,2f
            Optimized   ETB nauxao=  69  structure= 12s,6p,5d,2f
            Reference JKFIT nauxao=  70  structure= 10s,7p,5d,2f

            **** Molecular Accuracy ****
            Init    AutoAux cost= 1.289e-04  cost_vec= 1.385e-06, 4.183e-05, 2.204e-05, 1.289e-04, 5.517e-05, 9.586e-05
            Optimized   ETB cost= 1.578e-04  cost_vec= 1.331e-06, 4.666e-05, 9.344e-06, 1.578e-04, 5.466e-05, 8.623e-05
            Reference JKFIT cost= 1.733e-04  cost_vec= 2.534e-06, 5.629e-05, 1.959e-05, 1.733e-04, 2.011e-05, 1.151e-04

        The optimized ETB and reference cc-pVDZ-JKFIT bases have comparable sizes and
        atomic errors. The unoptimized AutoAux basis gives similar accuracy but uses
        substantially more auxiliary functions.
    '''
    spec_ref = BasisSpec.init_from_basis(f'{aobasis}-jkfit', atm)
    cost['ref'] = cost_func(spec_ref, True)

    spec.log_note('**** Atomic Accuracy ****')
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

    spec.log_note('**** Molecular Accuracy ****')
    ''' Reuse the same error metric for N2 to assess molecular transferability.

        The cost function returns errors for the complete molecule. The reported values
        below are divided by two to give errors per nitrogen atom.
    '''
    atom = 'N 0 0 0; N 1.098 0 0'
    cost_func_mol = lib.pyscf_helper.get_cost_func_auxopt(
        atom, aobasis, scf.ROHF, mol_settings={'spin':0},
        corr_settings={'frozen':frozen*2},
    )
    cost, cost_vec = cost_func_mol(spec_init, True)
    spec.log_note('Init    AutoAux cost= %.3e  cost_vec= %s' % (
        cost*0.5, ', '.join([f'{x*0.5:.3e}' for x in cost_vec])))
    cost, cost_vec = cost_func_mol(spec, True)
    spec.log_note('Optimized   ETB cost= %.3e  cost_vec= %s' % (
        cost*0.5, ', '.join([f'{x*0.5:.3e}' for x in cost_vec])))
    cost, cost_vec = cost_func_mol(spec_ref, True)
    spec.log_note('Reference JKFIT cost= %.3e  cost_vec= %s' % (
        cost*0.5, ', '.join([f'{x*0.5:.3e}' for x in cost_vec])))
    spec.log_note('')
