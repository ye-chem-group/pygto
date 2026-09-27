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


def format_cost_vec(cost_details):
    return ' '.join([
        f'{scaled_error:.3e}'
        for name, error, scaled_error in cost_details
    ])


if __name__ == '__main__':
    atm = 'N'
    spin = get_spin(atm)
    aobasis = 'cc-pvdz'
    frozen = 1

    ''' Construct the auxiliary-basis cost function.

        cost = max(
            HF Coulomb matrix error * gamma_vjk,
            HF exchange matrix error * gamma_vjk,
            HF Coulomb energy error,
            HF exchange energy error,
            MP2 correlation energy error,
            same-spin T2 error * gamma_mp2,
            opposite-spin T2 error * gamma_mp2,
        )

        The matrix- and energy-error terms have different numerical scales. The
        matrix errors are therefore multiplied by `gamma_vjk`, whose default value
        is 0.1.
        The normalized T2-error terms are multiplied by `gamma_mp2`, whose default
        value is 10.
    '''
    cost_func = lib.pyscf_helper.get_cost_func_auxopt(
        atm, aobasis, scf.ROHF, mol_settings={'spin':spin},
        corr_settings={'frozen':frozen},
    )

    cost = {}
    ''' Generate an initial auxiliary basis with PySCF AutoAux.

        AutoAux supplies broad initial exponent ranges and angular-momentum coverage.
        AuxOpt converts each channel to a Legendre representation before optimization,
        which reduces the number of independent parameters for large channels while
        retaining a flexible exponent distribution.
    '''
    auxbasis = df.autoaux(gto.M(atom=atm, basis=aobasis, spin=None))[atm]
    spec_init = BasisSpec.init_from_basis(auxbasis, atm)
    cost['init'] = cost_func(spec_init, True)

    ''' Optimize and reduce the initial auxiliary basis to the default target error
        of `1e-5`.
    '''
    spec = spec_init.copy()
    opt = AuxOpt(spec, cost_func, ftol=1e-5).set(verbose=4)
    opt.kernel()
    cost['opt'] = (opt.cost, opt.cost_details)

    ''' Compare with the reference cc-pVDZ-JKFIT auxiliary basis.

        Reference output:

**** Atomic Accuracy ****
Init AutoAux cost= 2.539e-05  cost_vec= 5.654e-06 7.021e-06 2.539e-05 9.296e-06 4.918e-06 8.233e-07 2.216e-10
Opt Legendre cost= 9.254e-06  cost_vec= 4.494e-07 9.254e-06 2.248e-06 9.254e-06 8.411e-06 2.204e-06 1.155e-07
Ref    JKFIT cost= 4.680e-05  cost_vec= 2.236e-06 4.680e-05 8.819e-06 1.443e-05 2.642e-06 1.147e-05 6.305e-06

**** AuxBasis Size ****
Init AutoAux nauxao= 110  structure= 13s,11p,10d,2f
Opt Legendre nauxao=  62  structure= 13s,5p,4d,2f
Ref    JKFIT nauxao=  70  structure= 10s,7p,5d,2f

**** Molecular Accuracy ****
Init AutoAux cost= 2.579e-04  cost_vec= 5.542e-06 8.365e-05 4.407e-05 2.579e-04 1.103e-04 1.134e-04 4.336e-05
Opt Legendre cost= 3.603e-04  cost_vec= 7.502e-06 9.683e-05 1.310e-05 3.603e-04 9.700e-05 1.120e-04 5.063e-05
Ref    JKFIT cost= 3.465e-04  cost_vec= 1.014e-05 1.126e-04 3.918e-05 3.465e-04 4.022e-05 1.375e-04 6.099e-05

        The optimized aux basis and reference cc-pVDZ-JKFIT bases have comparable sizes
        and atomic errors. The unoptimized AutoAux basis gives similar accuracy but uses
        substantially more auxiliary functions.
    '''
    spec_ref = BasisSpec.init_from_basis(f'{aobasis}-jkfit', atm)
    cost['ref'] = cost_func(spec_ref, True)

    spec.log_note('**** Atomic Accuracy ****')
    spec.log_note('Init AutoAux cost= %.3e  cost_vec= %s' % (
        cost['init'][0], format_cost_vec(cost['init'][1]))
    )
    spec.log_note('Opt Legendre cost= %.3e  cost_vec= %s' % (
        cost['opt'][0], format_cost_vec(cost['opt'][1]))
    )
    spec.log_note('Ref    JKFIT cost= %.3e  cost_vec= %s' % (
        cost['ref'][0], format_cost_vec(cost['ref'][1]))
    )
    spec.log_note('')

    spec.log_note('**** AuxBasis Size ****')
    spec.log_note('Init AutoAux nauxao= %3d  structure= %s' % (
        spec_init.nao, spec_init.structure
    ))
    spec.log_note('Opt Legendre nauxao= %3d  structure= %s' % (
        spec.nao, spec.structure
    ))
    spec.log_note('Ref    JKFIT nauxao= %3d  structure= %s' % (
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
    cost_mol, cost_details = cost_func_mol(spec_init, True)
    spec.log_note('Init AutoAux cost= %.3e  cost_vec= %s' % (
        cost_mol, format_cost_vec(cost_details))
    )
    cost_mol, cost_details = cost_func_mol(spec, True)
    spec.log_note('Opt Legendre cost= %.3e  cost_vec= %s' % (
        cost_mol, format_cost_vec(cost_details))
    )
    cost_mol, cost_details = cost_func_mol(spec_ref, True)
    spec.log_note('Ref    JKFIT cost= %.3e  cost_vec= %s' % (
        cost_mol, format_cost_vec(cost_details))
    )
    spec.log_note('')
