''' This example shows how to use the ANO workflow to contract primitive GTOs.

    We generate a contraction for the s and p channels of cc-pVQZ for carbon and
    compare it with the corresponding reference cc-pVQZ contraction.

    NOTE: This example relies on PySCF for atomic ROHF and CCSD calculations.
'''

from pygto import lib
from pygto.basis import BasisSpec, ContractedBasis
from pygto.workflow import ANO
from pygto.data.elements import get_spin

from pyscf import scf


if __name__ == '__main__':
    atm = 'C'
    spin = get_spin(atm)
    basis = 'cc-pvqz'
    val_l = [0, 1]
    frozen = 1

    ''' Initialize BasisSpec from the s and p channels of cc-pVQZ.
    '''
    spec = BasisSpec.init_from_basis(basis, atm, keep_l=val_l)

    ''' Construct function for getting ANO input.

        For customized function, follow the following function signature:

            def ano_input_func(spec):
                # your implementation
                return dm, s

        where `dm` is the spin-summed AO density matrix and `s` is the AO overlap matrix,
        both of shape `(spec.nao, spec.nao)`.
    '''
    ano_input_func = lib.pyscf_helper.get_ano_input_func(
        atm, scf.ROHF, mol_settings={'spin':spin},
    )
    # The generated ANO-input function is equivalent to the following implementation:
    # def ano_input_func(spec):
    #     import numpy as np
    #     from pyscf import gto
    #     basis = spec.get_pyscf_basis()
    #     mol = gto.M(atom=atm, basis=basis, spin=spin).set(verbose=0)
    #     mf = scf.ROHF(mol).run()
    #     dm = mf.make_rdm1()
    #     nao = mol.nao_nr()
    #     dm = np.asarray(dm).reshape(-1, nao, nao).sum(axis=0)
    #     assert dm.shape == (nao, nao)
    #     s = mf.get_ovlp()
    #     return dm, s

    ''' Perform the atomic natural orbital (ANO) contraction.
    '''
    ano = ANO(spec, ano_input_func, atm)
    # Leave the three most diffuse primitives uncontracted in each channel.
    ano.nfree_by_l = {0:3, 1:3}
    ano.kernel()
    cgto = ano.basis

    ''' Compare HF and CCSD correlation energies among the uncontracted, ANO-contracted,
        and reference-contracted s/p subsets of cc-pVQZ.

        Reference output:

            Uncontracted cc-pVQZ (s/p) HF energy= -37.6882343494
            ROHF-ANO-ctr cc-pVQZ (s/p) HF energy= -37.6882343494
            Reference    cc-pVQZ (s/p) HF energy= -37.6882343292

            Uncontracted cc-pVQZ (s/p) CCSD corr energy= -0.0452495657
            ROHF-ANO-ctr cc-pVQZ (s/p) CCSD corr energy= -0.0449117626
            Reference    cc-pVQZ (s/p) CCSD corr energy= -0.0449117672
    '''
    cgto_ref = ContractedBasis.init_from_basis(basis, atm, keep_l=val_l)
    cost_func_hf = lib.pyscf_helper.get_cost_func(
        atm, scf.ROHF, mol_settings={'spin':spin}, keep_l=val_l,
    )
    ehf_unc = cost_func_hf(spec)
    ehf_ctr = cost_func_hf(cgto)
    ehf_ref = cost_func_hf(cgto_ref)
    spec.log_note('Uncontracted cc-pVQZ (s/p) HF energy= %.10f' % (ehf_unc))
    spec.log_note('ROHF-ANO-ctr cc-pVQZ (s/p) HF energy= %.10f' % (ehf_ctr))
    spec.log_note('Reference    cc-pVQZ (s/p) HF energy= %.10f' % (ehf_ref))
    spec.log_note('')

    cost_func_ccsd = lib.pyscf_helper.get_cost_func(
        atm, scf.ROHF, mol_settings={'spin':spin},
        CORR=cc.CCSD, corr_settings={'frozen':frozen},
    )
    ecorr_unc = cost_func_ccsd(spec)
    ecorr_ctr = cost_func_ccsd(cgto)
    ecorr_ref = cost_func_ccsd(cgto_ref)
    spec.log_note('Uncontracted cc-pVQZ (s/p) CCSD corr energy= %.10f' % (ecorr_unc))
    spec.log_note('ROHF-ANO-ctr cc-pVQZ (s/p) CCSD corr energy= %.10f' % (ecorr_ctr))
    spec.log_note('Reference    cc-pVQZ (s/p) CCSD corr energy= %.10f' % (ecorr_ref))
    spec.log_note('')

    ''' Compare the ROHF-ANO and reference contractions of the cc-pVQZ s/p subset.
    '''
    spec.log_note('Reference cc-pVQZ (s/p) basis:')
    cgto_ref.dump_basis()
    spec.log_note('')
    spec.log_note('ROHF-ANO-ctr cc-pVQZ (s/p) basis:')
    cgto.dump_basis()
