import unittest
import numpy as np

from pygto import lib
from pygto.basis import BasisSpec
from pyscf import gto, scf, cc


class AtomicSCFTest(unittest.TestCase):

    def setUp(self):
        mol = gto.Mole()
        mol.atom = 'Cu'
        mol.spin = 1
        mol.basis = 'cc-pvdz'
        mol.verbose = 0
        mol.build()

        self.mol = mol

    def tearDown(self):
        del self.mol

    def test_ground_state(self):
        ''' For the 3d^10 4s^1 ground state, select-l calculations should
            match unconstrained ground-state calculations.
        '''
        mol = self.mol
        config = [
            (4, 6, 5, 0),   # spin alpha: 1s^1 2s^1 2p^3 3s^1 3p^3 3d^5 4s^1
            (3, 6, 5, 0),   # spin beta : 1s^1 2s^1 2p^3 3s^1 3p^3 3d^5
        ]

        # unconstrained ground-state calculation
        mf_uncon = scf.ROHF(mol)
        mf_uncon.kernel()

        # pure-l occupation
        mf_purel = scf.ROHF(mol)
        lib.pyscf_helper.atomic_scf_with_pure_l_config_(mf_purel, config)
        mf_purel.kernel()
        self.assertAlmostEqual(mf_uncon.e_tot, mf_purel.e_tot, 6)

        # dominant-l occupation
        mf_doml = scf.ROHF(mol)
        lib.pyscf_helper.atomic_scf_with_dominant_l_config_(mf_doml, config)
        mf_doml.kernel()
        self.assertAlmostEqual(mf_uncon.e_tot, mf_doml.e_tot, 6)

    def test_excited_state(self):
        ''' For the 3d^9 4s^2 excited state, we compared to known values.
        '''
        mol = self.mol
        config = [
            (4, 6, 5, 0),   # spin alpha: 1s^1 2s^1 2p^3 3s^1 3p^3 3d^5 4s^1
            (4, 6, 4, 0),   # spin beta : 1s^1 2s^1 2p^3 3s^1 3p^3 3d^4 4s^1
        ]

        # pure-l occupation
        mf_purel = scf.UHF(mol)
        lib.pyscf_helper.atomic_scf_with_pure_l_config_(mf_purel, config)
        mf_purel.kernel()
        self.assertAlmostEqual(mf_purel.e_tot, -1638.94973350937, 6)

        # dominant-l occupation
        mf_doml = scf.UHF(mol)
        lib.pyscf_helper.atomic_scf_with_dominant_l_config_(mf_doml, config)
        mf_doml.kernel()
        self.assertAlmostEqual(mf_doml.e_tot, -1638.95368170136, 6)


class CostFuncTest(unittest.TestCase):

    def test_cost_func_mf(self):
        atm = 'C'
        spin = 2
        basis = 'cc-pvdz'
        val_l = [0,1]

        # from cost_func
        cost_func = lib.pyscf_helper.get_cost_func(
            atm, scf.ROHF, mol_settings={'spin': spin}, keep_l=val_l)
        spec = BasisSpec.init_from_pyscf_basis(
            gto.basis.load(basis, atm), atm=atm,
        )
        e_tot = cost_func(spec)

        # from pyscf
        mol = gto.M(atom=atm, basis=spec.get_pyscf_basis(keep_l=val_l), spin=spin).set(verbose=0)
        mf = scf.ROHF(mol)
        mf.kernel()

        self.assertAlmostEqual(mf.e_tot, e_tot, 6)

    def test_cost_func_corr(self):
        atm = 'B'
        spin = 1
        basis = 'cc-pvdz'
        frozen = 1

        # from cost_func
        cost_func = lib.pyscf_helper.get_cost_func(
            atm, scf.UHF, mol_settings={'spin': spin},
            CORR=cc.UCCSD, corr_settings={'frozen': frozen},
        )
        spec = BasisSpec.init_from_pyscf_basis(
            gto.basis.load(basis, atm), atm=atm,
        )
        e_corr = cost_func(spec)

        # from pyscf
        mol = gto.M(atom=atm, basis=spec.get_pyscf_basis(), spin=spin).set(verbose=0)
        mf = scf.UHF(mol)
        mf.kernel()

        mcc = cc.UCCSD(mf, frozen=frozen)
        mcc.kernel()

        self.assertAlmostEqual(mcc.e_corr, e_corr, 6)

    def test_customized_SCF(self):
        atm = 'Be'
        spin = 2
        config = [  # Ms=1 triplet 1s^2 2s^1 2p^1
            (2, 1, 0, 0),   # spin alpha: 1s^1 2s^1 2p^1
            (1, 0, 0, 0),   # spin beta : 1s^1
        ]
        basis = 'cc-pvdz'
        val_l = [0,1]

        # customized SCF
        def SCF(mol):
            mf = scf.ROHF(mol)
            lib.pyscf_helper.atomic_scf_with_pure_l_config_(mf, config)
            return mf

        # from cost_func
        cost_func = lib.pyscf_helper.get_cost_func(
            atm, SCF, mol_settings={'spin': spin}, keep_l=val_l)
        spec = BasisSpec.init_from_pyscf_basis(
            gto.basis.load(basis, atm), atm=atm,
        )
        e_tot = cost_func(spec)

        # from pyscf
        mol = gto.M(atom=atm, basis=spec.get_pyscf_basis(keep_l=val_l), spin=spin).set(verbose=0)
        mf = scf.ROHF(mol)
        lib.pyscf_helper.atomic_scf_with_pure_l_config_(mf, config)
        mf.kernel()

        self.assertAlmostEqual(mf.e_tot, e_tot, 6)

if __name__ == '__main__':
    print('Run tests for pyscf_helper')
    unittest.main()
