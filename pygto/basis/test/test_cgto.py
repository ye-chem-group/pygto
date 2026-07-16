import unittest
import numpy as np

from pygto.basis import ContractedBasis, ContractedChannel


class BasisSpecTest(unittest.TestCase):
    def test_channel(self):
        l = 2
        nprim = 4
        nctr = 3
        exponents = np.asarray([0.37, 3.87, 0.92, 1.55])
        coefficients = np.random.rand(len(exponents), nctr)
        order = np.argsort(exponents)

        channel = ContractedChannel(l, exponents, coefficients)

        self.assertEqual(channel.l, l)
        self.assertEqual(channel.nprim, nprim)
        self.assertEqual(channel.nctr, nctr)
        self.assertEqual(channel.nao, nctr * (2*l+1))

        # exponents and coefficients are sorted in ascending order by exponents
        np.testing.assert_allclose(channel.exponents, exponents[order])
        np.testing.assert_allclose(channel.coefficients, coefficients[order])
        ecs = np.hstack((exponents[order].reshape(-1,1), coefficients[order]))
        np.testing.assert_allclose(channel.ecs, ecs)

        # pyscf_basis is sorted in descending order by exponents by default...
        ecs = np.asarray(channel.get_pyscf_basis()[0][1:])
        np.testing.assert_allclose(channel.ecs[::-1], ecs)
        # ...but this can be turned off
        ecs = np.asarray(channel.get_pyscf_basis(sort=False)[0][1:])
        np.testing.assert_allclose(channel.ecs, ecs)

        # filter by range
        emin = exponents.min()+1e-6
        emax = exponents.max()-1e-6
        ecs = np.asarray(channel.get_pyscf_basis(emin=emin)[0][1:])
        self.assertEqual(ecs.shape[0], nprim-1)
        ecs = np.asarray(channel.get_pyscf_basis(emax=emax)[0][1:])
        self.assertEqual(ecs.shape[0], nprim-1)
        ecs = np.asarray(channel.get_pyscf_basis(emin=emin, emax=emax)[0][1:])
        self.assertEqual(ecs.shape[0], nprim-2)

        # copy does not affect original
        new = channel.copy()
        new.coefficients = np.random.rand(nprim, nctr-1)
        np.testing.assert_allclose(channel.coefficients, coefficients[order])

        # setting values
        with self.assertRaises(ValueError):
            channel.exponents = np.random.rand(nprim-1)
        with self.assertRaises(ValueError):
            channel.coefficients = np.random.rand(nprim-1, nctr)

    def test_basis(self):
        data = [
            # l, nprim, nctr
            (0, 9, 2),
            (0, 5, 1),
            (1, 3, 2)
        ]
        channels = []
        for l,nprim,nctr in data:
            exponents = np.random.rand(nprim) * 100
            coefficients = np.random.rand(nprim,nctr)-0.5
            channels.append( ContractedChannel(l, exponents, coefficients) )
        basis = ContractedBasis(channels)
        for i in range(basis.nchannel):
            l,nprim,nctr = data[i]
            self.assertEqual(basis.channel_l(i), l)
            self.assertEqual(basis.channel_nprim(i), nprim)
            self.assertEqual(basis.channel_nctr(i), nctr)

if __name__ == '__main__':
    print('Run tests for CGTO')
    unittest.main()
