import unittest
import numpy as np

from pygto.basis import BasisSpec, ETB, Full


class BasisSpecTest(unittest.TestCase):
    def test_spec_merge_angular_momentum(self):
        channels = [
            Full(1, [0.3, 0.9, 2.7]),
            Full(1, [2.7, 4.5]),
        ]
        spec = BasisSpec(channels)
        spec = spec.merge_angular_momentum()

        self.assertEqual(spec.nbas, sum([c.nbas for c in channels])-1)

        channels = [
            Full(0, [0.3, 0.9, 2.7]),
            ETB(0, [2.7, 4.5]),
        ]
        spec = BasisSpec(channels)

        with self.assertRaises(TypeError):
            spec.merge_angular_momentum()

    def test_parameters(self):
        spec = BasisSpec([
            Full(0, [0.3, 0.9]),
            ETB(1, [0.2, 0.6, 1.8]),
        ])

        # check parameter updating with wrong size raises ValueError
        with self.assertRaises(ValueError):
            spec.parameters = np.zeros(spec.nparam + 1)

        # check with parameter does not mutate spec and does update parameters
        params = spec.parameters
        params[0] += 0.1
        spec1 = spec.with_parameters(params)
        self.assertIsNot(spec1, spec)
        self.assertNotAlmostEqual(spec1.parameters[0], spec.parameters[0])
        np.testing.assert_allclose(spec1.channels[1].exponents, spec.channels[1].exponents)

    def test_active_channel(self):
        channels = [
            Full(0, [0.3, 0.9, 2.7]),
            Full(1, [2.7, 4.5]),
            Full(1, [3.5]),
        ]
        spec = BasisSpec(channels)

        # specifying active channels
        spec.set_active_channel(0)
        np.testing.assert_allclose( spec.get_active_mask(), np.array([True,False,False]) )

        spec.active_channel = [0, 2]
        np.testing.assert_allclose( spec.get_active_mask(), np.array([True,False,True]) )

        # nparam only counts active parameters
        mask = spec.get_active_mask()
        nparam = sum([spec.channels[i].nparam for i in np.where(mask)[0]])
        self.assertEqual( spec.nparam, nparam )

        # copy preserves active_channel
        spec1 = spec.copy()
        np.testing.assert_allclose( spec1.active_channel, spec.active_channel )

        # with_parameters preserve active_channel
        np.testing.assert_allclose( spec.with_parameters([3., 4., 5., 6.]).active_channel,
                                    spec.active_channel )

        # replace_channel preserve active_channel
        np.testing.assert_allclose( spec.replace_channel(Full(1, [3., 4.]), 1).active_channel,
                                    spec.active_channel )

        # reset
        spec.set_active_channel()
        self.assertEqual( spec.active_channel, None )
        spec1.set_active_channel(None)
        self.assertEqual( spec1.active_channel, None )

        # exceptions
        for x in [-1, 10, [0,10], [-1,0]]:
            with self.assertRaises(ValueError):
                spec.set_active_channel(x)

        for x in ['1', 1.0]:
            with self.assertRaises(TypeError):
                spec.set_active_channel(x)

        # temporary
        spec.set_active_channel()
        with spec.temporary_active_channel([0,1]):
            np.testing.assert_allclose( spec.get_active_mask(), np.array([True,True,False]) )
        np.testing.assert_allclose( spec.get_active_mask(), np.array([True,True,True]) )

    def test_active_l(self):
        channels = [
            Full(0, [0.3, 0.9, 2.7]),
            Full(1, [2.7, 4.5]),
            ETB(1, [3.5]),
            ETB(2, [3.5])
        ]
        spec = BasisSpec(channels)

        # specifying active l
        spec.set_active_l(1)
        np.testing.assert_allclose( spec.get_active_mask(), np.array([False,True,True,False]) )

        spec.set_active_l([1,2])
        np.testing.assert_allclose( spec.get_active_mask(), np.array([False,True,True,True]) )

        # active_l cannot be assigned directly
        with self.assertRaises(RuntimeError):
            spec.active_l = [1,2]

        # copy preserves active_l
        spec1 = spec.copy()
        np.testing.assert_allclose( spec1.active_l, spec.active_l )

        # reset
        spec.set_active_l()
        self.assertEqual( spec.active_l, None )
        spec1.set_active_l(None)
        self.assertEqual( spec1.active_l, None )

        # exceptions
        for x in [-1, 10, [0,10], [-1,0]]:
            with self.assertRaises(ValueError):
                spec.set_active_l(x)

        for x in ['1', 1.0]:
            with self.assertRaises(TypeError):
                spec.set_active_l(x)

        # temporary
        spec.set_active_l()
        with spec.temporary_active_l([0,1]):
            np.testing.assert_allclose( spec.get_active_mask(), np.array([True,True,True,False]) )
        np.testing.assert_allclose( spec.get_active_mask(), np.array([True,True,True,True]) )


if __name__ == '__main__':
    print('Run tests for Spec')
    unittest.main()
