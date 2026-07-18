import os
import tempfile
import unittest
import h5py
import numpy as np

from pygto import lib
from pygto.basis import BasisSpec, ETB, Full


class ChkfileTest(unittest.TestCase):

    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.chkfile = os.path.join(self.tmpdir.name, 'basis.chk')

    def tearDown(self):
        self.tmpdir.cleanup()

    def test_scalar_string(self):
        lib.chkfile_helper.dump(self.chkfile, 'atom', 'Cu')
        atom = lib.chkfile_helper.load(self.chkfile, 'atom')

        self.assertEqual(atom, 'Cu')
        self.assertIsInstance(atom, str)

    def test_channel(self):
        channel = ETB(1, [0.3, 0.9, 2.7])
        channel.dump_chkfile(self.chkfile)
        channel1 = ETB.init_from_chkfile(self.chkfile)

        self.assertEqual(channel1.l, channel.l)
        np.testing.assert_allclose(channel1.exponents, channel.exponents)

    def test_spec(self):
        spec = BasisSpec([
            Full(0, [0.3, 0.9]),
            ETB(1, [0.2, 0.6, 1.8]),
        ]).set(atm='C')
        spec.dump_chkfile(self.chkfile)
        spec1 = BasisSpec.init_from_chkfile(self.chkfile)

        self.assertEqual(spec1.atm, 'C')
        self.assertIsInstance(spec1.channels[0], Full)
        self.assertIsInstance(spec1.channels[1], ETB)
        for c, c1 in zip(spec.channels, spec1.channels):
            self.assertEqual(c1.l, c.l)
            np.testing.assert_allclose(c1.exponents, c.exponents)
        self.assertIn('\nC  S\n', spec1.get_basis_str())

    def test_none_atom(self):
        spec = BasisSpec([Full(0, [0.3])])
        spec.dump_chkfile(self.chkfile)
        spec1 = BasisSpec.init_from_chkfile(self.chkfile)

        self.assertIsNone(spec1.atm)

    def test_convert_channel_type(self):
        spec = BasisSpec([
            Full(0, [0.3, 0.8, 2.7]),
            ETB(1, [0.2, 0.6, 1.8]),
        ])
        spec.dump_chkfile(self.chkfile)

        spec1 = BasisSpec.init_from_chkfile(self.chkfile, channel_type='full')
        self.assertTrue(all(isinstance(c, Full) for c in spec1.channels))

        spec1 = BasisSpec.init_from_chkfile(self.chkfile, channel_type='ETB')
        self.assertTrue(all(isinstance(c, ETB) for c in spec1.channels))

        with self.assertRaises(ValueError):
            BasisSpec.init_from_chkfile(self.chkfile, channel_type='unknown')

    def test_channel_type_is_required(self):
        spec = BasisSpec([Full(0, [0.3])])
        spec.dump_chkfile(self.chkfile)
        with h5py.File(self.chkfile, 'a') as f:
            del f['spec/channel_0/type']

        with self.assertRaises(KeyError):
            BasisSpec.init_from_chkfile(self.chkfile)

    def test_overwrite_with_fewer_channels(self):
        prefix = 'basis/spec'
        spec = BasisSpec([
            Full(0, [0.3]),
            Full(1, [0.2]),
            Full(2, [0.1]),
        ])
        spec.dump_chkfile(self.chkfile, prefix)

        spec = BasisSpec([ETB(1, [0.2, 0.6, 1.8])])
        spec.dump_chkfile(self.chkfile, prefix)
        spec1 = BasisSpec.init_from_chkfile(self.chkfile, prefix)

        self.assertEqual(spec1.nchannel, 1)
        self.assertIsInstance(spec1.channels[0], ETB)
        self.assertEqual(spec1.channels[0].l, 1)


if __name__ == '__main__':
    print('Run tests for chkfile')
    unittest.main()
