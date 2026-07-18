import os
import tempfile
import textwrap
import unittest

from pygto import lib


class LoadBasisNWChemTest(unittest.TestCase):

    def test_indented_inline_basis(self):
        basis_text = '''
            #BASIS SET: (2s,1p) -> [2s,1p]
            C    S
                10.0D+00    0.25
            C    S
                 1.0        1.00
            C    P
                 0.5        1.00
        '''

        basis = lib.load_basis_nwchem(basis_text, 'C')

        self.assertEqual(basis, [
            [0, [10.0, 0.25]],
            [0, [1.0, 1.0]],
            [1, [0.5, 1.0]],
        ])

    def test_last_element_in_file(self):
        basis_text = textwrap.dedent('''
            BASIS "ao basis" PRINT
            #BASIS SET: (1s) -> [1s]
            H    S
                 1.0        1.0
            #BASIS SET: (1s,1p) -> [1s,1p]
            He   S
                 2.0        1.0
            He   P
                 0.5        1.0
            END
        ''')

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, 'basis.dat')
            with open(path, 'w') as f:
                f.write(basis_text)
            basis = lib.load_basis_nwchem(path, 'He')

        self.assertEqual(basis, [
            [0, [2.0, 1.0]],
            [1, [0.5, 1.0]],
        ])


if __name__ == '__main__':
    unittest.main()
