import unittest
import numpy as np

from pygto.basis import ETB, Full


class ChannelTest(unittest.TestCase):
    def test_ETB(self):
        def test1(es):
            n = len(es)
            amin = es.min()
            if n > 1:
                amax = es.max()
                beta = np.exp(np.log(amax/amin) / (n-1))

            for l in [0,1]:
                c = ETB(l, es)

                self.assertEqual(c.l, l)
                self.assertEqual(c.nao, (2*l+1) * n)
                self.assertEqual(c.nbas, n)
                self.assertEqual(c.nparam, 1 if n == 1 else 2)
                self.assertAlmostEqual(c.amin, amin, 6)
                if n > 1:
                    self.assertAlmostEqual(c.beta, beta, 6)

        es = np.asarray([0.3, 0.9, 2.7])
        test1(es)

        es = np.asarray([0.3])
        test1(es)

    def test_ETB_merge(self):
        l = 2
        es1 = [0.3, 0.9, 2.7]
        es2 = [3.0, 4.8]

        es = np.asarray(es1+es2)
        n = len(es)
        amin = es.min()
        amax = es.max()
        beta = np.exp(np.log(amax/amin) / (n-1))

        c1 = ETB(l, es1)
        c2 = ETB(l, es2)

        c = c1.merge(c2)

        self.assertEqual( c.nao, c1.nao+c2.nao )

    def test_merge_failure(self):
        l1 = 2
        l2 = 1
        es1 = [0.3, 0.9, 2.7]
        es2 = [3.0, 4.8]

        c1 = ETB(l1, es1)
        c2 = ETB(l2, es2)

        with self.assertRaises(TypeError):
            c = c1.merge(c2)

        c1 = ETB(l1, es1)
        c2 = Full(l1, es2)

        with self.assertRaises(TypeError):
            c = c1.merge(c2)

    def test_Full(self):
        es = [0.3, 0.9, 2.7]
        n = len(es)
        for l in [0,1]:
            c = Full(l, es)

            self.assertEqual(c.l, l)
            self.assertEqual(c.nao, (2*l+1) * n)
            self.assertEqual(c.nbas, n)
            self.assertEqual(c.nparam, n)
            self.assertAlmostEqual(c.exponents[0], min(es), 6)
            self.assertAlmostEqual(c.exponents[-1], max(es), 6)

    def test_Full_merge(self):
        l = 2
        es1 = [0.3, 0.9, 2.7]
        es2 = [3.0, 4.8]

        es = np.sort(np.asarray(es1+es2))
        n = len(es)

        c1 = Full(l, es1)
        c2 = Full(l, es2)

        c = c1.merge(c2)

        self.assertAlmostEqual(abs(c.exponents - es).max(), 0, 6)

    def test_Full_merge_repeat(self):
        l = 2
        es1 = [0.3, 0.9, 2.7]
        es2 = [2.7, 4.8]

        es = np.sort(np.unique(np.asarray(es1+es2)))
        n = len(es)

        c1 = Full(l, es1)
        c2 = Full(l, es2)

        c = c1.merge(c2)

        self.assertAlmostEqual(abs(c.exponents - es).max(), 0, 6)

    def test_parameters(self):
        c = Full(0, [0.3, 0.9, 2.7])
        params = c.parameters
        params[0] += 0.1

        # check parameters are copied
        self.assertAlmostEqual(c.exponents[0], 0.3, 6)
        self.assertNotAlmostEqual(c.parameters[0], params[0])

        # check parameter update with wrong size raises a ValueError
        with self.assertRaises(ValueError):
            c.parameters = [1.0, 2.0]

        # check `with_parameters` create a new copy with updated parameters
        c1 = c.with_parameters(params)
        self.assertIsNot(c1, c)
        self.assertNotAlmostEqual(c1.exponents[0], c.exponents[0])


if __name__ == '__main__':
    print('Run tests for Channel')
    unittest.main()
