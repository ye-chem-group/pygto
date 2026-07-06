''' Nelder-Mead optimizer.
'''

import numpy as np

from pygto.optimizer.optimizer import Optimizer


class NelderMead(Optimizer):

    def __init__(self, spec, cost_func, verbose=4):
        super().__init__(spec, cost_func, verbose=verbose)

        self.initial_step_abs = 2.5e-4
        self.initial_step_rel = 0.05
        self.reflection = 1.0
        self.expansion = 2.0
        self.contraction = 0.5
        self.shrink = 0.5
        self.simplex_spread_tol = 1e-8
        self.max_inner = 100

        self.xs = None
        self.fs = None

    def dump_flags(self):
        Optimizer.dump_flags(self)
        self.log_info('initial_step_abs= %.3e' % self.initial_step_abs)
        self.log_info('initial_step_rel= %.3e' % self.initial_step_rel)
        self.log_info('reflection= %.3e' % self.reflection)
        self.log_info('expansion= %.3e' % self.expansion)
        self.log_info('contraction= %.3e' % self.contraction)
        self.log_info('shrink= %.3e' % self.shrink)
        self.log_info('simplex_spread_tol= %.3e' % self.simplex_spread_tol)
        self.log_info('max_inner= %d' % self.max_inner)

    def initialize(self):
        super().initialize()

        xs, fs = self.init_simplex()
        self.update_simplex(xs, fs)
        self.status = 'initialized'

    def init_simplex(self):
        x0 = self.parameters
        xs = [x0.copy()]
        fs = [self.objective]

        for i in range(x0.size):
            xi = x0.copy()
            if abs(x0[i]) < 1e-3:
                xi[i] += self.initial_step_abs
            else:
                xi[i] += abs(x0[i]) * self.initial_step_rel
            xs.append(xi)
            fs.append(self.safe_eval(xi))

        return np.asarray(xs), np.asarray(fs)

    def update_simplex(self, xs, fs):
        self.xs, self.fs = sort_simplex(xs, fs)
        self.parameters = self.xs[0].copy()
        self.objective = float(self.fs[0])

    def next_step(self):
        self.status = 'started'
        self.message = None

        xs = self.xs.copy()
        fs = self.fs.copy()
        fmin_now = self.objective
        n = len(self.parameters) + 1

        if not np.isfinite(fmin_now):
            self.status = 'evaluation_failed'
            self.message = 'nonfinite_objective'
            return

        for _ in range(self.max_inner):
            xc = xs[:-1].mean(axis=0)
            xd = xc - xs[-1]
            x1 = xc + self.reflection * xd
            f1 = self.safe_eval(x1)

            fm = fs[0]
            ft = fs[-2]

            shrink = False
            if f1 < fm:
                x2 = xc + self.expansion * xd
                f2 = self.safe_eval(x2)
                if f2 < f1:
                    xnew = x2
                    fnew = f2
                else:
                    xnew = x1
                    fnew = f1
            elif f1 < ft:
                xnew = x1
                fnew = f1
            else:
                xh1 = xc - xd * self.contraction
                xh2 = xc + xd * self.contraction
                fh1 = self.safe_eval(xh1)
                fh2 = self.safe_eval(xh2)
                if fh1 > fh2:
                    xh1, xh2 = xh2, xh1
                    fh1, fh2 = fh2, fh1
                if fh1 < ft:
                    xnew = xh1
                    fnew = fh1
                else:
                    shrink = True
                    xs[1:] = (xs[1:] - xs[0]) * self.shrink + xs[0]
                    for i in range(1, n):
                        fs[i] = self.safe_eval(xs[i])

            if not shrink:
                xs[-1] = xnew
                fs[-1] = fnew

            xs, fs = sort_simplex(xs, fs)
            if not np.isfinite(fs[0]):
                self.status = 'evaluation_failed'
                break

            fmin = fs[0]
            df = fmin - fmin_now
            if abs(df) > 1e-30:
                self.status = 'accepted'
                break

            xave = xs.mean(axis=0)
            spread = (((xs - xave)**2.).mean(axis=1)**0.5).max()
            if spread < self.simplex_spread_tol:
                self.status = 'simplex_converged'
                break
        else:
            if np.isfinite(fs[0]) and abs(float(fs[0]) - float(fmin_now)) <= 1e-30:
                self.status = 'no_improvement'
                self.message = (
                    'Nelder-Mead reached max_inner without improving the best vertex'
                )
            else:
                self.status = 'simplex_stalled'
                self.message = (
                    'Nelder-Mead inner loop reached max_inner=%d' % self.max_inner
                )

        self.update_simplex(xs, fs)


def sort_simplex(xs, fs):
    xs = np.asarray(xs)
    fs = np.asarray(fs)
    order = np.argsort(fs)
    return xs[order], fs[order]
