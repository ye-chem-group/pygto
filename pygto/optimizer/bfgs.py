''' BFGS optimizer.
'''

import numpy as np

from pygto.lib.numdiff_helper import numer_grad
from pygto.optimizer.optimizer import Optimizer


class BFGS(Optimizer):

    def __init__(self, spec, cost_func, grad_func=None, verbose=4):
        super().__init__(spec, cost_func, verbose=verbose)

        self.grad_func = grad_func
        self.memory = 20
        self.max_step = 0.5
        self.armijo_c1 = 1e-4
        self.step_min = 1e-8
        self.max_backtracks = 60
        self.curvature_tol = 1e-10
        self.grad_abs_step = 1e-4
        self.grad_rel_step = 1e-4

        self.s_list = None
        self.y_list = None
        self.rho_list = None

    def dump_flags(self):
        Optimizer.dump_flags(self)
        self.log('grad_func= %s' % ('None' if self.grad_func is None else 'user'))
        self.log('memory= %d' % self.memory)
        self.log('max_step= %.3e' % self.max_step)
        self.log('armijo_c1= %.3e' % self.armijo_c1)
        self.log('step_min= %.3e' % self.step_min)
        self.log('max_backtracks= %d' % self.max_backtracks)
        self.log('curvature_tol= %.3e' % self.curvature_tol)
        self.log('grad_abs_step= %.3e' % self.grad_abs_step)
        self.log('grad_rel_step= %.3e' % self.grad_rel_step)

    def initialize(self):
        super().initialize()
        self.gradient = self.get_gradient(self.parameters)
        self.s_list = []
        self.y_list = []
        self.rho_list = []
        self.status = 'initialized'

    def get_gradient(self, parameters):
        if self.grad_func is None:
            return numer_grad(
                self.get_objective,
                parameters,
                abs_step=self.grad_abs_step,
                rel_step=self.grad_rel_step,
            )

        spec = self.spec.with_parameters(parameters)
        return np.asarray(self.grad_func(spec), dtype=float)

    def safe_eval_gradient(self, parameters):
        objective = self.safe_eval(parameters)
        if not np.isfinite(objective):
            return None, None, False

        try:
            gradient = self.get_gradient(parameters)
            if not np.all(np.isfinite(gradient)):
                self.message = 'nonfinite_gradient'
                return None, None, False
            return objective, gradient, True
        except Exception as err:
            self.message = str(err)
            return None, None, False

    def next_step(self):
        self.status = 'started'
        self.message = None

        x = self.parameters.copy()
        f = self.objective
        g = self.gradient.copy()

        s_list = self.s_list
        y_list = self.y_list
        rho_list = self.rho_list

        q = g.copy()
        alpha = []
        for s, y, rho in zip(reversed(s_list), reversed(y_list), reversed(rho_list)):
            a = rho * np.dot(s, q)
            alpha.append(a)
            q -= a * y

        if len(s_list) > 0:
            sy = np.dot(s_list[-1], y_list[-1])
            yy = np.dot(y_list[-1], y_list[-1])
            gamma = sy / yy if yy > 0 else 1.0
        else:
            gamma = 1.0

        r = gamma * q
        for s, y, rho, a in zip(s_list, y_list, rho_list, reversed(alpha)):
            beta = rho * np.dot(y, r)
            r += s * (a - beta)

        p = -r
        gp = np.dot(g, p)
        if gp >= 0 or not np.isfinite(gp):
            self.log('BFGS direction is not descent. Resetting history.')
            self.s_list = []
            self.y_list = []
            self.rho_list = []
            s_list = self.s_list
            y_list = self.y_list
            rho_list = self.rho_list
            p = -g
            gp = np.dot(g, p)
            if gp >= 0 or not np.isfinite(gp):
                self.status = 'line_search_failed'
                self.message = 'no_descent_direction'
                return

        pmax = np.max(np.abs(p))
        if self.max_step is not None and pmax > self.max_step:
            p *= self.max_step / pmax
            gp = np.dot(g, p)

        step = 1.0
        accepted = False
        ftrial = None
        gtrial = None
        xtrial = None

        for _ in range(self.max_backtracks):
            xtrial = x + step * p
            if np.max(np.abs(xtrial - x)) == 0:
                self.message = 'zero_trial_step'
                break

            ftrial, gtrial, ok = self.safe_eval_gradient(xtrial)
            if ok and ftrial <= f + self.armijo_c1 * step * gp:
                accepted = True
                break

            step *= 0.5
            if step < self.step_min:
                self.message = 'step_below_step_min'
                break

        if not accepted:
            self.log('Line search failed. Keeping current parameters.')
            self.s_list = []
            self.y_list = []
            self.rho_list = []
            self.status = 'line_search_failed'
            return

        self.parameters = xtrial
        self.objective = ftrial
        self.cost = ftrial
        self.gradient = gtrial

        s = self.parameters - x
        y = self.gradient - g

        sy = np.dot(s, y)
        ss = np.dot(s, s)
        yy = np.dot(y, y)
        if sy > self.curvature_tol * np.sqrt(ss * yy):
            if len(s_list) >= self.memory:
                s_list.pop(0)
                y_list.pop(0)
                rho_list.pop(0)

            s_list.append(s)
            y_list.append(y)
            rho_list.append(1.0 / sy)
        else:
            self.log('Skipping BFGS update due to poor curvature condition.')

        self.status = 'accepted'
