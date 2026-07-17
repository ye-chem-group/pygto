''' BFGS optimizer.
'''

import numpy as np

from pygto.lib.numdiff_helper import numer_grad
from pygto.optimizer.optimizer import Optimizer


class BFGS(Optimizer):
    ''' Limited-memory BFGS optimizer with Armijo backtracking.

        Args:
            spec (BasisSpec):
                Basis specification to optimize.
            cost_func (callable):
                Function that accepts a BasisSpec and returns its scalar cost.
            grad_func (callable):
                Function that accepts a BasisSpec and returns the gradient of the
                total objective. Default is None, which uses numerical gradients.
            verbose (int):
                Logging verbosity. Default is None.

        Attributes:
            memory (int):
                Maximum number of stored correction pairs. Default is 20.
            max_step (float or None):
                Maximum absolute component of the search direction. Default is 0.5;
                None disables direction scaling.
            armijo_c1 (float):
                Armijo sufficient-decrease coefficient. Default is `1e-4`.
            step_min (float):
                Minimum line-search step. Default is `1e-8`.
            max_backtracks (int):
                Maximum number of backtracking evaluations. Default is 60.
            curvature_tol (float):
                Relative curvature threshold for accepting correction pairs. Default
                is `1e-10`.
            grad_abs_step (float):
                Absolute step for numerical gradients. Default is `1e-4`.
            grad_rel_step (float):
                Relative step for numerical gradients. Default is `1e-4`.
    '''

    support_grad = True

    def __init__(self, spec, cost_func, grad_func=None, verbose=None):
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
        ''' Log general and BFGS-specific optimizer settings. '''
        Optimizer.dump_flags(self)
        self.log_info('grad_func= %s' % ('None' if self.grad_func is None else 'user'))
        self.log_info('memory= %d' % self.memory)
        self.log_info('max_step= %.3e' % self.max_step)
        self.log_info('armijo_c1= %.3e' % self.armijo_c1)
        self.log_info('step_min= %.3e' % self.step_min)
        self.log_info('max_backtracks= %d' % self.max_backtracks)
        self.log_info('curvature_tol= %.3e' % self.curvature_tol)
        self.log_info('grad_abs_step= %.3e' % self.grad_abs_step)
        self.log_info('grad_rel_step= %.3e' % self.grad_rel_step)
        self.log_info('')

    def initialize(self):
        ''' Initialize objective, gradient, and BFGS correction history. '''
        super().initialize()
        self.gradient = self.get_gradient(self.parameters)
        self.s_list = []
        self.y_list = []
        self.rho_list = []
        self.status = 'initialized'

    def get_gradient(self, parameters):
        ''' Evaluate the objective gradient.

            Args:
                parameters (array_like):
                    Optimization parameters.

            Return:
                gradient (ndarray):
                    Analytic gradient from `grad_func`, or a numerical gradient when
                    `grad_func` is None.
        '''
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
        ''' Safely evaluate the objective and gradient.

            Args:
                parameters (array_like):
                    Optimization parameters.

            Return:
                objective (float or None):
                    Objective value, or None when evaluation fails.
                gradient (ndarray or None):
                    Gradient, or None when evaluation fails.
                success (bool):
                    Whether both values are finite and were evaluated successfully.
        '''
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
        ''' Perform one BFGS step with Armijo backtracking.

            Note:
                Failed line searches retain the current parameters and clear the
                correction history.
        '''
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
            self.log_warn('BFGS direction is not descent. Resetting history.')
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
            self.log_warn('Line search failed. Keeping current parameters.')
            self.s_list = []
            self.y_list = []
            self.rho_list = []
            self.status = 'line_search_failed'
            return

        self.parameters = xtrial
        self.objective = ftrial
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
            self.log_warn('Skipping BFGS update due to poor curvature condition.')

        self.status = 'accepted'
