import sys
import numpy as np

from pygto.lib import StreamObject, FloatSum
from pygto.lib import chkfile_helper


class Optimizer(StreamObject):
    ''' Base class for PyGTO optimizers.

        Subclasses should implement `next_step`, which updates at least
        `self.parameters`, `self.objective`, and `self.status`.
    '''

    support_grad = False

    def __init__(self, spec, cost_func, verbose=None):
        self.spec = spec
        self.cost_func = cost_func
        if verbose is not None: self.verbose = verbose

        self._accuracy = None
        self.ftol = None
        self.xtol = None
        self.gtol = None
        self.max_cycle = 1000

        # Attributes for optimization control
        self.accuracy = 'medium'    # this sets ftol, xtol, and gtol

        self.ratio_min = 1.7
        self.ratio_penalty_strength = 10.   # Hartree
        self.ratio_penalty_warning_thresh = 1e-6    # 1 micro-Hartree

        self.chkfile = None # save most recent spec

        # Attributes set by `kernel`. Do not set them.
        self.parameters = None
        self.cost = None
        self.ratio_penalty = None
        self.objective = None
        self.objective_info = None
        self.gradient = None
        self.converged = False
        self.stop_reason = None
        self.status = None
        self.message = None
        self.cycle = 0
        self.feval = 0
        self.history = []

    @property
    def accuracy(self):
        return self._accuracy

    @accuracy.setter
    def accuracy(self, accuracy):
        accuracy = accuracy.lower()
        if accuracy == 'low':
            self.ftol = 1e-6
            self.xtol = 1e-2
        elif accuracy == 'medium':
            self.ftol = 3e-8
            self.xtol = 1e-2
        elif accuracy == 'high':
            self.ftol = 1e-9
            self.xtol = 1e-3
        elif accuracy == 'ultra':
            self.ftol = 1e-10
            self.xtol = 1e-4
        else:
            raise ValueError('Unknown accuracy "%s"' % accuracy)
        self.gtol = self.ftol**0.5
        self._accuracy = accuracy

    def format_cost(self):
        return f'cost= {self.cost:.10f}'

    def ratio_penalty_warning(self):
        if self.ratio_penalty > self.ratio_penalty_warning_thresh:
            self.log_warn('ratio_penalty= %.2e' % self.ratio_penalty)

    def get_cost(self, parameters):
        self.feval += 1
        spec = self.spec.with_parameters(parameters)
        return float(self.cost_func(spec))

    def get_ratio_penalty(self, parameters):
        spec = self.spec.with_parameters(parameters)
        return float(spec.get_ratio_penalty(self.ratio_min, self.ratio_penalty_strength))

    def get_objective_info(self, parameters):
        cost = self.get_cost(parameters)
        ratio_penalty = self.get_ratio_penalty(parameters)
        objective_info = FloatSum(cost=cost, ratio_penalty=ratio_penalty)
        return objective_info

    def update_objective_info_(self, parameters):
        self.objective_info = self.get_objective_info(parameters)
        self.objective = self.objective_info.value
        self.cost = self.objective_info.cost
        self.ratio_penalty = self.objective_info.ratio_penalty

    def get_objective(self, parameters):
        return self.get_objective_info(parameters).value

    def safe_eval(self, parameters):
        try:
            objective = self.get_objective(parameters)
            if not np.isfinite(objective):
                self.message = 'nonfinite_objective'
                return np.inf
            return objective
        except Exception as err:
            self.message = str(err)
            return np.inf

    def initialize(self):
        self.parameters = self.spec.parameters
        self.feval = 0
        self.update_objective_info_(self.parameters)
        self.gradient = None
        self.converged = False
        self.stop_reason = None
        self.status = 'initialized'
        self.message = None
        self.cycle = 0
        self.history = []

    def next_step(self):
        raise NotImplementedError

    def check_convergence(self, df, dx):
        if not np.isfinite(self.objective):
            self.stop_reason = 'nonfinite_objective'
            return True

        if abs(df) < self.ftol and dx < self.xtol:
            if self.gradient is None:
                self.converged = True
                self.stop_reason = 'df+dx'
                return True

            gmax = np.max(np.abs(self.gradient))
            if gmax < self.gtol:
                self.converged = True
                self.stop_reason = 'df+dx+dg'
                return True

        return False

    def dump_flags(self):
        self.log_info('\n')
        self.log_info('******** %s ********' % (self.__class__.__name__))
        self.log_info('ftol= %.3e' % self.ftol)
        self.log_info('xtol= %.3e' % self.xtol)
        self.log_info('gtol= %.3e' % self.gtol)
        self.log_info('max_cycle= %d' % self.max_cycle)
        self.log_info('chkfile= %s' % (str(self.chkfile)))
        self.log_info('')

    def kernel(self, **kwargs):
        self.set(**kwargs)

        if getattr(self, 'grad_func', None) is not None and self.ratio_penalty_strength is not None:
            self.log_warn(
                'User-provided `grad_func` is assumed to be the gradient of the full '
                'objective, including ratio penalty. PyGTO does not add the ratio-penalty '
                'gradient automatically at this point.'
            )

        self.initialize()
        self.dump_flags()
        self.print_init()
        self.save_history(df=0.0, dx=0.0)

        spec = self.spec.with_parameters(self.parameters)

        if not np.isfinite(self.objective):
            self.stop_reason = 'nonfinite_objective'
        else:
            for cycle in range(1, self.max_cycle+1):
                self.cycle = cycle
                objective_old = self.objective
                spec_old = spec

                self.next_step()

                # update objective, cost, penalty, ...
                self.update_objective_info_(self.parameters)

                df = self.objective - objective_old
                spec = self.spec.with_parameters(self.parameters)
                dx = np.max(np.abs(
                    spec.convergence_parameters - spec_old.convergence_parameters
                ))

                self.print_step(df, dx)
                self.save_history(df, dx)
                self.dump_chkfile(spec)

                if self.status in ('failed', 'line_search_failed', 'evaluation_failed'):
                    self.stop_reason = self.status
                    break

                if self.check_convergence(df, dx):
                    break
            else:
                self.stop_reason = 'max_cycle'

        self.spec.parameters = self.parameters

        self.print_final()
        return self.cost, self.spec

    def save_history(self, df=None, dx=None):
        data = {
            'cycle': self.cycle,
            'cost': self.cost,
            'objective': self.objective,
            'objective_info': dict(self.objective_info.__dict__),
            'df': df,
            'dx': dx,
            'status': self.status,
        }
        if self.gradient is not None:
            data['gmax'] = np.max(np.abs(self.gradient))
        self.history.append(data)

    def print_init(self):
        self.log_info('')
        self.log_info('Init basis:')
        if self.verbose >= 4:   # info
            self.spec.dump_basis(stdout=self.stdout)
        self.log_info('')
        self.log_note('Init %s' % (self.format_cost()))
        if hasattr(self, 'ratio_penalty'): self.ratio_penalty_warning()

    def print_step(self, df, dx):
        if self.gradient is None:
            self.log_info(
                'cycle= %d  %s  df= % .2e  dx= %.2e  stat= %s'
                % (self.cycle, (self.format_cost()), df, dx, self.status)
            )
        else:
            gmax = np.max(np.abs(self.gradient))
            self.log_info(
                'cycle= %d  %s  df= % .2e  dx= %.2e  |g|= %.2e  stat= %s'
                % (self.cycle, (self.format_cost()), df, dx, gmax, self.status)
            )
        if hasattr(self, 'ratio_penalty'): self.ratio_penalty_warning()

    def print_final(self):
        if self.converged:
            self.log_note('Convergence is reached for %s' % (self.__class__.__name__))
        else:
            self.log_warn('Convergence is not reached for %s: %s' % (
                self.__class__.__name__, self.stop_reason
            ))
        self.log_note('Final %s' % (self.format_cost()))
        if hasattr(self, 'ratio_penalty'): self.ratio_penalty_warning()
        self.log_info('Final basis:')
        if self.verbose >= 4:   # info
            self.spec.dump_basis(stdout=self.stdout)
        self.log_info('')

    def dump_chkfile(self, spec=None, chkfile=None, prefix=None):
        if chkfile is None: chkfile = self.chkfile
        if chkfile is not None:
            if spec is None: spec = self.spec
            spec.dump_chkfile(chkfile, prefix=prefix)
