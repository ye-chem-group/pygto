import sys
import numpy as np


class Optimizer:
    ''' Base class for PyGTO optimizers.

        Subclasses should implement `next_step`, which updates at least
        `self.parameters`, `self.objective`, and `self.status`.
    '''

    def __init__(self, spec, cost_func, verbose=4):
        self.spec = spec.copy()
        self.cost_func = cost_func
        self.verbose = verbose

        self._accuracy = None
        self.ftol = None
        self.xtol = None
        self.gtol = None
        self.max_cycle = 1000

        self.accuracy = 'medium'    # this sets ftol, xtol, and gtol

        # Attributes set by `kernel`. Do not set them.
        self.parameters = None
        self.cost = None
        self.objective = None
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
        else:
            raise ValueError('Unknown accuracy "%s"' % accuracy)
        self.gtol = self.ftol**0.5
        self._accuracy = accuracy

    def set(self, **kwargs):
        for key, val in kwargs.items():
            if not hasattr(self, key):
                raise AttributeError(
                    '%s has no attribute "%s"' % (self.__class__.__name__, key)
                )
            setattr(self, key, val)
        return self

    def get_cost(self, spec):
        return float(self.cost_func(spec))

    def get_objective(self, parameters):
        spec = self.spec.with_parameters(parameters)
        cost = self.get_cost(spec)
        self.feval += 1
        return cost

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
        self.objective = self.get_objective(self.parameters)
        self.cost = self.objective
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
        self.log('Optimizer= %s' % self.__class__.__name__)
        self.log('ftol= %.3e' % self.ftol)
        self.log('xtol= %.3e' % self.xtol)
        self.log('gtol= %.3e' % self.gtol)
        self.log('max_cycle= %d' % self.max_cycle)

    def kernel(self, **kwargs):
        self.set(**kwargs)

        self.initialize()
        self.dump_flags()
        self.print_init()
        self.save_history(df=0.0, dx=0.0)

        if not np.isfinite(self.objective):
            self.stop_reason = 'nonfinite_objective'
        else:
            for cycle in range(1, self.max_cycle+1):
                self.cycle = cycle
                objective_old = self.objective
                convergence_parameters_old = self.spec.convergence_parameters.copy()

                self.next_step()

                df = self.objective - objective_old
                self.cost = self.objective
                self.spec = self.spec.with_parameters(self.parameters)
                dx = np.max(np.abs(
                    self.spec.convergence_parameters - convergence_parameters_old
                ))

                self.print_step(df, dx)
                self.save_history(df, dx)

                if self.status in ('failed', 'line_search_failed', 'evaluation_failed'):
                    self.stop_reason = self.status
                    break

                if self.check_convergence(df, dx):
                    break
            else:
                self.stop_reason = 'max_cycle'

        self.print_final()
        return self.cost, self.spec

    def save_history(self, df=None, dx=None):
        data = {
            'cycle': self.cycle,
            'cost': self.cost,
            'objective': self.objective,
            'df': df,
            'dx': dx,
            'status': self.status,
        }
        if self.gradient is not None:
            data['gmax'] = np.max(np.abs(self.gradient))
        self.history.append(data)

    def print_init(self):
        self.log('init cost= %.12f' % self.cost)

    def print_step(self, df, dx):
        if self.gradient is None:
            self.log(
                'cycle= %4d  cost= %.12f  df= % .2e  dx= % .2e  stat= %s'
                % (self.cycle, self.cost, df, dx, self.status)
            )
        else:
            gmax = np.max(np.abs(self.gradient))
            self.log(
                'cycle= %4d  cost= %.12f  df= % .2e  dx= % .2e  '
                '|g|= %.2e  stat= %s'
                % (self.cycle, self.cost, df, dx, gmax, self.status)
            )

    def print_final(self):
        if self.converged:
            self.log('Convergence is reached')
        else:
            self.log('Convergence is not reached: %s' % self.stop_reason)
        self.log('Final cost= %.12f' % self.cost)

    def log(self, msg, level=4):
        if self.verbose >= level:
            print(msg, file=sys.stdout)


if __name__ == '__main__':
    pass
