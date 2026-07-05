import copy
import numpy as np

from pygto.optimizer import Optimizer, NelderMead, BFGS
from pygto.lib import StreamObject


PRESET_STAGES = {
    'loose': [
        {
            'optimizer': NelderMead,
            'optimizer_settings': {
                'accuracy': 'low',
            },
        },
        {
            'optimizer': NelderMead,
            'optimizer_settings': {
                'accuracy': 'medium',
            },
        },
    ],

    'default': [
        {
            'optimizer': NelderMead,
            'optimizer_settings': {
                'accuracy': 'medium',
            },
        },
        {
            'optimizer': BFGS,
            'optimizer_settings': {
                'accuracy': 'high',
            },
        },
    ],

    'refine': [
        {
            'optimizer': BFGS,
            'optimizer_settings': {
                'accuracy': 'medium',
            },
        },
        {
            'optimizer': BFGS,
            'optimizer_settings': {
                'accuracy': 'high',
            },
        },
    ]
}

class ScheduledOptimizer(StreamObject):
    def __init__(self, spec, cost_func, stages='default', grad_func=None, verbose=4):
        self.spec = spec.copy()
        self.cost_func = cost_func
        self.grad_func = grad_func
        self._stages = None
        self.verbose = verbose

        self.set_stages(stages)

        # attributes set by kernel; do not modify
        self.optimizers = []
        self.history = []
        self.feval = 0
        self.converged = False
        self.stop_reason = None

    @property
    def stages(self):
        return self._stages

    @stages.setter
    def stages(self, stages):
        self.set_stages(stages)

    def set_stages(self, stages):
        if isinstance(stages, str):
            stages = self.get_preset_stages(stages)
        elif isinstance(stages, dict):
            stages = [stages]
        elif isinstance(stages, (list,tuple)):
            if not all([isinstance(stage, dict) for stage in stages]):
                raise TypeError(f'Individual stage must be dict.')
        else:
            raise TypeError(f'Stages must be str or list/tuple.')

        # check stages is non-empty
        if len(stages) == 0:
            raise ValueError('Stages must not be empty.')

        # validate keys for each stage
        allowed_keys = {'optimizer', 'optimizer_settings', 'spec_settings', 'cost_func', 'grad_func'}
        for istage,stage in enumerate(stages):
            if 'optimizer' not in stage:
                raise KeyError('stages[%d] does not have "optimizer" key.' % (istage))
            if not set(stage.keys()).issubset(allowed_keys):
                raise KeyError('stages[%d] contains invalid keys. Allowed keys are %s' %
                               (istage, ', '.join(['"%s"'%k for k in allowed_keys])))

        # formalizing optimizer for each stage
        stages = copy.deepcopy(stages)
        for stage in stages:
            stage['optimizer'] = _formalize_optimizer(stage['optimizer'])

        self._stages = stages

    @staticmethod
    def get_preset_stages(name):
        if isinstance(name, str):
            if name.lower() not in PRESET_STAGES:
                raise ValueError(f'Unknown named stages "{name}"')
            stages = copy.deepcopy(PRESET_STAGES[name.lower()])
        else:
            raise TypeError('Stage name must be string.')

        return stages


    def kernel(self, **kwargs):
        self.set(**kwargs)

        self.initialize()

        for istage,stage in enumerate(self.stages):
            spec = self.spec

            OPT = stage['optimizer']

            self.log_note('Stage %d/%d'%(istage+1, len(self.stages)))
            self.print_stage_info(stage)

            self.apply_spec_settings_(spec, stage)

            cost_func = stage.get('cost_func', self.cost_func)
            if getattr(OPT, 'support_grad', False):
                grad_func = stage.get('grad_func', self.grad_func)
                opt = OPT(spec, cost_func, grad_func)
            else:
                opt = OPT(spec, cost_func)
            opt.set(verbose=self.verbose, stdout=self.stdout)

            self.apply_optimizer_settings_(opt, stage)

            self.cost, self.spec = opt.kernel()

            self.print_step(istage, opt)
            self.save_history(opt, stage)

        self.converged = all(opt.converged for opt in self.optimizers)
        self.stop_reason = self.history[-1]['stop_reason']

        self.print_final()

        return self.cost, self.spec

    @staticmethod
    def apply_spec_settings_(spec, stage):
        ''' Apply settings for BasisSpec in plance to `spec`.
        '''
        spec_settings = stage.get('spec_settings', None)
        if spec_settings is None:
            return spec

        allowed_keys = {'active_l', 'active_channel', 'channel_type'}
        if not set(spec_settings.keys()).issubset(allowed_keys):
            raise KeyError('"spec_settings" contains invalid keys. Allowed keys are: %s'
                           % (', '.join(['"%s"'%(k) for k in allowed_keys])))

        if 'active_l' in spec_settings:
            spec.set_active_l(spec_settings['active_l'])
        if 'active_channel' in spec_settings:
            spec.set_active_channel(spec_settings['active_channel'])
        if 'channel_type' in spec_settings:
            spec.convert_to_(spec_settings['channel_type'])

        return spec

    @staticmethod
    def apply_optimizer_settings_(optimizer, stage):
        ''' Apply settings for Optimizer in plance to `opt`.
        '''
        optimizer_settings = stage.get('optimizer_settings', None)
        if optimizer_settings is None:
            return optimizer

        optimizer.set(**optimizer_settings)

        return optimizer

    def print_stage_info(self, stage):
        self.log_note('optimizer= %s' % (stage['optimizer'].__name__))
        self.log_info('optimizer_settings= %s' % (str(stage.get('optimizer_settings', None))))
        self.log_info('spec_settings= %s' % (str(stage.get('spec_settings', None))))
        self.log_info('')

    def print_step(self, istage, opt):
        self.log_note('stage= %d  cost= %.12f  cycle= %d  feval= %d  converged= %s' %
                      (istage+1, self.cost, opt.cycle, opt.feval, str(opt.converged)))
        self.log_note('')

    def print_final(self):
        if self.converged:
            self.log_note('Convergence is reached for %s' % (self.__class__.__name__))
        else:
            self.log_warn('Convergence is not reached for %s: %s' %
                          (self.__class__.__name__, self.stop_reason))

    def initialize(self):
        self.optimizers = []
        self.history = []
        self.feval = 0
        self.converged = False
        self.stop_reason = None

    def save_history(self, opt, stage):
        self.optimizers.append(opt)
        self.feval += opt.feval

        data = {
            'optimizer': opt.__class__.__name__,
            'cost': opt.cost,
            'converged': opt.converged,
            'stop_reason': opt.stop_reason,
            'cycle': opt.cycle,
            'feval': opt.feval,
            'optimizer_settings': copy.deepcopy(stage.get('optimizer_settings', None)),
            'spec_settings': copy.deepcopy(stage.get('spec_settings', None)),
        }
        if opt.gradient is not None:
            data['gmax'] = np.max(np.abs(opt.gradient))

        self.history.append(data)



def _formalize_optimizer(opt):
    if isinstance(opt, str):
        opt = opt.lower()
        if opt in ['nm', 'nelder', 'neldermead']:
            OPT = NelderMead
        elif opt in ['bfgs']:
            OPT = BFGS
        else:
            raise ValueError('Unknown named optimizer. Accepted values are '
                             '"neldermead" or "bfgs".')
    elif issubclass(opt, Optimizer):
        OPT = opt   # allow customized optimizer derived from the base Optimizer class.
    else:
        raise TypeError('Unknown optimizer type.')

    return OPT
