import sys
import copy
import numpy as np

from pygto.lib import StreamObject
from pygto.optimizer import ScheduledOptimizer, Optimizer


class MaterialConstraintAtomicOptimization(StreamObject):
    ''' Optimize an atomic basis under periodic linear-dependence constraints.

        Args:
            spec (BasisSpec):
                Initial atomic basis specification.
            stages (list of dict):
                Electronic-structure optimization stages. Each stage requires `prefix`
                and `cost_func`, and may define `penalty_rescale` and `active_l`.
            lindep_penalty_func (callable):
                Function with signature `(spec, scale) -> (penalty, condition_number)`.

        Attributes:
            penalty_strength (float):
                Weight of the linear-dependence penalty. Default is 0.01.
            max_cycle (int):
                Number of MCAO passes at each lattice scale. Default is 5.
            lattice_scaling_step_size (float):
                Decrement between successive lattice scales. Default is 0.03.
            lattice_scaling_target_penalty (float):
                Penalty used to determine the initial lattice scale. Default is 0.1.
            verbose_optimizer (int or None):
                Verbosity of stage optimizers. Default is None, which derives it from
                this object's verbosity.
            basis_to_save (str or None):
                Path for writing the current basis after each scale. Default is None.
            chkfile (str or None):
                Checkpoint path. Default is None, which disables checkpoint output.
    '''

    def __init__(self, spec, stages, lindep_penalty_func):
        self.spec = spec
        self.lindep_penalty_func = lindep_penalty_func

        # attributes with recommended default
        self.penalty_strength = 0.01
        self.max_cycle = 5
        self.lattice_scaling_step_size = 0.03
        self.lattice_scaling_target_penalty = 0.1
        self.verbose_optimizer = None

        self.basis_to_save = None

        self.chkfile = None

        # attributes set by kernel; do not modify
        self.cost = None
        self.penalty = None
        self.cond = None
        self._stages = None

        self.set_stages(stages)

    def dump_flags(self):
        ''' Log MCAO settings. '''
        self.log_info('\n')
        self.log_info('******** %s ********' % (self.__class__.__name__))
        self.log_info('penalty_strength= %.15g' % self.penalty_strength)
        self.log_info('max_cycle= %d' % self.max_cycle)
        self.log_info('lattice_scaling_step_size= %.15g' % self.lattice_scaling_step_size)
        self.log_info('lattice_scaling_target_penalty= %.15g' % self.lattice_scaling_target_penalty)
        self.log_info('verbose_optimizer= %s' % self.verbose_optimizer)
        self.log_info('basis_to_save= %s' % self.basis_to_save)
        self.log_info('chkfile= %s' % (str(self.chkfile)))
        self.log_info('')

    def initialize(self):
        ''' Evaluate initial stage costs and linear-dependence metrics. '''
        spec = self.spec.copy()
        self.cost = self.cost_func(spec)
        self.penalty, self.cond = self.lindep_penalty_func(spec)

    @property
    def stages(self):
        ''' Return MCAO stage configurations.

            Return:
                stages (list of dict):
                    Copied and formalized stage configurations.
        '''
        return self._stages

    @stages.setter
    def stages(self, value):
        ''' Validate and set MCAO stage configurations. '''
        self.set_stages(value)

    def set_stages(self, stages):
        ''' Validate, copy, and store MCAO stage configurations.

            Args:
                stages (list of dict):
                    Stages requiring `prefix` and `cost_func`. Missing
                    `penalty_rescale` and `active_l` default to 1 and None.
        '''
        must_have_keys = ['prefix', 'cost_func']
        optional_keys = ['penalty_rescale', 'active_l']
        stage_default = {'penalty_rescale': 1., 'active_l': None}

        stages = copy.deepcopy(stages)

        for stage in stages:
            if not isinstance(stage, dict):
                raise TypeError('stage must be dict.')
            if not all([k in stage for k in must_have_keys]):
                raise KeyError('stage must have the following keys: %s' %
                               (', '.join([f'"{k}"' for k in must_have_keys])))
            for k in optional_keys:
                if k not in stage:
                    stage[k] = stage_default[k]

        self._stages = stages

    def cost_func(self, spec, stages=None):
        ''' Evaluate all electronic-structure costs in selected stages.

            Args:
                spec (BasisSpec):
                    Basis specification to evaluate.
                stages (list of dict):
                    Stage configurations. Default is None, which uses `self.stages`.

            Return:
                costs (list of tuple):
                    `(prefix, cost)` pairs in stage order.
        '''
        if stages is None: stages = self.stages

        return [
            (stage['prefix'], stage['cost_func'](spec)) for stage in stages
        ]

    def format_result(self, cost=None, penalty=None, cond=None):
        ''' Format costs and linear-dependence metrics for logging.

            Args:
                cost (list of tuple):
                    Named costs. Default is None, which uses `self.cost`.
                penalty (float):
                    Linear-dependence penalty. Default is None, which uses
                    `self.penalty`.
                cond (float):
                    Overlap condition number. Default is None, which uses `self.cond`.

            Return:
                result (str):
                    Formatted metrics.
        '''
        if cost is None: cost = self.cost
        if penalty is None: penalty = self.penalty
        if cond is None: cond = self.cond
        sout = [f'{name}= {val:.9f}' for name,val in cost]
        sout += [f'penalty= {penalty:.3e}', f'cond= {cond:.3e}']
        return '  '.join(sout)

    def get_lattice_scaling_schedule(self, spec, scale0=None):
        ''' Construct a descending lattice-scaling schedule ending at one.

            Args:
                spec (BasisSpec):
                    Basis used to determine the initial scale.
                scale0 (float):
                    Initial scale. Default is None, which solves for the target penalty.

            Return:
                scales (ndarray):
                    Lattice scales ending at 1.
        '''
        if scale0 is not None and not isinstance(scale0, float):
            raise TypeError('scale0 must be float.')

        if scale0 is None:
            scale0 = solve_scale_for_penalty(spec, self.lindep_penalty_func,
                                             self.lattice_scaling_target_penalty)

        if np.isclose(scale0, 1.):
            return np.asarray([1.])
        elif scale0 > 1.:
            scales = np.arange(scale0, 1., -self.lattice_scaling_step_size)
            if not np.isclose(scales[-1], 1.):
                scales = np.hstack((scales, [1.]))
        else:
            raise ValueError('scale0 must be greater than 1.')

        return scales

    def kernel(self, **kwargs):
        ''' Run MCAO over the lattice-scaling schedule.

            Args:
                kwargs (dict):
                    Attribute overrides applied before execution.

            Return:
                cost (list of tuple):
                    Final named electronic-structure costs.
                penalty (float):
                    Final linear-dependence penalty.
                cond (float):
                    Final overlap condition number.
                spec (BasisSpec):
                    Optimized basis specification.
        '''
        self.set(**kwargs)

        self.dump_flags()
        self.initialize()
        self.print_init()

        spec = self.spec.copy()

        scales = self.get_lattice_scaling_schedule(spec)
        self.log_info('Lattice scaling schedule: %s' % (', '.join([f'{x:.3f}' for x in scales])))

        for scale in scales:
            self.penalty, self.cond = self.lindep_penalty_func(spec, scale)

            self.log_info('Enter MCAO cycle  scale= %.3f  %s' % (scale, self.format_result()))

            self.cost, self.penalty, self.cond, spec = self.kernel_mcao(spec, scale)

            self.log_info('Leaving MCAO cycle  scale= %.3f  %s' % (scale, self.format_result()))

            self.log_info('')
            self.log_info('Current basis:')
            if self.verbose >= 4:
                spec.dump_basis(stdout=self.stdout)
            self.log_info('')

            if self.basis_to_save is not None:
                with open(self.basis_to_save, 'w') as f:
                    spec.dump_basis(stdout=f)

            self.dump_chkfile(spec)

            if np.isclose(scale, 1.):
                break

        self.spec.parameters = spec.parameters

        self.print_final()

        return self.cost, self.penalty, self.cond, self.spec

    def kernel_mcao(self, spec, scale=1., stages=None, lindep_penalty_func=None,
                    penalty_strength=None, max_cycle=None, verbose_optimizer=None):
        ''' Perform MCAO at a fixed lattice scale.

            Args:
                spec (BasisSpec):
                    Basis specification to optimize.
                scale (float):
                    Lattice scale. Default is 1.
                stages (list of dict):
                    Stage configurations. Default is None, which uses `self.stages`.
                lindep_penalty_func (callable):
                    Penalty function. Default is None, which uses the configured
                    function.
                penalty_strength (float):
                    Penalty weight. Default is None, which uses `self.penalty_strength`.
                max_cycle (int):
                    Number of MCAO passes. Default is None, which uses `self.max_cycle`.
                verbose_optimizer (int):
                    Stage-optimizer verbosity. Default is None.

            Return:
                cost (list of tuple):
                    Named electronic-structure costs.
                penalty (float):
                    Linear-dependence penalty.
                cond (float):
                    Overlap condition number.
                spec (BasisSpec):
                    Optimized basis specification.
        '''
        if stages is None: stages = self.stages
        if penalty_strength is None: penalty_strength = self.penalty_strength
        if lindep_penalty_func is None: lindep_penalty_func = self.lindep_penalty_func
        if max_cycle is None: max_cycle = self.max_cycle
        if verbose_optimizer is None: verbose_optimizer = self.verbose_optimizer
        if verbose_optimizer is None: verbose_optimizer = max(2, self.verbose-3)

        lindep_penalty_func_fixed_scale = lambda x: lindep_penalty_func(x, scale)

        def penalty_func(spec, full_output=False):
            ''' Evaluate the fixed-scale penalty with optional condition number. '''
            penalty, cond = lindep_penalty_func_fixed_scale(spec)
            if full_output:
                return penalty, cond
            else:
                return penalty

        for cycle in range(1, max_cycle+1):

            for stage in stages:
                prefix = stage['prefix']
                cost_func = stage['cost_func']
                penalty_rescale = stage['penalty_rescale']
                active_l = stage['active_l']
                verbose = stage.get('verbose', verbose_optimizer)

                def cost_func_with_penalty(spec):
                    ''' Return one stage's cost plus its scaled penalty. '''
                    cost = cost_func(spec)
                    penalty = penalty_func(spec)
                    objective = cost + penalty * penalty_strength * penalty_rescale
                    return objective

                def format_cost(s):
                    ''' Format one optimizer's objective components for logging. '''
                    energy = cost_func(s.spec)
                    penalty, cond = penalty_func(s.spec, True)
                    return 'cost= %.10f  e= %.10f  penalty= %.3e  cond= %.3e' % (
                        s.cost, energy, penalty, cond
                    )

                with spec.temporary_active_l(active_l):
                    opt_stages = ScheduledOptimizer.get_preset_stages()
                    for opt_stage in opt_stages:
                        opt_stage['optimizer_settings'] = {
                            'format_cost': format_cost,
                        }
                    opt = ScheduledOptimizer(spec, cost_func_with_penalty,
                                             stages=opt_stages).set(verbose=verbose_optimizer)
                    opt.kernel()

                spec = opt.spec

            cost = [(stage['prefix'], stage['cost_func'](spec)) for stage in stages]
            penalty, cond = lindep_penalty_func_fixed_scale(spec)

            self.log_debug('MCAO cycle= %d  %s' % (cycle, self.format_result(cost, penalty, cond)),
                          indent=1)

        return cost, penalty, cond, spec

    def print_init(self):
        ''' Log the initial basis and metrics. '''
        self.log_note('')
        self.log_note('Init basis:')
        if self.verbose >= 3:
            self.spec.dump_basis(stdout=self.stdout)
        self.log_note('Init %s' % (self.format_result()), space=True)

    def print_step(self):
        ''' Placeholder hook for MCAO step logging. '''
        pass

    def print_final(self):
        ''' Log the final basis and metrics. '''
        self.log_note('Final %s' % (self.format_result()), space=True)
        self.log_note('Final basis:')
        if self.verbose >= 3:
            self.spec.dump_basis(stdout=self.stdout)
        self.log_note('')

    dump_chkfile = Optimizer.dump_chkfile

MCAO = MaterialConstraintAtomicOptimization


def solve_scale_for_penalty(spec, get_lindep_penalty, target_penalty, xtol=0.01, b=2.):
    ''' Solve for a lattice scale producing a target penalty.

        Args:
            spec (BasisSpec):
                Basis specification passed to the penalty function.
            get_lindep_penalty (callable):
                Function returning `(penalty, condition_number)` for `(spec, scale)`.
            target_penalty (float):
                Desired penalty.
            xtol (float):
                Scale tolerance for bisection. Default is 0.01.
            b (float):
                Initial upper bracket. Default is 2.

        Return:
            scale (float):
                Lattice scale, bounded below by 1.
    '''
    penalty = get_lindep_penalty(spec, 1.)[0]
    if penalty < target_penalty:
        return 1.

    from scipy.optimize import bisect
    func = lambda x: get_lindep_penalty(spec, x)[0] - target_penalty

    find_b = False
    for cycle in range(10):
        if func(b) < 0:
            find_b = True
            break
        b *= 1.5
    if not find_b:
        raise RuntimeError('Failed to bracket lattice scale for target penalty in [1, %.10g].' % b)

    scale, res = bisect(func, 1., b, xtol=xtol, full_output=True)
    if not res.converged:
        raise RuntimeError('Bisect does not converge. Reason= %s' % (res.flag))

    if abs(scale-1.) < xtol:
        return 1.

    return scale


if __name__ == '__main__':
    from pyscf import gto, scf, mp, cc
    from pygto.basis import BasisSpec
    from pygto.lib import pyscf_helper
    from pygto.lib import Lattice

    try:
        atm = sys.argv[1]
        spin = int(sys.argv[2])
        basis = sys.argv[3]
        pseudo = sys.argv[4]
        if pseudo.lower() == 'none': pseudo = None
        fvasp = sys.argv[5]
        kappa0 = float(sys.argv[6])
        try:
            basis_X = sys.argv[7]
        except:
            basis_X = None
    except:
        print('Usage: atm, spin, basis, pseudo, fvasp, kappa0, [basis_X=None]', flush=True)
        sys.exit(1)

    print('Input arguments:')
    print('atm= %s' % atm)
    print('spin= %d' % spin)
    print('basis= %s' % basis)
    print('pseudo= %s' % (str(pseudo)))
    print('fvasp= %s' % fvasp)
    print('kappa0= %.3e' % kappa0)
    print('basis_X= %s' % (str(basis_X)))
    print('\n', flush=True)

    basis = gto.basis.load(basis, atm)
    spec = BasisSpec.init_from_pyscf_basis(basis, atm=atm)

    lat = Lattice.init_from_vasp_poscar(fvasp)
    basis_full = {}
    for at in list(set(lat.atms)):
        if at == atm:
            basis_full[at] = basis
        else:
            basis_full[at] = basis_X
    cell = lat.get_pyscf_cell(basis=basis_full)

    from pyscf.data.elements import chemcore
    frozen = chemcore(gto.M(atom=atm, basis=basis, spin=None))

    stages = [
        # valence
        {
            'prefix': 'ehf',
            'cost_func': pyscf_helper.get_cost_func(
                atm, scf.RHF, mol_settings={'spin': spin}, keep_l=[0,1]
            ),
            'penalty_rescale': 1.,
            'active_l': [0,1],
        },
        # polarization
        {
            'prefix': 'ecorr',
            'cost_func': pyscf_helper.get_cost_func(
                atm, scf.RHF, mol_settings={'spin': spin},
                CORR=cc.CCSD, corr_settings={'frozen': frozen},
            ),
            'penalty_rescale': 0.1,
            'active_l': sorted(list(set([2,3,4,5]) & set(spec.angular_momenta))),
        },
    ]

    lindep_penalty_func = pyscf_helper.get_lindep_penalty_func(atm, cell, kappa0)
    mcao = MCAO(spec, stages, lindep_penalty_func).set(verbose=5)
    mcao.kernel()
