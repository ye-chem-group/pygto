import sys
import numpy as np

from pygto.lib import StreamObject
from pygto.optimizer import ScheduledOptimizer


class TargetCostAtomicOptimization(StreamObject):

    def __init__(self, spec, cost_func, ftol=1e-3):

        self.spec = spec
        self.cost_func = cost_func
        self.ftol = ftol

        # attribute with default
        self.max_cycle = 5
        self.init_ftol_rescaling = 0.1
        self.opt_ftol_rescaling = 10.
        self.force_block_filter = True
        self.freeze_and_thaw = True
        self.verbose_optimizer = None

        # attribute set by kernel
        self.cost = None
        self.cost_init = None
        self.converged = False

    def dump_flags(self):
        self.log_info('\n')
        self.log_info('******** %s ********' % (self.__class__.__name__))
        self.log_info('ftol= %.10g' % self.ftol)
        self.log_info('max_cycle= %d' % self.max_cycle)
        self.log_info('init_ftol_rescaling= %.10g' % self.init_ftol_rescaling)
        self.log_info('opt_ftol_rescaling= %.10g' % self.opt_ftol_rescaling)
        self.log_info('force_block_filter= %s' % (str(self.force_block_filter)))
        self.log_info('freeze_and_thaw= %s' % (str(self.freeze_and_thaw)))
        self.log_info('verbose_optimizer= %s' % (str(self.verbose_optimizer)))
        self.log_info('')

    def filter_rigid(self, spec, ftol=None, force_block_filter=None):
        if ftol is None: ftol = self.ftol
        if force_block_filter is None: force_block_filter = self.force_block_filter

        structure_old = spec.structure

        cost_func = self.cost_func
        cost_init = cost_func(spec) if self.cost_init is None else self.cost_init

        self.log_debug('')
        self.log_debug('Find exponents to keep with ftol= %.3e' % ftol, indent=1)

        for channel_idx,channel in enumerate(spec.channels):
            self.log_debug('Channel= %d  l= %d' % (channel_idx, channel.l), indent=2)

            delta_fs = np.asarray([
                cost_func(spec1) - cost_init
                for spec1 in spec.remove_one_exponent_candidates_rigid(channel_idx)
            ])
            mask = delta_fs > ftol
            index = np.where(mask)[0]
            if len(index) == 0: # no exponents left
                raise RuntimeError('All exponents are below threshold in Channel= %d l= %d; '
                                   'refusing to remove entire channel.' %
                                   (channel_idx, channel.l))

            if force_block_filter:
                index = list(range(index.min(), index.max()+1))
                mask[index] = True

            for ie,(exponent,delta_f) in enumerate(zip(channel.exponents, delta_fs)):
                keep_s = {True: 'keep', False: 'discard'}[mask[ie]]
                self.log_debug(f'exponent= %.3e  delta_f= %.3e  status= %s' %
                               (exponent, delta_f, keep_s), indent=3)

            if len(index) == len(delta_fs):
                self.log_debug('No exponent to discard.', indent=3)
                continue

            self.log_debug('Keep %d exponents: %s' % (
                len(index), ', '.join([f'{x:.6g}' for x in channel.exponents[index]])), indent=3)

            spec.filter_channel_by_index_(channel_idx, index)

        nochange = structure_old == spec.structure
        if not nochange:
            self.log_debug('Basis structure change: %s -> %s' % (structure_old, spec.structure),
                          indent=1)
        self.log_debug('')

        return spec, nochange

    def filter_optimization(self, spec):
        structure_old = spec.structure

        cost_func = self.cost_func
        cost_init = self.cost_init
        ftol = self.ftol * self.opt_ftol_rescaling

        self.log_debug('')
        self.log_debug('Find exponents for removing tests with ftol= %.3e' % ftol, indent=1)

        for channel_idx,channel in enumerate(spec.channels):
            self.log_debug('Channel= %d  l= %d' % (channel_idx, channel.l), indent=2)

            delta_fs = np.asarray([
                cost_func(spec1) - cost_init
                for spec1 in spec.remove_one_exponent_candidates_rigid(channel_idx)
            ])
            idx_min = delta_fs.argmin()
            delta_f = delta_fs[idx_min]
            if delta_f < ftol:
                exponent_index = [i for i in range(channel.nexponent) if i != idx_min]
                spec1 = spec.filter_channel_by_index(channel_idx, exponent_index)
                if self.freeze_and_thaw:
                    cost, spec1 = self.optimize_candidate(spec1, active_channel=channel_idx)
                else:
                    cost, spec1 = self.optimize_candidate(spec1)
                delta_f = cost - cost_init
                if delta_f < self.ftol:
                    spec = spec1
                    status = 'accepted'
                else:
                    status = 'rejected'
                self.log_debug('Remove exponent= %.3e  cost= %.10f  delta_f= %.3e  status= %s' %
                              (channel.exponents[idx_min], cost, delta_f, status), indent=3)
            else:
                self.log_debug('No exponent to test removing.', indent=3)

        nochange = structure_old == spec.structure
        if not nochange:
            self.log_debug('Basis structure change: %s -> %s' % (structure_old, spec.structure),
                          indent=1)
        self.log_debug('')

        return spec, nochange

    def initialize(self):
        spec = self.spec.copy()
        spec = self.filter_rigid(spec, ftol=self.ftol*self.init_ftol_rescaling)[0]
        self.cost_init, spec = self.optimize_candidate(spec)
        self.cost = self.cost_init
        for i,c in enumerate(spec.channels):
            self.spec.replace_channel_(i, c)

    def kernel(self, **kwargs):
        self.set(**kwargs)

        self.dump_flags()
        self.initialize()
        self.print_init()

        spec = self.spec.copy()

        self.converged = False
        for cycle in range(1, self.max_cycle+1):

            spec = self.filter_rigid(spec)[0]
            self.cost, spec = self.optimize_candidate(spec)

            spec, nochange = self.filter_optimization(spec)
            if self.freeze_and_thaw:
                # Optimizing all channels together
                self.cost, spec = self.optimize_candidate(spec)
            else:
                self.cost = self.cost_func(spec)

            self.print_step(cycle, spec)

            if nochange:
                self.converged = True
                break

        for i,c in enumerate(spec.channels):
            self.spec.replace_channel_(i, c)

        self.print_final()

        return self.cost, self.spec

    def optimize_candidate(self, spec, active_channel=None, verbose=None):
        if verbose is None: verbose = self.verbose_optimizer
        if verbose is None: verbose = max(2, self.verbose-3)

        with spec.temporary_active_channel(active_channel):
            opt = ScheduledOptimizer(spec, self.cost_func).set(verbose=verbose)
            cost, spec = opt.kernel()

        return cost, spec

    def print_init(self):
        self.log_note('')
        self.log_note('Init basis:')
        if self.verbose >= 3:
            self.spec.dump_basis(stdout=self.stdout)
        self.log_note('Init cost= %.10f  structure= %s' % (self.cost_init, self.spec.structure),
                      space=True)

    def print_step(self, cycle, spec):
        self.log_info('TCAO cycle= %d  cost= %.10f  structure= %s' %
                      (cycle, self.cost, spec.structure))

    def print_final(self):
        self.log_note('Final cost= %.10f  delta_f= %.3e  structure= %s' %
                      (self.cost, self.cost-self.cost_init, self.spec.structure), space=True)
        self.log_note('Final basis:')
        if self.verbose >= 3:
            self.spec.dump_basis(stdout=self.stdout)
        self.log_note('')


TCAO = TargetCostAtomicOptimization


if __name__ == '__main__':
    from pygto.basis import BasisSpec
    from pygto.optimizer import ScheduledOptimizer
    from pygto.lib.pyscf_helper import get_cost_func
    from pyscf import gto, scf, cc
    from pyscf.data.elements import chemcore

    atm = 'Cl'
    spin = 1
    # fbas = 'cc-pvqz'
    fbas = 'def2-qzvp'
    ftol = 1e-3
    # pseudo = 'gth-hf-rev'
    pseudo = 'ccecp'
    # pseudo = None
    valence_l = [0,1]

    cost_func = get_cost_func(
        atm, scf.RHF, mol_settings={'spin': spin, 'pseudo': pseudo},
        keep_l=valence_l,
    )

    basis = gto.basis.load(fbas, atm)
    spec = BasisSpec.init_from_pyscf_basis(basis, atm=atm, keep_l=valence_l)

    opt = TCAO(spec, cost_func, ftol=ftol).set(verbose=5)
    opt.kernel()
