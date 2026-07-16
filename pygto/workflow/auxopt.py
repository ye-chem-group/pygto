import numpy as np

from pygto.lib import StreamObject
from pygto.optimizer import ScheduledOptimizer
from pygto.workflow import TCAO


class AuxiliaryBasisOptimization(TCAO):

    def __init__(self, spec, cost_func, ftol=1e-5, verbose=None):
        TCAO.__init__(self, spec, cost_func, ftol, verbose)

        self.init_ftol_rescaling = 0.5

    def cost_func_full(self, spec):
        return self.cost_func(spec, True)

    def initialize(self):
        spec = self.spec.copy()
        self.cost_init, spec = self.optimize_candidate(spec)
        self.cost_vec = self.cost_func_full(spec)[1]
        self.cost = self.cost_init

        ftol = self.ftol * self.init_ftol_rescaling

        if self.cost_init > ftol:
            self.log_info('Enter AuxBasExpansion cycle cost= %.3e  structure= %s' % (
                self.cost, spec.structure), indent=1)
            self.log_debug('costvec= %s' % (
                ' '.join(['%.3e'%x for x in self.cost_vec])), indent=2)
            stages = (
                (1, False),
                (2, False),
                (3, False),
                (1, True),
                (2, True),
                (3, True),
            )
            found = False
            for repeat, add_high_l in stages:
                spec1 = increase_basis_size(spec, repeat, add_high_l)
                cost, spec1 = self.optimize_candidate(spec1)
                cost_vec = self.cost_func_full(spec1)[1]

                self.log_info('repeat= %d  add_high_l= %s  cost= %.3e  structure= %s' % (
                    repeat, str(add_high_l), cost, spec1.structure), indent=2)
                self.log_debug('costvec= %s' % (
                    ' '.join(['%.3e'%x for x in cost_vec])), indent=3)

                if cost < ftol:
                    found = True
                    spec = spec1
                    break

            self.log_info('Leaving AuxBasExpansion cycle cost= %.3e  structure= %s' % (
                cost, spec.structure), indent=1)
            self.log_info('costvec= %s' % (
                ' '.join(['%.3e'%x for x in cost_vec])), indent=2)
            self.log_info('')

            if not found:
                self.log_error('Failed to generate an init basis with the desired ftol %.3e' % (
                    ftol))
                raise RuntimeError

            self.cost = cost
            self.cost_vec = cost_vec

        self.spec.channels = spec.channels

    def filter_rigid(self, spec, ftol=None, select_channel=None, cost_init=None):
        if cost_init is None: cost_init = 0
        cost, spec, nochange = TCAO.filter_rigid(self, spec, ftol, select_channel, cost_init)
        cost_vec = self.cost_func_full(spec)[1]
        return cost, cost_vec, spec, nochange

    def filter_optimization(self, spec, select_channel=None, cost_init=None, force_accept=False):
        if cost_init is None: cost_init = 0
        cost, spec, nochange = TCAO.filter_optimization(self, spec, select_channel, cost_init,
                                                        force_accept)
        cost_vec = self.cost_func_full(spec)[1]
        return cost, cost_vec, spec, nochange

    def optimize_candidate(self, spec, cost_func=None, active_channel=None, verbose=None):
        if cost_func is None: cost_func = self.cost_func
        if verbose is None: verbose = self.verbose_optimizer
        if verbose is None: verbose = max(2, self.verbose-3)

        stages = [
            {
                'optimizer': 'NelderMead', 'optimizer_settings': {'accuracy': 'low'},
            },
            {
                'optimizer': 'NelderMead', 'optimizer_settings': {'accuracy': 'medium'},
            },
            {
                'optimizer': 'NelderMead', 'optimizer_settings': {'accuracy': 'high'},
            },
        ]

        with spec.temporary_active_channel(active_channel):
            if spec.nparam == 0:    # The selected channels contain nothing to optimize.
                return float(cost_func(spec)), spec

            opt = ScheduledOptimizer(spec, cost_func, stages=stages).set(verbose=verbose)
            cost, spec = opt.kernel()

        return cost, spec

    def kernel(self, **kwargs):
        self.set(**kwargs)

        self.dump_flags()
        self.initialize()
        self.print_init()

        spec = self.spec.copy()

        self.converged = False
        for cycle in range(1, self.max_cycle+1):

            self.cost, self.cost_vec, spec, nochange = self.filter_optimization(spec)

            self.print_step(cycle, spec)
            self.dump_chkfile(spec)

            if nochange:
                self.converged = True
                break

        for i,c in enumerate(spec.channels):
            self.spec.replace_channel_(i, c)

        self.print_final()

        return self.cost, self.cost_vec, self.spec

    def print_init(self):
        self.log_note('Init basis:')
        if self.verbose >= 3:
            self.spec.dump_basis(stdout=self.stdout)
        self.log_note('Init cost= %.3e  structure= %s' % (self.cost_init, self.spec.structure))
        self.log_note('costvec= %s' % (' '.join(['%.3e'%x for x in self.cost_vec])), indent=1)
        self.log_debug('')

    def print_step(self, cycle, spec):
        self.log_info('AuxOpt cycle= %d  cost= %.3e  structure= %s' % (
            cycle, self.cost, spec.structure))
        self.log_info('costvec= %s' % (' '.join(['%.3e'%x for x in self.cost_vec])), indent=1)
        self.log_debug('')

    def print_final(self):
        self.log_note('Final cost= %.3e  structure= %s' % (
            self.cost, self.spec.structure))
        self.log_note('costvec= %s' % (' '.join(['%.3e'%x for x in self.cost_vec])), indent=1)
        self.log_note('Final basis:')
        if self.verbose >= 3:
            self.spec.dump_basis(stdout=self.stdout)
        self.log_note('')


def increase_basis_size(spec, repeat, add_high_l):
    if len(spec.angular_momenta) != spec.nchannel:
        raise NotImplementedError

    if abs(np.asarray(spec.angular_momenta) - np.arange(spec.nchannel)).max() > 0.1:
        raise NotImplementedError

    if repeat == 0 and not add_high_l:
        return spec

    channels = []
    for channel in spec.channels:
        c = channel.copy()
        for i in range(repeat):
            c = c.add_one_exponent_candidates()[1]
        channels.append(c)

    if add_high_l:
        from pygto.basis import ETB
        l = max(spec.angular_momenta)+1
        c = ETB(l, [])
        for i in range(repeat):
            c = c.add_one_exponent_candidates()[1]
        channels.append(c)

    return spec.with_channels(channels)


AuxOpt = AuxiliaryBasisOptimization


if __name__ == '__main__':
    from pyscf import gto, scf, df
    from pygto.basis import BasisSpec
    from pygto.optimizer import ScheduledOptimizer
    from pygto.lib import pyscf_helper

    atm = 'C'
    spin = 2
    frozen = 1
    val_l = [0,1]
    aobasis = 'cc-pvdz'

    mol = gto.M(atom=atm, basis=aobasis, spin=spin)
    auxbasis = df.autoaux(mol)[atm]
    auxspec = BasisSpec.init_from_pyscf_basis(auxbasis, atm=atm, channel_type='etb')

    cost_func = pyscf_helper.get_cost_func_auxopt(
        atm, aobasis, scf.UHF, mol_settings={'spin':spin},
        corr_settings={'frozen':frozen},
    )

    opt = AuxOpt(auxspec, cost_func, ftol=1e-5).set(verbose=4)
    opt.kernel()
