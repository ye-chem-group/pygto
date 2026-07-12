import sys
import numpy as np

from pygto.lib import StreamObject, to_int_list
from pygto.optimizer import ScheduledOptimizer, Optimizer


class TargetCostAtomicOptimization(StreamObject):
    ''' Optimizing basis to the target cost value.

        Starting from a sufficiently large initial basis, this module progressively reduce
        the size of the basis and monitor the cost value increase. It stops by identifying
        the smallest basis whose error (defined as the cost of that basis subtract the cost
        of the initial basis after optimization) lies within given tolerance.

        Input:
            spec (BasisSpec):
                Initial basis for reduction. We strongly recommend using a known, large
                basis, such as the all-electron cc-pVQZ set. The module is "smart" enough
                to trim the initial overcomplete basis.
            cost_func (Callable):
                cost_func(spec) -> cost.
            ftol (float):
                Tolerance for aborting the recursive basis reduction. Default is 1e-3 (Ha).

        Args:
            max_cycle (int):
                Maximum cycle for basis reduction. Each cycle will trim multiple primitives
                so this arg does not need to be very large for a reasonable input basis.
                Default is 10.
    '''

    def __init__(self, spec, cost_func, ftol=1e-3, verbose=None):

        self.spec = spec
        self.cost_func = cost_func
        self.ftol = ftol
        if verbose is not None: self.verbose = verbose

        # attribute with default
        self.max_cycle = 10
        self.init_ftol_rescaling = 0.1
        self.opt_ftol_rescaling = 10.
        self.ftol_exponent = None
        self.force_block_filter = True
        self.freeze_and_thaw = True
        self.verbose_optimizer = None

        self.chkfile = None

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
        if self.ftol_exponent is not None:
            self.log_info('ftol_exponent= %.10g' % self.ftol_exponent)
        self.log_info('force_block_filter= %s' % (str(self.force_block_filter)))
        self.log_info('freeze_and_thaw= %s' % (str(self.freeze_and_thaw)))
        self.log_info('verbose_optimizer= %s' % (str(self.verbose_optimizer)))
        self.log_info('chkfile= %s' % (str(self.chkfile)))
        self.log_info('')

    def filter_rigid(self, spec, ftol=None, ftol_exponent=None, force_block_filter=None):
        if ftol is None: ftol = self.ftol
        if ftol_exponent is None: ftol_exponent = self.ftol_exponent
        if ftol_exponent is not None: ftol = min(ftol, ftol_exponent)
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

    def filter_optimization(self, spec, cost_func=None, cost_init=None, ftol=None,
                            ftol_rescaling=None, freeze_and_thaw=None, select_channel=None,
                            force_accept=False):
        structure_old = spec.structure

        if cost_func is None: cost_func = self.cost_func
        if cost_init is None: cost_init = self.cost_init
        if ftol is None: ftol = self.ftol
        if ftol_rescaling is None: ftol_rescaling = self.opt_ftol_rescaling
        ftol_filter = ftol * ftol_rescaling
        if freeze_and_thaw is None: freeze_and_thaw = self.freeze_and_thaw

        if select_channel is None:
            select_channel = range(spec.nchannel)
        else:
            select_channel = to_int_list(select_channel)

        self.log_debug('')
        if force_accept:
            self.log_debug('Find exponents for removing tests', indent=1)
        else:
            self.log_debug('Find exponents for removing tests with ftol= %.3e' % ftol, indent=1)

        for channel_idx,channel in enumerate(spec.channels):
            if not channel_idx in select_channel:
                continue

            self.log_debug('Channel= %d  l= %d' % (channel_idx, channel.l), indent=2)

            delta_fs = np.asarray([
                cost_func(spec1) - cost_init
                for spec1 in spec.remove_one_exponent_candidates_rigid(channel_idx)
            ])
            idx_min = delta_fs.argmin()
            delta_f = delta_fs[idx_min]
            if force_accept or delta_f < ftol_filter:
                exponent_index = [i for i in range(channel.nexponent) if i != idx_min]
                spec1 = spec.filter_channel_by_index(channel_idx, exponent_index)
                if freeze_and_thaw:
                    cost, spec1 = self.optimize_candidate(spec1, cost_func=cost_func,
                                                          active_channel=channel_idx)
                else:
                    cost, spec1 = self.optimize_candidate(spec1, cost_func=cost_func)
                delta_f = cost - cost_init
                if force_accept:
                    spec = spec1
                    status = 'forced-accepted'
                elif delta_f < ftol:
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
        ftol_init = self.ftol*self.init_ftol_rescaling
        spec = self.filter_rigid(spec, ftol=ftol_init)[0]
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
            self.dump_chkfile(spec)

            if nochange:
                self.converged = True
                break

        for i,c in enumerate(spec.channels):
            self.spec.replace_channel_(i, c)

        self.print_final()

        return self.cost, self.spec

    def optimize_candidate(self, spec, cost_func=None, active_channel=None, verbose=None):
        if cost_func is None: cost_func = self.cost_func
        if verbose is None: verbose = self.verbose_optimizer
        if verbose is None: verbose = max(2, self.verbose-3)

        with spec.temporary_active_channel(active_channel):
            opt = ScheduledOptimizer(spec, cost_func).set(verbose=verbose)
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
        self.log_info('TCAO cycle= %d  cost= %.10f  delta_f= %.3e  structure= %s' %
                      (cycle, self.cost, self.cost-self.cost_init, spec.structure))

    def print_final(self):
        self.log_note('Final cost= %.10f  delta_f= %.3e  structure= %s' %
                      (self.cost, self.cost-self.cost_init, self.spec.structure), space=True)
        self.log_note('Final basis:')
        if self.verbose >= 3:
            self.spec.dump_basis(stdout=self.stdout)
        self.log_note('')

    dump_chkfile = Optimizer.dump_chkfile


TCAO = TargetCostAtomicOptimization


class ChannelReduction(TCAO):
    ''' Progressively reduction of channel size by cost criterion.

        Given the initial basis, each channel will be progressively compressed by removing
        an exponent followed by reoptimization of that channel (with other channels kept
        frozen). This process recursively generates a series of channels of decreasing size
        and increasing cost valuee. The process aborts when the error of the current channel,
        defined as the current cost subtract the initial cost, exceeds a given tolerance.

        Input:
            spec (BasisSpec):
                Initial basis for reduction. We strongly recommend using initial basis
                generated by the TargetCostAtomicOptimization (TCAO) with a reasonably
                low `ftol`, which gaurantees a reasonable starting point.
            cost_func (Callable):
                cost_func(spec) -> cost.
            ftol (float):
                Tolerance for aborting the recursive channel reduction. Default is 1e-2 (Ha).

        Args:
            channel_idxs (int or list of int):
                Specify the channels to perform reduction. Default is None, which means
                all channels in `spec` will be reduced.
            chkfile (str):
                The file where initial and optimized basis will be saved.
                - "spec": input BasisSpec
                - "channel_[idx]/nbas_[n]": channel series generated by reduction
                Default is None, meaning that chkfile is not written.
            outdir (str):
                Directory where basis data and summary will be written. For each channel,
                a subdir "[outdir]/channel_[idx]" will be created. Inside this subdir, one
                will find:
                - "summary.csv": a CSV file summarizes the structure, cost, and delta_f
                  for that channel.
                - "[n].dat": basis set data for that channel of `n` primitives.
    '''

    def __init__(self, spec, cost_func, ftol=1e-2, verbose=None):
        TCAO.__init__(self, spec, cost_func, ftol, verbose)

        self.channel_idxs = None    # default: performing reduction for all channels
        self.outdir = None

        self.stop_reason = None
        self.results = None

    def dump_flags(self):
        TCAO.dump_flags(self)
        self.log_info('channel_idxs= %s' % (str(self.channel_idxs)))
        self.log_info('outdir= %s' % (str(self.outdir)))
        self.log_info('')

    def initialize(self):
        self.converged = []
        self.stop_reason = []
        self.results = []
        self.cost_init = self.cost = self.cost_func(self.spec)

    def kernel(self, **kwargs):
        self.set(**kwargs)

        self.initialize()
        self.dump_flags()
        self.print_init()

        channel_idxs = self.channel_idxs
        if channel_idxs is None:
            channel_idxs = range(self.spec.nchannel)
        else:
            channel_idxs = to_int_list(channel_idxs)

        if self.chkfile is not None:
            spec.dump_chkfile(self.chkfile, prefix='spec')

        for channel_idx in channel_idxs:
            spec = self.spec.copy()
            spec._check_channel_idx(channel_idx)

            cost_init = self.cost_func(spec)

            self.log_note('')
            self.log_note('Series generation for Channel= %d  l= %d' % (
                channel_idx, spec.channels[channel_idx].l
            ))
            self.log_note('Init channel structure= %s  cost= %.10f' % (
                spec.channels[channel_idx].structure, cost_init
            ))

            results = [ (spec.channels[channel_idx].copy(), cost_init, 0.) ]
            self.dump_chkfile(spec.channels[channel_idx], channel_idx, first_pass=True)

            converged = False
            stop_reason = 'Only one exponent left'
            while spec.channels[channel_idx].nbas > 1:
                # use a larger ftol to ensure filtering success
                spec = self.filter_optimization(
                    spec, cost_init=cost_init,
                    force_accept=True, select_channel=channel_idx,
                )[0]
                cost = self.cost_func(spec)
                df = cost - cost_init
                results.append( (spec.channels[channel_idx].copy(), cost, df) )

                self.log_info('Current channel structure= %s  cost= %.10f  delta_f= %.3e' % (
                    spec.channels[channel_idx].structure, cost, df
                ))
                self.log_info('Current channel:')
                if self.verbose >= 4:
                    spec.dump_channel_basis(channel_idx)
                    self.log_info('')

                self.dump_chkfile(spec.channels[channel_idx], channel_idx)

                if df > self.ftol:
                    converged = True
                    stop_reason = 'ftol met'
                    break

            if converged:
                self.log_note('Convergence is reached for Channel= %d  l= %d' % (
                    channel_idx, spec.channels[channel_idx].l
                ))
            else:
                self.log_note('Convergence is not reached for Channel= %d  l= %d: %s' % (
                    channel_idx, spec.channels[channel_idx].l, stop_reason
                ))
            self.log_note('Series generated for Channel= %d  l= %d:' % (
                channel_idx, spec.channels[channel_idx].l
            ))
            for channel, cost, df in results:
                self.log_note('structure= %s  cost= %.10f  delta_f= %.3e' % (
                    channel.structure, cost, df
                ))
            self.log_note('')

            self.dump_summary(channel_idx, results)

            self.converged.append( (channel_idx, converged) )
            self.stop_reason.append( (channel_idx, stop_reason) )
            self.results.append( (channel_idx, results) )

        return self.results

    def print_init(self):
        self.log_note('Init basis structure: %s  cost= %.10f' % (
            self.spec.structure, self.cost_init
        ))
        self.log_info('Init basis:')
        if self.verbose >= 4:
            self.spec.dump_basis()
        self.log_info('')

    def dump_chkfile(self, channel, channel_idx, chkfile=None, first_pass=False):
        if chkfile is None: chkfile = self.chkfile
        if chkfile is not None:
            if first_pass:
                # remove all existing results
                import h5py
                with h5py.File(chkfile, 'a') as f:
                    key = f'channel_{channel_idx}'
                    if key in f:
                        del f[key]
            prefix = f'channel_{channel_idx}/nbas_{channel.nbas}'
            channel.dump_chkfile(chkfile, prefix=prefix)

    def dump_summary(self, channel_idx, results, outdir=None):
        ''' Save channel summary to disk.

            For each channel, a directory "[outdir]/channel_[idx]" is created.
                - A "summary.csv" file will be created in this directory that summarizes
                  the structure, cost and df of the series generated for this channel.
                - A series of "[nbas].dat" files will be generated, which are NWChem
                  format basis files for the series generated for this channel.

            Note:
                - If `outdir` is None, outdir will be set to self.oudir. If still None,
                  "summary" will be used as default.
                - If "[outdir]/channel_[idx]" already contains "summary.csv" and "*.dat"
                  files, they will be removed. Other files will be retained.
        '''
        import os
        from pygto.lib import mkdir

        if outdir is None: outdir = self.outdir
        if outdir is None: outdir = 'summary'
        subdir = f'{outdir}/channel_{channel_idx}'
        # remove existng files in subdir
        if os.path.isdir(subdir):
            for fname in os.listdir(subdir):
                if fname == 'summary.csv' or fname.endswith('.dat'):
                    path = os.path.join(subdir, fname)
                    if os.path.isfile(path) or os.path.islink(path):
                        os.remove(path)
        mkdir(subdir, recursive=True)

        # write summary to "summary.csv" and channel basis data
        fconv = f'{subdir}/summary.csv'
        sout = ['structure,cost,delta_f']
        for channel, cost, df in results:
            fbas = f'{subdir}/{channel.nbas}.dat'
            with open(fbas, 'w') as fb:
                channel.dump_basis(atm=self.spec.atm, stdout=fb)
            sout.append('%s,%.10f,%.3e' % (channel.structure, cost, df))
        sout = '\n'.join(sout)
        open(fconv, 'w').write(sout+'\n')


if __name__ == '__main__':
    from pygto.basis import BasisSpec
    from pygto.optimizer import ScheduledOptimizer
    from pygto.lib.pyscf_helper import get_cost_func
    from pyscf import gto, scf, cc
    from pyscf.data.elements import chemcore

    atm = 'C'
    spin = 2
    fbas = 'cc-pvqz'
    ftol = 1e-3
    pseudo = 'gth-hf-rev'
    valence_l = [0,1]

    cost_func = get_cost_func(
        atm, scf.RHF, mol_settings={'spin': spin, 'pseudo': pseudo},
        keep_l=valence_l,
    )

    basis = gto.basis.load(fbas, atm)
    spec = BasisSpec.init_from_pyscf_basis(basis, atm=atm, keep_l=valence_l)

    opt = TCAO(spec, cost_func, ftol=ftol).set(verbose=5)
    opt.kernel()
