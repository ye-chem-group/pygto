import numpy as np

from pygto import lib
from pygto.optimizer import ScheduledOptimizer, Optimizer


class TargetCostAtomicOptimization(lib.StreamObject):
    ''' Reduce a basis while keeping its cost increase below a target tolerance.

        Starting from a sufficiently large basis, this workflow progressively removes
        primitives and reoptimizes exponents. The error is the cost increase relative
        to the optimized initial basis.

        Args:
            spec (BasisSpec):
                Initial basis specification for reduction.
            cost_func (callable):
                Function accepting a BasisSpec and returning a scalar cost.
            ftol (float):
                Maximum accepted cost increase. Default is `1e-3` Hartree.
            verbose (int):
                Logging verbosity. Default is None.

        Attributes:
            max_cycle (int):
                Maximum basis-reduction cycles. Default is 10.
            init_ftol_rescaling (float):
                Factor applied to `ftol` during initial rigid filtering. Default is 0.1.
            force_block_filter (bool):
                Whether retained rigid candidates must form a contiguous exponent block.
                Default is True.
            freeze_and_thaw (bool):
                Whether to optimize a reduced channel before jointly relaxing selected
                channels. Default is True.
            candidate_generation (str):
                Candidate strategy, "heuristic" or "rigid". Default is "heuristic".
            candidate_maxnum (int):
                Maximum number of rigid candidates optimized per channel. Default is 2.
            verbose_optimizer (int or None):
                Candidate-optimizer verbosity. Default is None, which derives it from
                this object's verbosity.
            chkfile (str or None):
                Checkpoint path. Default is None, which disables checkpoint output.
    '''

    def __init__(self, spec, cost_func, ftol=1e-3, verbose=None):

        self.spec = spec
        self.cost_func = cost_func
        self.ftol = ftol
        if verbose is not None: self.verbose = verbose

        # attribute with default
        self.max_cycle = 10
        self.init_ftol_rescaling = 0.1
        self.force_block_filter = True
        self.freeze_and_thaw = True
        self.candidate_generation = 'heuristic'  # Options: heuristic, rigid
        self.candidate_maxnum = 2
        self.verbose_optimizer = None

        self.chkfile = None

        # attribute set by kernel
        self.cost = None
        self.cost_init = None
        self.converged = False

    def dump_flags(self):
        ''' Log TCAO settings. '''
        self.log_info('\n')
        self.log_info('******** %s ********' % (self.__class__.__name__))
        self.log_info('ftol= %.10g' % self.ftol)
        self.log_info('max_cycle= %d' % self.max_cycle)
        self.log_info('init_ftol_rescaling= %.10g' % self.init_ftol_rescaling)
        self.log_info('force_block_filter= %s' % (str(self.force_block_filter)))
        self.log_info('freeze_and_thaw= %s' % (str(self.freeze_and_thaw)))
        self.log_info('candidate_generation= %s' % (self.candidate_generation))
        self.log_info('candidate_maxnum= %d' % (self.candidate_maxnum))
        self.log_info('verbose_optimizer= %s' % (str(self.verbose_optimizer)))
        self.log_info('chkfile= %s' % (str(self.chkfile)))
        self.log_info('')

    def filter_rigid(self, spec, ftol=None, select_channel=None, cost_init=None):
        ''' Remove individually safe exponents before reoptimizing selected channels.

            Args:
                spec (BasisSpec):
                    Basis specification to filter.
                ftol (float):
                    Maximum accepted cost increase. Default is None, which uses
                    `self.ftol`.
                select_channel (int or list of int):
                    Channels eligible for filtering. Default is None, which selects all.
                cost_init (float):
                    Reference cost. Default is None, which uses the stored initial cost
                    when available.

            Return:
                cost (float):
                    Cost after filtering and optional reoptimization.
                spec (BasisSpec):
                    Filtered basis specification.
                nochange (bool):
                    Whether no exponent was removed.
        '''
        if ftol is None: ftol = self.ftol

        if select_channel is None:
            select_channel = list(range(spec.nchannel))
        else:
            select_channel = lib.to_int_list(select_channel)

        cost_func = self.cost_func
        cost = cost_func(spec)
        if cost_init is None:
            cost_init = cost if self.cost_init is None else self.cost_init

        self.log_debug('Enter RigidTrim cycle  structure= %s  cost= %.10f  delta_f= %.3e' % (
            spec.structure, cost, cost-cost_init), indent=1)

        nochange = True
        for channel_idx,channel in enumerate(spec.channels):
            if not channel_idx in select_channel:
                continue

            self.log_debug('Channel= %d  l= %d' % (channel_idx, channel.l), indent=2)

            delta_fs = np.asarray([
                cost_func(spec1) - cost_init
                for spec1 in spec.remove_one_exponent_candidates_rigid(channel_idx)
            ])
            mask = delta_fs > ftol
            index = np.where(mask)[0]

            if self.force_block_filter and len(index) > 0:
                index = list(range(index.min(), index.max()+1))
                mask[index] = True

            for ie,(exponent,delta_f) in enumerate(zip(channel.exponents, delta_fs)):
                keep_s = {True: 'keep', False: 'discard'}[mask[ie]]
                self.log_debug(f'exponent= %.3e  delta_f= %.3e  status= %s' %
                               (exponent, delta_f, keep_s), indent=3)

            if len(index) == len(delta_fs):
                self.log_debug('No exponent to discard.', indent=3)
                continue

            if len(index) > 0:
                self.log_debug('Keep %d exponents: %s' % (
                    len(index), ', '.join([f'{x:.6g}' for x in channel.exponents[index]])
                ), indent=3)
            else:
                self.log_debug('All exponents are discarded.', indent=3)

            spec.filter_channel_by_index_(channel_idx, index)
            nochange = False

        if not nochange:
            cost, spec = self.optimize_candidate(spec, active_channel=select_channel)

        self.log_debug('Leaving RigidTrim cycle  structure= %s  cost= %.10f  delta_f= %.3e' % (
            spec.structure, cost, cost-cost_init), indent=1)
        self.log_debug('')

        return cost, spec, nochange

    def _candidates_by_heuristic(self, spec, channel_idx):
        ''' Generate rescaled one-exponent-removal candidates for a channel. '''
        return spec.remove_one_exponent_candidates(channel_idx)

    def _candidates_generation_rigid(self, spec, channel_idx):
        ''' Return the lowest-cost rigid removal candidates for a channel. '''
        ncandidate = max(1,self.candidate_maxnum)
        candidates = spec.remove_one_exponent_candidates_rigid(channel_idx)
        fs = [self.cost_func(spec1) for spec1 in candidates]
        return [candidates[i] for i in np.argsort(fs)[:ncandidate]]

    def filter_optimization(self, spec, select_channel=None, cost_init=None, force_accept=False):
        ''' Select optimized one-exponent-removal candidates channel by channel.

            Args:
                spec (BasisSpec):
                    Basis specification to reduce.
                select_channel (int or list of int):
                    Channels eligible for reduction. Default is None, which selects all.
                cost_init (float):
                    Reference cost. Default is None, which uses the stored initial cost
                    when available.
                force_accept (bool):
                    Whether to accept the lowest-cost candidate regardless of `ftol`.
                    Default is False.

            Return:
                cost (float):
                    Cost of the resulting basis.
                spec (BasisSpec):
                    Reduced and optimized basis specification.
                nochange (bool):
                    Whether no candidate was accepted.
        '''
        if select_channel is None:
            select_channel = list(range(spec.nchannel))
        else:
            select_channel = lib.to_int_list(select_channel)

        cost = self.cost_func(spec)
        if cost_init is None:
            cost_init = cost if self.cost_init is None else self.cost_init
        self.log_debug('Enter OptTrim cycle  structure= %s  cost= %.10f  delta_f= %.3e' % (
            spec.structure, cost, cost-cost_init), indent=1)

        nochange = True
        for channel_idx,channel in enumerate(spec.channels):
            if not channel_idx in select_channel:
                continue

            self.log_debug('Channel= %d  l= %d' % (channel_idx, channel.l), indent=2)

            if channel.nbas == 0:
                self.log_debug('Skip empty channel.', indent=3)
                continue

            if self.candidate_generation.lower().startswith('heur'):
                candidates = self._candidates_by_heuristic(spec, channel_idx)
            elif self.candidate_generation.lower().startswith('rig'):
                candidates = self._candidates_generation_rigid(spec, channel_idx)
            else:
                raise ValueError('Unknown candidate generation method %s' % (
                    str(self.candidate_generation)))

            idx_min = None
            cost_min = float('inf')
            for idx_cand,spec1 in enumerate(candidates):
                if self.freeze_and_thaw:
                    cost1, spec1 = self.optimize_candidate(spec1, active_channel=channel_idx)
                else:
                    cost1, spec1 = self.optimize_candidate(spec1)
                delta_f1 = cost1 - cost_init
                status = 'acceptable' if delta_f1 < self.ftol else 'rejected'
                self.log_debug('Candidate #%d  cost= %.10f  delta_f= %.3e  status= %s' % (
                    idx_cand+1, cost1, delta_f1, status), indent=3)
                self.log_debug('Exponents= %s' % (
                    ', '.join([f'{x:.4g}' for x in spec1.channels[channel_idx].exponents])
                ), indent=4)
                if cost1 < cost_min:
                    idx_min = idx_cand
                    cost_min = cost1
                    spec_min = spec1

            delta_f_min = cost_min - cost_init
            if force_accept or delta_f_min < self.ftol:
                nochange = False
                spec = spec_min
                cost = cost_min
                self.log_debug('Accepting candidate #%d  cost= %.10f  delta_f= %.3e' % (
                    idx_min+1, cost, cost-cost_init
                ), indent=3)
            else:
                self.log_debug('No candidate is accepted', indent=3)

        if not nochange and self.freeze_and_thaw:
            cost, spec = self.optimize_candidate(spec, active_channel=select_channel)

        self.log_debug('Leaving OptTrim cycle  structure= %s  cost= %.10f  delta_f= %.3e' % (
            spec.structure, cost, cost-cost_init), indent=1)
        self.log_debug('')

        return cost, spec, nochange

    def initialize(self):
        ''' Rigidly filter and optimize the initial basis. '''
        spec = self.spec.copy()
        ftol_init = self.ftol*self.init_ftol_rescaling
        self.cost_init, spec, nochange = self.filter_rigid(spec, ftol=ftol_init)
        if nochange:    # force optimization if not done in `filter_rigid`
            self.cost_init, spec = self.optimize_candidate(spec)
        self.cost = self.cost_init
        for i,c in enumerate(spec.channels):
            self.spec.replace_channel_(i, c)

    def kernel(self, **kwargs):
        ''' Run target-cost basis reduction.

            Args:
                kwargs (dict):
                    Attribute overrides applied before execution.

            Return:
                cost (float):
                    Final scalar cost.
                spec (BasisSpec):
                    Reduced and optimized basis specification.
        '''
        self.set(**kwargs)

        self.dump_flags()
        self.initialize()
        self.print_init()

        spec = self.spec.copy()

        self.converged = False
        for cycle in range(1, self.max_cycle+1):

            # step 1: a rapid rigid filter by exponentwise error
            self.cost, spec, nochange = self.filter_rigid(spec)

            # step 2: a dedicated optimization-based filter
            self.cost, spec, nochange = self.filter_optimization(spec)

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
        ''' Optimize a candidate basis with ScheduledOptimizer.

            Args:
                spec (BasisSpec):
                    Candidate basis specification.
                cost_func (callable):
                    Cost function. Default is None, which uses `self.cost_func`.
                active_channel (int or list of int):
                    Channels to optimize. Default is None, which activates all.
                verbose (int):
                    Optimizer verbosity. Default is None, which derives it from this
                    object's verbosity.

            Return:
                cost (float):
                    Optimized candidate cost.
                spec (BasisSpec):
                    Optimized candidate basis.
        '''
        if cost_func is None: cost_func = self.cost_func
        if verbose is None: verbose = self.verbose_optimizer
        if verbose is None: verbose = max(2, self.verbose-3)

        with spec.temporary_active_channel(active_channel):
            if spec.nparam == 0:    # The selected channels contain nothing to optimize.
                return float(cost_func(spec)), spec

            opt = ScheduledOptimizer(spec, cost_func).set(verbose=verbose)
            cost, spec = opt.kernel()

        return cost, spec

    def print_init(self):
        ''' Log the optimized initial basis and reference cost. '''
        self.log_note('Init basis:')
        if self.verbose >= 3:
            self.spec.dump_basis(stdout=self.stdout)
        self.log_note('Init cost= %.10f  structure= %s' % (self.cost_init, self.spec.structure),
                      space=True)
        self.log_debug('')

    def print_step(self, cycle, spec):
        ''' Log one TCAO reduction cycle.

            Args:
                cycle (int):
                    One-based cycle index.
                spec (BasisSpec):
                    Current basis specification.
        '''
        self.log_info('TCAO cycle= %d  cost= %.10f  delta_f= %.3e  structure= %s' %
                      (cycle, self.cost, self.cost-self.cost_init, spec.structure))
        self.log_debug('')

    def print_final(self):
        ''' Log the final reduced basis and cost increase. '''
        self.log_note('Final cost= %.10f  delta_f= %.3e  structure= %s' %
                      (self.cost, self.cost-self.cost_init, self.spec.structure), space=True)
        self.log_note('Final basis:')
        if self.verbose >= 3:
            self.spec.dump_basis(stdout=self.stdout)
        self.log_note('')

    dump_chkfile = Optimizer.dump_chkfile


TCAO = TargetCostAtomicOptimization


class ChannelReduction(TCAO):
    ''' Generate progressively smaller channels ordered by increasing cost.

        Each selected channel is compressed by removing one exponent and reoptimizing
        that channel while keeping other channels frozen. Reduction stops after the cost
        increase exceeds `ftol` or only one exponent remains.

        Args:
            spec (BasisSpec):
                Initial basis specification.
            cost_func (callable):
                Function accepting a BasisSpec and returning a scalar cost.
            ftol (float):
                Cost-increase stopping tolerance. Default is `1e-2` Hartree.
            verbose (int):
                Logging verbosity. Default is None.

        Attributes:
            channel_idxs (int or list of int):
                Channels to reduce. Default is None, which selects all channels.
            chkfile (str or None):
                Checkpoint file for the input BasisSpec and generated channel series.
                Default is None.
            outdir (str or None):
                Directory for channel summaries and NWChem basis files. Default is None,
                which uses "summary".
    '''

    def __init__(self, spec, cost_func, ftol=1e-2, verbose=None):
        TCAO.__init__(self, spec, cost_func, ftol, verbose)

        self.channel_idxs = None    # default: performing reduction for all channels
        self.outdir = None

        self.stop_reason = None
        self.results = None

    def dump_flags(self):
        ''' Log channel-reduction settings. '''
        TCAO.dump_flags(self)
        self.log_info('channel_idxs= %s' % (str(self.channel_idxs)))
        self.log_info('outdir= %s' % (str(self.outdir)))
        self.log_info('')

    def initialize(self):
        ''' Reset channel-reduction results and evaluate the reference cost. '''
        self.converged = []
        self.stop_reason = []
        self.results = []
        self.cost_init = self.cost_func(self.spec)
        self.cost = None    # cost is intentionally not used for ChannelReduction

    def kernel(self, **kwargs):
        ''' Generate reduction series for selected channels.

            Args:
                kwargs (dict):
                    Attribute overrides applied before execution.

            Return:
                results (list of tuple):
                    `(channel_idx, series)` pairs, where each series contains
                    `(channel, cost, delta_f)` entries.
        '''
        self.set(**kwargs)

        self.initialize()
        self.dump_flags()
        self.print_init()

        channel_idxs = self.channel_idxs
        if channel_idxs is None:
            channel_idxs = range(self.spec.nchannel)
        else:
            channel_idxs = lib.to_int_list(channel_idxs)

        if self.chkfile is not None:
            self.spec.dump_chkfile(self.chkfile, prefix='spec')

        for channel_idx in channel_idxs:
            spec = self.spec.copy()
            spec._check_channel_idx(channel_idx)

            self.log_note('Series generation for Channel= %d  l= %d' % (
                channel_idx, spec.channels[channel_idx].l
            ))
            self.log_note('Init channel structure= %s  cost= %.10f' % (
                spec.channels[channel_idx].structure, self.cost_init
            ))

            results = [ (spec.channels[channel_idx].copy(), self.cost_init, 0.) ]
            self.dump_chkfile(spec.channels[channel_idx], channel_idx, first_pass=True)

            converged = False
            stop_reason = 'Only one exponent left'
            while spec.channels[channel_idx].nbas > 1:
                cost, spec = self.filter_optimization(
                    spec, force_accept=True, select_channel=channel_idx,
                )[:2]
                df = cost - self.cost_init
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
        ''' Log the initial basis and reference cost. '''
        self.log_note('Init basis structure: %s  cost= %.10f' % (
            self.spec.structure, self.cost_init
        ))
        self.log_note('Init basis:')
        if self.verbose >= 3:
            self.spec.dump_basis()
        self.log_info('')

    def dump_chkfile(self, channel, channel_idx, chkfile=None, first_pass=False):
        ''' Save one generated channel to a checkpoint file.

            Args:
                channel (Channel):
                    Channel to save.
                channel_idx (int):
                    Index of the reduced channel in the parent basis.
                chkfile (str):
                    Checkpoint path. Default is None, which uses `self.chkfile`.
                first_pass (bool):
                    Whether to clear earlier results for this channel. Default is False.
        '''
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
        ''' Save a channel-reduction series and CSV summary to disk.

            Args:
                channel_idx (int):
                    Index of the reduced channel.
                results (list of tuple):
                    `(channel, cost, delta_f)` entries.
                outdir (str):
                    Parent output directory. Default is None, which uses `self.outdir`
                    and then "summary".

            Note:
                Existing `summary.csv` and `.dat` files in this channel's output
                directory are replaced; unrelated files are retained.
        '''
        import os
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
        lib.mkdir(subdir, recursive=True)

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
    from pyscf import gto, scf

    atm = 'C'
    spin = 2
    fbas = 'cc-pvqz'
    ftol = 1e-4
    pseudo = 'gth-hf-rev'
    valence_l = [0,1]

    cost_func = lib.pyscf_helper.get_cost_func(
        atm, scf.RHF, mol_settings={'spin': spin, 'pseudo': pseudo},
        keep_l=valence_l,
    )

    basis = gto.basis.load(fbas, atm)
    spec = BasisSpec.init_from_pyscf_basis(basis, atm=atm, keep_l=valence_l)

    opt = TCAO(spec, cost_func, ftol=ftol).set(verbose=5)
    opt.kernel()
