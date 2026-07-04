from pygto.optimizer import NelderMead, BFGS


PRESET_SCHEDULE = {
    'loose': [
        (NelderMead, 'low'),
        (NelderMead, 'medium'),
    ],
    'default': [
        (NelderMead, 'low'),
        (NelderMead, 'medium'),
        (BFGS, 'high'),
    ],
    'refine': [
        (BFGS, 'high'),
        (BFGS, 'ultra'),
    ]
}
PRESET_SCHEDULE['default'] = PRESET_SCHEDULE['explore']


def scheduled_optimize(spec, cost_func, schedule='default', grad_func=None, verbose=4):
    ''' Optimize with schedule.
    '''
    if isinstance(schedule, str):
        if schedule not in PRESET_SCHEDULE:
            raise ValueError(f'Unknown named schedule "{schedule}"')
        tasks = PRESET_SCHEDULE[schedule]
    elif isinstance(schedule, (list,tuple)):
        tasks = schedule
    else:
        raise TypeError(f'Schedule must be str or list/tuple.')

    def set_opt(OPT):
        if isinstance(OPT, str):
            if OPT.lower().startswith('nel'):
                OPT = NelderMead
            elif OPT.lower().startswith('bf'):
                OPT = BFGS
            else:
                raise ValueError(f'Unknown optimizer type "{OPT}".')
        elif OPT in [NelderMead,BFGS]:
            pass
        else:
            raise TypeError(f'Unknown optimizer "{OPT.__name__}"')

        return OPT


    def create_opt_obj(spec, OPT):
        if isinstance(OPT, str):
            if OPT.lower().startswith('nel'):
                OPT = NelderMead
            elif OPT.lower().startswith('bf'):
                OPT = BFGS
            else:
                raise ValueError(f'Unknown optimizer type "{OPT}".')

        if OPT == NelderMead:
            opt = OPT(spec, cost_func, verbose=verbose)
        elif OPT == BFGS:
            opt = OPT(spec, cost_func, verbose=verbose, grad_func=grad_func)
        else:
            raise TypeError(f'Unknown optimizer "{OPT.__name__}"')

        return opt

    ntask = len(tasks)
    opts = []
    spec1 = spec.copy()
    for itask,task in enumerate(tasks):
        OPT, acc = task
        OPT = set_opt(OPT)

        spec.log_info(f'Schedule {itask+1}/{ntask}  optimizer= {OPT.__name__}  '
                      f'accuracy= {acc}')

        if OPT == NelderMead:
            opt = OPT(spec, cost_func, verbose=verbose)
        else:
            opt = OPT(spec, cost_func, grad_func=grad_func, verbose=verbose)

        opt.kernel()
        opts.append( opt )

        spec1 = opt.spec.copy()

    return spec1, opts
