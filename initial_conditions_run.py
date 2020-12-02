#!/usr/bin/python3
'''For each of the acute and chronic models, for the parameter values
for each of the 3 SATs, and for the initial conditions for each of the
3 SATs, run 1,000 simulations. This produces a file called
`initial_conditions.h5`.'''


import h5
import herd
import run


def _copy_run(model, SAT, initial_conditions, nruns, hdfstore_out):
    '''Copy the data from 'run.h5'.'''
    filename = 'run.h5'
    where = f'model={model} & SAT={SAT} & run<{nruns}'
    with h5.HDFStore(filename, mode='r') as hdfstore_in:
        for chunk in hdfstore_in.select(where=where, iterator=True):
            run._insert_index_levels(chunk, 2,
                                     initial_conditions=initial_conditions)
            hdfstore_out.put(chunk, min_itemsize=run._min_itemsize)


def run_initial_conditions(model, SAT, initial_conditions, tmax,
                           nruns, hdfstore):
    if initial_conditions == SAT:
        _copy_run(model, SAT, initial_conditions, nruns, hdfstore)
    else:
        p = herd.Parameters(model=model, SAT=SAT,
                            _initial_conditions=initial_conditions)
        logging_prefix = (
            ', '.join((f'model {model}',
                       f'SAT {SAT}',
                       f'initial_conditions {initial_conditions}'))
            + ', ')
        df = run.run_many(p, tmax, nruns,
                          logging_prefix=logging_prefix)
        run._prepend_index_levels(df, model=model, SAT=SAT,
                                  initial_conditions=initial_conditions)
        hdfstore.put(df, min_itemsize=run._min_itemsize)


if __name__ == '__main__':
    SATs = (1, 2, 3)
    nruns = 1000
    tmax = 10

    filename = 'initial_conditions.h5'
    with h5.HDFStore(filename) as store:
        for model in ('acute', 'chronic'):
            for SAT in SATs:
                for initial_conditions in SATs:
                    run_initial_conditions(model, SAT, initial_conditions,
                                           tmax, nruns, store)
        store.repack()
