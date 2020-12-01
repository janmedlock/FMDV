#!/usr/bin/python3
'''For each of the acute and chronic models, for each of the 3 SATs,
and for 12 different times of year to start the simulations, run 1,000
simulations using the baseline parameter values. This produces a file
called `start_time.h5`.'''


import numpy

import h5
import herd
import run


def _copy_run(model, SAT, start_time, nruns, hdfstore_out):
    '''Copy the data from 'run.h5'.'''
    filename = 'run.h5'
    where = f'model={model} & SAT={SAT} & run<{nruns}'
    with h5.HDFStore(filename, mode='r') as hdfstore_in:
        for chunk in hdfstore_in.select(where=where, iterator=True):
            run._insert_index_levels(chunk, 2,
                                     start_time=start_time)
            hdfstore_out.put(chunk, min_itemsize=run._min_itemsize)


def run_start_time(model, SAT, start_time, tmax, nruns, hdfstore):
    if start_time == 0.5:
        _copy_run(model, SAT, start_time, nruns, hdfstore)
    else:
        p = herd.Parameters(model=model, SAT=SAT)
        p.start_time = start_time
        logging_prefix = (', '.join((f'model {model}',
                                     f'SAT {SAT}',
                                     f'start_time {start_time}'))
                          + ', ')
        df = run.run_many(p, tmax, nruns,
                          logging_prefix=logging_prefix)
        run._prepend_index_levels(df, model=model, SAT=SAT,
                                  start_time=start_time)
        hdfstore.put(df, min_itemsize=run._min_itemsize)


if __name__ == '__main__':
    start_times = numpy.arange(0, 1, 1 / 12)
    nruns = 1000
    tmax = 10

    filename = 'start_time.h5'
    with h5.HDFStore(filename) as store:
        for start_time in start_times:
            for model in ('acute', 'chronic'):
                for SAT in (1, 2, 3):
                    run_start_time(model, SAT, start_time,
                                   tmax, nruns, store)
        store.repack()
