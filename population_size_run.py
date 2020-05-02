#!/usr/bin/python3

import numpy

import h5
import herd
import run


def _copy_run(model, SAT, population_size, nruns, hdfstore_out):
    '''Copy the data from 'run.h5'.'''
    filename = 'run.h5'
    where = f'model={model} & SAT={SAT} & run<{nruns}'
    with h5.HDFStore(filename, mode='r') as hdfstore_in:
        for chunk in hdfstore_in.select(where=where, iterator=True):
            run._insert_index_levels(chunk, 2,
                                     population_size=population_size)
            hdfstore_out.put(chunk, min_itemsize=run._min_itemsize)


def run_population_size(model, SAT, population_size, tmax, nruns, hdfstore):
    if population_size == 1000:
        _copy_run_SATs(model, SAT, population_size, nruns, hdfstore)
    else:
        p = herd.Parameters(model=model, SAT=SAT)
        p.population_size = population_size
        logging_prefix = (', '.join((f'model {model}',
                                     f'SAT {SAT}',
                                     f'population_size {population_size}'))
                          + ', ')
        df = run.run_many(p, tmax, nruns,
                          logging_prefix=logging_prefix)
        run._prepend_index_levels(df, model=model, SAT=SAT,
                                  population_size=population_size)
        hdfstore.put(df, min_itemsize=run._min_itemsize)


def arange(start, stop, step):
    '''Like `numpy.arange()`, but with `stop` in the range.'''
    return numpy.arange(start, stop + step, step)


if __name__ == '__main__':
    population_sizes = numpy.hstack((arange(100, 900, 100),
                                     arange(1000, 5000, 1000)))
    nruns = 1000
    tmax = 10

    filename = 'population_size.h5'
    with h5.HDFStore(_filename) as store:
        for population_size in population_sizes:
            for model in ('acute', 'chronic'):
                for SAT in (1, 2, 3):
                    run_population_size(model, SAT, population_size,
                                        tmax, nruns, store)
        store.repack()
