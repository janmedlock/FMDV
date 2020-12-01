#!/usr/bin/python3
'''For each of the acute and chronic models, for each of the 3 SATs,
and for 5 different levels of birth seasonality, run 1,000 simulations
using the baseline parameter values. This produces a file called
`birth_seasonality.h5`.'''


import numpy

import h5
import herd
import run


def _copy_run(model, SAT, bscov, nruns, hdfstore_out):
    '''Copy the data from 'run.h5'.'''
    filename = 'run.h5'
    where = f'model={model} & SAT={SAT} & run<{nruns}'
    with h5.HDFStore(filename, mode='r') as hdfstore_in:
        for chunk in hdfstore_in.select(where=where, iterator=True):
            run._insert_index_levels(
                chunk, 2,
                birth_seasonal_coefficient_of_variation=bscov)
            hdfstore_out.put(chunk, min_itemsize=run._min_itemsize)


def run_birth_seasonality(model, SAT, birth_seasonality_scaling, tmax, nruns,
                          hdfstore):
    p = herd.Parameters(model=model, SAT=SAT)
    p.birth_seasonal_coefficient_of_variation *= birth_seasonality_scaling
    bscov = p.birth_seasonal_coefficient_of_variation
    if birth_seasonality_scaling == 1:
        _copy_run(model, SAT, bscov, nruns, hdfstore)
    else:
        logging_prefix = (
            ', '.join((
                f'model {model}',
                f'SAT {SAT}',
                f'birth_seasonality_scaling {birth_seasonality_scaling}'))
            + ', ')
        df = run.run_many(p, tmax, nruns,
                          logging_prefix=logging_prefix)
        run._prepend_index_levels(
            df, model=model, SAT=SAT,
            birth_seasonal_coefficient_of_variation=bscov)
        hdfstore.put(df, min_itemsize=run._min_itemsize)


if __name__ == '__main__':
    birth_scalings = numpy.linspace(0, 2, 5)
    nruns = 1000
    tmax = 10

    filename = 'birth_seasonality.h5'
    with h5.HDFStore(filename) as store:
        for birth_scaling in birth_scalings:
            for model in ('acute', 'chronic'):
                for SAT in (1, 2, 3):
                    run_birth_seasonality(model, SAT, birth_scaling,
                                          tmax, nruns, store)
        store.repack()
