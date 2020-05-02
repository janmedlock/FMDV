#!/usr/bin/python3

import sys
import os.path
import csv

sys.path.append('..')
import herd
from herd import birth
import extinction_time
sys.path.pop()


def search_parameter(parameter_name, values, parameters, tmax, nruns,
                    *args, **kwargs):
    assert hasattr(parameters, parameter_name)

    (basename, ext) = os.path.splitext(os.path.basename(sys.argv[0]))
    filename = basename + '.csv'

    new = not os.path.exists(filename)
    fd = open(filename, 'a')
    w = csv.writer(fd)

    paramkeys = sorted(parameters.__dict__.keys())

    if new:
        # Write header.
        w.writerow(paramkeys + ['extinction_time (years)'])
        fd.flush()

    for v in values:
        setattr(parameters, parameter_name, v)
        print('{} = {}'.format(parameter_name, v))
        ets = extinction_time.find(parameters, tmax, nruns, *args, **kwargs)
        w.writerow([getattr(parameters, k) for k in paramkeys]
                   + ets)
        fd.flush()

    fd.close()


if __name__ == '__main__':
    import numpy

    model = 'chronic'

    population_sizes = (100, 200, 500, 1000)
    birth_seasonal_coefficients_of_variation = (
        0.61 * numpy.array([1, 0.75, 0.5, 2, 3, 0.25, 4, 0.1, 0]))

    nruns = 1000
    tmax = 10
    debug = False

    for bscov in birth_seasonal_coefficients_of_variation:
        for SAT in (1, 2, 3):
            parameters = herd.Parameters(model=model, SAT=SAT)
            parameters.birth_seasonal_coefficient_of_variation = bscov
            search_parameter('population_size',
                             population_sizes,
                             parameters,
                             tmax,
                             nruns,
                             debug=debug)
