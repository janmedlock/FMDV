#!/usr/bin/python3
'''Time the Floquet solver.'''

import subprocess
from time import perf_counter, process_time

from matplotlib import pyplot
import numpy
from scipy.integrate import quad

# Rebuild the 'monodromy' module, if necessary.
subprocess.run(['make'], cwd='../herd/floquet', check=True)

from context import herd
import herd.age_structure
import herd.mortality
import herd.floquet
import herd.floquet.floquet


def find_stable_age_structure(parameters, fast=False):
    solver_parameters = herd.floquet.monodromy.Parameters(parameters)
    # Temporarily monkeypatch non-caching version of
    # `_find_dominant_eigen` into place to force recomputation.
    _find_dominant_eigen = herd.floquet.floquet._find_dominant_eigen
    herd.floquet.floquet._find_dominant_eigen = _find_dominant_eigen.func
    try:
        t0, p0 = perf_counter(), process_time()
        if fast:
            birth_scaling = 0.9378975738425385
            r, v, ages = herd.floquet.floquet._find_dominant_eigen(
                birth_scaling, solver_parameters,
                herd.floquet.floquet._agemax_default,
                herd.floquet.floquet._agestep_default)
        else:
            # Force a recomputation of the birth scaling.
            birth_scaling = herd.floquet.floquet._find_birth_scaling.func(
                solver_parameters,
                herd.floquet.floquet._step_default,
                herd.floquet.floquet._age_max_default)
        t1, p1 = perf_counter(), process_time()
    finally:
        # Restore caching version of `_find_dominant_eigen`.
        herd.floquet.floquet._find_dominant_eigen = _find_dominant_eigen
    print('{} benchmark took {:g} s clock, {:g} s CPU'.format(
        'Fast' if fast else 'Slow', t1 - t0, p1 - p0))
    if fast:
        # assert numpy.isclose(r, 0), 'r = {:g}'.format(r)
        if not numpy.isclose(r, 0):
            print(f'r = {r:g}')
    else:
        v, ages = herd.age_structure.find_stable_age_structure(parameters)
    return (v, ages)


def plot(ages, stable_age_structure):
    fig, ax = pyplot.subplots()
    agemax = 25
    which = (ages <= agemax)
    ax.plot(ages[which], stable_age_structure[which],
            label='stable age structure')
    mortality_sf = herd.mortality.from_param_values().sf
    mortality_sf_scale, _ = quad(mortality_sf, ages[0], ages[-1])
    ax.plot(ages[which], mortality_sf(ages[which]) / mortality_sf_scale,
            label='scaled mortality survival',
            color='black', linestyle='dotted')
    ax.set_xlabel('age (y)')
    ax.set_ylabel('density (y$^{-1}$)')
    ax.legend()
    fig.tight_layout()
    pyplot.show()


if __name__ == '__main__':
    parameters = herd.Parameters()
    stable_age_structure, ages = find_stable_age_structure(parameters,
                                                           fast=True)
    plot(ages, stable_age_structure)
