'''Tools for the probability of being in each immune state.'''

import numpy

from . import _solver


class Interpolant:
    '''The solution from `_solver.solve()`, that then gets interpolated to
    different ages as needed.'''

    def __init__(self, params):
        self._probability = _solver.solve(params)

    def __call__(self, age):
        # Interpolate `self._probability` to `age`.
        ages = numpy.atleast_1d(age)
        # For some reason, duplicates may be present after `.union()`.
        index = self._probability.index.union(ages) \
                                       .drop_duplicates()
        prob = self._probability.reindex(index) \
                                .interpolate() \
                                .loc[age]
        assert prob.shape[:-1] == numpy.shape(age)
        return prob


def probability(params, age):
    '''The probability of being in each immune status at age `a`.'''
    return Interpolant(params)(age)