'''Common code to work with simulation extinction times.'''


import numpy
import pandas

import h5
import run


def _get_extinction(infected, tmax=10):
    t = infected.index.get_level_values('time (y)')
    time = t.max() - t.min()
    observed = (infected.iloc[-1] == 0)
    assert observed or (time == tmax)
    return dict(time=time, observed=observed)


def load_extinction_time(filename, by, **kwargs):
    columns = ['exposed', 'infectious', 'chronic']
    extinction = {}
    with h5.HDFStore(filename, mode='r') as store:
        for (ix, chunk) in store.groupby(by, columns=columns, **kwargs):
            infected = chunk.sum(axis='columns')
            extinction[ix] = _get_extinction(infected)
    extinction = pandas.DataFrame.from_dict(extinction,
                                            orient='index')
    extinction.index.names = by
    extinction.sort_index(inplace=True)
    return extinction


def number_infected(x):
    M, S, E, I, R = x
    return (E + I)


def find(parameters, tmax, nruns, *args, **kwds):
    data = run.run_many(parameters, tmax, nruns, *args, **kwds)
    (T, X) = zip(*(zip(*d) for d in data))
    extinction_times = [t[-1] if (number_infected(x[-1]) == 0) else None
                        for (t, x) in zip(T, X)]
    return extinction_times


def ppf(D, q, a=0):
    Daug = numpy.asarray(sorted(D) + [a])
    indices = numpy.ceil(numpy.asarray(q) * len(D) - 1).astype(int)
    return Daug[indices]


def proportion_ge_x(D, x):
    return float(len(numpy.compress(numpy.asarray(D) >= x, D))) / float(len(D))


def get_stats(extinction_times):
    mystats = {}
    mystats['median'] = numpy.median(extinction_times)
    mystats['mean'] = numpy.mean(extinction_times)
    mystats['q_90'] = ppf(extinction_times, 0.9)
    mystats['q_95'] = ppf(extinction_times, 0.95)
    mystats['q_99'] = ppf(extinction_times, 0.99)
    mystats['proportion >= 1'] = proportion_ge_x(extinction_times, 1)
    mystats['proportion >= 10'] = proportion_ge_x(extinction_times, 10)
    return mystats