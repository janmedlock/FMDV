'''Common plotting code.'''


import os.path

import matplotlib.collections
import matplotlib.colors
import matplotlib.pyplot
import matplotlib.ticker
import numpy
import pandas
import statsmodels.nonparametric.api

import h5
from herd.utility import arange
import run


# Science
rc = {}
# Widths: 89mm, 183mm, 120mm, 136mm.
# Sans-serif, preferably Helvetica or Arial.
rc['font.family'] = 'sans-serif'
rc['font.sans-serif'] = 'DejaVu Sans'
# Fonts between 5pt and 7pt.
# Separate panels in multi-part figures should be labelled with 8
# pt bold, upright (not italic) a, b, c...


t_name = 'time (y)'


def _build_downsampled_group(group, t, t_step, by):
    # Only keep time index.
    group = group.reset_index(by, drop=True)
    # Shift start to 0.
    group.index -= group.index.min()
    # Only interpolate between start and extinction.
    # Round up to the next multiple of `t_step`.
    mask = (t <= (numpy.ceil(group.index.max() / t_step) * t_step))
    # Interpolate from the closest point <= t.
    return group.reindex(t[mask], method='ffill')


def build_downsampled(filename_in, t_min=0, t_max=10, t_step=1/365, by=None):
    t = arange(t_min, t_max, t_step, endpoint=True)
    base, ext = os.path.splitext(filename_in)
    filename_out = base + '_downsampled' + ext
    with h5.HDFStore(filename_in, mode='r') as store_in, \
         h5.HDFStore(filename_out, mode='w') as store_out:
        if by is None:
            by = [n for n in store_in.get_index_names() if n != t_name]
        for (ix, group) in store_in.groupby(by):
            downsampled = _build_downsampled_group(group, t, t_step, by)
            # Append `ix` to the index levels.
            downsampled = pandas.concat({ix: downsampled},
                                        names=by, copy=False)
            store_out.put(downsampled.dropna(), index=False,
                          min_itemsize=run._min_itemsize)
        store_out.create_table_index()
        store_out.repack()


def get_downsampled(filename, by=None):
    base, ext = os.path.splitext(filename)
    filename_ds = base + '_downsampled' + ext
    if not os.path.exists(filename_ds):
        build_downsampled(filename, by=by)
    return h5.HDFStore(filename_ds, mode='r')


def _build_infected(filename, filename_out, by=None):
    store = get_downsampled(filename, by=by)
    columns = ['exposed', 'infectious', 'chronic']
    infected = []
    for chunk in store.select(columns=columns, iterator=True):
        infected.append(chunk.sum(axis='columns'))
    infected = pandas.concat(infected, copy=False)
    infected.name = 'infected'
    h5.dump(infected, filename_out, mode='w',
            min_itemsize=run._min_itemsize)


def get_infected(filename, by=None):
    base, ext = os.path.splitext(filename)
    filename_infected = base + '_infected' + ext
    try:
        infected = h5.load(filename_infected)
    except OSError:
        _build_infected(filename, filename_infected, by=by)
        infected = h5.load(filename_infected)
    return infected


def _build_extinction_time_group(infected, tmax=10):
    t = infected.index.get_level_values(t_name)
    time = t.max() - t.min()
    observed = (infected.iloc[-1] == 0)
    assert observed or (time == tmax)
    return dict(time=time, observed=observed)


def _build_extinction_time(filename, filename_out, by=None):
    # Only the infected columns.
    columns = ['exposed', 'infectious', 'chronic']
    extinction = {}
    with h5.HDFStore(filename, mode='r') as store:
        if by is None:
            by = [n for n in store.get_index_names() if n != t_name]
        for (ix, group) in store.groupby(by, columns=columns):
            infected = group.sum(axis='columns')
            extinction[ix] = _build_extinction_time_group(infected)
    extinction = pandas.DataFrame.from_dict(extinction, orient='index')
    extinction.index.names = by
    extinction.sort_index(level=by, inplace=True)
    h5.dump(extinction, filename_out, mode='w',
            min_itemsize=run._min_itemsize)


def get_extinction_time(filename, by=None):
    base, ext = os.path.splitext(filename)
    filename_et = base + '_extinction_time' + ext
    try:
        extinction_time = h5.load(filename_et)
    except OSError:
        _build_extinction_time(filename, filename_et, by=by)
        extinction_time = h5.load(filename_et)
    return extinction_time


def set_violins_linewidth(ax, lw):
    for col in ax.collections:
        if isinstance(col, matplotlib.collections.PolyCollection):
            col.set_linewidth(0)


def get_density(endog, times):
    # Avoid errors if endog is empty.
    if len(endog) > 0:
        kde = statsmodels.nonparametric.api.KDEUnivariate(endog)
        kde.fit(cut=0)
        return kde.evaluate(times)
    else:
        return numpy.zeros_like(times)


def kdeplot(endog, ax=None, shade=False, cut=0, **kwds):
    if ax is None:
        ax = matplotlib.pyplot.gca()
    endog = endog.dropna()
    if len(endog) > 0:
        kde = statsmodels.nonparametric.api.KDEUnivariate(endog)
        kde.fit(cut=cut)
        x = numpy.linspace(kde.support.min(), kde.support.max(), 301)
        y = kde.evaluate(x)
        line, = ax.plot(x, y, **kwds)
        if shade:
            shade_kws = dict(
                facecolor=kwds.get('facecolor', line.get_color()),
                alpha=kwds.get('alpha', 0.25),
                clip_on=kwds.get('clip_on', True),
                zorder=kwds.get('zorder', 1))
            ax.fill_between(x, 0, y, **shade_kws)
    return ax


# Erin's colors.
SAT_colors = {
    1: '#2271b5',
    2: '#ef3b2c',
    3: '#807dba'
}


def get_cmap_SAT(SAT):
    '''White to `SAT_colors[SAT]`.'''
    return matplotlib.colors.LinearSegmentedColormap.from_list(
        'name', ['white', SAT_colors[SAT]])
