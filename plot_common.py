import os.path

import matplotlib.collections
import matplotlib.colors
import matplotlib.ticker
import numpy
import pandas
import statsmodels.nonparametric.api

import h5
import run_common


# Nature.
rc = {}
# Widths: 89mm, 183mm, 120mm, 136mm.
# Sans-serif, preferably Helvetica or Arial.
rc['font.family'] = 'sans-serif'
rc['font.sans-serif'] = 'DejaVu Sans'
# Fonts between 5pt and 7pt.
# Separate panels in multi-part figures should be labelled with 8
# pt bold, upright (not italic) a, b, c...


# All files are relative to this source file.
_path = os.path.dirname(__file__)

filename = os.path.join(_path, 'run.h5')

t_name = 'time (y)'


def _build_downsample_group(group, t, by):
    # Only keep time index.
    group = group.reset_index(by, drop=True)
    # Only interpolate between start and extinction.
    mask = ((t >= group.index.min()) & (t <= group.index.max()))
    # Interpolate from the closest point <= t.
    return group.reindex(t[mask], method='ffill')


def build_downsample(filename_in, t_min=0, t_max=10, t_step=1/365):
    t = numpy.arange(t_min, t_max, t_step)
    base, ext = os.path.splitext(filename_in)
    filename_out = base + '_downsampled' + ext
    with h5.HDFStore(filename_in, mode='r') as store_in, \
         h5.HDFStore(filename_out, mode='w') as store_out:
        by = [n for n in store_in.get_index_names() if n != t_name]
        for (ix, group) in store_in.groupby(by):
            downsample = _build_downsample_group(group, t, by)
            # Append `ix` to the index levels.
            downsample = pandas.concat({ix: downsample},
                                       names=by, copy=False)
            store_out.put(downsample, dropna=True, index=False,
                          min_itemsize=run_common._min_itemsize)
        store_out.create_table_index()
        store_out.repack()


def get_downsampled(filename):
    t_max = 10 + 11 / 12
    base, ext = os.path.splitext(filename)
    filename_ds = base + '_downsampled' + ext
    if not os.path.exists(filename_ds):
        build_downsampled(filename, t_max=t_max)
    return h5.HDFStore(filename_ds, mode='r')


def _build_infected(filename, filename_out):
    store = get_downsampled(filename)
    columns = ['exposed', 'infectious', 'chronic']
    infected = []
    for chunk in store.select(columns=columns, iterator=True):
        infected.append(chunk.sum(axis='columns'))
    infected = pandas.concat(infected, copy=False)
    infected.name = 'infected'
    h5.dump(infected, filename_out, mode='w',
            min_itemsize=run_common._min_itemsize)


def get_infected():
    filename_infected = os.path.join(_path, 'plot_common_infected.h5')
    try:
        infected = h5.load(filename_infected)
    except OSError:
        _build_infected(filename, filename_infected)
        infected = h5.load(filename_infected)
    return infected


def _build_extinction_time_group(infected):
    if infected.iloc[-1] == 0:
        t = infected.index.get_level_values(t_name)
        return t.max() - t.min()
    else:
        return numpy.nan


def _build_extinction_time(filename, filename_out):
    with h5.HDFStore(filename, mode='r') as store:
        by = [n for n in store.get_index_names() if n != t_name]
        # Only the infected columns.
        columns = ['exposed', 'infectious', 'chronic']
        ser = {}
        for (ix, group) in store.groupby(by, columns=columns):
            infected = group.sum(axis='columns')
            ser[ix] = _build_extinction_time_group(infected)
    ser = pandas.Series(ser, name='extinction time (days)')
    ser.rename_axis(by, inplace=True)
    h5.dump(ser, filename_out, mode='w',
            min_itemsize=run_common._min_itemsize)


def get_extinction_time():
    filename_et = os.path.join(_path,
                               'plot_common_extinction_time.h5')
    try:
        extinction_time = h5.load(filename_et)
    except OSError:
        _build_extinction_time(filename, filename_et)
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
