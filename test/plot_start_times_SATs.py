#!/usr/bin/python3

import os.path
import sys

from matplotlib import pyplot
import numpy
import pandas
import seaborn

sys.path.append('..')
import h5
import plot_common
import run_common
sys.path.pop()


# All files are relative to this source file.
_path = os.path.dirname(__file__)

filename = os.path.join(_path, 'run_start_times_SATs.h5')


def get_infected(model='acute'):
    filename_infected = os.path.join(_path,
                                     'plot_start_times_SATs_infected.h5')
    try:
        infected = h5.load(filename_infected)
    except FileNotFoundError:
        plot_common._build_infected(filename, filename_infected)
        infected = h5.load(filename_infected)
    return infected.loc[model]


def plot_infected_facet(infected, color=None, alpha=1, **kwargs):
    # `color` is ignored so each run gets a different color.
    infected = infected.unstack('run')
    t = infected.index.get_level_values(plot_common.t_name).sort_values()
    t -= t.min()
    # Convert to days.
    t *= 365
    pyplot.plot(t, infected, alpha=alpha, **kwargs)
    # `infection.fillna(0)` gives mean including those that have gone extinct.
    # Use `alpha=1` for the mean, not the `alpha` value passed in.
    pyplot.plot(t, infected.fillna(0).mean(axis=1),
                color='black', alpha=1, **kwargs)


def plot_infected(model='acute'):
    infected = get_infected(model=model)
    infected.reset_index(inplace=True)
    pyplot.figure()
    # g = seaborn.FacetGrid(data=infected,
    #                       row='SAT', col='start_time',
    #                       sharey='row')
    g = seaborn.FacetGrid(data=infected, col='SAT')
    g.map(plot_infected_facet, 'infected', alpha=0.3)
    g.set_axis_labels('time (days)', 'number infected')
    g.set_titles('{col_var} {col_name}')
    pyplot.tight_layout()
    pyplot.savefig(f'plot_start_times_SATs_infected_{model}.pdf')


def get_extinction_time(model='acute'):
    filename_et = os.path.join(_path,
                               'plot_start_times_SATs_extinction_time.h5')
    try:
        extinction_time = h5.load(filename_et)
    except FileNotFoundError:
        plot_common._build_extinction_time(filename, filename_et)
        extinction_time = h5.load(filename_et)
    return extinction_time.loc[model]


def plot_extinction_time(model='acute'):
    extinction_time = get_extinction_time(model=model)
    extinction_time.reset_index(inplace=True)
    pyplot.figure()
    # seaborn.factorplot(data=extinction_time,
    #                    x='extinction time (days)', y='start_time', col='SAT',
    #                    kind='box', orient='horizontal', sharey=False)
    ax = seaborn.violinplot(data=extinction_time,
                            x='extinction time (days)', y='SAT',
                            orient='horizontal', width=0.95,
                            cut=0, linewidth=1)
    plot_common.set_violins_linewidth(ax, 0)
    pyplot.ylabel('')
    locs, labels = pyplot.yticks()
    pyplot.yticks(locs, ['SAT {}'.format(i.get_text()) for i in labels])
    pyplot.xlim(left=0)
    pyplot.tight_layout()
    pyplot.savefig(f'plot_start_times_SATs_extinction_time_{model}.pdf')


def _build_time_to_peak_group(infected):
    t = infected.index.get_level_values(plot_common.t_name)
    m = infected.index.get_loc(infected.idxmax())
    return (t[m] - t.min())


def _build_time_to_peak(filename_out):
    with h5.HDFStore(filename, mode='r') as store:
        by = [n for n in store.get_index_names() if n != plot_common.t_name]
        # Only the first start time.
        where = 'start_time=0'
        # Only the infected columns.
        columns = ['exposed', 'infectious', 'chronic']
        ser = {}
        for (ix, group) in store.groupby(by, where=where, columns=columns):
            infected = group.sum(axis='columns')
            ser[ix] = _build_time_to_peak_group(infected)
    ser = pandas.Series(ser, name='time to peak (days)')
    ser.rename_axis(by, inplace=True)
    ser *= 365
    h5.dump(ser, filename_out, mode='w',
            min_itemsize=run_common._min_itemsize)


def get_time_to_peak(model='acute'):
    filename_ttp = os.path.join(_path,
                                'plot_start_times_SATs_time_to_peak.h5')
    try:
        extinction_time = h5.load(filename_ttp)
    except FileNotFoundError:
        _build_time_to_peak(filename_ttp)
        time_to_peak = h5.load(filename_ttp)
    return time_to_peak.loc[model]


def plot_time_to_peak(model='acute'):
    time_to_peak = get_time_to_peak(model=model)
    time_to_peak.reset_index(inplace=True)
    pyplot.figure()
    # seaborn.factorplot(data=time_to_peak,
    #                    x='time to peak (days)', y='start_time', col='SAT',
    #                    sharey=False,
    #                    kind='violin', orient='horizontal',
    #                    cut=0, linewidth=1)
    ax = seaborn.violinplot(data=time_to_peak,
                            x='time to peak (days)', y='SAT',
                            orient='horizontal', width=0.95,
                            cut=0, linewidth=1)
    plot_common.set_violins_linewidth(ax, 0)
    pyplot.ylabel('')
    locs, labels = pyplot.yticks()
    pyplot.yticks(locs, ['SAT {}'.format(i.get_text()) for i in labels])
    pyplot.tight_layout()
    pyplot.savefig(f'plot_start_times_SATs_time_to_peak_{model}.pdf')


def _build_total_infected_group(df):
    R = df['recovered']
    # This sucks to approximate total infected.
    return R.iloc[-1] - R.iloc[0]


def _build_total_infected(filename_out):
    with h5.HDFStore(filename, mode='r') as store:
        by = [n for n in store.get_index_names() if n != plot_common.t_name]
        # Only the first start time.
        where = 'start_time=0'
        # Only the recovered column.
        columns = ['recovered']
        ser = {}
        for (ix, group) in store.groupby(by, where=where, columns=columns):
            ser[ix] = _build_total_infected_group(group)
    ser = pandas.Series(ser, name='total infected')
    ser.rename_axis(by, inplace=True)
    ser.clip_lower(0, inplace=True)
    h5.dump(ser, filename_out, mode='w',
            min_itemsize=run_common._min_itemsize)


def get_total_infected(model='acute'):
    filename_ti = os.path.join(_path,
                               'plot_start_times_SATs_total_infected.h5')
    try:
        total_infected = h5.load(filename_ti)
    except FileNotFoundError:
        _build_total_infected(filename_ti)
        total_infected = h5.load(filename_ti)
    return total_infected.loc[model]


def plot_total_infected(model='acute'):
    total_infected = get_total_infected(model=model)
    total_infected.reset_index(inplace=True)
    pyplot.figure()
    # seaborn.factorplot(data=total_infected,
    #                    x='total infected', y='start_time', col='SAT',
    #                    sharey=False,
    #                    kind='violin', orient='horizontal',
    #                    cut=0, linewidth=1)
    ax = seaborn.violinplot(data=total_infected,
                            x='total infected', y='SAT',
                            orient='horizontal', width=0.95,
                            cut=0, linewidth=1)
    plot_common.set_violins_linewidth(ax, 0)
    pyplot.ylabel('')
    locs, labels = pyplot.yticks()
    pyplot.yticks(locs, ['SAT {}'.format(i.get_text()) for i in labels])
    pyplot.tight_layout()
    pyplot.savefig(f'plot_start_times_SATs_total_infected_{model}.pdf')


if __name__ == '__main__':
    model = 'chronic'

    plot_infected(model)
    # plot_extinction_time(model)
    # plot_time_to_peak(model)
    # plot_total_infected(model)
    # pyplot.show()