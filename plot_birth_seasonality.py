#!/usr/bin/python3

from matplotlib import colors, pyplot, ticker
import numpy
import seaborn
import statsmodels.nonparametric.api

import extinction_times
import h5
import plot_common
import stats


def load_extinction_times():
    filename = 'plot_birth_seasonality.h5'
    try:
        df = h5.load(filename)
    except OSError:
        df = extinction_times.load_extinction_times(
            'run_birth_seasonality.h5',
            ['model', 'SAT', 'birth_seasonal_coefficient_of_variation', 'run'])
        h5.dump(df, filename)
    return df


def plot_median(df, CI=0.5):
    row = dict(acute=0, chronic=1)
    column = dict(enumerate(range(3), 1))
    levels = [CI / 2, 1 - CI / 2]
    with seaborn.axes_style('darkgrid'):
        fig, axes = pyplot.subplots(len(row), len(column),
                                    sharex='col', sharey='row')
        for ((model, SAT), group) in df.groupby(['model', 'SAT']):
            i, j = row[model], column[SAT]
            ax = axes[i, j]
            by = 'birth_seasonal_coefficient_of_variation'
            times = group.groupby(by).time
            median = times.median()
            ax.plot(median.index, median,
                    color=plot_common.SAT_colors[SAT])
            CI_ = times.quantile(levels).unstack()
            ax.fill_between(CI_.index, CI_[levels[0]], CI_[levels[1]],
                            color=plot_common.SAT_colors[SAT],
                            alpha=0.5)
            if ax.is_first_row():
                ax.set_title(f'SAT{SAT}', fontsize='medium')
            else:
                ax.set_title('')
            if ax.is_last_row():
                if j == 1:
                    ax.set_xlabel('Birth seasonal\ncoefficient of variation')
            else:
                ax.set_xlabel('')
            if ax.is_first_col():
                ax.set_ylabel(
                    f'{model.capitalize()} model\n\nextinction time (y)')
        for ax in axes[:, -1]:
            ax.set_ylim(bottom=0)
        fig.align_labels()
        fig.tight_layout()


def plot_survival(df):
    row = dict(enumerate(range(3), 1))
    column = dict(acute=0, chronic=1)
    fig, axes = pyplot.subplots(3, 2, sharex='col', sharey='row')
    for ((model, SAT), group) in df.groupby(['model', 'SAT']):
        i, j = row[SAT], column[model]
        ax = axes[i, j]
        for (b, g) in group.groupby('birth_seasonal_coefficient_of_variation'):
            survival = stats.get_survival(g, 'time', 'observed')
            ax.step(survival.index, survival,
                    where='post',
                    label=f'birth_seasonal_coefficient_of_variation {b}')


def plot_kde(df):
    row = dict(enumerate(range(3), 1))
    column = dict(acute=0, chronic=1)
    with seaborn.axes_style('darkgrid'):
        fig, axes = pyplot.subplots(3, 2, sharex='col')
        for ((model, SAT), group) in df.groupby(['model', 'SAT']):
            i, j = row[SAT], column[model]
            ax = axes[i, j]
            for (b, g) in group.groupby(
                    'birth_seasonal_coefficient_of_variation'):
                ser = g.time[g.observed]
                proportion_observed = len(ser) / len(g)
                if proportion_observed > 0:
                    kde = statsmodels.nonparametric.api.KDEUnivariate(ser)
                    kde.fit(cut=0)
                    x = kde.support
                    y = proportion_observed * kde.density
                else:
                    x, y = [], []
                label = b if i == j == 0 else ''
                ax.plot(x, y, label=label, alpha=0.7)
            ax.yaxis.set_major_locator(ticker.NullLocator())
            if ax.is_first_row():
                ax.set_title(f'{model.capitalize()} model',
                             fontdict=dict(fontsize='medium'))
            if ax.is_last_row():
                ax.set_xlim(left=0)
                ax.set_xlabel('extinction time (y)')
            if ax.is_first_col():
                ylabel = '\ndensity' if i == 1 else ''
                ax.set_ylabel(f'SAT{SAT}{ylabel}')
        leg = fig.legend(loc='center left', bbox_to_anchor=(0.8, 0.5),
                         title='Birth seasonal\ncoefficient of\nvariation')
        fig.tight_layout(rect=(0, 0, 0.82, 1))


def _get_cmap(color):
    '''White to `color`.'''
    return colors.LinearSegmentedColormap.from_list('name',
                                                    ['white', color])


def plot_kde_2d(df):
    persistence_time_max = dict(acute=0.5, chronic=10)
    bscovs = (df.index
                .get_level_values('birth_seasonal_coefficient_of_variation')
                .unique()
                .sort_values())
    bscov_baseline = bscovs[len(bscovs) // 2]
    rc = {'figure.figsize': (5, 4),
          'xtick.labelsize': 'small',
          'ytick.labelsize': 'small',
          'axes.labelsize': 'small',
          'axes.titlesize': 'medium'}
    with pyplot.rc_context(rc):
        fig, axes = pyplot.subplots(2 + 1, 3, sharex='col', sharey='row',
                                    gridspec_kw=dict(height_ratios=(1, 1, 0.5)))
        ylabelpad = 0
        for (i, (model, group_model)) in enumerate(df.groupby('model')):
            persistence_time = numpy.linspace(0, persistence_time_max[model],
                                              301)
            for (j, (SAT, group_SAT)) in enumerate(group_model.groupby('SAT')):
                ax = axes[i, j]
                density = numpy.zeros((len(persistence_time),
                                       len(bscovs)))
                proportion_observed = numpy.zeros_like(bscovs, dtype=float)
                for (k, (b, g)) in enumerate(group_SAT.groupby(
                        'birth_seasonal_coefficient_of_variation')):
                    ser = g.time[g.observed]
                    nruns = len(g)
                    proportion_observed[k] = len(ser) / nruns
                    if proportion_observed[k] > 0:
                        kde = statsmodels.nonparametric.api.KDEUnivariate(ser)
                        kde.fit(cut=0)
                        density[:, k] = kde.evaluate(persistence_time)
                    else:
                        density[:, k] = 0
                cmap = _get_cmap(plot_common.SAT_colors[SAT])
                # Use raw `density` for color,
                # but plot `density * proportion_observed`.
                norm = colors.Normalize(vmin=0, vmax=numpy.max(density))
                ax.imshow(density * proportion_observed,
                          cmap=cmap, norm=norm, interpolation='bilinear',
                          extent=(min(bscovs), max(bscovs),
                                  min(persistence_time), max(persistence_time)),
                          aspect='auto', origin='lower', clip_on=False)
                ax.autoscale(tight=True)
                if model == 'chronic':
                    ax_po = axes[-1, j]
                    ax_po.plot(bscovs, 1 - proportion_observed,
                               color=plot_common.SAT_colors[SAT],
                               clip_on=False, zorder=3)
                    ax_po.autoscale(tight=True)
                    if ax.is_first_col():
                        ax_po.set_ylabel('persisting\n10 y',
                                         labelpad=ylabelpad)
                        ax_po.yaxis.set_major_formatter(
                            plot_common.PercentFormatter())
                        ax_po.yaxis.set_minor_locator(
                            ticker.AutoMinorLocator(2))
                    ax_po.xaxis.set_minor_locator(ticker.AutoMinorLocator(2))
                    if j == 1:
                        ax_po.set_xlabel(
                            'birth seasonal\ncoefficient of variation')
                if ax.is_first_row():
                    ax.set_title(f'SAT{SAT}')
                if ax.is_last_row():
                    ax.set_xlabel('extinction time (y)')
                    ax.xaxis.set_major_locator(
                    ticker.MultipleLocator(max(persistence_time) / 5))
                    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator(2))
                if ax.is_first_col():
                    ax.set_ylabel('extinction\ntime (y)',
                                  labelpad=ylabelpad)
                    ax.yaxis.set_major_locator(
                        ticker.MultipleLocator(max(persistence_time) / 5))
                    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator(2))
        for ax in fig.axes:
            ax.axvline(bscov_baseline,
                       color='black', linestyle='dotted', alpha=0.7)
            for sp in ('top', 'right'):
                ax.spines[sp].set_visible(False)
        fig.align_labels()
        title_x = 0.01
        fig.text(title_x, 0.72, 'Acute model',
                 rotation=90, size='small')
        fig.text(title_x, 0.25, 'Chronic model',
                 rotation=90, size='small')
        fig.tight_layout(pad=0, rect=(0.03, 0, 1, 1))
        fig.savefig('plot_birth_seasonality.pdf')
        fig.savefig('plot_birth_seasonality.png', dpi=300)


if __name__ == '__main__':
    df = load_extinction_times()
    # plot_median(df)
    # plot_survival(df)
    # plot_kde(df)
    plot_kde_2d(df)
    pyplot.show()
