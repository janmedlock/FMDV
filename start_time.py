#!/usr/bin/python3
'''Analyze and plot the results of varying the time of year to start
the simluations. This requires the file `start_time.h5`, which is
built by `start_time_run.py`.'''


from matplotlib import colors, pyplot, ticker
import numpy
import seaborn
import statsmodels.nonparametric.api

import plot_common
import stats


def _get_index(df, **kwds):
    ix = tuple(kwds[level] if level in kwds else slice(None)
               for level in df.index.names)
    return ix


def _make_periodic(df):
    '''Make periodic by copying the values from `start_time` = 0 to
    `start_time` = 1.'''
    new = df.loc[_get_index(df, start_time=0.), :].copy()
    new.reset_index(inplace=True)
    new['start_time'] = 1.
    new.set_index(df.index.names, inplace=True)
    df = df.append(new)
    df.sort_index(level=df.index.names, inplace=True)
    return df


def load():
    filename = 'start_time.h5'
    df = plot_common.get_extinction_time(filename)
    df = _make_periodic(df)
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
            by = 'start_time'
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
                    ax.set_xlabel('Start time (y)')
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
        for (s, g) in group.groupby('start_time'):
            survival = stats.get_survival(g, 'time', 'observed')
            ax.step(survival.index, survival,
                    where='post',
                    label=f'start_time {s}')


def plot_kde(df):
    row = dict(enumerate(range(3), 1))
    column = dict(acute=0, chronic=1)
    with seaborn.axes_style('darkgrid'):
        fig, axes = pyplot.subplots(3, 2, sharex='col')
        for ((model, SAT), group) in df.groupby(['model', 'SAT']):
            i, j = row[SAT], column[model]
            ax = axes[i, j]
            for (s, g) in group.groupby('start_time'):
                ser = g.time[g.observed]
                proportion_observed = len(ser) / len(g)
                if proportion_observed > 0:
                    kde = statsmodels.nonparametric.api.KDEUnivariate(ser)
                    kde.fit(cut=0)
                    x = kde.support
                    y = proportion_observed * kde.density
                else:
                    x, y = [], []
                label = s if i == j == 0 else ''
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
                         title='Start time (y)')
        fig.tight_layout(rect=(0, 0, 0.82, 1))


def plot_kde_2d(df):
    persistence_time_max = dict(acute=0.5, chronic=10)
    start_times = (df.index
                     .get_level_values('start_time')
                     .unique()
                     .sort_values())
    start_time_baseline = 0.5
    width = 390 / 72.27
    height = 0.8 * width
    rc = plot_common.rc.copy()
    rc['figure.figsize'] = (width, height)
    rc['xtick.labelsize'] = rc['ytick.labelsize'] = 7
    rc['axes.labelsize'] = 8
    rc['axes.titlesize'] = 9
    nrows = 2 + 1
    ncols = 3
    height_ratios = (1, 1, 0.5)
    w_pad = 8 / 72
    with pyplot.rc_context(rc):
        fig = pyplot.figure(constrained_layout=True)
        fig.set_constrained_layout_pads(w_pad=w_pad)
        gs = fig.add_gridspec(nrows, ncols,
                              height_ratios=height_ratios)
        axes = numpy.empty((nrows, ncols), dtype=object)
        axes[0, 0] = None  # Make sharex & sharey work for axes[0, 0].
        for row in range(nrows):
            for col in range(ncols):
                # Columns share the x scale.
                sharex = axes[0, col]
                # Rows share the y scale.
                sharey = axes[row, 0]
                axes[row, col] = fig.add_subplot(gs[row, col],
                                                 sharex=sharex,
                                                 sharey=sharey)
        for (i, (model, group_model)) in enumerate(df.groupby('model')):
            persistence_time = numpy.linspace(0, persistence_time_max[model],
                                              301)
            for (j, (SAT, group_SAT)) in enumerate(group_model.groupby('SAT')):
                ax = axes[i, j]
                density = numpy.zeros((len(persistence_time),
                                       len(start_times)))
                proportion_observed = numpy.zeros_like(start_times,
                                                       dtype=float)
                grouper = group_SAT.groupby('start_time')
                for (k, (s, g)) in enumerate(grouper):
                    ser = g.time[g.observed]
                    nruns = len(g)
                    proportion_observed[k] = len(ser) / nruns
                    density[:, k] = plot_common.get_density(ser,
                                                            persistence_time)
                cmap = plot_common.get_cmap_SAT(SAT)
                # Use raw `density` for color,
                # but plot `density * proportion_observed`.
                norm = colors.Normalize(vmin=0, vmax=numpy.max(density))
                ax.imshow(density * proportion_observed,
                          cmap=cmap, norm=norm, interpolation='bilinear',
                          extent=(min(start_times), max(start_times),
                                  min(persistence_time), max(persistence_time)),
                          aspect='auto', origin='lower', clip_on=False)
                # ax shares the xaxis with ax_po.
                ax.xaxis.set_tick_params(which='both',
                                         labelbottom=False, labeltop=False)
                ax.xaxis.offsetText.set_visible(False)
                ax.xaxis.set_minor_locator(ticker.AutoMinorLocator(2))
                ax.yaxis.set_major_locator(
                    ticker.MultipleLocator(max(persistence_time) / 5))
                ax.yaxis.set_minor_locator(ticker.AutoMinorLocator(2))
                if ax.is_first_col():
                    ax.set_ylabel('Extinction time (y)')
                if ax.is_first_row():
                    ax.set_title(f'SAT{SAT}')
                if model == 'chronic':
                    ax_po = axes[-1, j]
                    ax_po.plot(start_times, 1 - proportion_observed,
                               color=plot_common.SAT_colors[SAT],
                               clip_on=False, zorder=3)
                    ax_po.set_xlabel('Start time (y)')
                    ax_po.xaxis.set_minor_locator(ticker.AutoMinorLocator(2))
                    ax_po.yaxis.set_major_formatter(
                        ticker.PercentFormatter(xmax=1))
                    ax_po.yaxis.set_minor_locator(ticker.AutoMinorLocator(2))
                    if ax_po.is_first_col():
                        ax_po.set_ylabel('Persisting 10 y')
        for ax in fig.axes:
            ax.axvline(start_time_baseline,
                       color='black', linestyle='dotted', alpha=0.7)
            ax.autoscale(tight=True)
            if not ax.is_first_col():
                ax.yaxis.set_tick_params(which='both',
                                         labelleft=False, labelright=False)
                ax.yaxis.offsetText.set_visible(False)
            for sp in ('top', 'right'):
                ax.spines[sp].set_visible(False)
        fig.align_ylabels()
        label_x = 0
        label_kws = dict(fontsize=8,
                         rotation=90,
                         horizontalalignment='left',
                         verticalalignment='center')
        fig.text(label_x, 0.79, 'Acute model', **label_kws)
        fig.text(label_x, 0.31, 'Carrier model', **label_kws)
        fig.savefig('start_time.pdf')


if __name__ == '__main__':
    df = load()
    # plot_median(df)
    # plot_survival(df)
    # plot_kde(df)
    plot_kde_2d(df)
    pyplot.show()
