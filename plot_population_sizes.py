#!/usr/bin/python3

from matplotlib import colors, pyplot, ticker
import numpy
import seaborn
import statsmodels.nonparametric.api

import extinction_times
import h5
import plot_common
import stats


# Nature
rc = {}
# Sans-serif, preferably Helvetica or Arial.
rc['font.family'] = 'sans-serif'
rc['font.sans-serif'] = 'DejaVu Sans'


def load_extinction_times():
    filename = 'plot_population_sizes.h5'
    try:
        df = h5.load(filename)
    except OSError:
        df = extinction_times.load_extinction_times(
            'run_population_sizes.h5',
             ['model', 'SAT', 'population_size', 'run'])
        h5.dump(df, filename)
    return df


def plot_survival(df):
    row = dict(enumerate(range(3), 1))
    column = dict(acute=0, chronic=1)
    fig, axes = pyplot.subplots(3, 2, sharex='col', sharey='row')
    for ((model, SAT), group) in df.groupby(['model', 'SAT']):
        i, j = row[SAT], column[model]
        ax = axes[i, j]
        for (p, g) in group.groupby('population_size'):
            survival = stats.get_survival(g, 'time', 'observed')
            ax.step(survival.index, survival,
                    where='post',
                    label=f'population size {p}')


def plot_kde(df):
    row = dict(enumerate(range(3), 1))
    column = dict(acute=0, chronic=1)
    with seaborn.axes_style('darkgrid'):
        fig, axes = pyplot.subplots(3, 2, sharex='col')
        for ((model, SAT), group) in df.groupby(['model', 'SAT']):
            i, j = row[SAT], column[model]
            ax = axes[i, j]
            for (p, g) in group.groupby('population_size'):
                ser = g.time[g.observed]
                proportion_observed = len(ser) / len(g)
                if proportion_observed > 0:
                    kde = statsmodels.nonparametric.api.KDEUnivariate(ser)
                    kde.fit(cut=0)
                    x = kde.support
                    y = proportion_observed * kde.density
                else:
                    x, y = [], []
                label = p if i == j == 0 else ''
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
                         handletextpad=3, title='Population size')
        for text in leg.get_texts():
            text.set_horizontalalignment('right')
        fig.tight_layout(rect=(0, 0, 0.82, 1))


def plot_kde_2d(df):
    persistence_time_max = dict(acute=0.5, chronic=10)
    population_sizes = (df.index
                          .get_level_values('population_size')
                          .unique()
                          .sort_values())
    population_size_baseline = 1000
    width = 390 / 72.27
    height = 0.8 * width
    rc_ = rc.copy()
    rc_['figure.figsize'] = (width, height)
    rc_['xtick.labelsize'] = 7
    rc_['ytick.labelsize'] = 7
    rc_['axes.labelsize'] = 8
    rc_['axes.titlesize'] = 9
    nrows = 2 + 1
    ncols = 3
    height_ratios = (1, 1, 0.5)
    w_pad = 8 / 72
    with pyplot.rc_context(rc_):
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
                                       len(population_sizes)))
                proportion_observed = numpy.zeros_like(population_sizes,
                                                       dtype=float)
                grouper = group_SAT.groupby('population_size')
                for (k, (p, g)) in enumerate(grouper):
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
                          extent=(min(population_sizes), max(population_sizes),
                                  min(persistence_time), max(persistence_time)),
                          aspect='auto', origin='lower', clip_on=False)
                ax.set_xscale('log')
                # ax shares the xaxis with ax_po.
                ax.xaxis.set_tick_params(which='both',
                                         labelbottom=False, labeltop=False)
                ax.xaxis.offsetText.set_visible(False)
                ax.yaxis.set_major_locator(
                    ticker.MultipleLocator(max(persistence_time) / 5))
                ax.yaxis.set_minor_locator(ticker.AutoMinorLocator(2))
                if ax.is_first_col():
                    ax.set_ylabel('Extinction time (y)')
                if ax.is_first_row():
                    ax.set_title(f'SAT{SAT}')
                if model == 'chronic':
                    ax_po = axes[-1, j]
                    ax_po.plot(population_sizes, 1 - proportion_observed,
                               color=plot_common.SAT_colors[SAT],
                               clip_on=False, zorder=3)
                    ax_po.set_xscale('log')
                    ax_po.set_xlabel('Population size')
                    ax_po.xaxis.set_major_formatter(ticker.LogFormatter())
                    ax_po.yaxis.set_major_formatter(
                        ticker.PercentFormatter(xmax=1))
                    ax_po.yaxis.set_minor_locator(ticker.AutoMinorLocator(2))
                    if ax_po.is_first_col():
                        ax_po.set_ylabel('Persisting 10 y')
        for ax in fig.axes:
            ax.axvline(population_size_baseline,
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
        fig.text(label_x, 0.31, 'Chronic model', **label_kws)
        fig.savefig('population_sizes.pdf')


if __name__ == '__main__':
    df = load_extinction_times()
    # plot_survival(df)
    # plot_kde(df)
    plot_kde_2d(df)
    pyplot.show()
