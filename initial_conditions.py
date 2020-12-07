#!/usr/bin/python3
'''Analyze and plot the results of varying the initial conditions.
the simluations. This requires the file `initial_conditions.h5`, which
is built by `initial_conditions_run.py`.'''


from matplotlib import pyplot, ticker
import numpy
import seaborn

import plot_common


def load():
    filename = 'initial_conditions.h5'
    infected = plot_common.get_infected(filename)
    extinction_time = plot_common.get_extinction_time(filename)
    return (infected, extinction_time)


def plot_infected(ax, infected, model, SAT, IC, draft=False):
    # .unstack('run') puts 'run' on columns, time on rows.
    i = infected.loc[(model, SAT, IC)].unstack('run')
    if draft:
        # Only plot the first 100 runs for speed.
        i = i.iloc[:, :100]
    # Start time at 0.
    t = i.index - i.index.min()
    ax.plot(365 * t, i, color=plot_common.SAT_colors[SAT],
            alpha=0.15, linewidth=0.5, drawstyle='steps-pre')
    # `i.fillna(0)` gives mean including those that
    # have gone extinct.
    ax.plot(365 * t, i.fillna(0).mean(axis='columns'), color='black',
            alpha=1)
    # Tighten y-axis limits.
    ax.margins(y=0)
    # Shared x-axis with extinction time.
    ax.xaxis.set_tick_params(which='both',
                             labelbottom=False, labeltop=False)
    ax.xaxis.offsetText.set_visible(False)
    # Shared y-axis between models.
    if ax.is_first_col():
        ax.set_ylabel('Number\ninfected')
        ax.annotate(f'SAT{SAT}', (-0.5, 0),
                    xycoords='axes fraction',
                    rotation=90,
                    fontsize=pyplot.rcParams['axes.titlesize'])
    else:
        ax.yaxis.set_tick_params(which='both',
                                 labelleft=False, labelright=False)
        ax.yaxis.offsetText.set_visible(False)
    if ax.is_first_row():
        ax.set_title(f'Initial condition {IC}', loc='center')


def plot_extinction_time(ax, extinction_time, model, SAT, IC):
    et = extinction_time.loc[(model, SAT, IC)]
    e = et.time.copy()
    e[~et.observed] = numpy.nan
    color = plot_common.SAT_colors[SAT]
    plot_common.kdeplot(365 * e.dropna(), ax=ax, color=color, shade=True)
    not_extinct = len(e[e.isnull()]) / len(e)
    arrow_loc = {'chronic': {1: (0.9, 0.5),
                             2: (0.96, 0.65),
                             3: (0.92, 0.9)}}
    if not_extinct > 0:
        (ne_min, p_min) = (0.001, 0.1)
        (ne_max, p_max) = (1, 0.5)
        pad = ((p_max - p_min) / (ne_max - ne_min) * (not_extinct - ne_min)
               + p_min)
        bbox = dict(boxstyle=f'rarrow, pad={pad}',
                    facecolor=color, linewidth=0)
        ax.annotate('{:g}%'.format(not_extinct * 100),
                    arrow_loc[model][SAT], xycoords='axes fraction',
                    bbox=bbox, color='white',
                    verticalalignment='bottom',
                    horizontalalignment='right')
    # No y ticks.
    ax.yaxis.set_major_locator(ticker.NullLocator())
    # Shared x-axes between SATs.
    if ax.is_last_row():
        ax.set_xlabel('Time (d)')
    else:
        ax.xaxis.set_tick_params(which='both',
                                 labelbottom=False, labeltop=False)
        ax.xaxis.offsetText.set_visible(False)
    # Shared y-axis between models.
    if ax.is_first_col():
        ax.set_ylabel('Extinction\ntime')


def plot(infected, extinction_time, draft=False):
    models = infected.index.get_level_values('model').unique()
    SATs = infected.index.get_level_values('SAT').unique()
    ICs = infected.index.get_level_values('initial_conditions').unique()
    width = 390 / 72.27
    height = 4.4
    rc = plot_common.rc.copy()
    rc['figure.figsize'] = (width, height)
    rc['xtick.labelsize'] = rc['ytick.labelsize'] = 7
    rc['axes.labelsize'] = 8
    rc['axes.titlesize'] = 9
    nrows = len(SATs) * 2
    ncols = len(ICs)
    height_ratios = (3, 1) * (nrows // 2)
    width_ratios = (1, ) * ncols
    with seaborn.axes_style('whitegrid'), pyplot.rc_context(rc=rc):
        # Common upper y limit for infected plots.
        infected_max = (infected.max()
                        * (1 + pyplot.rcParams['axes.ymargin']))
        for model in models:
            fig = pyplot.figure(constrained_layout=True)
            gs = fig.add_gridspec(nrows, ncols,
                                  height_ratios=height_ratios,
                                  width_ratios=width_ratios)
            axes = numpy.empty((nrows, ncols), dtype=object)
            axes[0, 0] = None  # Make sharex & sharey work for axes[0, 0].
            for row in range(nrows):
                for (col, IC) in enumerate(ICs):
                    # Share the x scale.
                    sharex = axes[0, 0]
                    # The infection plots share the y scale.
                    # The extinction-time plots do *not* share the y scale.
                    sharey = axes[0, 0] if (row % 2 == 0) else None
                    axes[row, col] = fig.add_subplot(gs[row, col],
                                                     sharex=sharex,
                                                     sharey=sharey)
            for (i, SAT) in enumerate(SATs):
                row_i = 2 * i
                row_e = 2 * i + 1
                for (col, IC) in enumerate(ICs):
                    plot_infected(axes[row_i, col], infected, model, SAT, IC,
                                  draft=draft)
                    plot_extinction_time(axes[row_e, col], extinction_time,
                                         model, SAT, IC)
            # Shade time region from plot of acute model
            # in plot of chronic model.
            if model == 'chronic':
                e_acute = extinction_time.loc['acute'].time
                assert e_acute.notnull().all()
                e_acute_mask = e_acute.max(level=['SAT', 'initial_conditions'])
                color = pyplot.rcParams['grid.color']
                for (i, SAT) in enumerate(SATs):
                    for row in range(2 * i, 2 * i + 2):
                        for (col, IC) in enumerate(ICs):
                            ax = axes[row, col]
                            ax.axvspan(0, 365 * e_acute_mask[(SAT, IC)],
                                       color=color, alpha=0.5, linewidth=0)
            # I get weird results if I set these limits individually.
            for (i, SAT) in enumerate(SATs):
                for row in (2 * i, 2 * i + 1):
                    for col in range(ncols):
                        ax = axes[row, col]
                        ax.set_xlim(left=0)
                        ax.set_ylim(bottom=0)
                        unit = {'acute': 50, 'chronic': 1000}[model]
                        ax.xaxis.set_major_locator(
                            ticker.MultipleLocator(unit))
                        if row == 2 * i:
                            ax.set_ylim(top=infected_max)
                            ax.yaxis.set_major_locator(
                                ticker.MultipleLocator(50))
            seaborn.despine(fig=fig, top=True, right=True,
                            bottom=False, left=False)
            # For some reason, aligning the rows and columns works better
            # than aligning all axes.
            fig.align_xlabels(axes[-1, :])
            for (i, SAT) in enumerate(SATs):
                row_i = 2 * i
                row_e = 2 * i + 1
                fig.align_ylabels(axes[[row_i, row_e], 0])
            fig.savefig(f'initial_conditions_{model}.pdf')


if __name__ == '__main__':
    draft = False
    infected, extinction_time = load()
    plot(infected, extinction_time, draft=draft)
    pyplot.show()
