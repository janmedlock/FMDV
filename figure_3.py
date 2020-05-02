#!/usr/bin/python3
'''This script builds 'figure_3.pdf' and 'figure_3.png' by plotting
 simulation data into 'figure_3_nodiagram.pdf' with `plot()`, then
 combining that with 'diagram/diagram.tex' through 'figure_3.tex'
in `build()`.'''


import subprocess

from matplotlib import pyplot, ticker
import numpy
import seaborn
import statsmodels.nonparametric.api

import plot_common



# Nature.
rc = plot_common.rc.copy()
total_width = 183 / 25.4  # inches
# 184.983 pts is from `pdfinfo diagram/diagram_standalone.pdf'.
# 1.4 is the scaling in figure_3.tex.
diagram_width = 184.983 * 1.4 / 72  # inches
fig_width = total_width - diagram_width
# There's whitespace left...
fig_width *= 1.13
fig_height = 6  # inches
rc['figure.figsize'] = (fig_width, fig_height)
# Between 5pt and 7pt.
rc['font.size'] = 6
rc['axes.titlesize'] = 7
rc['axes.labelsize'] = 6
rc['xtick.labelsize'] = rc['ytick.labelsize'] = 5


def load():
    infected = plot_common.get_infected()
    extinction_time = plot_common.get_extinction_time()
    # Convert from years to days.
    i = infected.index.names.index('time (y)')
    times = infected.index.levels[i] * 365
    infected.index.set_levels(times, level='time (y)', inplace=True)
    infected.rename_axis(index={'time (y)': 'time (d)'}, inplace=True)
    extinction_time *= 365
    return (infected, extinction_time)


def plot_infected(ax, infected, model, SAT, draft=False):
    # .unstack('run') puts 'run' on columns, time on rows.
    i = infected.loc[(model, SAT)].unstack('run')
    if draft:
        # Only plot the first 100 runs for speed.
        i = i.iloc[:, :100]
    # Start time at 0.
    t = i.index - i.index.min()
    ax.plot(t, i, color=plot_common.SAT_colors[SAT],
            alpha=0.15, linewidth=0.5, drawstyle='steps-pre')
    # `i.fillna(0)` gives mean including those that
    # have gone extinct.
    ax.plot(t, i.fillna(0).mean(axis='columns'), color='black',
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
        ax.annotate(f'SAT{SAT}', (-0.35, 0.05),
                    xycoords='axes fraction',
                    rotation=90, fontsize=rc['axes.titlesize'])
    else:
        ax.yaxis.set_tick_params(which='both',
                                 labelleft=False, labelright=False)
        ax.yaxis.offsetText.set_visible(False)
    if ax.is_first_row():
        ax.set_title(f'{model.capitalize()} model', loc='center')


def kdeplot(endog, ax=None, shade=False, cut=0, **kwds):
    if ax is None:
        ax = pyplot.gca()
    endog = endog.dropna()
    if len(endog) > 0:
        kde = statsmodels.nonparametric.api.KDEUnivariate(endog)
        kde.fit(cut=cut)
        x = numpy.linspace(kde.support.min(), kde.support.max(), 301)
        y = kde.evaluate(x)
        line, = ax.plot(x, y, **kwds)
        if shade:
            shade_kws = dict(
                facecolor = kwds.get('facecolor', line.get_color()),
                alpha=kwds.get('alpha', 0.25),
                clip_on=kwds.get('clip_on', True),
                zorder=kwds.get('zorder', 1))
            ax.fill_between(x, 0, y, **shade_kws)
    return ax


def plot_extinction_time(ax, extinction_time, model, SAT):
    e = extinction_time.loc[(model, SAT)]
    color = plot_common.SAT_colors[SAT]
    kdeplot(e.dropna(), ax=ax, color=color, shade=True)
    not_extinct = len(e[e.isnull()]) / len(e)
    arrow_loc = {'chronic': {1: (0.92, 0.65),
                             3: (0.96, 0.8)}}
    if not_extinct > 0:
        # 0.6 -> 0.3, 1 -> 1.
        pad = (1 - 0.3) / (1 - 0.6) * (not_extinct - 0.6) + 0.3
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
    SATs = infected.index.get_level_values('SAT').unique()
    models = infected.index.get_level_values('model').unique()
    nrows = len(SATs) * 2
    ncols = len(models)
    height_ratios = (3, 1) * (nrows // 2)
    width_ratios = (1, 1)
    with seaborn.axes_style('whitegrid'), pyplot.rc_context(rc=rc):
        fig = pyplot.figure(constrained_layout=True)
        gs = fig.add_gridspec(nrows, ncols,
                              height_ratios=height_ratios,
                              width_ratios=width_ratios)
        axes = numpy.empty((nrows, ncols), dtype=object)
        axes[0, 0] = None  # Make sharex & sharey work for axes[0, 0].
        for row in range(nrows):
            for col in range(ncols):
                # Columns share the x scale.
                sharex = axes[0, col]
                # The infection plots share the y scale.
                # The extinction-time plots do *not* share the y scale.
                sharey = axes[0, 0] if (row % 2 == 0) else None
                axes[row, col] = fig.add_subplot(gs[row, col],
                                                 sharex=sharex,
                                                 sharey=sharey)
        for (i, SAT) in enumerate(SATs):
            for (col, model) in enumerate(models):
                row_i = 2 * i
                row_e = 2 * i + 1
                plot_infected(axes[row_i, col], infected, model, SAT,
                              draft=draft)
                plot_extinction_time(axes[row_e, col], extinction_time,
                                     model, SAT)
        # Shade time region from acute-model column
        # in chronic-model column.
        col_chronic = numpy.where(models == 'chronic')[0][0]
        e_acute = extinction_time.loc['acute']
        assert e_acute.notnull().all()
        e_acute_mask = e_acute.max(level='SAT')
        color = pyplot.rcParams['grid.color']
        for (i, SAT) in enumerate(SATs):
            for row in range(2 * i, 2 * i + 2):
                ax = axes[row, col_chronic]
                ax.axvspan(0, e_acute_mask[SAT],
                           color=color, alpha=0.5, linewidth=0)
        # I get weird results if I set these limits individually.
        for (i, SAT) in enumerate(SATs):
            for (col, model) in enumerate(models):
                for row in (2 * i, 2 * i + 1):
                    ax = axes[row, col]
                    ax.set_xlim(left=0)
                    ax.set_ylim(bottom=0)
        seaborn.despine(fig=fig, top=True, right=True, bottom=False, left=False)
        # For some reason, aligning the rows and columns works better
        # than aligning all axes.
        fig.align_xlabels(axes[-1, :])
        for (i, SAT) in enumerate(SATs):
            row_i = 2 * i
            row_e = 2 * i + 1
            fig.align_ylabels(axes[[row_i, row_e], 0])
        fig.savefig('figure_3_no_diagram.pdf')


def build(draft=False):
    # Build PDF super-figure.
    subprocess.check_call(['latexmk', '-pdf', 'figure_3'])
    if not draft:
        # Clean up build files.
        subprocess.check_call(['latexmk', '-c', 'figure_3'])
        # Convert PDF to 300dpi PNG.
        subprocess.check_call(['pdftocairo', '-png', '-r', '300',
                               '-singlefile', 'figure_3.pdf'])


if __name__ == '__main__':
    draft = False
    infected, extinction_time = load()
    plot(infected, extinction_time, draft=draft)
    build(draft=draft)
    if not draft:
        pyplot.show()
