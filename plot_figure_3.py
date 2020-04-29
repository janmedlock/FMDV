#!/usr/bin/python3
#
# TODO
# * Check width of figure with diagram.

import subprocess

from matplotlib import gridspec, pyplot, ticker
import numpy
import pandas
import seaborn
import statsmodels.nonparametric.api

import plot_common
import run_common


# From `pdfinfo notes/diagram_standalone.pdf'.
diagram_width = 184.763 / 72  # inches
diagram_height = 279.456 / 72  # inches

# Nature.
rc = {}
# Widths: 89mm, 183mm, 120mm, 136mm.
total_width = 183 / 25.4  # inches
fig_width = total_width - diagram_width
fig_height = 6  # inches
rc['figure.figsize'] = (fig_width, fig_height)
# Sans-serif, preferably Helvetica or Arial.
rc['font.family'] = 'sans-serif'
rc['font.sans-serif'] = 'DejaVu Sans'
# Between 5pt and 7pt.
rc['font.size'] = 6
rc['axes.titlesize'] = 7
rc['axes.labelsize'] = 6
rc['xtick.labelsize'] = rc['ytick.labelsize'] = 5
# Separate panels in multi-part figures should be labelled with 8
# pt bold, upright (not italic) a, b, c...
# I'm gonna try to avoid this.


def load():
    infected = []
    extinction_time = []
    for model in ('acute', 'chronic'):
        i = plot_common.get_infected(model=model)
        run_common._prepend_index_levels(i, model=model)
        infected.append(i)
        e = plot_common.get_extinction_time(model=model)
        run_common._prepend_index_levels(e, model=model)
        extinction_time.append(e)
    infected = pandas.concat(infected)
    extinction_time = pandas.concat(extinction_time)
    return (infected, extinction_time)


def plot_infected(ax, infected, model, SAT):
    # .unstack('run') puts 'run' on columns, time on rows.
    i = infected.loc[(model, SAT)].unstack('run')
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
        ax.annotate(f'SAT{SAT}', (-0.35, 0.1),
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
    if not_extinct > 0:
        # 0.6 -> 0.3, 1 -> 1.
        pad = (1 - 0.3) / (1 - 0.6) * (not_extinct - 0.6) + 0.3
        bbox = dict(boxstyle=f'rarrow, pad={pad}',
                    facecolor=color, linewidth=0)
        ax.annotate('{:g}%'.format(not_extinct * 100),
                    (0.92, 0.8), xycoords='axes fraction',
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


def plot(infected, extinction_time):
    SATs = infected.index.get_level_values('SAT').unique()
    models = infected.index.get_level_values('model').unique()
    nrows = len(SATs) * 2
    ncols = len(models)
    height_ratios = (3, 1) * (nrows // 2)
    width_ratios = (1, 1)
    with seaborn.axes_style('whitegrid'), pyplot.rc_context(rc=rc):
        fig = pyplot.figure(constrained_layout=True)
        gs = gridspec.GridSpec(nrows, ncols, figure=fig,
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
                plot_infected(axes[row_i, col], infected, model, SAT)
                plot_extinction_time(axes[row_e, col], extinction_time,
                                     model, SAT)
        # Shade time region from acute-model column
        # in chronic-model column.
        col_chronic = numpy.where(models == 'chronic')[0][0]
        e_acute = extinction_time.loc['acute']
        assert e_acute.notnull().all()
        e_acute_mask = e_acute.max()
        color = pyplot.rcParams['grid.color']
        for (i, SAT) in enumerate(SATs):
            for row in range(2 * i, 2 * i + 2):
                _, margin = axes[row, col_chronic].margins()
                axes[row, col_chronic].axvspan(0, e_acute_mask * (1 + margin),
                                               color=color, alpha=0.5,
                                               linewidth=0)
        # I get weird results if I set these limits individually.
        for row in range(nrows):
            for col in range(ncols):
                axes[row, col].set_xlim(left=0)
                axes[row, col].set_ylim(bottom=0)
        seaborn.despine(fig=fig, top=True, right=True, bottom=False, left=False)
        # fig.align_labels()
        # For some reason, aligning the rows and columns works better
        # than aligning all axes.
        fig.align_xlabels(axes[-1, :])
        for (i, SAT) in enumerate(SATs):
            row_i = 2 * i
            row_e = 2 * i + 1
            fig.align_ylabels(axes[[row_i, row_e], 0])
        fig.savefig('figure_3_no_diagram.pdf')


def build():
    subprocess.check_call(['pdflatex', '--synctex=15', 'figure_3.tex'])
    subprocess.check_call(['pdftocairo', '-png', '-r', '300', '-singlefile',
                           'figure_3.pdf'])


if __name__ == '__main__':
    infected, extinction_time = load()
    plot(infected, extinction_time)
    build()
    pyplot.show()
