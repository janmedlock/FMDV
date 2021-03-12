#!/usr/bin/python3
'''Build Figure 3 from our paper. This requires the file `run.h5`,
which is built by `run.py`.'''


from matplotlib import pyplot, ticker
import numpy
import seaborn

import plot_common


# Science
rc = plot_common.rc.copy()
width = 183 / 25.4  # convert mm to in
height = 6  # in
rc['figure.figsize'] = (width, height)
# Between 5pt and 7pt.
rc['font.size'] = 6
rc['axes.titlesize'] = 9
rc['axes.labelsize'] = 8
rc['xtick.labelsize'] = rc['ytick.labelsize'] = 7


def load():
    filename = 'run.h5'
    infected = plot_common.get_infected(filename)
    extinction_time = plot_common.get_extinction_time(filename)
    return (infected, extinction_time)


def plot_infected(ax, infected, model, SAT, draft=False):
    # .unstack('run') puts 'run' on columns, time on rows.
    i = infected.loc[(model, SAT)].unstack('run')
    if draft:
        # Only plot the first 100 runs for speed.
        i = i.iloc[:, :100]
    # Start time at 0.
    t = 365 * (i.index - i.index.min())
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
        ax.annotate(f'SAT{SAT}', (-0.25, 0.05),
                    xycoords='axes fraction',
                    rotation=90, fontsize=rc['axes.titlesize'])
    else:
        ax.yaxis.set_tick_params(which='both',
                                 labelleft=False, labelright=False)
        ax.yaxis.offsetText.set_visible(False)
    if ax.is_first_row():
        model_ = {'chronic': 'carrier'}.get(model, model)
        ax.set_title(f'{model_.capitalize()} model', loc='center')


def plot_extinction_time(ax, extinction_time, model, SAT):
    et = extinction_time.loc[(model, SAT)]
    e = et.time.copy()
    e[~et.observed] = numpy.nan
    color = plot_common.SAT_colors[SAT]
    plot_common.kdeplot(365 * e.dropna(), ax=ax, color=color, shade=True)
    not_extinct = len(e[e.isnull()]) / len(e)
    arrow_loc = {'chronic': {1: (0.96, 0.65),
                             3: (0.99, 0.8)}}
    if not_extinct > 0:
        (ne_min, p_min) = (0.6, 0.3)
        (ne_max, p_max) = (1, 1)
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
            row_i = 2 * i
            row_e = 2 * i + 1
            for (col, model) in enumerate(models):
                plot_infected(axes[row_i, col], infected, model, SAT,
                              draft=draft)
                plot_extinction_time(axes[row_e, col], extinction_time,
                                     model, SAT)
        # Shade time region from acute-model column
        # in chronic-model column.
        col_chronic = numpy.where(models == 'chronic')[0][0]
        e_acute = extinction_time.loc['acute'].time
        assert e_acute.notnull().all()
        e_acute_mask = e_acute.max(level='SAT')
        color = pyplot.rcParams['grid.color']
        for (i, SAT) in enumerate(SATs):
            for row in range(2 * i, 2 * i + 2):
                ax = axes[row, col_chronic]
                ax.axvspan(0, 365 * e_acute_mask[SAT],
                           color=color, alpha=0.5, linewidth=0)
        # I get weird results if I set these limits individually.
        for (i, SAT) in enumerate(SATs):
            for row in (2 * i, 2 * i + 1):
                for (col, model) in enumerate(models):
                    ax = axes[row, col]
                    ax.set_xlim(left=0)
                    ax.set_ylim(bottom=0)
                    unit = {'acute': 50, 'chronic': 1000}[model]
                    ax.xaxis.set_major_locator(ticker.MultipleLocator(unit))
                    if row == 2 * i:
                        ax.yaxis.set_major_locator(ticker.MultipleLocator(100))
        seaborn.despine(fig=fig, top=True, right=True,
                        bottom=False, left=False)
        # For some reason, aligning the rows and columns works better
        # than aligning all axes.
        fig.align_xlabels(axes[-1, :])
        for (i, SAT) in enumerate(SATs):
            row_i = 2 * i
            row_e = 2 * i + 1
            fig.align_ylabels(axes[[row_i, row_e], 0])
        # Separate panels in multi-part figures should be labelled with 8
        # pt bold, upright (not italic) a, b, c...
        label_kws = dict(fontsize=8,
                         fontweight='bold',
                         horizontalalignment='left',
                         verticalalignment='top')
        label_y = 1
        fig.text(0, label_y, 'a', **label_kws)
        fig.text(0.555, label_y, 'b', **label_kws)
        fig.savefig('figure_3.pdf')
        fig.savefig('figure_3.png', dpi=300)


if __name__ == '__main__':
    draft = False
    infected, extinction_time = load()
    plot(infected, extinction_time, draft=draft)
    pyplot.show()
