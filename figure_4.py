#!/usr/bin/python3
'''Build Figure 4 from our paper. The requires the files
`population_size.h5`, which is built by `population_size_run.py`, and
`samples.h5`, which is built by `samples_run.py`.'''


from matplotlib import colors, pyplot, ticker
import numpy

import plot_common
import population_size
import samples
import stats


# Science
rc = plot_common.rc.copy()
width = 183 / 25.4  # convert mm to in
height = 6
rc['figure.figsize'] = (width, height)
# Between 5pt and 7pt.
rc['font.size'] = 6
rc['axes.titlesize'] = 8
rc['axes.labelsize'] = 6
rc['xtick.labelsize'] = rc['ytick.labelsize'] = 5


def load_population_size():
    df = population_size.load()
    return df.loc['chronic']


def plot_sensitivity_population_sizes(axes):
    df = load_population_size()
    population_sizes = (df.index
                          .get_level_values('population_size')
                          .unique()
                          .sort_values())
    population_size_baseline = 1000
    persistence_time_max = 10
    persistence_time = numpy.linspace(0, persistence_time_max, 301)
    ylabelpad = 0
    for (j, (SAT, group_SAT)) in enumerate(df.groupby('SAT')):
        (ax, ax_po) = axes[:, j]
        density = numpy.zeros((len(persistence_time),
                               len(population_sizes)))
        proportion_observed = numpy.zeros_like(population_sizes,
                                               dtype=float)
        grouper = group_SAT.groupby('population_size')
        for (k, (p, g)) in enumerate(grouper):
            ser = g.time[g.observed]
            nruns = len(g)
            proportion_observed[k] = len(ser) / nruns
            density[:, k] = plot_common.get_density(ser, persistence_time)
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
            ax.set_ylabel('Extinction time (y)', labelpad=ylabelpad)
        ax_po.plot(population_sizes, 1 - proportion_observed,
                   color=plot_common.SAT_colors[SAT],
                   clip_on=False, zorder=3)
        ax_po.set_xscale('log')
        ax_po.set_xlabel('Population size', labelpad=1.5)
        ax_po.xaxis.set_major_formatter(ticker.LogFormatter())
        ax_po.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1))
        ax_po.yaxis.set_minor_locator(ticker.AutoMinorLocator(2))
        if ax_po.is_first_col():
            ax_po.set_ylabel('Persisting 10 y', labelpad=ylabelpad)
        for ax_ in {ax, ax_po}:
            ax_.axvline(population_size_baseline,
                        color='black', linestyle='dotted', alpha=0.7)
            ax_.autoscale(tight=True)
            if not ax_.is_first_col():
                ax_.yaxis.set_tick_params(which='both',
                                          labelleft=False, labelright=False)
                ax_.yaxis.offsetText.set_visible(False)
            for sp in ('top', 'right'):
                ax_.spines[sp].set_visible(False)


def load_samples():
    df = samples.load()
    return df.loc['chronic']


def _get_prcc(df, params, outcome):
    return stats.prcc(df[params].dropna(axis='columns', how='all'),
                      df[outcome])


def plot_sensitivity_samples(axes):
    df = load_samples()
    outcome = 'time'
    params = df.columns.drop([outcome, 'observed'])
    colors_ = [f'C{j}' for j in range(len(params))][::-1]
    rho = df.groupby('SAT').apply(_get_prcc, params, outcome).T
    rho.dropna(axis='index', how='all', inplace=True)
    # Sort rows on mean absolute values.
    order = rho.abs().mean(axis='columns').sort_values().index
    rho = rho.loc[order]
    xabsmax = rho.abs().max().max()
    y = range(len(rho))
    ylabels = [samples.param_transforms.get(p, p)
                                       .capitalize()
                                       .replace('_', ' ')
               for p in rho.index]
    for ((SAT, rho_SAT), ax) in zip(rho.items(), axes):
        ax.barh(y, rho_SAT, height=1, left=0,
                align='center', color=colors_, edgecolor=colors_)
        ax.set_xlabel('PRCC')
        ax.set_xlim(- xabsmax, xabsmax)
        ax.set_ylim(- 0.5, len(rho) - 0.5)
        ax.xaxis.set_minor_locator(ticker.AutoMinorLocator(2))
        ax.yaxis.set_tick_params(which='both', left=False, right=False,
                                 pad=25)
        if ax.is_first_col():
            ax.set_yticks(y)
            ax.set_yticklabels(ylabels, horizontalalignment='left')
        else:
            ax.yaxis.set_tick_params(which='both',
                                     labelleft=False, labelright=False)
            ax.yaxis.offsetText.set_visible(False)
        for sp in ('top', 'left', 'right'):
            ax.spines[sp].set_visible(False)


def plot():
    SATs = (1, 2, 3)
    height_ratios = (0.5, 0.25, 1)
    nrows = len(height_ratios)
    ncols = len(SATs)
    with pyplot.rc_context(rc):
        fig = pyplot.figure(constrained_layout=True)
        gs = fig.add_gridspec(nrows, ncols,
                              height_ratios=height_ratios)
        axes = numpy.empty((nrows, ncols), dtype=object)
        axes[0, 0] = None  # Make sharex & sharey work for axes[0, 0].
        for row in range(nrows):
            for col in range(ncols):
                # The population size plots share the x scale.
                # The sample plots do *not* share the x scale.
                sharex = axes[0, col] if (row < 2) else None
                # Rows share the y scale.
                sharey = axes[row, 0]
                axes[row, col] = fig.add_subplot(gs[row, col],
                                                 sharex=sharex,
                                                 sharey=sharey)
        axes_population_sizes = axes[:-1]
        axes_samples = axes[-1]
        plot_sensitivity_population_sizes(axes_population_sizes)
        plot_sensitivity_samples(axes_samples)
        for (SAT, ax) in zip(SATs, axes[0]):
            ax.set_title(f'SAT{SAT}')
        fig.align_ylabels(axes_population_sizes)
        label_x = 0
        # Separate panels in multi-part figures should be labelled with 8
        # pt bold, upright (not italic) a, b, c...
        label_kws = dict(fontsize=8,
                         fontweight='bold',
                         horizontalalignment='left',
                         verticalalignment='top')
        fig.text(label_x, 1, 'a', **label_kws)
        fig.text(label_x, 0.535, 'b', **label_kws)
        fig.savefig('figure_4.pdf')
        fig.savefig('figure_4.png', dpi=300)
    return fig


if __name__ == '__main__':
    fig = plot()
    pyplot.show()
