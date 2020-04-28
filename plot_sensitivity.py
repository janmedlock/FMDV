#!/usr/bin/python3

from matplotlib import colors, pyplot, ticker
import numpy
import statsmodels.nonparametric.api

import plot_common
import plot_population_sizes
import plot_samples
import stats


# Nature
rc = {}
# Widths: 89mm, 183mm, 120mm, 136mm.
width = 183 / 25.4  # convert mm to in
height = 6
rc['figure.figsize'] = (width, height)
# Sans-serif, preferably Helvetica or Arial.
rc['font.family'] = 'sans-serif'
rc['font.sans-serif'] = 'DejaVu Sans'
# Between 5pt and 7pt.
rc['font.size'] = 6
rc['axes.titlesize'] = 8
rc['axes.labelsize'] = 6
rc['xtick.labelsize'] = rc['ytick.labelsize'] = 5


def load_population_sizes():
    df = plot_population_sizes.load_extinction_times()
    return df.loc['chronic']


def _get_density(endog, times):
    # Avoid errors if endog is empty.
    if len(endog) > 0:
        kde = statsmodels.nonparametric.api.KDEUnivariate(endog)
        kde.fit(cut=0)
        return kde.evaluate(times)
    else:
        return numpy.zeros_like(times)


def plot_population_sizes_(axes):
    df = load_population_sizes()
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
        for (k, (p, g)) in enumerate(group_SAT.groupby('population_size')):
            ser = g.time[g.observed]
            nruns = len(g)
            proportion_observed[k] = len(ser) / nruns
            density[:, k] = _get_density(ser, persistence_time)
        cmap = plot_population_sizes._get_cmap(plot_common.SAT_colors[SAT])
        # Use raw `density` for color,
        # but plot `density * proportion_observed`.
        norm = colors.Normalize(vmin=0, vmax=numpy.max(density))
        ax.imshow(density * proportion_observed,
                  cmap=cmap, norm=norm, interpolation='bilinear',
                  extent=(min(population_sizes), max(population_sizes),
                          min(persistence_time), max(persistence_time)),
                  aspect='auto', origin='lower', clip_on=False)
        ax.set_xscale('log')
        ax.xaxis.set_tick_params(which='both',
                                 labelbottom=False, labeltop=False)
        ax.xaxis.offsetText.set_visible(False)
        if ax.is_first_col():
            ax.set_ylabel('extinction\ntime (y)', labelpad=ylabelpad)
            ax.yaxis.set_major_locator(
                ticker.MultipleLocator(max(persistence_time) / 5))
            ax.yaxis.set_minor_locator(ticker.AutoMinorLocator(2))
        ax_po.plot(population_sizes, 1 - proportion_observed,
                   color=plot_common.SAT_colors[SAT],
                   clip_on=False, zorder=3)
        ax_po.set_xlabel('population size')
        ax_po.set_xscale('log')
        ax_po.xaxis.set_major_formatter(ticker.LogFormatter())
        ax_po.yaxis.set_major_formatter(plot_common.PercentFormatter())
        ax_po.yaxis.set_minor_locator(ticker.AutoMinorLocator(2))
        if ax_po.is_first_col():
            ax_po.set_ylabel('persisting\n10 y', labelpad=ylabelpad)
        for ax_ in {ax, ax_po}:
            ax_.axvline(population_size_baseline,
                        color='black', linestyle='dotted', alpha=0.7)
            ax_.autoscale(tight=True)
            for sp in ('top', 'right'):
                ax_.spines[sp].set_visible(False)


def load_samples():
    df = plot_samples.load_extinction_times()
    return df.loc['chronic']


def _get_prcc(df, params, outcome):
    return stats.prcc(df[params].dropna(axis='columns', how='all'),
                      df[outcome])


def plot_samples_(axes):
    df = load_samples()
    outcome = 'extinction_time'
    SATs = df.index.get_level_values('SAT').unique()
    samples = df.index.get_level_values('sample').unique()
    params = df.columns.drop([outcome, 'extinction_observed'])
    colors_ = [f'C{j}' for j in range(len(params))][::-1]
    rho = df.groupby('SAT').apply(_get_prcc, params, outcome).T
    rho.dropna(axis='index', how='all', inplace=True)
    # Sort rows on mean absolute values.
    order = rho.abs().mean(axis='columns').sort_values().index
    rho = rho.loc[order]
    xabsmax = rho.abs().max().max()
    y = range(len(rho))
    ylabels = [plot_samples.param_transforms.get(p, p).replace('_', ' ')
               for p in rho.index]
    ylabelpad = 30
    for ((SAT, rho_SAT), ax) in zip(rho.items(), axes):
        ax.barh(y, rho_SAT, height=1, left=0,
                align='center', color=colors_, edgecolor=colors_)
        ax.set_xlabel('PRCC')
        ax.set_xlim(- xabsmax, xabsmax)
        ax.set_ylim(- 0.5, len(rho) - 0.5)
        ax.xaxis.set_minor_locator(ticker.AutoMinorLocator(n=2))
        ax.yaxis.set_tick_params(which='both', left=False, right=False,
                                 pad=ylabelpad)
        if ax.is_first_col():
            ax.set_yticks(y)
            ax.set_yticklabels(ylabels, horizontalalignment='left')
        for sp in ('top', 'left', 'right'):
            ax.spines[sp].set_visible(False)



def plot():
    height_ratios = (1, 0.5, 0.25)
    SATs = (1, 2, 3)
    with pyplot.rc_context(rc):
        fig, axes = pyplot.subplots(
            3, len(SATs), sharey='row',
            gridspec_kw=dict(height_ratios=height_ratios))
        axes_samples = axes[0]
        axes_population_sizes = axes[1:]
        plot_samples_(axes_samples)
        plot_population_sizes_(axes_population_sizes)
        for (SAT, ax) in zip(SATs, axes[0]):
            ax.set_title(f'SAT{SAT}')
        fig.align_labels(axes_samples)
        fig.align_labels(axes_population_sizes)
        fig.tight_layout(pad=0.5)
        fig.savefig('plot_sensitivity.pdf')
        fig.savefig('plot_sensitivity.png', dpi=300)
    return fig


if __name__ == '__main__':
    fig = plot()
    pyplot.show()
