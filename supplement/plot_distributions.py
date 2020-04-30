#!/usr/bin/python3
import sys

import numpy
from matplotlib import lines, pyplot, ticker

sys.path.append('..')
from herd import Parameters, RandomVariables
import plot_common
sys.path.pop()


# The of these determines the order of the legend.
SATs = ('all', 1, 2, 3)

colors = {SAT: 'black' if (SAT == 'all') else plot_common.SAT_colors[SAT]
          for SAT in SATs}
labels = {SAT: 'All' if (SAT == 'all') else f'SAT{SAT}'
          for SAT in SATs}

# For 'all', use the first non-'all' SAT.
SAT_map = ((SAT, SATs[1] if (SAT == 'all') else SAT)
           for SAT in SATs)
RVs = {SAT: RandomVariables(Parameters(SAT=v))
       for (SAT, v) in SAT_map}

# Common to all SATs.
common = ('mortality', 'birth', 'maternal_immunity_waning')


def get_RV(RVs, name):
    if name in common:
        which = ('all', )
    else:
        which = (SAT for SAT in SATs if (SAT != 'all'))
    return {SAT: getattr(RVs[SAT], name) for SAT in which}


width = 390 / 72.27
height = 0.6 * width
rc = plot_common.rc.copy()
rc['figure.figsize'] = (width, height)
rc['font.size'] = 7
rc['axes.titlesize'] = 'large'
ncols = 6
nrows = 2
with pyplot.rc_context(rc=rc):
    fig = pyplot.figure(constrained_layout=True)
    # Add a dummy row to make space for the legend.
    height_ratios = (1, 1, 0.13)
    gs = fig.add_gridspec(nrows + 1, ncols,
                          height_ratios=height_ratios)
    axes = numpy.empty((nrows, ncols), dtype=object)
    axes[0, 0] = None  # Make sharex & sharey work for axes[0, 0].
    for row in range(nrows):
        for col in range(ncols):
            # Columns share the x scale.
            sharex = axes[0, col]
            axes[row, col] = fig.add_subplot(gs[row, col],
                                             sharex=sharex)
    axes_hazards = axes[0]
    axes_survivals = axes[1]
    j = -1

    RV = get_RV(RVs, 'mortality')
    title = 'Death'
    xlabel = 'Age ($\mathrm{y}$)'
    t_max = 20
    t = numpy.linspace(0, t_max, 1001)
    j += 1
    for SAT, v in RV.items():
        axes_hazards[j].plot(t, v.hazard(t), color=colors[SAT],
                             drawstyle='steps-post')
        axes_survivals[j].plot(t, v.sf(t), color=colors[SAT])
    axes_hazards[j].set_title(title)
    axes_survivals[j].set_xlabel(xlabel)

    RV = get_RV(RVs, 'birth')
    title = 'Birth'
    xlabel = 'Time ($\mathrm{y}$)'
    t_max = 3
    t = numpy.linspace(0, t_max, 1001)
    j += 1
    for SAT, v in RV.items():
        axes_hazards[j].plot(t, v.hazard(t, 4 + t), color=colors[SAT])
        axes_survivals[j].plot(t, v.sf(t, 0, 4), color=colors[SAT])
    axes_hazards[j].set_title(title)
    axes_survivals[j].set_xlabel(xlabel)

    RV = get_RV(RVs, 'maternal_immunity_waning')
    title = 'Waning'
    xlabel = 'Age ($\mathrm{y}$)'
    t_max = 1
    t = numpy.linspace(0, t_max, 1001)
    j += 1
    for SAT, v in RV.items():
        axes_hazards[j].plot(t, v.hazard(t), color=colors[SAT])
        axes_survivals[j].plot(t, v.sf(t), color=colors[SAT])
    axes_hazards[j].set_title(title)
    axes_survivals[j].set_xlabel(xlabel)

    RV = get_RV(RVs, 'progression')
    title = 'Progression'
    xlabel = 'Time ($\mathrm{d}$)'
    t_max = 10
    t = numpy.linspace(0, t_max, 1001)
    j += 1
    for SAT, v in RV.items():
        axes_hazards[j].plot(t, v.hazard(t / 365), color=colors[SAT])
        axes_survivals[j].plot(t, v.sf(t / 365), color=colors[SAT])
    axes_hazards[j].set_title(title)
    axes_survivals[j].set_xlabel(xlabel)

    RV = get_RV(RVs, 'recovery')
    title = 'Recovery'
    xlabel = 'Time ($\mathrm{d}$)'
    t_max = 15
    t = numpy.linspace(0, t_max, 1001)
    j += 1
    for SAT, v in RV.items():
        axes_hazards[j].plot(t, v.hazard(t / 365), color=colors[SAT])
        axes_survivals[j].plot(t, v.sf(t / 365), color=colors[SAT])
    axes_hazards[j].set_title(title)
    axes_survivals[j].set_xlabel(xlabel)

    RV = get_RV(RVs, 'chronic_recovery')
    title = 'Chronic \nrecovery'
    xlabel = 'Time ($\mathrm{y}$)'
    t_max = 1
    t = numpy.linspace(0, t_max, 1001)
    j += 1
    for SAT, v in RV.items():
        axes_hazards[j].plot(t, v.hazard(t), color=colors[SAT])
        axes_survivals[j].plot(t, v.sf(t), color=colors[SAT])
    axes_hazards[j].set_title(title)
    axes_survivals[j].set_xlabel(xlabel)

    for ax in axes.flat:
        for l in ('top', 'right'):
            ax.spines[l].set_visible(False)
        ax.autoscale(tight=True)
        ax.xaxis.set_minor_locator(ticker.AutoMinorLocator(2))
        ax.yaxis.set_minor_locator(ticker.AutoMinorLocator(2))
        for line in ax.lines:
            line.set_clip_on(False)

    for ax in axes_hazards:
        ax.set_ylim(bottom=0)
        ax.yaxis.set_major_formatter(ticker.StrMethodFormatter('{x:g}'))
        # axes_hazards shares the xaxis with axes_survivals.
        ax.xaxis.set_tick_params(which='both',
                                 labelbottom=False, labeltop=False)
        ax.xaxis.offsetText.set_visible(False)
    for ax in axes_survivals:
        ax.set_ylim(0, 1)
        ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1))

    axes_hazards[0].set_ylabel(r'Hazard ($\mathrm{y}^{-1}$)')
    axes_survivals[0].set_ylabel('Survival')

    handles = [lines.Line2D([], [], color=color, label=labels[SAT])
               for (SAT, color) in colors.items()]
    fig.legend(handles=handles, markerfirst=False, loc='lower center',
               ncol=len(handles))

    fig.align_labels()

    fig.savefig('distributions.pgf')
    pyplot.show()
