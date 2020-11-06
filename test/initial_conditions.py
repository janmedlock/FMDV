#!/usr/bin/python3

import itertools
import sys

from matplotlib import pyplot
import numpy

sys.path.append('..')
from herd import initial_conditions, Parameters, utility
sys.path.pop()


def plot_conditional_probability(ICs, ages, ax):
    '''Plot probability of being in each class vs. age,
    conditioned on being alive.'''
    p = ICs.immune_status_conditional_pdf(ages)
    ax.stackplot(ages, p.T, labels=p.columns)
    ax.set_ylabel('probability\ngiven alive')


def plot_probability(ICs, ages, ax):
    '''Plot probability of being in each class vs. age,
    *not* conditioned on being alive.'''
    p = ICs.pdf(ages)
    ax.stackplot(ages, p.T)
    ax.set_ylabel('density')


def plot_samples(ICs, ages, ax):
    '''For a sample initial condition,
    plot the number in each class vs. age.'''
    numpy.random.seed(1)  # Make `ICs.rvs()` reproducible.
    samples = ICs.rvs()
    bins = ages
    left = bins[:-1]
    width = ages[1] - ages[0]
    bottom = numpy.zeros_like(left)
    for (immune_status, ages_in_immune_status) in samples.items():
        (height, _) = numpy.histogram(ages_in_immune_status, bins=bins)
        ax.bar(left, height, width, bottom, align='edge')
        bottom += height
    ax.set_ylabel('count in\nsample')


def reorder_for_lr(items, ncol):
    return itertools.chain(*[items[i::ncol] for i in range(ncol)])


def plot_ICs(SAT):
    parameters = Parameters(SAT=SAT)
    ICs = initial_conditions.gen(parameters)
    ages = utility.arange(0, 20, 0.1, endpoint=True)
    (fig, axes) = pyplot.subplots(3, 1, sharex=True)
    plot_fcns = (plot_conditional_probability,
                 plot_probability,
                 plot_samples)
    for (ax, plot_fcn) in zip(axes, plot_fcns):
        plot_fcn(ICs, ages, ax)
    axes[-1].set_xlabel('age')
    for ax in axes:
        ax.margins(0)
    fig.align_ylabels()
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    (handles, labels) = axes[0].get_legend_handles_labels()
    nrow = 2
    ncol = (len(labels) + nrow - 1) // nrow
    fig.legend(reorder_for_lr(handles, ncol),
               reorder_for_lr(labels, ncol),
               ncol=ncol, loc='lower center')


if __name__ == '__main__':
    for SAT in (1, 2, 3):
        plot_ICs(SAT)
    pyplot.show()
