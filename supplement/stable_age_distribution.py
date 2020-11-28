#!/usr/bin/python3
'''Plot the stable age distribution.'''


import sys

from matplotlib import pyplot, ticker
import numpy

sys.path.append('..')
from herd import Parameters
import herd.age_structure
import plot_common
sys.path.pop()


def plot_age_structure():
    ages = numpy.linspace(0, 20, 301)
    parameters = Parameters()
    age_structure = herd.age_structure.gen(parameters).pdf(ages)
    width = 390 / 72.27
    height = 0.6 * width
    rc = plot_common.rc.copy()
    rc['figure.figsize'] = (width, height)
    with pyplot.rc_context(rc):
        fig = pyplot.figure(constrained_layout=True)
        ax = fig.add_subplot()
        ax.plot(ages, age_structure, color='black', clip_on=False)
        ax.autoscale(tight=True)
        ax.set_xlabel('Age (y)')
        ax.set_ylabel('Density (y$^{-1}$)')
        ax.xaxis.get_major_locator().set_params(integer=True)
        ax.xaxis.set_minor_locator(ticker.AutoMinorLocator(2))
        ax.yaxis.set_minor_locator(ticker.AutoMinorLocator(2))
        for l in ('top', 'right'):
            ax.spines[l].set_visible(False)
        fig.savefig('stable_age_distribution.pgf')
        return fig


if __name__ == '__main__':
    fig = plot_age_structure()
    pyplot.show()
