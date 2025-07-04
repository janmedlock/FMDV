#!/usr/bin/python3
'''Plot infection hazards.'''

from matplotlib import pyplot

from context import herd
from herd._initial_conditions import find_hazard_infection, plot


models = ('acute', 'chronic')


def plot_hazards(show=True):
    for model in models:
        params = herd.Parameters(model=model)
        hazard_infection = find_hazard_infection(params)
        plot(hazard_infection, params, show=False, label=model.capitalize())
    pyplot.legend()
    if show:
        pyplot.show()


if __name__ == '__main__':
    plot_hazards()
