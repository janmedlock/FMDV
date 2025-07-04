#!/usr/bin/python3
'''Run one simulation.'''

import time

from matplotlib import pyplot

from context import herd
from context import run


def make_plot(data, show=True):
    (fig, axes) = pyplot.subplots()
    for (name, ser) in data.items():
        axes.plot(ser, label=name,
                  drawstyle='steps-pre',
                  alpha=0.9, linewidth=1)
    axes.set_xlabel(data.index.name)
    axes.set_ylabel('number')
    axes.legend(loc='center right')
    if show:
        pyplot.show()
    return fig


if __name__ == '__main__':
    SAT = 1
    MODEL = 'chronic'
    SEED = 1
    TMAX = 10
    DEBUG = False

    p = herd.Parameters(model=MODEL, SAT=SAT)
    t0 = time.time()
    data = run.run_one(p, TMAX, SEED, debug=DEBUG)
    t = time.time() - t0
    print(f'Run time: {t} seconds.')

    make_plot(data)
