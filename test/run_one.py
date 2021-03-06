#!/usr/bin/python3

import sys
import time

from matplotlib import pyplot
import seaborn

sys.path.append('..')
import herd
import run
sys.path.pop()


def make_plot(data, show=True):
    (fig, ax) = pyplot.subplots()
    seaborn.set_palette(seaborn.color_palette('deep', 6))
    for (k, x) in data.items():
        ax.step(365 * x.index, x, where='post', label=k)
    ax.set_xlabel(data.index.name)
    ax.set_ylabel('number')
    ax.legend()
    if show:
        pyplot.show()


if __name__ == '__main__':
    SAT = 1
    model = 'chronic'
    seed = 1
    tmax = 10
    debug = False

    p = herd.Parameters(model=model, SAT=SAT)
    t0 = time.time()
    data = run.run_one(p, tmax, seed, debug=debug)
    t1 = time.time()
    print('Run time: {} seconds.'.format(t1 - t0))

    make_plot(data)
