#!/usr/bin/python3

import numpy
import time
import pandas

import herd
 

def make_plot(data, show = True):
    from matplotlib import pyplot
    import seaborn
    from scipy import integrate

    (fig, ax) = pyplot.subplots()
    seaborn.set_palette(seaborn.color_palette('husl', 4))

    (t, x) = map(numpy.array, zip(*data))  
    # apply numpy.array to items in zip(*data).  numpy.array turns vector in array, zip(
    # zip() takes in two lists and returns them in tuples, the same length as the initials;  zip(*data) unzips
    for (j, l) in enumerate(('M', 'S', 'I', 'R')):
        ax.step(365 * t, x[:, j], where = 'post', label = l)

    ax.set_xlabel('time (days)')
    ax.set_ylabel('number')

    ax.legend()

    if show:
        pyplot.show()

def make_datasheet(data):
	(t, x) = map(numpy.array, zip(*data))
	data2 = pandas.DataFrame(data=x, index=t, columns=['M', 'S', 'I', 'R'])
	print(data2.head(5))
	data2.to_csv("testfile.csv", sep=',')
	

if __name__ == '__main__':
    numpy.random.seed(1)

    p = herd.Parameters()
    p.population_size = 1000
    p.birth_seasonal_coefficient_of_variation = 1

    tmax = 1
    debug = False

    t0 = time.time()
    data = herd.Herd(p, debug = debug).run(tmax)
    t1 = time.time()
    print('Run time: {} seconds.'.format(t1 - t0))

    #make_plot(data)
    make_datasheet(data)