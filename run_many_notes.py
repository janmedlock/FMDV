#!/usr/bin/python3

import numpy
import functools
import multiprocessing
import pandas


import herd


def run_one(parameters, tmax, *args, **kwds):
	# *args does ???  Use *args when you are not sure how many arguments might be passed to your function.
	# **kwds is similar in that it allows you to handle named arguments that you haven't defined in advance. 
	# Note: the syntax is * and **; args and kwargs are only convention, could use *mylist for example.
	
    '''
    Run one simulation, among multiple running in parallel.
    run_number is at the end of *args to allow easy use of
    multiprocessing.Pool().map() with functools.partial().
    '''
    run_number = args[-1]
    args_other = args[ : -1]
    return herd.Herd(parameters, *args_other,
                     run_number = run_number, **kwds).run(tmax)

# THIS FUNCTION CALLED IN MAIN as: data = run_many(nruns, p, tmax, debug = debug);
# where nruns=100, p= parameters defined in parameters.py 
def run_many(nruns, parameters, tmax, *args, **kwds):
    'Run many simulations in parallel.'

    # Build the RVs once to make sure the caches are seeded.
    rvs = herd.RandomVariables(parameters)

    # Set the random seed on each worker to a different seed.
    # numpy.random.seed() with no argument uses /dev/urandom to seed.
    with multiprocessing.Pool(initializer = numpy.random.seed) as pool:
        # f is _doOne with all the args and kwargs set, except runNumber,
        # which will be appended to args.
        f = functools.partial(run_one, parameters, tmax, *args, **kwds)
        # functools.partial allows you to call the function (run_one) with the positional arguments and keywords arguments. 
        

        # f(0), f(1), f(2), ..., f(nruns - 1),
        # computed in parallel.
        return pool.map(f, range(nruns))
        # recall map is an apply function, here you are applying run_one (via the functools package)
        # so appy run_one nruns times; use pool. when using multiprocessing; so splits the apply function between cores. 


def get_mean(T, X):
    T_mean = numpy.unique(numpy.hstack(T))
    X_mean = numpy.zeros((len(T_mean), len(X[0][0])))
    n = numpy.zeros_like(T_mean)
    for (Tk, Xk) in zip(T, X):
        Tk = numpy.array(Tk)
        Xk = numpy.array(Xk)

        # Only go to the end of this simulation.
        T_ = T_mean.compress(T_mean <= Tk[-1])

        # Find the indicies i[j] of the largest Tk with Tk[i[j]] <= T_[j]
        indices = [(Tk <= t).nonzero()[0][-1] for t in T_]

        X_mean[ : len(T_)] += Xk[indices]
        n[ : len(T_)] += 1
    X_mean /= n[:, numpy.newaxis]

    return (T_mean, X_mean)

def make_datasheet(data):
    (T, X) = zip(*(zip(*d) for d in data))
    index = 0
    appended_data = []
    # need to turn T, X into an array before it can be converted into a datasheet
    for (t, x) in zip(T, X):
        t = numpy.array(t)
        x = numpy.array(x)
        # Add column for total and index, append each
        n = x.sum(-1)
        rep = numpy.array([index]* len(t))
        x = numpy.column_stack((x, n, rep))
        print(x.shape)   # generates num_interations arrays containing 5 columns and time dependent number of rows. 
        data = pandas.DataFrame(data=x, index=t, columns=['M', 'S', 'I', 'R', 'Total', 'Rep'])
        appended_data.append(data)  # stores dataframes in a list
        index += 1
    final_data = pandas.concat(appended_data)  
        
    #data = pandas.DataFrame(data=x, index=t, columns=['M', 'S', 'I', 'R', 'Total', 'Rep'])  ##print(data2.head(5))
    final_data.to_csv("testfile_manyruns.csv", sep=',')


def make_plots(data, show = True):
    from matplotlib import pyplot
    import seaborn
    import itertools
    from scipy import integrate

    from herd import pde

    (fig, ax) = pyplot.subplots(5, sharex = True)
    colors = itertools.cycle(seaborn.color_palette('husl', 8))

    (T, X) = zip(*(zip(*d) for d in data))
    
    # Recall zip takes to lists and makes them into 1 list of tuples... *zip is the reverse???
    # Recall use * (as in *args) when unsure how many arguments will get passed (called a 'splat' operator
    
    for (t, x) in zip(T, X):
        c = next(colors)
        t = numpy.array(t)
        x = numpy.array(x)
        # Add column for total.
        n = x.sum(-1)
        x = numpy.column_stack((x, n))
        for j in range(x.shape[-1]):
            ax[j].step(365 * t, x[:, j], where = 'post',
                       color = c, alpha = 0.5)

    (T_mean, X_mean) = get_mean(T, X)
    # Add column for total.
    N_mean = X_mean.sum(-1)
    X_mean = numpy.column_stack((X_mean, N_mean))
    for j in range(X_mean.shape[-1]):
        ax[j].step(365 * T_mean, X_mean[:, j], where = 'post',
                   color = 'black')

    (t_, a_, X_) = pde.solve(20, 20, 0.01, p)
    x_ = numpy.zeros((len(X_), len(t_)))
    for j in range(len(X_)):
        x_[j] = integrate.trapz(X_[j], a_, axis = 1)
    # Add column for total.
    n_ = x_.sum(0)
    x_ = numpy.row_stack((x_, n_))
    Tmax_ = numpy.ceil(numpy.max(numpy.hstack(T)))
    dt_ = t_[1] - t_[0]
    j_ = int(- Tmax_ / dt_ - 1)
    t_ -= t_[j_]
    for k in range(len(x_)):
        ax[k].plot(365 * t_[j_ : ], x_[k, j_ : ] / n_[j_] * p.population_size,
                   linestyle = ':', color = 'black')

    ax[0].set_ylabel('maternal immunity')
    ax[1].set_ylabel('susceptible')
    ax[2].set_ylabel('infected')
    ax[3].set_ylabel('recovered')
    ax[4].set_ylabel('total')

    ax[4].set_xlabel('time (days)')

    for ax_ in ax:
        yl = ax_.get_ylim()
        if yl[0] < 0:
            ax_.set_ylim(ymin = 0)

    if show:
        pyplot.show()

# function not called here
def get_I_terminal(data):
    tX_terminal = (d[-1] for d in data)
    X_terminal = (tX[-1] for tX in tX_terminal)
    I_terminal = (X[2] for X in X_terminal)
    return numpy.fromiter(I_terminal, numpy.int, len(data))


if __name__ == '__main__':
    import time

    numpy.random.seed(1)

    p = herd.Parameters()
    p.population_size = 1000
    p.birth_seasonal_coefficient_of_variation = 1

    tmax = 1
    nruns = 3  # was 100, reduced for play
    debug = False
    
    t0 = time.time()
    data = run_many(nruns, p, tmax, debug = debug)
    t1 = time.time()
    print('Run time: {} seconds.'.format(t1 - t0))
    
    #make_plots(data)
    make_datasheet(data)