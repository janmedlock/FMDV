#!/usr/bin/python3
'''Check simuation output for consistency.'''

from context import h5


T_NAME = 'time (y)'


def check_completed(path, by, cols_infected, t_max, **kwds):
    with h5.HDFStore(path, mode='r') as store:
        grouper = store.groupby(by, columns=cols_infected, **kwds)
        for (_, group) in grouper:
            infected = group.sum(axis='columns')
            extinction = infected.iloc[-1] == 0
            t = group.index.get_level_values(T_NAME)
            time = t.max() - t.min()
            completed = extinction or (time == t_max)
            assert completed


if __name__ == '__main__':
    check_completed(
        '../population_size.h5',
        ['model', 'SAT', 'population_size', 'run'],
        ['exposed', 'infectious', 'chronic'],
        10,
        where=' & '.join(
            f'{k}={v}' for (k, v) in {'model': 'acute', 'SAT': 1}.items()
        ),
    )
