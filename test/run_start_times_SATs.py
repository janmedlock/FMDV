#!/usr/bin/python3

import os.path
import sys

sys.path.append('..')
import h5
import run_common
sys.path.pop()


if __name__ == '__main__':
    nruns = 1000
    tmax = 10

    _filebase, _ = os.path.splitext(__file__)
    _filename = _filebase + '.h5'
    with h5.HDFStore(_filename) as store:
        for model in ('acute', 'chronic'):
            run_common.run_start_times_SATs(model, tmax, nruns, store)
        store.repack(_filename)
