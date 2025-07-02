# FMDV

## A transmission model of foot-and-model-disease viruses in African buffalo

**Jan Medlock [\<medlockj@oregonstate.edu\>](mailto:medlockj@oregonstate.edu),
Erin Gorisch, Anna Jolles, Simon Gubbins, Brianna Beechler,
Peter Buss, Nick Juleff, Lin-Mari de Klerk-Lorist, Francois Maree,
Eva Perez-Martin, OL van Schalkwyk, Katherine Scott, Fuquan Zhang,
Bryan Charleston.**

**Copyright 2014–2025, Jan Medlock et al.  All rights reserved.
Released under the [GNU AGPL 3](LICENSE).**

This repository contains Python code and data used to simulate and
analyze FMDV transmission in African buffalo for our paper
> Jolles A, Gorsich E, Gubbins S, Beechler B, Buss P, Juleff N,
> deKlerk-Lorist L-M, Maree F, Perez-Martin E, van Schalkwyk OL,
> Scott K, Zhang F, Medlock J, Charleston B.
> Endemic persistence of a highly contagious pathogen:
> Foot-and-mouth disease in its wildlife host.
> *Science*. 2021. 374(6563): 104–109.
> [doi:10.1126/science.abd2475](https://doi.org/10.1126/science.abd2475).

The scripts and model code are written in Python3, using many
third-party libraries.  Most notably:
[Python3](https://www.python.org/),
[NumPy & SciPy](https://www.scipy.org/),
[statsmodels](https://www.statsmodels.org/),
[pandas](https://pandas.pydata.org/),
[PyTables](https://www.pytables.org/),
[Sorted Containers](http://www.grantjenks.com/docs/sortedcontainers/),
[matplotlib](https://matplotlib.org/),
& [Seaborn](https://seaborn.pydata.org/).

### Main simulation code

The module that simulates the FMDV model is in the Python module
[herd](herd).

The submodule [herd.floquet](herd/floquet) contains the solver to find
the population stable age distribution with birth seasonality. In the
folder [herd/floquet](herd/floquet) is an optional faster
implementation in [Cython](https://cython.org/) of
`herd.floquet.monodromy` that can be built using the included
[Makefile](herd/floquet/Makefile).

The folder [herd/data](herd/data) contains the parameter
posterior samples from the statistical analysis of our experimental
data and a summary of the data from Hedger (1972) used to initialize
our simulations.

### Simulation scripts

These scripts run the model simulations. **Each of these takes many
cpu-days to run.**

* [run.py](run.py), for each of the acute and chronic models and for
  each of the 3 SATs, runs 1,000 simulations using the baseline
  parameter values. It produces a file called `run.h5`.

* [samples_run.py](samples_run.py), for each of the acute and chronic
  models, for each of the 3 SATs, and for each of 20,000 parameter
  posterior samples, runs 1 simulation. It produces a file called
  `samples.h5`.

* [birth_seasonality_run.py](birth_seasonality_run.py), for each of
  the acute and chronic models, for each of the 3 SATs, and for 5
  different levels of birth seasonality, runs 1,000 simulations using
  the baseline parameter values. It produces a file called
  `birth_seasonality.h5`.

* [population_size_run.py](population_size_run.py), for each of the
  acute and chronic models, for each of the 3 SATs, and for 14
  different population sizes, runs 1,000 simulations using the
  baseline parameter values. It produces a file called
  `population_size.h5`.

* [start_time_run.py](start_time_run.py), for each of the acute and
  chronic models, for each of the 3 SATs, and for 12 different times
  of year to start the simulations, runs 1,000 simulations using the
  baseline parameter values. It produces a file called
  `start_time.h5`.

### Analysis and plotting scripts

These scripts analyze and plot the simulation results. Most of them
require having run the simulation scripts above.

* [R0.py](R0.py) computes the basic reproduction number,
  *R*<sub>0</sub>, for each of the acute and chronic models and each
  of the 3 SATs.

* [susceptible_recruitment.py](susceptible_recruitment.py) plots the
  inflow of susceptibles vs. time of year that occurs in our model due
  to the seasonality of births and subsequent waning of maternal
  immunity.

* [samples.py](samples.py) analyzes and plots the results of the
  simulations over the parameter posterior samples. This requires the
  file `samples.h5`, which is built by
  [samples_run.py](samples_run.py).

* [birth_seasonality.py](birth_seasonality.py) analyzes and plots the
  results of varying the level of birth seasonality. This requires the
  file `birth_seasonality.h5`, which is built by
  [birth_seasonality_run.py](birth_seasonality_run.py).

* [population_size.py](population_size.py) analyzes and plots the
  results of varying the population size. This requires the file
  `population_size.h5`, which is built by
  [population_size_run.py](population_size_run.py).

* [start_time.py](start_time.py) analyzes and plots the results of
  varying the time of year to start the simluations. This requires the
  file `start_time.h5`, which is built by
  [start_time_run.py](start_time_run.py).

* [figure_3.py](figure_3.py) builds Figure 3 from our paper. This
  requires the file `run.h5`, which is built by [run.py](run.py).

* [figure_4.py](figure_4.py) builds Figure 4 from our paper. The
  requires the files `population_size.h5`, which is built by
  [population_size_run.py](population_size_run.py), and `samples.h5`,
  which is built by [samples_run.py](samples_run.py).

### Test scripts

The [test](test) directory contains some scripts to test various parts
of the model code.

* [run_one.py](test/run_one.py) runs 1 model simulation and plots the
  results.

* [run_many.py](test/run_many.py) runs 100 model simulations for one
  model and one SAT and plots the results.

* [samples_run_test.py](test/samples_run_test.py) sequentially runs
  simulations with the parameter posterior samples for one model and
  one SAT.

* [age_structure.py](test/age_structure.py) and
  [age_structure_3d.py](test/age_structure_3d.py) plot the stable age
  structure of the model buffalo population.

* [benchmark.py](test/benchmark.py) times the Floquet solver in
  [herd.floquet](herd/floquet) for the stable age structure.

* [find_infection_hazard.py](test/find_infection_hazard.py) finds the
  best-fit infection hazard.

* [initial_conditions.py](test/initial_conditions.py) plots the model
  initial conditions.

* [h5_test.py](test/h5_test.py) tests the simulation output files for
  consistency.

### Supplementary material

[supplement](supplement) contains files and scripts for building the
modeling section of the supplementary material from our paper.
