r'''The McKendrick–von Foerster age-structured PDE model
for the density u(t, a) of buffalo of age a at time t is
(d/dt + d/da) u(t, a) = - d(a) u(t, a),
u(t, 0) = \int_0^{inf} b(t, a) u(t, a) da
u(0, a) = u_0(a).
The birth rate `b(t, a)`, which is the product of
`female_probability_at_birth` and `birth.gen.hazard(t, a)` in our
model, is periodic in `t` with period T = 1 year.

After discretizing in age, the monodromy matrix Phi(T) is the
fundamental solution of the PDEs after one period, starting from the
initial condition
Phi(0) = I,
the identity matrix.  The fundamental solution requires solving the
matrix-valued version of the orignal PDE.

Here, the matrix-valued PDE is solved using the Crank–Nicolson method
on characteristics, following the work of Fabio Milner.

So that the cache keys only depend on the relevant parts of
`herd.parameters.Parameters()` and so that the setup can be reused in
`_find_birth_scaling()`, the solver is called in 3 steps:
>>> solver_parameters = monodromy.Parameters(parameters)
>>> solver = monodromy.Solver(solver_parameters, agemax, agestep)
>>> PhiT = solver.solve(birth_scaling)
where `parameters` is a `herd.parameters.Parameters()` instance.'''

import numpy
from scipy import sparse

from herd import birth, mortality, parameters, utility
from herd.floquet.period import period


class _CSR_Matrix(sparse.csr_matrix):
    def matvecs(self, B, C):
        '''Compute the matrix multiplication `C += A @ B`, where
        `A` is a `scipy.sparse.csr_matrix()`, and
        `B` & `C` are `numpy.ndarray()`s.'''
        # Use the private function
        # `scipy.sparse._sparsetools.csr_matvecs()` so we can specify
        # the output array `C` to avoid the building of a new matrix
        # for the output.
        n_row, n_col = self.shape
        n_vecs = B.shape[1]
        sparse._sparsetools.csr_matvecs(n_row, n_col, n_vecs,
                                        self.indptr, self.indices, self.data,
                                        B.ravel(), C.ravel())


class Parameters(parameters.Parameters):
    '''Build a `herd.parameters.Parameters()`-like object that
    only has the parameters needed by `Solver()`
    so that it can be efficiently cached.'''
    def __init__(self, params):
        # Generally, the values of these parameters should be
        # floats, so explicitly convert them so the cache doesn't
        # get duplicated keys for the float and int representation
        # of the same number, e.g. `float(0)` and `int(0)`.
        # Normalize `params.birth_peak_time_of_year` by making
        # it relative to `params.start_time` and then modulo
        # `period` so that it is in [0, period).
        self.birth_normalized_peak_time_of_year = float(
            (params.birth_peak_time_of_year - params.start_time)
            % period)
        self.birth_seasonal_coefficient_of_variation = float(
            params.birth_seasonal_coefficient_of_variation)
        self.female_probability_at_birth = float(
            params.female_probability_at_birth)


class _Solution:
    '''A `_Solution()` is a sequence of length `1 + order`, with
    `solution[0]` storing the solution at the current time step,
    `solution[1]` storing the solution 1 time steps ago,
    ...
    `solution[order]` storing the solution `order` time steps ago.
    Calling `_Solution.update()` in sync with iterating through
    rearranges the elements of `solution` so that its elements stay in
    the above order at the next time step:
    the old `solution[0]` becomes the new `solution[1]`;
    ...
    the old `solution[order - 1]` becomes the new `solution[order]`;
    the old `solution[order]` becomes the new `solution[0]`,
    recycled and ready to be set to the value of the solution at the
    new time step.'''
    def __init__(self, order, shape):
        self._array = numpy.empty((1 + order, ) + shape)
        # '_front' points to the `0` element in the current iteration.
        self._front = 0

    def __getitem__(self, i):
        '''Get the solution `i` time steps ago.'''
        j = (i + self._front) % self._array.shape[0]
        return self._array[j]

    def update(self):
        '''Move the entries forward one position,
        wrapping the last entry to the front.'''
        # Decrement `_front` by 1 and wrap around if it's negative.
        self._front = (self._front - 1) % self._array.shape[0]


class Solver:
    '''Solve the monodromy problem.'''
    # Crank–Nicolson is 1st order because the solution at t_n
    # depends on the solution at t_{n - 1}.
    _order = 1

    def __init__(self, solver_params, agemax, agestep):
        self.params = solver_params
        self.ages = utility.arange(0, agemax, agestep, endpoint=True)
        tstep = agestep
        self._t = utility.arange(0, period, tstep, endpoint=True)
        mortalityRV = mortality.from_param_values()
        self._init_crank_nicolson(tstep, mortalityRV)
        self._init_births(agestep)

    @classmethod
    def from_parameters(cls, params, agemax, agestep):
        '''Build a `Solver()` instance using `herd.parameters.Parameters()`
        directly.'''
        solver_params = Parameters(params)
        return cls(solver_params, agemax, agestep)

    def _set_initial_condition(self, t_n, solution, birth_rate, temp):
        '''The initial condition for the fundamental solution is the
        identity matrix.'''
        # Avoid creating a new matrix, like `numpy.eye()` does.
        solution[0][:] = 0
        numpy.fill_diagonal(solution[0], 1)

    def _init_births(self, agestep):
        r'''The trapezoid rule for the birth integral for i = 0,
        u_0^n = \sum_j (b_j^n u_j^n + b_{j + 1}^n u_{j + 1}^n) * da / 2.
        This can be written as
        u_0^n = (v * b^n) @ u^n,
        with
        v = da * [0.5, 1, 1, ..., 1, 1, 0.5].
        Put `female_probability_at_birth` in there, too, for
        simplicity & efficiency.'''
        self._v_trapezoid = numpy.empty(self.ages.shape[0])
        self._v_trapezoid[:] = (agestep
                                * self.params.female_probability_at_birth)
        self._v_trapezoid[[0, -1]] /= 2

    def _step_births(self, t_n, solution, birth_rate, temp):
        r'''Calculate the birth integral
        B(t) = \int_0^{inf} b(t, a) U(t, a) da
        using the composite trapezoid rule,
        where U is `solution[0]`.'''
        # The simple version is
        # `U[0] = (v_trapezoid * birth_rate) @ U`
        # but avoid building new vectors.
        birth_rate(t_n, self.ages, out=temp)
        temp *= self._v_trapezoid
        temp.dot(solution[0], out=solution[0][0])

    def _init_crank_nicolson(self, tstep, mortalityRV):
        '''The Crank–Nicolson method is
        (u_i^n - u_{i - 1}^{n - 1}) / dt
        = - d_{i - 1 / 2} * (u_i^n + u_{i - 1}^{n - 1}) / 2,
        for i = 1, 2, ....
        This can be written as
        u^n = M @ u^{n - 1},
        with
        M[i, i - 1] = (1 - k_i) / (1 + k_i),
        k_i = d_{i - 1 / 2} * dt / 2,
        and, to prevent the last age group from aging out of the population,
        M[-1, -1] = (1 - k_{-1}) / (1 + k_{-1}),
        k_{-1} = d_{-1} * dt / 2.'''
        M = sparse.lil_matrix((self.ages.shape[0], self.ages.shape[0]))
        # Midpoints between adjacent ages.
        ages_mid = (self.ages[1:] + self.ages[:-1]) / 2
        k = mortalityRV.hazard(ages_mid) * tstep / 2
        # Set the first subdiagonal.
        M.setdiag((1 - k) / (1 + k), -1)
        # Keep the last age group from ageing out.
        k_last = mortalityRV.hazard(self.ages[-1]) * tstep / 2
        M[-1, -1] = (1 - k_last) / (1 + k_last)
        self._M_crank_nicolson = _CSR_Matrix(M)

    def _step_crank_nicolson(self, t_n, solution, birth_rate, temp):
        '''Do a Crank–Nicolson step.'''
        # The simple version is
        # `solution[0] = M @ solution[1]`
        # but avoid building a new matrix.
        solution[0][:] = 0
        # solution[0] += M @ solution[1]
        self._M_crank_nicolson.matvecs(solution[1], solution[0])
        self._step_births(t_n, solution, birth_rate, temp)

    def _iterate(self, solution, birth_rate, temp):
        '''The core of the solver, iterating over `self._t`.'''
        if self._t.shape[0] == 0: return None
        ## n = 0 ##
        t_n = self._t[0]
        self._set_initial_condition(t_n, solution, birth_rate, temp)
        if self._t.shape[0] == 1: return solution[0]
        ## n = 1, 2, ... ##
        for t_n in self._t[1:]:
            solution.update()
            self._step_crank_nicolson(t_n, solution, birth_rate, temp)
        # Return the solution at the final time.
        return solution[0]

    def solve(self, birth_scaling):
        '''Find the monodromy matrix Phi(T), where T is the period.'''
        # The fundamental solution is an `n_ages` x `n_ages` matrix.
        # One matrix for the current time step, plus one for each
        # order of the solver.
        solution = _Solution(self._order,
                             (self.ages.shape[0], self.ages.shape[0]))
        # Set up birth rate.
        birthRV = birth.from_param_values(
            self.params.birth_normalized_peak_time_of_year,
            self.params.birth_seasonal_coefficient_of_variation,
            _scaling=birth_scaling)
        temp = numpy.empty(self.ages.shape[0])
        return self._iterate(solution, birthRV.hazard, temp)
