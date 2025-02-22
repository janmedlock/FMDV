'''A data structure to hold the model parameters and the baseline
values of those parameters.'''


from numpy import inf


class Parameters:
    def __init__(self, model='acute', SAT=1, _set_defaults=True,
                 _initial_conditions=None):
        'Initialize with default values.'
        self.SAT = SAT
        self.model = model
        chronic_model = (self.model == 'chronic')
        if _set_defaults:
            self.population_size = 1000
            self.initial_infectious = 2
            self.start_time = 0.5  # 6 months after birth peak.
            self.female_probability_at_birth = 0.5
            # I should have used birth_peak_time_of_year = 0.041667,
            # about 16 January, to make things simpler.
            self.birth_peak_time_of_year = 0
            # self.birth_seasonal_coefficient_of_variation = 0.505  # 1st year.
            # self.birth_seasonal_coefficient_of_variation = 0.984  # 2nd year.
            self.birth_seasonal_coefficient_of_variation = 0.613  # Both years.
            # During in M before moving to S.
            self.maternal_immunity_duration_mean = 0.37
            self.maternal_immunity_duration_shape = 1.19
            # Duration in R before returning to S.
            self.immunity_waning_duration = inf
            if self.SAT == 1:
                self.progression_shape = 1.2
                self.progression_mean = 0.5 / 365
                self.recovery_shape = 11.8
                self.recovery_mean = 5.7 / 365
                self.transmission_rate = 2.8 * 365
                # Proportion leaving I that become C.
                self.probability_chronic = 0.90 if chronic_model else 0
                # Duration in C before leaving to R.
                self.chronic_recovery_mean = 243 / 365
                self.chronic_recovery_shape = 3.2
                self.chronic_transmission_rate = 0.028 * 365
            elif self.SAT == 2:
                self.progression_shape = 1.6
                self.progression_mean = 1.3 / 365
                self.recovery_shape = 8.7
                self.recovery_mean = 4.6 / 365
                self.transmission_rate = 1.6 * 365
                # Proportion leaving I that become C.
                self.probability_chronic = 0.44 if chronic_model else 0
                # Duration in C before leaving to R.
                self.chronic_recovery_mean = 180 / 365
                self.chronic_recovery_shape = 3.2
                self.chronic_transmission_rate = 0.003 * 365
            elif self.SAT == 3:
                self.progression_shape = 1.6
                self.progression_mean = 2.8 / 365
                self.recovery_shape = 11.8
                self.recovery_mean = 4.2 / 365
                self.transmission_rate = 1.2 * 365
                # Proportion leaving I that become C.
                self.probability_chronic = 0.67 if chronic_model else 0
                # Duration in C before leaving to R.
                self.chronic_recovery_mean = 174 / 365
                self.chronic_recovery_shape = 3.2
                self.chronic_transmission_rate = 0.012 * 365
            else:
                raise ValueError("Unknown SAT '{}'!".format(self.SAT))
            if _initial_conditions is not None:
                # Use the initial conditions for the SAT in
                # `_initial_conditions` to build the initial
                # conditions for a simulation of SAT `SAT`.
                self.initial_conditions = self.__class__(
                    model=model, SAT=_initial_conditions,
                    _set_defaults=_set_defaults)

    def __repr__(self):
        'Make instances print nicely.'
        clsname = '{}.{}'.format(self.__module__, self.__class__.__name__)
        paramreprs = ['{!r}: {!r}'.format(k, self.__dict__[k])
                      for k in sorted(self.__dict__.keys())]
        return '<{}: {{{}}}>'.format(clsname, ', '.join(paramreprs))

    @classmethod
    def from_repr(cls, r_, set_defaults=True):
        assert r_.startswith('<')
        assert r_.endswith('>')
        r = r_[1 : -1]
        l = r.find(':')
        name = r[ : l]
        # Can be different depending on how the module is imported.
        clsname = '{}.{}'.format(cls.__module__, cls.__name__)
        assert (name == clsname)
        paramstr_ = r[l + 2 : ]
        assert paramstr_.startswith('{')
        assert paramstr_.endswith('}')
        paramstr = paramstr_[1 : -1]
        p = cls(_set_defaults=set_defaults)
        for s in paramstr.split(', '):
            (k_, vstr) = s.split(': ')
            assert k_.startswith("'")
            assert k_.endswith("'")
            k = k_[1 : -1]
            v = float(vstr)
            if int(v) == v:
                v = int(v)
            setattr(p, k, v)
        return p
