import numpy

from . import birth_super


def fracpart(x):
    return numpy.mod(x, 1.)


def get_seasonal_variance_from_gap_size(g):
    'g in months'
    if g is None:
        return 0.
    else:
        return 4. / 3. / (1. - g / 12.) - 1.

def get_gap_size_from_seasonal_variance(b):
    'in months'
    if b == 0.:
        return None
    else:
        return 12. * (1. - 4. / 3. / (b + 1.))


class gen(birth_super.gen):
    def _getparams(self):
        if self.seasonal_variance < 1. / 3.:
            alpha = 1. + numpy.sqrt(3. * self.seasonal_variance)
            beta = (2. * numpy.sqrt(3. * self.seasonal_variance)
                    / (1. + numpy.sqrt(3. * self.seasonal_variance)))

        else:
            alpha = 3. * (1. + self.seasonal_variance) / 2.
            beta = 3. * (1. + self.seasonal_variance) / 4.
            
        return (alpha, beta)

    def hazard(self, time, time0, age0):
        (alpha, beta) = self._getparams()

        tau = fracpart(time + time0)

        fdown = alpha * numpy.clip(1. - 2. * beta * tau, 0., numpy.inf)
        fup = alpha * numpy.clip(1. - 2. * beta * (1 - tau), 0., numpy.inf)
        # 0 if current age (age0 + time) < 4
        # else: alpha if (time + time0 - beta / 2) mod 1 <= beta
        #       else: 0
        return self.scaling * numpy.where(age0 + time < 4., 0.,
                                          numpy.where(tau <= 0.5, fdown, fup))

    def _cdf(self, time, time0, age0):
        (alpha, beta) = self._getparams()

        c = numpy.clip(4 - age0, 0., numpy.inf) + time0
        d = time + time0
        
        if beta < 1:
            H1 = (1 - beta / 2.) * (numpy.floor(d) - numpy.floor(c) - 1.)
            H2 = numpy.where(fracpart(c) < 1. / 2.,
                             1. / 2. * (1. - beta / 2.) + 
                             (1. / 2. - fracpart(c))
                             * (1. - beta / 2. - beta * fracpart(c)),
                             (1. - fracpart(c))
                             * (1 - beta * fracpart(c)))
            H3 = numpy.where(fracpart(d) < 1. / 2.,
                             fracpart(d) * (1. - beta * fracpart(d)),
                             1. / 2. * (1. - beta / 2.)
                             + (fracpart(d) - 1. / 2.)
                             * (1. - 3. / 2. * beta + beta * fracpart(d)))

            H = H1 + H2 + H3
        else:
            H4 = 1 / 2. / beta * (numpy.floor(d) - numpy.floor(c) - 1.)
            H5 = numpy.where(fracpart(c) < 1. / 2. / beta,
                             1. / 4. / beta
                             + beta * (1. / 2. / beta - fracpart(c)) ** 2,
                             numpy.where(fracpart(c) < 1. - 1. / 2. / beta,
                                         1. / 4. / beta,
                                         (1. - fracpart(c))
                                         * (1. - beta
                                            * (1. - fracpart(c)))))
            H6 = numpy.where(fracpart(d) < 1. / 2. / beta,
                             fracpart(d) * (1. - beta * fracpart(d)),
                             numpy.where(fracpart(d) < 1. - 1. / 2. / beta,
                                         1. / 4. / beta,
                                         1. / 4. / beta
                                         + beta * (fracpart(d) - 1
                                                   + 1. / 2. / beta) ** 2))
                                     
            H = H4 + H5 + H6

        I = self.scaling * numpy.where(time < 4 - age0,
                                       0.,
                                       alpha * H)

        return 1. - numpy.exp(- I)