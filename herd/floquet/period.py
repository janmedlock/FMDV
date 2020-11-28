'''The period of the model forcing.'''


from herd import birth


# `herd.birth.gen.hazard` is the only time-dependent function.
period = birth._period
