'''Monkeypatch `tables.attributeset.AttributeSet.__getattr__()` to fix
opening files written when there were a different number of global
compression methods: https://github.com/PyTables/PyTables/issues/811'''


import functools

import tables


_getattr_old = tables.attributeset.AttributeSet.__getattr__


@functools.wraps(_getattr_old)
def _getattr_new(self, name):
    try:
        return _getattr_old(self, name)
    except ValueError:
        return None


tables.attributeset.AttributeSet.__getattr__ = _getattr_new
