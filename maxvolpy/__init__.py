"""
Module `maxvolpy` is designed for constructing different low-rank
skeleton and cross approximations. There are different strategies of
doing this, but this module is focused on approximations, based on
submatrices of good volume.

Right now, cross approximations are not implemented, but all kinds of
algorithms of finding good submatrices to build skeleton approximations
are presented in `maxvol` submodule.
"""

from __future__ import absolute_import

__all__ = ['maxvol']

from . import maxvol
from .__version__ import __version__
