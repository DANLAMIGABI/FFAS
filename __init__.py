"""
FFAS - "Fuzzy systems For Architecture Selection"
        A Fuzzy Expert Systems Toolbox for systems architecture selection.

Recommended Use
---------------
>>> import FFAS as ffas
"""
__all__ = []

#try:
#    from .version import version as __version__
#except ImportError:
#    __version__ = "unbuilt-dev"
#else:
#    del version

######################
# Subpackage imports #
######################



# fuzzy systems
import FFAS.systems as _systems
from FFAS.systems import *
__all__.extend(_systems.__all__)

# fuzzy training
import FFAS.training as _training
from FFAS.training import *
__all__.extend(_training.__all__)

# fuzzy operations
import FFAS.fuzzy_operations as _fuzzyOps
from FFAS.fuzzy_operations import *
__all__.extend(_fuzzyOps.__all__)

# timer
import FFAS.timer as _timer
from FFAS.timer import Timer
__all__.append('Timer')
