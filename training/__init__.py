"""
subpackage with trianing modules
"""

__all__ = [
           'readExpertData',
           'readFuzzyInputData',
           'buildInputs',
           'combine_inputs',
           'write_expert_data',
           'generate_MFs',
           'train_system',
           'train_system_mp',
           'plot_rule_grid',
           'plot_parallel',
           'write_fcl_file_FRBS',
           'run_optimization',
           'run_optimization_mp',
           'train_NEFPROX',
           'write_fcl_file_NFS',
           'fuzErrorAC',
           'fuzErrorInt',
           'fuzDistAC',
           'getError',
           'getRangeError',
          ]

from .read_data import *
from .train_numerical import *
from .train_numerical_mp import * #multi-processing version
from .optimize_numerical import *
from .optimize_numerical_mp import * #multi-processing version
from .train_nfs import *
from .fuzzy_error import *
