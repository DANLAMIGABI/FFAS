"""
   fuzzy_math.py
   
   Author: Frank Patterson
   21-Dec-14
"""

from openmdao.main.api import Component
from openmdao.lib.datatypes.api import Float, Str, Dict, Int

import numpy as np
import skfuzzy as fuzz

class fuzzy_mean(Component):

    fuzzy_inputs = Dict({}, iotype='in', desc='input dict of fuzzy inputs {key:fuzzyVal, key2:fuzzyVal2, ...]')
    output_key = Str('', iotype='in', desc='key for output value')
    fuzzy_output = Dict( {}, iotype='out', desc='output {output_key:fuzzy average}')
    
    def exceute(self):
        pass