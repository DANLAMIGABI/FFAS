# author - Frank Patterson


from openmdao.main.api import Component, VariableTree
from openmdao.lib.datatypes.api import Float, Str, Dict, Int, List, VarTree

import time 

import copy
import numpy as np
import skfuzzy as fuzz
import csv

import fuzzy_operations as fuzzyOps

import matplotlib.pylab as plt 

optionN = [6,4,4,5,4,3,4,6,4]#option counts for each function
cMatrix = [ [0,0,0,0,0,0,0,0,0,1,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0],
            [0,0,0,0,0,0,0,0,0,1,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
            [0,0,0,0,0,0,0,0,0,1,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0],
            [0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0],
            [0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0],
            [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,1,0,0,0,0,0],
            [0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
            [0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
            [0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0],
            [0,0,0,0,0,0,0,0,0,0,1,1,1,0,0,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,1],
            [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0],
            [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,1],
            [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,1],
            [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,1],
            [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0],
            [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0],
            [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
            [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0],
            [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0],
            [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
            [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,1,0,0,0,0,0,0,0,0,0,0],
            [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
            [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,1,0,0,1],
            [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0],
            [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,1],
            [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,1],
            [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0],
            [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
            [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0],
            [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
            [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
            [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
            [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
            [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
            [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
            [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
            [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
            [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
            [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
            [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0], ]

class checkCompatibility(Component):
    """
    Checks compatibility of morph matrix selections
    """
    
    #inputs are outputs from systems
    gen_num = Int(0, iotype='in', desc='counter for generation in optimization: does nothing but exist')
    option1 = Float(1, iotype='in', low=0.5, high=6.5, desc='option selection for functional aspect VL_SYS_TYPE (1-6)')
    option2 = Float(1, iotype='in', low=0.5, high=3.5, desc='option selection for functional aspect VL_SYS_PROP (1-3)')
    option3 = Float(1, iotype='in', low=0.5, high=3.5, desc='option selection for functional aspect VL_SYS_DRV (1-3)')
    option4 = Float(1, iotype='in', low=0.5, high=5.5, desc='option selection for functional aspect VL_SYS_TECH (1-5)')
    option5 = Float(1, iotype='in', low=0.5, high=4.5, desc='option selection for functional aspect FWD_SYS_PROP (1-4)')
    option6 = Float(1, iotype='in', low=0.5, high=3.5, desc='option selection for functional aspect FWD_SYS_DRV (1-3)')
    option7 = Float(1, iotype='in', low=0.5, high=4.5, desc='option selection for functional aspect FWD_SYS_TYPE (1-4)')
    option8 = Float(1, iotype='in', low=0.5, high=6.5, desc='option selection for functional aspect WING_SYS_TYPE (1-6)')
    option9 = Float(1, iotype='in', low=0.5, high=4.5, desc='option selection for functional aspect ENG_SYS_TYPE (1-4)')
    options = List([], iotype='in', desc='list of options selected [0-n]')
    count   = Int(0, iotype='in', desc='if count = 1: return the total number of incompatibilities and set compatibility to 0')

    incompatCount = Int(0, iotype='out', desc='Cont incompatibility options')
    compatibility = Int(0, iotype='out', desc='Compatibility flag: 0-compatible, 1-incompatible')
    option1_out = Float(1, iotype='out', desc='option selection for functional aspect VL_SYS_TYPE (1-6)')
    option2_out = Float(1, iotype='out', desc='option selection for functional aspect VL_SYS_PROP (1-3)')
    option3_out = Float(1, iotype='out', desc='option selection for functional aspect VL_SYS_DRV (1-3)')
    option4_out = Float(1, iotype='out', desc='option selection for functional aspect VL_SYS_TECH (1-5)')
    option5_out = Float(1, iotype='out', desc='option selection for functional aspect FWD_SYS_PROP (1-4)')
    option6_out = Float(1, iotype='out', desc='option selection for functional aspect FWD_SYS_DRV (1-3)')
    option7_out = Float(1, iotype='out', desc='option selection for functional aspect FWD_SYS_TYPE (1-4)')
    option8_out = Float(1, iotype='out', desc='option selection for functional aspect WING_SYS_TYPE (1-6)')
    option9_out = Float(1, iotype='out', desc='option selection for functional aspect ENG_SYS_TYPE (1-4)')


    def execute(self):
        """
        Check Matrix (1-incompatible, 0-compatible)
        """     
        total = 0
        self.options = [int(self.option1), int(self.option2), int(self.option3), 
        				int(self.option4), int(self.option5), int(self.option6), 
        				int(self.option7), int(self.option8), int(self.option9)]

        self.option1_out = self.option1
        self.option2_out = self.option2
        self.option3_out = self.option3
        self.option4_out = self.option4
        self.option5_out = self.option5
        self.option6_out = self.option6
        self.option7_out = self.option7
        self.option8_out = self.option8
        self.option9_out = self.option9

        #print "CHECKING OPTIONS:", self.options

        for i in range(len(self.options)):
            #print "Checking", i
            if self.options[i] < 1:
                self.compatibility = 1
                total = len(self.options)
                #break

        	if self.options[i] > optionN[i]: #if option > allowed option count then incompatible
        		self.compatibility = 1
                total = len(self.options)
                #break
        		#if self.count == 0: break

            for j in range(i,len(self.options)):
            	x = sum(optionN[:i]) + self.options[i] - 1 #position in compatibility matrix for option (i)
            	y = sum(optionN[:j]) + self.options[j] - 1 #position in compatibility matrix for option (j)
            	c = cMatrix[x][y]
                #print "compatibility check at", x, y, "is", c
                total = total + c
                if c > 0:  break

        if total == 0:
        	self.compatibility = 0
        else:
        	self.compatibility = 1

        if self.count == 1: 
            self.incompatCount = total
            self.compatibility = 0 #make compatible if just counting incompatible combos (stops passthrough)
