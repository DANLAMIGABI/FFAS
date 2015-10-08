#author - Frank Patterson
from openmdao.main.api import Component, VariableTree
from openmdao.lib.datatypes.api import Float, Str, Dict, Int, List, VarTree

import time 

import copy
import numpy as np
import skfuzzy as fuzz
import csv
import math

import fuzzy_operations as fuzzyOps

import matplotlib.pylab as plt 

#-----------------------------------------------------------------------------#
#-----------------------------------------------------------------------------#
class FuzzyMF(VariableTree):
    """ 
    Membershipt function for fuzzy output (or input) 'key', [x_vals, y_vals]
    """
    mf_key 		= Str('', desc='Key for Output Membership Function of interest')
    mf_dict		= Dict({}, desc = 'Dict of for MF(s) (key:[crisp] or key:[fuzzy_x, fuzzy_y])')


class Postprocess_Fuzzy_Outputs(Component):
    """
    Takes in some outputs from the fuzzy systems and creates crisp values by which 
    to find optimal solutions. Uses alpha cuts at the the given alpha_val level.
    """
    
    
    # set up interface to the framework
    #inputs are outputs from systems
    fuzzSys_in_1 = VarTree(FuzzyMF(), iotype='in', desc='fuzzy system output')
    fuzzSys_in_2 = VarTree(FuzzyMF(), iotype='in', desc='fuzzy system output')
    fuzzSys_in_3 = VarTree(FuzzyMF(), iotype='in', desc='fuzzy system output')
    fuzzSys_in_4 = VarTree(FuzzyMF(), iotype='in', desc='fuzzy system output')
    fuzzSys_in_5 = VarTree(FuzzyMF(), iotype='in', desc='fuzzy system output')
    fuzzSys_in_6 = VarTree(FuzzyMF(), iotype='in', desc='fuzzy system output')
    fuzzSys_in_7 = VarTree(FuzzyMF(), iotype='in', desc='fuzzy system output')
    fuzzSys_in_8 = VarTree(FuzzyMF(), iotype='in', desc='fuzzy system output')
    fuzzSys_in_9 = VarTree(FuzzyMF(), iotype='in', desc='fuzzy system output')

    alpha_val = Float(0.7, iotype='in', desc='alpha-cut to perform range post processing at')
    goalVals = Dict({'sys_phi'  :6.7, 
                     'sys_FoM'  :0.775, 
                     'sys_LoD'  :12.5, 
                     'sys_etaP' :0.875, 
                     'sys_Pin'  :3500.0,
                     'sys_GWT'  :10000,
                     'sys_VH'   :325}, iotype='in', desc='crisp goals for each output')

    PLOTMODE = Int(0, iotype='in', desc='Flag for plotting, 1 for plot')
    printResults = Int(1, iotype='in', desc='print results each iteration?')

    passthrough = Int(0, iotype='in', low=0, high=1, desc='catch flag for incompatible options')
    incompatCount = Int(0, iotype='in', desc='count of incomatible options')

    runFlag_in = Int(0, iotype='in', desc='test')
    runFlag_out = Int(0, iotype='out', desc='test')

    ranges_out = Dict({}, iotype='out', desc='alpha cuts for each fuzzy input')
    response_1     = Float(0.0, iotype='out', desc='crisp measure 1 to perform optimization')
    response_1_r   = Float(0.0, iotype='out', desc='range for crisp measure 1')
    response_1_POS = Float(0.0, iotype='out', desc='fuzzy POS measure (dominance to crisp goal)')

    response_2     = Float(0.0, iotype='out', desc='crisp measure 2 to perform optimization')
    response_2_r   = Float(0.0, iotype='out', desc='range for crisp measure 2')
    response_2_POS = Float(0.0, iotype='out', desc='fuzzy POS measure (dominance to crisp goal)')   

    response_3     = Float(0.0, iotype='out', desc='crisp measure 3 to perform optimization')
    response_3_r   = Float(0.0, iotype='out', desc='range for crisp measure 3')
    response_3_POS = Float(0.0, iotype='out', desc='fuzzy POS measure (dominance to crisp goal)')

    response_4     = Float(0.0, iotype='out', desc='crisp measure 4 to perform optimization')	
    response_4_r   = Float(0.0, iotype='out', desc='range for crisp measure 4')
    response_4_POS = Float(0.0, iotype='out', desc='fuzzy POS measure (dominance to crisp goal)')

    response_5     = Float(0.0, iotype='out', desc='crisp measure 5 to perform optimization')
    response_5_r   = Float(0.0, iotype='out', desc='range for crisp measure 5')
    response_5_POS = Float(0.0, iotype='out', desc='fuzzy POS measure (dominance to crisp goal)')

    response_6     = Float(0.0, iotype='out', desc='crisp measure 6 to perform optimization')
    response_6_r   = Float(0.0, iotype='out', desc='range for crisp measure 6')
    response_6_POS = Float(0.0, iotype='out', desc='fuzzy POS measure (dominance to crisp goal)')

    response_7     = Float(0.0, iotype='out', desc='crisp measure 7 to perform optimization')
    response_7_r   = Float(0.0, iotype='out', desc='range for crisp measure 7')
    response_7_POS = Float(0.0, iotype='out', desc='fuzzy POS measure (dominance to crisp goal)')

    response_8     = Float(0.0, iotype='out', desc='crisp measure 8 to perform optimization')
    response_8_r   = Float(0.0, iotype='out', desc='range for crisp measure 8')
    response_8_POS = Float(0.0, iotype='out', desc='fuzzy POS measure (dominance to crisp goal)')

    response_9     = Float(0.0, iotype='out', desc='crisp measure 9 to perform optimization')
    response_9_r   = Float(0.0, iotype='out', desc='range for crisp measure 9')
    response_9_POS = Float(0.0, iotype='out', desc='fuzzy POS measure (dominance to crisp goal)')

    fuzzyPOS       = Float(0.0, iotype='out', desc='Fuzzy Measure for POS (product of all POS measures')

    def execute(self):
        """
        Translate fuzzy inputs to crisp values to optimize system.
        """      
        inputs = [self.fuzzSys_in_1, self.fuzzSys_in_2, self.fuzzSys_in_3, 
        		  self.fuzzSys_in_4, self.fuzzSys_in_5, self.fuzzSys_in_6,
                  self.fuzzSys_in_7, self.fuzzSys_in_8, self.fuzzSys_in_9]
        outs  = [[self.response_1][0], [self.response_2][0], self.response_3, 
                 self.response_4, self.response_5, self.response_6,
                 self.response_7, self.response_8, self.response_9]
        outs_r  = [self.response_1_r, self.response_2_r, self.response_3_r, 
                   self.response_4_r, self.response_5_r, self.response_6_r,
                   self.response_7_r, self.response_8_r, self.response_9_r]
        
        if self.passthrough == 1:
            if self.printResults == 1: print "Incompatible combo found..."
            self.response_1     = 0.0 #phi
            self.response_1_r   = 0.0 
            self.response_1_POS = -1.0

            self.response_2     = 0.0 #FoM
            self.response_2_r   = 0.0
            self.response_2_POS = -1.0

            self.response_3     = 0.0 #LoD
            self.response_3_r   = 0.0
            self.response_3_POS = -1.0

            self.response_4     = 0.0 #etaP
            self.response_4_r   = 0.0
            self.response_4_POS = -1.0

            self.response_5     = 0.0 #GWT
            self.response_5_r   = 0.0
            self.response_5_POS = -1.0

            self.response_6     = -99999.0 #P
            self.response_6_r   = 0.0
            self.response_6_POS = 0.0 

            self.response_7     = 0.0 #VH
            self.response_7_r   = 0.0
            self.response_7_POS = -1.0

            return None

        else:
            #get alpha cuts for crisp responses and ranges 
            for i in range(len(inputs)):
                
                if inputs[i].mf_key	<> '':
                    if len(inputs[i].mf_dict[inputs[i].mf_key]) 	 == 1: #crisp value
                        self.ranges_out[inputs[i].mf_key] = [inputs[i].mf_dict[inputs[i].mf_key][0], inputs[i].mf_dict[inputs[i].mf_key][0]]
                    elif len(inputs[i].mf_dict[inputs[i].mf_key])  == 2: #fuzzy function
                        self.ranges_out[inputs[i].mf_key] = fuzzyOps.alpha_cut(self.alpha_val, inputs[i].mf_dict[inputs[i].mf_key])

                    #capture results for crisp measures
                    if self.ranges_out[inputs[i].mf_key] <> None: y = self.ranges_out[inputs[i].mf_key]
                    else:                                         y = [0.0, 0.0]

                    if inputs[i].mf_key == 'sys_phi': 
                        self.response_1 = fuzz.defuzz(inputs[i].mf_dict[inputs[i].mf_key][0],np.array(inputs[i].mf_dict[inputs[i].mf_key][1]),'centroid')
                        self.response_1 = self.response_1 * math.exp(-self.incompatCount)**0.5
                        self.response_1_r = max(y) - min(y)
                        self.response_1_POS = fuzzyOps.fuzzyPOS(inputs[i].mf_dict[inputs[i].mf_key][0],np.array(inputs[i].mf_dict[inputs[i].mf_key][1]), self.goalVals['sys_phi'])
                   
                    if inputs[i].mf_key == 'sys_FoM': 
                        self.response_2 = fuzz.defuzz(inputs[i].mf_dict[inputs[i].mf_key][0],np.array(inputs[i].mf_dict[inputs[i].mf_key][1]),'centroid')
                        self.response_2 = self.response_2 * math.exp(-self.incompatCount)**0.5
                        self.response_2_r = max(y) - min(y)
                        #if self.response_2 < 0.6:  
                        #    self.response_2_POS = fuzzyOps.fuzzyPOS(inputs[i].mf_dict[inputs[i].mf_key][0],np.array(inputs[i].mf_dict[inputs[i].mf_key][1]), self.goalVals['sys_FoM'], direction='max', plot=True)
                        self.response_2_POS = fuzzyOps.fuzzyPOS(inputs[i].mf_dict[inputs[i].mf_key][0],np.array(inputs[i].mf_dict[inputs[i].mf_key][1]), self.goalVals['sys_FoM'])

                    if inputs[i].mf_key == 'sys_LoD': 
                        self.response_3 = fuzz.defuzz(inputs[i].mf_dict[inputs[i].mf_key][0],np.array(inputs[i].mf_dict[inputs[i].mf_key][1]),'centroid')
                        self.response_2 = self.response_3 * math.exp(-self.incompatCount)**0.5
                        self.response_3_r = max(y) - min(y)
                        self.response_3_POS = fuzzyOps.fuzzyPOS(inputs[i].mf_dict[inputs[i].mf_key][0],np.array(inputs[i].mf_dict[inputs[i].mf_key][1]), self.goalVals['sys_LoD'])

                    if inputs[i].mf_key == 'sys_etaP': 
                        self.response_4 = fuzz.defuzz(inputs[i].mf_dict[inputs[i].mf_key][0],np.array(inputs[i].mf_dict[inputs[i].mf_key][1]),'centroid')
                        self.response_4 = self.response_4 * math.exp(-self.incompatCount)**0.5
                        self.response_4_r = max(y) - min(y)
                        self.response_4_POS = fuzzyOps.fuzzyPOS(inputs[i].mf_dict[inputs[i].mf_key][0],np.array(inputs[i].mf_dict[inputs[i].mf_key][1]), self.goalVals['sys_etaP'])

                    if inputs[i].mf_key == 'sys_GWT': #invert GWT to maximize all
                        self.response_5 = fuzz.defuzz(inputs[i].mf_dict[inputs[i].mf_key][0],np.array(inputs[i].mf_dict[inputs[i].mf_key][1]),'centroid')
                        self.response_5 = self.response_5 * math.exp(-self.incompatCount)**0.5
                        self.response_5_r = max(y) - min(y)
                        self.response_5_POS = fuzzyOps.fuzzyPOS(inputs[i].mf_dict[inputs[i].mf_key][0],np.array(inputs[i].mf_dict[inputs[i].mf_key][1]), self.goalVals['sys_GWT'], direction='min')

                    
                    if inputs[i].mf_key == 'sys_P': #invert GWT to maximize all
                        self.response_6 = 0.0-fuzz.defuzz(inputs[i].mf_dict[inputs[i].mf_key][0],np.array(inputs[i].mf_dict[inputs[i].mf_key][1]),'centroid')
                        self.response_6 = self.response_6 * math.exp(-self.incompatCount)**0.5
                        self.response_6_r = max(y) - min(y)
                        self.response_6_POS = fuzzyOps.fuzzyPOS(inputs[i].mf_dict[inputs[i].mf_key][0],np.array(inputs[i].mf_dict[inputs[i].mf_key][1]), self.goalVals['sys_Pin'], direction='min')

                    if inputs[i].mf_key == 'sys_VH': #invert GWT to maximize all
                        self.response_7 = fuzz.defuzz(inputs[i].mf_dict[inputs[i].mf_key][0],np.array(inputs[i].mf_dict[inputs[i].mf_key][1]),'centroid')
                        self.response_7 = self.response_7 * math.exp(-self.incompatCount)**0.5
                        self.response_7_r = max(y) - min(y)
                        self.response_7_POS = fuzzyOps.fuzzyPOS(inputs[i].mf_dict[inputs[i].mf_key][0],np.array(inputs[i].mf_dict[inputs[i].mf_key][1]), self.goalVals['sys_VH'])

                        self.fuzzyPOS = self.response_1_POS*self.response_2_POS*self.response_3_POS*self.response_4_POS*self.response_6_POS


        if self.printResults == 1: #print results
            print "Alternative:", self.passthrough, ":",
            print "PHI: %.1f, (%.3f)" % (self.response_1, self.response_1_POS),
            print "  FoM: %.3f, (%.3f)" % (self.response_2, self.response_2_POS), 
            print "  L/D: %.1f, (%.3f)" % (self.response_3, self.response_3_POS),
            print "  etaP: %.3f, (%.3f)" % (self.response_4, self.response_4_POS),
            print "  GWT: %.0f, (%.3f)" % (self.response_5, self.response_5_POS),
            print "  Pinst: %.0f, (%.3f)" % (self.response_6, self.response_6_POS),
            print "  VH: %.0f, (%.3f)" % (self.response_7, self.response_7_POS),
            print "  FPOS: %.3f" % self.fuzzyPOS

        #plotting for testing
        if self.PLOTMODE == 1: #plot results
            plt.figure()
            i=1
            for r in inputs:
                plt.subplot(3,2,i)
                if r.mf_key <> '':
                    if len(r.mf_dict[r.mf_key])      == 1: #crisp value
                        pass#self.ranges_out[r.mf_key] = [r.mf_dict[r.mf_key][0], r.mf_dict[r.mf_key][1]]
                    elif len(r.mf_dict[r.mf_key])  == 2: #fuzzy function
                        plt.plot(r.mf_dict[r.mf_key][0],r.mf_dict[r.mf_key][1])
                    i = i + 1

            plt.show()

        #print "OBJECTIVES:", self.response_1, self.response_2, self.response_3

