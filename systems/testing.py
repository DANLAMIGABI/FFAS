# -*- coding: utf-8 -*-
"""
Created on Sun Jun  7 13:28:38 2015

@author: frankpatterson
"""
import numpy as np
import skfuzzy as fuzz

########## TESTING fuzzy_system.py ##############

from systems import *

#testing build_fuzz_system
def test_build_fuzzy_system():
    inputs, outputs, rulebase, AND_operator, OR_operator, aggregator, implication, defuzz = \
        build_fuzz_system('FCL_files/test2.fcl')
    #PRINTING OUT SYSTEM FOR DISPLAY
    print ''
    print 'INPUTS:'    
    for key in inputs: 
        print inputs[key].name, inputs[key].data_type, inputs[key].data_range
        for key in inputs[key].MFs: print '    ', key
    print 'OUTPUTS:'
    for key in outputs:
        print outputs[key].name, outputs[key].data_type, outputs[key].data_range
        for key in outputs[key].MFs: print '    ', key
    for r in rulebase: 
        print 'RULE: ', r.rule_id
        print r.rule_list
    print 'AND operator: ', AND_operator
    print 'OR operator: ', OR_operator
    print 'AGGREGATOR: ', aggregator 
    print 'IMPLICATION: ', implication

#testing fuzzy system
def test_fuzzy_system():
    print '------------------------------------------------------------------------'
    print '                       TEST FUZZY SYSTEM                                '
    print '------------------------------------------------------------------------'
    
    inputs, outputs, rulebase, AND_operator, OR_operator, aggregator, implication, defuzz = \
        build_fuzz_system('FCL_files/test2.fcl')
    sys = Fuzzy_System(inputs, outputs, rulebase, AND_operator, OR_operator, aggregator, implication, defuzz)
    print sys.firing_strength('High', 6.3, inputs['Weight_VLsys'])
    x1 = np.arange(1.0,9.1,0.1)
    y1 = fuzz.trapmf(x1, [4,6,7,9])
    print sys.firing_strength('High', [x1,y1] , inputs['Weight_VLsys'])
    
    print 'TEST FUZZY RULE IMPLEMENTATION'
    input_list = {'Weight_VLsys': 5.0, 'Weight_VLprop': 4.0, 'Weight_VLdrive': 5.0}
    results = sys.run(input_list, TESTMODE=1)
    print len(results), 'results calculated.'


################ REAL FCL
def test_realFCL():
    longfile = 'FCL_files/test.fcl'    
    inputs, outputs, rulebase, AND_operator, OR_operator, aggregator, implication, defuzz = \
        build_fuzz_system(longfile)
    sys = Fuzzy_System(inputs, outputs, rulebase, AND_operator, OR_operator, aggregator, implication, defuzz)
    input_list = {'WING_LoD': 15.,
                  'SYSTEM_f': 5}
    results = sys.run(input_list, TESTMODE=1)



## TEST NEURO FUZZY SYSTEMS
import random
from timer import Timer

##BUILD TEST SYSTEM
def test_build_system():
    inputs, outputs, rulebase, AND_operator, OR_operator, \
    aggregator, implication, defuzz = build_fuzz_system('FCL_files/test.fcl')
    NFS = NEFPROX(inputs, outputs, rulebase, defuzz)
    
    print 'TEST FUZZY RULE IMPLEMENTATION'
    input_list = {'Weight_VLsys': 5.0, 'Weight_VLprop': 4.0, 'Weight_VLdrive': 5.0}
    results = NFS.run(input_list, TESTMODE=True)


### 
def test_real_NFS_FCL():       
    testFCL = 'FCL_files/FOMsys_trained_5In7Out_500tData_250vData.fcl'
    inputs, outputs, rulebase, AND_operator, OR_operator, aggregator, implication, defuzz = build_fuzz_system(testFCL)
    NFS = NEFPROX(inputs, outputs, rulebase, defuzz) #system built
    
    #PLOTTING
    for i in range(3):
        download = random.random()*0.3  #download (e_d)
        eta = 0.5 + random.random()*0.5 #eta
        sigma = 0.4*random.random()     #solidity
        DL = 5 + random.random()*145    #disk loading
        input_list = {'DATA_e_d': download, 'DATA_eta': eta, 'DATA_sigma': sigma, 'DATA_w': DL }
        results = NFS.run(input_list, TESTMODE=True)
    
    
    #TIMING
    n = 100
    inputs = []
    outputs = []
    with Timer() as t:
        for i in range(n):
            download = random.random()*0.3 #download (e_d)
            eta = 0.5 + random.random()*0.5 #eta
            sigma = 0.4*random.random()     #solidity
            DL = 5 + random.random()*145    #disk loading
            input_list = {'DATA_e_d': download, 'DATA_eta': eta, 'DATA_sigma': sigma, 'DATA_w': DL }
            results = NFS.run(input_list)
            inputs.append([input_list[k] for k in input_list])
            outputs.append(results)
    print "=> average calc time:", float(t.secs)/float(n), "s"
    





if __name__ == "__main__":
    test_build_fuzzy_system()
    test_fuzzy_system()
    test_realFCL()
    #test_build_system()
    #test_real_NFS_FCL()