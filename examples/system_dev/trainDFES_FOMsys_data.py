# -*- coding: utf-8 -*-
"""
@author: frankpatterson
"""
import copy

import numpy as np
import skfuzzy as fuzz

import fuzzy_operations as fuzzyOps

from systems import *
from training import *

import sys

class Logger(object):
    """
    log output to file and screen
    """
    def __init__(self):
        self.terminal = sys.stdout
        self.log = open("trainDFES_FOMsys_data.log", "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)  

sys.stdout = Logger()

            
            
ASPECT_list = ['VL_SYS_TYPE', 'VL_SYS_PROP', 'VL_SYS_DRV', 'VL_SYS_TECH', \
            'FWD_SYS_PROP', 'FWD_SYS_DRV', 'FWD_SYS_TYPE',\
            'WING_SYS_TYPE', \
            'ENG_SYS_TYPE']    #list of system functional aspects 

## Baseline Input MFs
input_ranges = \
{   'LD'   : [5, 25],
    'w'    : [0, 150],
    'e_d'  : [0.0, 0.3],
    'phi'  : [1, 9],
    'FM'   : [0.3, 1.0],
    'f'    : [1,9],
    'V_max': [150, 550],
    'eta_p': [0.6, 1.0],
    'sigma': [0.05, 0.4],
    'TP'   : [0.0, 20.0],
    'PW'   : [0.01, 5.0],
    'eta'  : [0.5,1.0],
    'WS'   : [15,300],
    'SFC'  : [1,9],     
    }
output_ranges = \
{   'sys_FoM' : [0.4, 1.0],
}

inputLimits = {inp: [input_ranges[inp][0] - 0.1*(input_ranges[inp][1]-input_ranges[inp][0]),
                     input_ranges[inp][1] + 0.1*(input_ranges[inp][1]-input_ranges[inp][0])]for inp in input_ranges}
outputLimits = {otp: [output_ranges[otp][0] - 0.1*(output_ranges[otp][1]-output_ranges[otp][0]),
                      output_ranges[otp][1] + 0.1*(output_ranges[otp][1]-output_ranges[otp][0])]for otp in output_ranges}


### FIGURE OF MERIT SYSETMS (FUZZY DATA) ###
inRanges = {    'DATA_e_d':     input_ranges['e_d'],
                'DATA_sigma':   input_ranges['sigma'],
                'DATA_w':       input_ranges['w'],
                'DATA_eta':     input_ranges['eta']}
outRanges = {'sys_FoM' : output_ranges['sys_FoM'] }

#CREATE DATA
ASPECT_list = ['VL_SYS_TYPE', 'VL_SYS_PROP', 'VL_SYS_DRV', 'VL_SYS_TECH', \
            'FWD_SYS_PROP', 'FWD_SYS_DRV', 'FWD_SYS_TYPE',\
            'WING_SYS_TYPE', \
            'ENG_SYS_TYPE']
            
combData = buildInputs(ASPECT_list, None, 'data/FoM_generatedData_15Jun15.csv', False,        #training data set
                    inputCols={'w':1, 'sigma':0, 'e_d':2, 'eta':3,},
                    outputCols={'sysFoM':4})

q=0 #use first (quant inputs)


#Turn data into fuzzy MFs
"""
fuzzData = []
for point in combData:

    fuzIn = {} #create input MFs for each input
    for inp in point[q]:
        #create singleton input MFs
        mean = sum(point[q][inp])/len(point[q][inp]) #get mean of range (single value)
        fuzIn[inp[0]+'_'+inp[1]] = fuzzyOps.rangeToMF([mean], 'gauss')
    
    fuzOut = {} #create trapezoidal output MFs
    fuzOut['sys_FoM'] = fuzzyOps.rangeToMF(point[2], 'gauss')
    
    fuzzData.append([fuzIn, fuzOut])    
"""
fuzzData_tri = []
for point in combData:

    fuzIn = {} #create input MFs for each input
    for inp in point[q]:
        #create singleton input MFs
        mean = sum(point[q][inp])/len(point[q][inp]) #get mean of range (single value)
        fuzIn[inp[0]+'_'+inp[1]] = fuzzyOps.rangeToMF([mean], 'tri')
    
    fuzOut = {} #create trapezoidal output MFs
    fuzOut['sys_FoM'] = fuzzyOps.rangeToMF(point[2], 'tri')
    
    fuzzData_tri.append([fuzIn, fuzOut])
    
fuzzData_trap = []
for point in combData:

    fuzIn = {} #create input MFs for each input
    for inp in point[q]:
        #create singleton input MFs
        mean = sum(point[q][inp])/len(point[q][inp]) #get mean of range (single value)
        fuzIn[inp[0]+'_'+inp[1]] = fuzzyOps.rangeToMF([mean], 'trap')
    
    fuzOut = {} #create trapezoidal output MFs
    fuzOut['sys_FoM'] = fuzzyOps.rangeToMF(point[2], 'trap')
    
    fuzzData_trap.append([fuzIn, fuzOut])

############### TRAINING CASES ###############     
maxIter = 30
xConverge = 0.0005


## TRAINING CASE 1:
nData = (500, 0.2)      #data to use, holdback rate
nNodes = (100, 30, 30)  #hidden nodes, input gran, output gran
learning = (0.1, 0.03) #learning rate, momentum val
fileName = "FCL_files/DFES_FOMdata_data(%d)_nodes(%d_%d_%d).nwf" % (nData[:1] + nNodes)
print "Case MF-2, Triangular: Data Points: %d (%.2f holdback)" % nData,
print "Hidden Nodes: %d, Input Granularity: %d, Output Granularity: %d, " % nNodes
fuzzData_tri1 = copy.deepcopy(fuzzData_tri[:nData[0]])
sys = DFES(inRanges, outRanges, 'sigmoid', hidNodes=nNodes[0], inGran=nNodes[1], outGran=nNodes[2])
sys.train(fuzzData_tri1, holdback=nData[1], LR=learning[0], M=learning[1], maxIterations=maxIter, xConverge=xConverge)
sys.write_weights(fileName) #write network weight foil
print "~~~~~~~~~~~~~~~ Optimization Complete ~~~~~~~~~~~~~~~"

"""
## TRAINING CASE 2:
nData = (500, 0.2)      #data to use, holdback rate
nNodes = (100, 30, 30)  #hidden nodes, input gran, output gran
learning = (0.1, 0.03) #learning rate, momentum val
fileName = "FCL_files/DFES_FOMdata_data(%d)_nodes(%d_%d_%d).nwf" % (nData[:1] + nNodes)
print "Case MF-3, Trapezoidal: Data Points: %d (%.2f holdback)" % nData,
print "Hidden Nodes: %d, Input Granularity: %d, Output Granularity: %d, " % nNodes
fuzzData_trap1 = copy.deepcopy(fuzzData_trap[:nData[0]])
sys = DFES(inRanges, outRanges, 'sigmoid', hidNodes=nNodes[0], inGran=nNodes[1], outGran=nNodes[2])
sys.train(fuzzData_trap1, holdback=nData[1], LR=learning[0], M=learning[1], maxIterations=maxIter, xConverge=xConverge)
sys.write_weights(fileName) #write network weight foil
print "~~~~~~~~~~~~~~~ Optimization Complete ~~~~~~~~~~~~~~~"


## TRAINING CASE 3:
nData = (500, 0.2)      #data to use, holdback rate
nNodes = (100, 30, 30)  #hidden nodes, input gran, output gran
learning = (0.1, 0.03) #learning rate, momentum val
fileName = "FCL_files/DFES_FOMdata_data(%d)_nodes(%d_%d_%d).nwf" % (nData[:1] + nNodes)
print "Case MF-1, Gaussian: Data Points: %d (%.2f holdback)" % nData,
print "Hidden Nodes: %d, Input Granularity: %d, Output Granularity: %d, " % nNodes
fuzzData1 = copy.deepcopy(fuzzData[:nData[0]])
sys = DFES(inRanges, outRanges, 'sigmoid', hidNodes=nNodes[0], inGran=nNodes[1], outGran=nNodes[2])
sys.train(fuzzData1, holdback=nData[1], LR=learning[0], M=learning[1], maxIterations=maxIter, xConverge=xConverge)
sys.write_weights(fileName) #write network weight foil
print "~~~~~~~~~~~~~~~ Optimization Complete ~~~~~~~~~~~~~~~"


## TRAINING CASE 2:
nData = (500, 0.2)      #data to use, holdback rate
nNodes = (130, 40, 40)  #hidden nodes, input gran, output gran
learning = (0.1, 0.03) #learning rate, momentum val
fileName = "FCL_files/DFES_FOMdata_data(%d)_nodes(%d_%d_%d).nwf" % (nData[:1] + nNodes)
print "Case 3-1, Data Points: %d (%.2f holdback)" % nData,
print "Hidden Nodes: %d, Input Granularity: %d, Output Granularity: %d, " % nNodes
fuzzData1 = copy.deepcopy(fuzzData[:nData[0]])
sys = DFES(inRanges, outRanges, 'sigmoid', hidNodes=nNodes[0], inGran=nNodes[1], outGran=nNodes[2])
sys.train(fuzzData, holdback=nData[1], LR=learning[0], M=learning[1], maxIterations=maxIter, xConverge=xConverge)
sys.write_weights(fileName) #write network weight foil
print "~~~~~~~~~~~~~~~ Optimization Complete ~~~~~~~~~~~~~~~"

## TRAINING CASE 3:
nData = (500, 0.2)      #data to use, holdback rate
nNodes = (160, 50, 50)  #hidden nodes, input gran, output gran
learning = (0.1, 0.03) #learning rate, momentum val
fileName = "FCL_files/DFES_FOMdata_data(%d)_nodes(%d_%d_%d).nwf" % (nData[:1] + nNodes)
print "Case 3-1, Data Points: %d (%.2f holdback)" % nData,
print "Hidden Nodes: %d, Input Granularity: %d, Output Granularity: %d, " % nNodes
fuzzData1 = copy.deepcopy(fuzzData[:nData[0]])
sys = DFES(inRanges, outRanges, 'sigmoid', hidNodes=nNodes[0], inGran=nNodes[1], outGran=nNodes[2])
sys.train(fuzzData, holdback=nData[1], LR=learning[0], M=learning[1], maxIterations=maxIter, xConverge=xConverge)
sys.write_weights(fileName) #write network weight foil
print "~~~~~~~~~~~~~~~ Optimization Complete ~~~~~~~~~~~~~~~"

## TRAINING CASE 4:
nData = (500, 0.2)      #data to use, holdback rate
nNodes = (160, 50, 30)  #hidden nodes, input gran, output gran
learning = (0.1, 0.03) #learning rate, momentum val
fileName = "FCL_files/DFES_FOMdata_data(%d)_nodes(%d_%d_%d).nwf" % (nData[:1] + nNodes)
print "Case 3-1, Data Points: %d (%.2f holdback)" % nData,
print "Hidden Nodes: %d, Input Granularity: %d, Output Granularity: %d, " % nNodes
fuzzData1 = copy.deepcopy(fuzzData[:nData[0]])
sys = DFES(inRanges, outRanges, 'sigmoid', hidNodes=nNodes[0], inGran=nNodes[1], outGran=nNodes[2])
sys.train(fuzzData, holdback=nData[1], LR=learning[0], M=learning[1], maxIterations=maxIter, xConverge=xConverge)
sys.write_weights(fileName) #write network weight foil
print "~~~~~~~~~~~~~~~ Optimization Complete ~~~~~~~~~~~~~~~"

## TRAINING CASE 5:
nData = (500, 0.2)      #data to use, holdback rate
nNodes = (120, 30, 50)  #hidden nodes, input gran, output gran
learning = (0.1, 0.03) #learning rate, momentum val
fileName = "FCL_files/DFES_FOMdata_data(%d)_nodes(%d_%d_%d).nwf" % (nData[:1] + nNodes)
print "Case 3-1, Data Points: %d (%.2f holdback)" % nData,
print "Hidden Nodes: %d, Input Granularity: %d, Output Granularity: %d, " % nNodes
fuzzData1 = copy.deepcopy(fuzzData[:nData[0]])
sys = DFES(inRanges, outRanges, 'sigmoid', hidNodes=nNodes[0], inGran=nNodes[1], outGran=nNodes[2])
sys.train(fuzzData, holdback=nData[1], LR=learning[0], M=learning[1], maxIterations=maxIter, xConverge=xConverge)
sys.write_weights(fileName) #write network weight foil
print "~~~~~~~~~~~~~~~ Optimization Complete ~~~~~~~~~~~~~~~"

## TRAINING CASE 6:
nData = (500, 0.2)      #data to use, holdback rate
nNodes = (160, 50, 30)  #hidden nodes, input gran, output gran
learning = (0.1, 0.03) #learning rate, momentum val
fileName = "FCL_files/DFES_FOMdata_data(%d)_nodes(%d_%d_%d).nwf" % (nData[:1] + nNodes)
print "Case 3-1, Data Points: %d (%.2f holdback)" % nData,
print "Hidden Nodes: %d, Input Granularity: %d, Output Granularity: %d, " % nNodes
fuzzData1 = copy.deepcopy(fuzzData[:nData[0]])
sys = DFES(inRanges, outRanges, 'sigmoid', hidNodes=nNodes[0], inGran=nNodes[1], outGran=nNodes[2])
sys.train(fuzzData, holdback=nData[1], LR=learning[0], M=learning[1], maxIterations=maxIter, xConverge=xConverge)
sys.write_weights(fileName) #write network weight foil
print "~~~~~~~~~~~~~~~ Optimization Complete ~~~~~~~~~~~~~~~"

"""


















