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
        self.log = open("trainFRBS_PHIwExpert_logfile.log", "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)  

sys.stdout = Logger()

SYS_list = ['VL_SYS', 'FWD_SYS', 'WING_SYS', 'ENG_SYS'] #a list of systems 
ASPECT_list = ['VL_SYS_TYPE', 'VL_SYS_PROP', 'VL_SYS_DRV', 'VL_SYS_TECH', \
            'FWD_SYS_PROP', 'FWD_SYS_DRV', 'FWD_SYS_TYPE',\
            'WING_SYS_TYPE', \
            'ENG_SYS_TYPE']    #list of system functional aspects 

## I/O Ranges and Limits
input_ranges = \
{   'LD'   : [5, 25],
    'w'    : [0, 150],
    'e_d'  : [0.0, 0.3],
    'phi'  : [1, 9],
    'FM'   : [0.3, 1.0],
    'f'    : [1,9],
    'V_max': [150, 550],
    'eta_p': [0.6, 1.0],
    'sigma': [0.05,0.4],
    'TP'   : [0.0, 20.0],
    'PW'   : [0.01, 5.0],
    'eta_d'  : [0.5,1.0],
    'WS'   : [15,300],
    'SFC'  : [1,9],     
    }
output_ranges = \
{   'sys_FoM' : [0.4, 1.0],
    'sys_phi' : [1.0, 9.0],
}

inputLimits = {inp: [input_ranges[inp][0] - 0.1*(input_ranges[inp][1]-input_ranges[inp][0]),
                     input_ranges[inp][1] + 0.1*(input_ranges[inp][1]-input_ranges[inp][0])]for inp in input_ranges}
outputLimits = {otp: [output_ranges[otp][0] - 0.1*(output_ranges[otp][1]-output_ranges[otp][0]),
                      output_ranges[otp][1] + 0.1*(output_ranges[otp][1]-output_ranges[otp][0])]for otp in output_ranges}

#Read in Input Data for Morph
dataIn = readFuzzyInputData('data/morphInputs_13Jun15.csv')
combinedData = buildInputs(ASPECT_list, dataIn, 'data/phiData_300pts.csv', True)
q=0 #use first (quant inputs) 

inputList = [   ('VL_SYS_TECH', 'phi'),
                ('FWD_SYS_DRV', 'eta_d'),
                ('FWD_SYS_TYPE', 'phi'),
                ('VL_SYS_TECH', 'f'),
                ('FWD_SYS_PROP', 'eta_p'),
                ('VL_SYS_TECH', 'w'),
                ('VL_SYS_TYPE', 'f'),
                ('VL_SYS_TECH', 'LD'),
                ('WING_SYS_TYPE', 'LD'),
                ('FWD_SYS_TYPE', 'TP'),
                ('VL_SYS_TYPE', 'w'),
                ('WING_SYS_TYPE', 'f'),
                ('VL_SYS_PROP', 'w'),
                ('VL_SYS_TYPE', 'phi'),
                ('VL_SYS_PROP', 'phi'), ]

inRanges = {inp[0]+"_"+inp[1] : input_ranges[inp[1]] for inp in inputList} #set exact inputs
outRanges = {'sys_phi' : output_ranges['sys_phi'] } 
 
#Turn data into fuzzy MFs
fuzzData = []
for point in combinedData:

    fuzIn = {} #create input MFs for each input
    for inp in point[q]:
        #create singleton input MFs
        mean = sum(point[q][inp])/len(point[q][inp]) #get mean of range (single value)
        fuzIn[inp[0]+'_'+inp[1]] = fuzzyOps.rangeToMF([mean], 'gauss')
    
    fuzOut = {} #create trapezoidal output MFs
    fuzOut['sys_phi'] = fuzzyOps.rangeToMF(point[2], 'gauss')
    
    fuzzData.append([fuzIn, fuzOut])

############### TRAINING CASES ###############     

############### TRAINING CASES ###############     
maxIter = 35
xConverge = 0.001

## TRAINING CASE 1:
nData = (300, 0.15)      #data to use, holdback rate
nNodes = (200, 30, 40)  #hidden nodes, input gran, output gran
learning = (0.05, 0.01) #learning rate, momentum val
fileName = "FCL_files/DFES_PHIwExperts_data(%d)_nodes(%d_%d_%d).nwf" % (nData[:1] + nNodes)
print "Case 1, Data Points: %d (%.2f holdback)" % nData,
print "Hidden Nodes: %d, Input Granularity: %d, Output Granularity: %d, " % nNodes
fuzzData1 = copy.deepcopy(fuzzData[:nData[0]])
sys = DFES(inRanges, outRanges, 'sigmoid', hidNodes=nNodes[0], inGran=nNodes[1], outGran=nNodes[2])
sys.train(fuzzData, holdback=nData[1], LR=learning[0], M=learning[1], maxIterations=maxIter, xConverge=xConverge)
sys.write_weights(fileName) #write network weight foil
print "~~~~~~~~~~~~~~~ Optimization Complete ~~~~~~~~~~~~~~~"

"""
## TRAINING CASE 2:
nData = (251, 0.2)      #data to use, holdback rate
nNodes = (400, 40, 30)  #hidden nodes, input gran, output gran
learning = (0.1, 0.03) #learning rate, momentum val
fileName = "FCL_files/DFES_PHIwExperts_data(%d)_nodes(%d_%d_%d).nwf" % (nData[:1] + nNodes)
print "Case 2, Data Points: %d (%.2f holdback)" % nData,
print "Hidden Nodes: %d, Input Granularity: %d, Output Granularity: %d, " % nNodes
fuzzData1 = copy.deepcopy(fuzzData[:nData[0]])
sys = DFES(inRanges, outRanges, 'sigmoid', hidNodes=nNodes[0], inGran=nNodes[1], outGran=nNodes[2])
sys.train(fuzzData, holdback=nData[1], LR=learning[0], M=learning[1], maxIterations=maxIter, xConverge=xConverge)
sys.write_weights(fileName) #write network weight foil
print "~~~~~~~~~~~~~~~ Optimization Complete ~~~~~~~~~~~~~~~"

## TRAINING CASE 3:
nData = (251, 0.2)      #data to use, holdback rate
nNodes = (300, 30, 40)  #hidden nodes, input gran, output gran
learning = (0.1, 0.03) #learning rate, momentum val
fileName = "FCL_files/DFES_PHIwExperts_data(%d)_nodes(%d_%d_%d).nwf" % (nData[:1] + nNodes)
print "Case 3, Data Points: %d (%.2f holdback)" % nData,
print "Hidden Nodes: %d, Input Granularity: %d, Output Granularity: %d, " % nNodes
fuzzData1 = copy.deepcopy(fuzzData[:nData[0]])
sys = DFES(inRanges, outRanges, 'sigmoid', hidNodes=nNodes[0], inGran=nNodes[1], outGran=nNodes[2])
sys.train(fuzzData, holdback=nData[1], LR=learning[0], M=learning[1], maxIterations=maxIter, xConverge=xConverge)
sys.write_weights(fileName) #write network weight foil
print "~~~~~~~~~~~~~~~ Optimization Complete ~~~~~~~~~~~~~~~"

## TRAINING CASE 4:
nData = (251, 0.2)      #data to use, holdback rate
nNodes = (300, 30, 50)  #hidden nodes, input gran, output gran
learning = (0.1, 0.03) #learning rate, momentum val
fileName = "FCL_files/DFES_PHIwExperts_data(%d)_nodes(%d_%d_%d).nwf" % (nData[:1] + nNodes)
print "Case 4, Data Points: %d (%.2f holdback)" % nData,
print "Hidden Nodes: %d, Input Granularity: %d, Output Granularity: %d, " % nNodes
fuzzData1 = copy.deepcopy(fuzzData[:nData[0]])
sys = DFES(inRanges, outRanges, 'sigmoid', hidNodes=nNodes[0], inGran=nNodes[1], outGran=nNodes[2])
sys.train(fuzzData, holdback=nData[1], LR=learning[0], M=learning[1], maxIterations=maxIter, xConverge=xConverge)
sys.write_weights(fileName) #write network weight foil
print "~~~~~~~~~~~~~~~ Optimization Complete ~~~~~~~~~~~~~~~"

## TRAINING CASE 5:
nData = (251, 0.2)      #data to use, holdback rate
nNodes = (350, 30, 50)  #hidden nodes, input gran, output gran
learning = (0.1, 0.03) #learning rate, momentum val
fileName = "FCL_files/DFES_PHIwExperts_data(%d)_nodes(%d_%d_%d).nwf" % (nData[:1] + nNodes)
print "Case 5, Data Points: %d (%.2f holdback)" % nData,
print "Hidden Nodes: %d, Input Granularity: %d, Output Granularity: %d, " % nNodes
fuzzData1 = copy.deepcopy(fuzzData[:nData[0]])
sys = DFES(inRanges, outRanges, 'sigmoid', hidNodes=nNodes[0], inGran=nNodes[1], outGran=nNodes[2])
sys.train(fuzzData, holdback=nData[1], LR=learning[0], M=learning[1], maxIterations=maxIter, xConverge=xConverge)
sys.write_weights(fileName) #write network weight foil
print "~~~~~~~~~~~~~~~ Optimization Complete ~~~~~~~~~~~~~~~"
"""









