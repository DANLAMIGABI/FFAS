"""
author: Frank Patterson - 4Apr2015
Testing training modules
"""
import copy
from training import *
from systems import *
import matplotlib.pyplot as plt
import sys
import random

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
{   'sys_FoM' : [0.0, 1.0],
    'sys_phi' : [1.0, 9.0],
}

inputLimits = {inp: [input_ranges[inp][0] - 0.1*(input_ranges[inp][1]-input_ranges[inp][0]),
                     input_ranges[inp][1] + 0.1*(input_ranges[inp][1]-input_ranges[inp][0])]for inp in input_ranges}
outputLimits = {otp: [output_ranges[otp][0] - 0.1*(output_ranges[otp][1]-output_ranges[otp][0]),
                      output_ranges[otp][1] + 0.1*(output_ranges[otp][1]-output_ranges[otp][0])]for otp in output_ranges}

#Read in Input Data for Morph
dataIn = readFuzzyInputData('data/morphInputs_13Jun15.csv')

### get data linked to inputs
combinedData = buildInputs(ASPECT_list, dataIn, 'data/phiData_300pts.csv', True)
#random.shuffle(combinedData)

#get union of FWD and VL system empty weight ratio and average wing loading
#operations1 = { ('VL_SYS_UNION', 'phi'):  ( [('VL_SYS_TYPE', 'phi'), ('VL_SYS_PROP', 'phi'), ('VL_SYS_DRV', 'phi'), ('VL_SYS_TECH', 'phi')], 'UNION' ),
#                ('FWD_SYS_UNION', 'phi'): ( [('FWD_SYS_TYPE', 'phi'), ('FWD_SYS_PROP', 'phi'), ('FWD_SYS_DRV', 'phi')], 'UNION' ),
#                ('VL_SYS_UNION', 'w'):  ( [('VL_SYS_TYPE', 'w'), ('VL_SYS_PROP', 'w'), ('VL_SYS_TECH', 'w')], 'AVERAGE' ),
#                }
#combinedData = combine_inputs(combinedData, operations1)

#get average system empty weight ratio
#operations1 = {('SYS_PHI_AVGofUNIONS', 'phi'):  ( [('VL_SYS_UNION', 'phi'), ('FWD_SYS_UNION', 'phi'), ('WING_SYS_TYPE', 'phi'), ('ENG_SYS_TYPE', 'phi')], 'AVERAGE' ),
#                }
#combinedData = combine_inputs(combinedData, operations1)

#write_expert_data(combinedData, 'data/POC_combinedPhiData.csv')

nMaxTrain = 250
nMaxVal = 50

inputList = [   ('VL_SYS_TECH', 'phi'),
                ('FWD_SYS_PROP', 'eta_p'),
                ('VL_SYS_DRV', 'phi'),
                ('VL_SYS_TECH', 'w'),
                ('WING_SYS_TYPE', 'WS'),
                ('FWD_SYS_TYPE', 'TP'),
                ('VL_SYS_TYPE', 'w'),
                ('VL_SYS_TYPE', 'e_d'),
                ('VL_SYS_TYPE', 'TP'),
                ('VL_SYS_TYPE', 'phi'),
                ('VL_SYS_PROP', 'phi'),     ]

################################################################################



### OPTIMIZE SYSTEM: 7 GAUSS IN => 7 GAUSS OUT
nMFin  = 5             #number of input MFs
MFinType = 'gauss'      #input MF type
nMFout = 9              #number of output MFs
MFoutType = 'gauss'     #output MF type
inMFform = 'tri'       #input data MF form
outMFform = 'tri'     #output data MF form
filename = 'FCL_files/PHIsys_trained_5-2In9-2Out_gauss250tData_tri50vData.fcl'

inputRanges = { inp: copy.deepcopy(input_ranges[inp[1]])  for inp in inputList }
MFstructIn = [ [inp, ['A'+str(i+1) for i in range(nMFin)], [MFinType]*nMFin] for inp in inputRanges ]   

outputRanges = {'sys_phi' : copy.deepcopy(outputLimits['sys_phi'])}
MFstructOut = [ [otp, ['A'+str(i) for i in range(nMFout)], [MFoutType]*nMFout] for otp in outputRanges ]

print "OPTIMIZING: ", filename

combinedData = copy.deepcopy(combinedData)

run_optimization(combinedData[:250], combinedData[250:],   
                 MFstructIn, inputRanges, MFstructOut, outputRanges, 
                 filename, nMaxTrain, nMaxVal,
                 inMFform=inMFform, outMFform=outMFform, defuzz=None,
                 optMethod = 'GA')

                                                    
print "*** OPTIMIZATION COMPLETE ***"
print filename
print "*********************************************************************"

"""
### OPTIMIZE SYSTEM: 7 GAUSS IN => 7 GAUSS OUT
nMFin  = 7              #number of input MFs
MFinType = 'gauss'      #input MF type
nMFout = 7              #number of output MFs
MFoutType = 'gauss'     #output MF type
inMFform = 'tri'       #input data MF form
outMFform = 'tri'     #output data MF form
filename = 'FCL_files/PHIsys_trained_7-2In7-2Out_tri200tData_tri50vData.fcl'

inputRanges = { inp: copy.deepcopy(input_ranges[inp[1]])  for inp in inputList }
MFstructIn = [ [inp, ['A'+str(i+1) for i in range(nMFin)], [MFinType]*nMFin] for inp in inputRanges ]   

outputRanges = {'sys_phi' : copy.deepcopy(outputLimits['sys_phi'])}
MFstructOut = [ [otp, ['A'+str(i) for i in range(nMFout)], [MFoutType]*nMFout] for otp in outputRanges ]

print "OPTIMIZING: ", filename

combinedData = copy.deepcopy(combinedData)

run_optimization_mp(combinedData[:201], combinedData[201:],   
                 MFstructIn, inputRanges, MFstructOut, outputRanges, 
                 filename, nMaxTrain, nMaxVal,
                 inMFform=inMFform, outMFform=outMFform, defuzz=None,
                 optMethod = 'diffEv')

                                                    
print "*** OPTIMIZATION COMPLETE ***"
print filename
print "*********************************************************************"

### OPTIMIZE SYSTEM: 7 GAUSS IN => 7 GAUSS OUT
nMFin  = 7              #number of input MFs
MFinType = 'tri'      #input MF type
nMFout = 7              #number of output MFs
MFoutType = 'tri'     #output MF type
inMFform = 'gauss'       #input data MF form
outMFform = 'gauss'     #output data MF form
filename = 'FCL_files/PHIsys_trained_7-3In7-3Out_gauss200tData_tri50vData.fcl'

inputRanges = { inp: copy.deepcopy(input_ranges[inp[1]])  for inp in inputList }
MFstructIn = [ [inp, ['A'+str(i+1) for i in range(nMFin)], [MFinType]*nMFin] for inp in inputRanges ]   

outputRanges = {'sys_phi' : copy.deepcopy(outputLimits['sys_phi'])}
MFstructOut = [ [otp, ['A'+str(i) for i in range(nMFout)], [MFoutType]*nMFout] for otp in outputRanges ]

print "OPTIMIZING: ", filename

combinedData = copy.deepcopy(combinedData)

run_optimization_mp(combinedData[:201], combinedData[201:],   
                 MFstructIn, inputRanges, MFstructOut, outputRanges, 
                 filename, nMaxTrain, nMaxVal,
                 inMFform=inMFform, outMFform=outMFform, defuzz=None,
                 optMethod = 'diffEv')

                                                    
print "*** OPTIMIZATION COMPLETE ***"
print filename
print "*********************************************************************"
"""
 