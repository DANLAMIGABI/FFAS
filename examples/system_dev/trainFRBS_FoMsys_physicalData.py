
"""
author: Frank Patterson - 4Apr2015
Testing training modules
"""
import copy
import random
from training import *
from systems import *
import matplotlib.pyplot as plt
from timer import Timer

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
    'eta_d': [0.5,1.0],
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


### FIGURE OF MERIT SYSETMS (FUZZY DATA) ###
#CREATE DATA
ASPECT_list = ['VL_SYS_TYPE', 'VL_SYS_PROP', 'VL_SYS_DRV', 'VL_SYS_TECH', \
            'FWD_SYS_PROP', 'FWD_SYS_DRV', 'FWD_SYS_TYPE',\
            'WING_SYS_TYPE', \
            'ENG_SYS_TYPE']
            
combinedData = buildInputs(ASPECT_list, None, 'data/FoM_generatedData_15Jun15.csv', False,        #training data set
                           inputCols={'w':1, 'sigma':0, 'e_d':2, 'eta':3,},
                           outputCols={'sysFoM':4})
nMaxTrain = 400
nMaxVal = 100

################################################################################

### OPTIMIZE SYSTEM: 
nMFin  = 5              #number of input MFs
MFinType = 'gauss'      #input MF type
nMFout = 9              #number of output MFs
MFoutType = 'gauss'     #output MF type
inMFform = 'sing'       #input data MF form
outMFform = 'gauss'     #output data MF form
filename = 'FCL_files/FoMsys_trained_5-2In9-2Out_sing-gauss400tData_100vData.fcl'

inputRanges = {    ('DATA', 'e_d'):     copy.deepcopy(input_ranges['e_d']),
                   ('DATA', 'sigma'):   copy.deepcopy(input_ranges['sigma']),
                   ('DATA', 'w'):       copy.deepcopy(input_ranges['w']),
                   ('DATA', 'eta'):     copy.deepcopy(input_ranges['eta_d'])
              }
MFstructIn = [  [('DATA', 'e_d'),   ['A'+str(i+1) for i in range(nMFin)], [MFinType]*nMFin ],
                [('DATA', 'sigma'), ['A'+str(i+1) for i in range(nMFin)], [MFinType]*nMFin ],
                [('DATA', 'w'),     ['A'+str(i+1) for i in range(nMFin)], [MFinType]*nMFin ],
                [('DATA', 'eta'),   ['A'+str(i+1) for i in range(nMFin)], [MFinType]*nMFin ],
             ]     
outputRanges = {'sys_FoM' : output_ranges['sys_FoM'] }
MFstructOut = [ [('sys_FoM'), ['A'+str(i) for i in range(nMFout)], [MFoutType]*nMFout],
              ]            
              
print "OPTIMIZING: ", filename

combData = copy.deepcopy(combinedData)

with Timer() as t:
    run_optimization(combData[:400], combData[400:500], 
                    MFstructIn, inputRanges, MFstructOut, outputRanges, 
                    filename, nMaxTrain, nMaxVal,
                    inMFform=inMFform, outMFform=outMFform, defuzz=None, 
                    optMethod='GA', popX=1.5)

print "*** OPTIMIZATION COMPLETE in", t.secs, "s ***"
print filename
print "*********************************************************************"

                                                    

### OPTIMIZE SYSTEM: 
nMFin  = 5              #number of input MFs
MFinType = 'tri'      #input MF type
nMFout = 9              #number of output MFs
MFoutType = 'tri'     #output MF type
inMFform = 'sing'       #input data MF form
outMFform = 'tri'     #output data MF form
filename = 'FCL_files/FoMsys_trained_5-3In9-3Out_sing-tri400tData_100vData.fcl'

inputRanges = {    ('DATA', 'e_d'):     copy.deepcopy(input_ranges['e_d']),
                   ('DATA', 'sigma'):   copy.deepcopy(input_ranges['sigma']),
                   ('DATA', 'w'):       copy.deepcopy(input_ranges['w']),
                   ('DATA', 'eta'):     copy.deepcopy(input_ranges['eta_d'])
              }
MFstructIn = [  [('DATA', 'e_d'),   ['A'+str(i+1) for i in range(nMFin)], [MFinType]*nMFin ],
                [('DATA', 'sigma'), ['A'+str(i+1) for i in range(nMFin)], [MFinType]*nMFin ],
                [('DATA', 'w'),     ['A'+str(i+1) for i in range(nMFin)], [MFinType]*nMFin ],
                [('DATA', 'eta'),   ['A'+str(i+1) for i in range(nMFin)], [MFinType]*nMFin ],
             ]     
outputRanges = {'sys_FoM' : output_ranges['sys_FoM'] }
MFstructOut = [ [('sys_FoM'), ['A'+str(i) for i in range(nMFout)], [MFoutType]*nMFout],
              ]            
              
print "OPTIMIZING: ", filename

combData = copy.deepcopy(combinedData)

with Timer() as t:
    run_optimization(combData[:400], combData[400:500], 
                    MFstructIn, inputRanges, MFstructOut, outputRanges, 
                    filename, nMaxTrain, nMaxVal,
                    inMFform=inMFform, outMFform=outMFform, defuzz=None, 
                    optMethod='diffEv', popX=1.0)

print "*** OPTIMIZATION COMPLETE in", t.secs, "s ***"
print filename
print "*********************************************************************"

 