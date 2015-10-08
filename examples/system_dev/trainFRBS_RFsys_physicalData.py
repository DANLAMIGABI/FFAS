
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
    'phi'  : [0.5, 0.85],
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
    'sys_GWT' : [1000,75000],
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
            
combData_GWT = buildInputs(ASPECT_list, None, 'data/RF_GWT-P-VH-T-eWT_genData_2Sep15_14Sep15_LIMITEDFINAL.csv', False,        #training data set
                           inputCols={'phi':0, 'w':1, 'WS':2, 'sys_etaP':3, 'eta_d':4, 'sys_FoM':5,
                                      'e_d':6, 'dragX':7, 'SFC_quant':8, 'type':9,},# 'tech':10, 'jet':11},
                           outputCols={'sys_GWT':12})
combData_Pin = buildInputs(ASPECT_list, None, 'data/RF_GWT-P-VH-T-eWT_genData_2Sep15_14Sep15_LIMITEDFINAL.csv', False,        #training data set
                           inputCols={'phi':0, 'w':1, 'WS':2, 'sys_etaP':3, 'eta_d':4, 'sys_FoM':5,
                                      'e_d':6, 'dragX':7, 'SFC_quant':8, 'type':9,},# 'tech':10, 'jet':11},
                           outputCols={'sys_Pinst':13})
combData_Tin = buildInputs(ASPECT_list, None, 'data/RF_GWT-P-VH-T-eWT_genData_2Sep15_14Sep15_LIMITEDFINAL.csv', False,        #training data set
                           inputCols={'phi':0, 'w':1, 'WS':2, 'sys_etaP':3, 'eta_d':4, 'sys_FoM':5,
                                      'e_d':6, 'dragX':7, 'SFC_quant':8, 'type':9,},# 'tech':10, 'jet':11},
                           outputCols={'sys_Tinst':14})
combData_VH  = buildInputs(ASPECT_list, None, 'data/RF_GWT-P-VH-T-eWT_genData_2Sep15_14Sep15_LIMITEDFINAL.csv', False,        #training data set
                           inputCols={'phi':0, 'w':1, 'WS':2, 'sys_etaP':3, 'eta_d':4, 'sys_FoM':5,
                                      'e_d':6, 'dragX':7, 'SFC_quant':8, 'type':9,},# 'tech':10, 'jet':11},
                           outputCols={'sys_VH':15})
combData_eWT = buildInputs(ASPECT_list, None, 'data/RF_GWT-P-VH-T-eWT_genData_2Sep15_14Sep15_LIMITEDFINAL.csv', False,        #training data set
                           inputCols={'phi':0, 'w':1, 'WS':2, 'sys_etaP':3, 'eta_d':4, 'sys_FoM':5,
                                      'e_d':6, 'dragX':7, 'SFC_quant':8, 'type':9,},# 'tech':10, 'jet':11},
                           outputCols={'sys_eWT':16})
q=0 #use first (quant inputs)

#print combData_GWT[0]

nMaxTrain = 500
nMaxVal = 100

inputRanges = { ('DATA', 'phi'):        [0.5, 0.95],
                ('DATA', 'w'):          [1.0, 150.0],
                ('DATA', 'WS'):         [15.0, 300],
                ('DATA', 'sys_etaP'):   [0.6, 1.0],
                ('DATA', 'eta_d'):      [0.4, 1.0],
                ('DATA', 'sys_FoM'):    [0.4, 1.0],
                ('DATA', 'e_d'):        [0.0, 0.3],
                ('DATA', 'SFC_quant'):  [0.45, 1.05],
                ('DATA', 'dragX'):      [0.6, 1.15],
                ('DATA', 'type'):       [0.5, 3.5],
                #'DATA_tech':       [-0.5, 4.5],
                #'DATA_jet':        [0.5, 2.5],
              }
            
outRanges_GWT = {'sys_GWT'   : [1000,75000] }
outRanges_Pin = {'sys_Pinst' : [] }
outRanges_Tin = {'sys_Tinst' : [] }
outRanges_VH  = {'sys_VH'    : [] }
outRanges_eWT = {'sys_eWT'   : [] }

################################################################################

### OPTIMIZE SYSTEM: 
nMFin  = 5              #number of input MFs
MFinType = 'gauss'      #input MF type
nMFout = 9              #number of output MFs
MFoutType = 'gauss'     #output MF type
inMFform = 'sing'       #input data MF form
outMFform = 'gauss'     #output data MF form
filename = 'FCL_files/RFsys_trained_5-2In9-2Out_sing-gauss400tData_100vData.fcl'

MFstructIn = [  [('DATA', 'phi'),        ['A'+str(i+1) for i in range(nMFin)], [MFinType]*nMFin ],
                [('DATA', 'w'),          ['A'+str(i+1) for i in range(nMFin)], [MFinType]*nMFin ],
                [('DATA', 'WS'),         ['A'+str(i+1) for i in range(nMFin)], [MFinType]*nMFin ],
                [('DATA', 'sys_etaP'),   ['A'+str(i+1) for i in range(nMFin)], [MFinType]*nMFin ],
                [('DATA', 'eta_d'),      ['A'+str(i+1) for i in range(nMFin)], [MFinType]*nMFin ],
                [('DATA', 'sys_FoM'),    ['A'+str(i+1) for i in range(nMFin)], [MFinType]*nMFin ],
                [('DATA', 'e_d'),        ['A'+str(i+1) for i in range(nMFin)], [MFinType]*nMFin ],
                [('DATA', 'SFC_quant'),  ['A'+str(i+1) for i in range(nMFin)], [MFinType]*nMFin ],
                [('DATA', 'dragX'),      ['A'+str(i+1) for i in range(nMFin)], [MFinType]*nMFin ],
                [('DATA', 'type'),       ['A'+str(i+1) for i in range(nMFin)], [MFinType]*nMFin ],
             ]     
outputRanges = {'sys_GWT' : output_ranges['sys_GWT'] }
MFstructOut = [ [('sys_GWT'), ['A'+str(i) for i in range(nMFout)], [MFoutType]*nMFout],
              ]            
              
print "OPTIMIZING: ", filename

combData = copy.deepcopy(combData_GWT)

with Timer() as t:
    run_optimization(combData[:500], combData[500:600], 
                    MFstructIn, inputRanges, MFstructOut, outputRanges, 
                    filename, nMaxTrain, nMaxVal,
                    inMFform=inMFform, outMFform=outMFform, defuzz=None, 
                    optMethod='GA', popX=1.1)

print "*** OPTIMIZATION COMPLETE in", t.secs, "s ***"
print filename
print "*********************************************************************"

                                                    

 