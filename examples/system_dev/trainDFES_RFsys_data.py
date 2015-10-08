#author - Frank Patterson
import copy
import random
import numpy as np
import skfuzzy as fuzz

import fuzzy_operations as fuzzyOps

from systems import *
from training import *
from fuzzy_error import fuzDistAC

import matplotlib.pyplot as plt
plt.ioff()

import sys

class Logger(object):
    """
    log output to file and screen
    """
    def __init__(self):
        self.terminal = sys.stdout
        self.log = open("trainDFES_RFsys_data_2.log", "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)  

sys.stdout = Logger()


def limitData(dataList, outputKey):
    """Limit input data prior to training"""
    print 'Limiting', outputKey, 'to', output_ranges[outputKey]
    limitedData = []
    for i in range(len(dataList)):
        ldi = [dataList[i][0], dataList[i][1]]
        flag = 0
        if dataList[i][2][0] < output_ranges[outputKey][0]:
            flag = 1
            dataList[i][2][0] = output_ranges[outputKey][0]
        if dataList[i][2][1] > output_ranges[outputKey][1]:
            flag = 1
            dataList[i][2][1] = output_ranges[outputKey][1]
        
        if flag == 0:
            ldi.append(dataList[i][2])
            limitedData.append(ldi)
        
    return limitedData #change to datalist to just limit all raw data

def fuzzifyData(dataFile, outputKey, inMFtype='gauss', outMFtype='gauss'):
    fuzzData = []
    
    for point in dataFile:
        fuzIn = {} #create input MFs for each input
        for inp in point[q]:
            #create singleton input MFs
            mean = sum(point[q][inp])/len(point[q][inp]) #get mean of range (single value)
            fuzIn[inp[0]+'_'+inp[1]] = fuzzyOps.rangeToMF([mean], inMFtype)
        fuzOut = {} 
        fuzOut[outputKey] = fuzzyOps.rangeToMF(point[2], outMFtype)
        fuzzData.append([fuzIn, fuzOut])
        
    return fuzzData
############################################################################################
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
{   'sys_FoM'   : [0.4, 1.0],
    'sys_phi'   : [1.0, 9.0],
    'sys_GWT'   : [1000., 30000.],
    'sys_Pinst' : [1000., 15000.],
    'sys_Tinst' : [1000,  25000.],
    'sys_VH'    : [100.,  500.],
    'sys_eWT'   : [500.,  15000.],
}

inputLimits = {inp: [input_ranges[inp][0] - 0.1*(input_ranges[inp][1]-input_ranges[inp][0]),
                     input_ranges[inp][1] + 0.1*(input_ranges[inp][1]-input_ranges[inp][0])]for inp in input_ranges}
outputLimits = {otp: [output_ranges[otp][0] - 0.1*(output_ranges[otp][1]-output_ranges[otp][0]),
                      output_ranges[otp][1] + 0.1*(output_ranges[otp][1]-output_ranges[otp][0])]for otp in output_ranges}

"""
DATA FILE: COLUMNS:
0  - ('DATA', 'phi')        [.5, .85]
1  - ('DATA', 'w')          [1, 150]
2  - ('DATA', 'WS')         [15, 300]
3  - ('DATA', 'eta_p')      [.6, 1.0]
4  - ('DATA', 'eta_d')      [.7, 1.0]
5  - ('DATA', 'sys_FoM')    [.3, 1.0]
6  - ('DATA', 'e_d')        [0.0, 0.3]
7  - ('DATA', 'SFC_quant')  [0.35, 0.75]
8  - ('DATA', 'dragX')      [0.6, 1.15]
9  - ('DATA', 'type')       (1-tilt, 2-compound, 3-other)
10  - ('DATA', 'tech')       (1-none, 2-varRPM, 3-varDIA, 4-autogyro)
11 - ('DATA', 'jet')        (1-false, 2-true)
12 - sys_GWT                [5000, 70000]
13 - sys_Pinst              [1000, 15000]
14 - sys_Tinst	            [1000, 25000]
15 - sys_VH                 [100,500]
16 - sys_eWT	            [0, 30000]
"""

### FIGURE OF MERIT SYSETMS (FUZZY DATA) ###
inRanges = {    'DATA_phi':        [0.5, 0.95],
                'DATA_w':          [1.0, 150.0],
                'DATA_WS':         [15.0, 300],
                'DATA_sys_etaP':   [0.6, 1.0],
                'DATA_eta_d':      [0.4, 1.0],
                'DATA_sys_FoM':    [0.4, 1.0],
                'DATA_e_d':        [0.0, 0.3],
                'DATA_SFC_quant':  [0.45, 1.05],
                'DATA_dragX':      [0.6, 1.15],
                'DATA_type':       [0.5, 3.5],
                #'DATA_tech':       [-0.5, 4.5],
                #'DATA_jet':        [0.5, 2.5],
            }
outRanges_GWT = {'sys_GWT' : output_ranges['sys_GWT'] }
outRanges_Pin = {'sys_Pinst' : output_ranges['sys_Pinst'] }
outRanges_Tin = {'sys_Tinst' : output_ranges['sys_Tinst'] }
outRanges_VH = {'sys_VH' : output_ranges['sys_VH'] }
outRanges_eWT = {'sys_eWT' : output_ranges['sys_eWT'] }

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

#for d in combData_eWT: print d[2], type(d[2])
 
plt.figure()
plt.scatter([d[2][0] for d in combData_GWT], [d[2][1] for d in combData_GWT])
plt.ylabel('maxGWT')
plt.xlabel('minGWT')

plt.figure()
plt.subplot(2,1,1)
hist, bins = np.histogram([d[2][0] for d in combData_GWT], bins=50)
width = 0.7 * (bins[1] - bins[0])
center = (bins[:-1] + bins[1:]) / 2
plt.bar(center, hist, align='center', width=width)

plt.subplot(2,1,2)
hist, bins = np.histogram([d[2][1] for d in combData_GWT], bins=50)
width = 0.7 * (bins[1] - bins[0])
center = (bins[:-1] + bins[1:]) / 2
plt.bar(center, hist, align='center', width=width)
plt.show()

print 'GWT has', len(combData_GWT), 'points =>', 
combData_GWT = limitData(combData_GWT, 'sys_GWT')
print 'GWT data limted to', len(combData_GWT), 'points'

print 'Pin has', len(combData_Pin), 'points =>', 
combData_Pin = limitData(combData_Pin, 'sys_Pinst')
print 'Pin data limted to', len(combData_Pin), 'points'

print 'Tin has', len(combData_Tin), 'points =>', 
combData_Tin = limitData(combData_Tin, 'sys_Tinst')
print 'Tin data limted to', len(combData_Tin), 'points'

print 'VH has', len(combData_VH), 'points =>', 
combData_VH  = limitData(combData_VH, 'sys_VH')
print 'VH data limted to', len(combData_VH), 'points'

print 'eWT has', len(combData_eWT), 'points =>', 
combData_eWT = limitData(combData_eWT, 'sys_eWT')
print 'eWT data limted to', len(combData_eWT), 'points'


plt.figure()
plt.scatter([d[2][0] for d in combData_GWT], [d[2][1] for d in combData_GWT])
plt.ylabel('maxGWT')
plt.xlabel('minGWT')

plt.figure()
plt.subplot(2,1,1)
hist, bins = np.histogram([d[2][0] for d in combData_GWT], bins=50)
width = 0.7 * (bins[1] - bins[0])
center = (bins[:-1] + bins[1:]) / 2
plt.bar(center, hist, align='center', width=width)

plt.subplot(2,1,2)
hist, bins = np.histogram([d[2][1] for d in combData_GWT], bins=50)
width = 0.7 * (bins[1] - bins[0])
center = (bins[:-1] + bins[1:]) / 2
plt.bar(center, hist, align='center', width=width)
plt.show()


#Turn data into fuzzy MFs
fuzzData_GWT = fuzzifyData(combData_GWT, 'sys_GWT')
fuzzData_Pin = fuzzifyData(combData_Pin, 'sys_Pinst')
fuzzData_Tin = fuzzifyData(combData_Tin, 'sys_Tinst')
fuzzData_VH  = fuzzifyData(combData_VH, 'sys_VH')
fuzzData_eWT = fuzzifyData(combData_eWT, 'sys_eWT')



############### TRAINING CASES ###############     
maxIter = 15
xConverge = 0.005

## TRAINING GWT:

nData = (360, 0.1)      #data to use, holdback rate
nNodes = (260, 40, 70)  #hidden nodes, input gran, output gran
learning = (0.001, 0.0005)  #learning rate, momentum val
fileName = "FCL_files/DFES_RFdata_GWT_data(%d)_nodes(%d_%d_%d)_valErr_16Sep15_short.nwf" % (nData[:1] + nNodes)
print "Case 1-1, Data Points: %d (%.2f holdback)" % nData,
print "Hidden Nodes: %d, Input Granularity: %d, Output Granularity: %d, " % nNodes
print "Learning Rate: %.3f, Momentum: %.3f" % learning
fuzzData = random.sample(fuzzData_GWT, nData[0])
sys = DFES(inRanges, outRanges_GWT, 'sigmoid', hidNodes=nNodes[0], inGran=nNodes[1], outGran=nNodes[2])
sys.train(fuzzData, holdback=nData[1], LR=learning[0], M=learning[1], maxIterations=maxIter, xConverge=xConverge, combError=False)
sys.write_weights(fileName) #write network weight foil
print "~~~~~~~~~~~~~~~ Optimization Complete ~~~~~~~~~~~~~~~"



## TRAINING Pisnt:
nData = (360, 0.1)      #data to use, holdback rate
nNodes = (260, 40, 70)  #hidden nodes, input gran, output gran
learning = (0.001, 0.0005)   #learning rate, momentum val
fileName = "FCL_files/DFES_RFdata_Pin_data(%d)_nodes(%d_%d_%d)_valErr_16Sep15_short.nwf.nwf" % (nData[:1] + nNodes)
print "Case 1-1, Data Points: %d (%.2f holdback)" % nData,
# print "Hidden Nodes: %d, Input Granularity: %d, Output Granularity: %d, " % nNodes
print "Learning Rate: %.3f, Momentum: %.3f" % learning
fuzzData = random.sample(fuzzData_Pin, nData[0])
sys = DFES(inRanges, outRanges_Pin, 'sigmoid', hidNodes=nNodes[0], inGran=nNodes[1], outGran=nNodes[2])
sys.train(fuzzData, holdback=nData[1], LR=learning[0], M=learning[1], maxIterations=maxIter, xConverge=xConverge, combError=True)
sys.write_weights(fileName) #write network weight foil
print "~~~~~~~~~~~~~~~ Optimization Complete ~~~~~~~~~~~~~~~"

"""
## TRAINING Tisnt:
nData = (630, 0.2)      #data to use, holdback rate
nNodes = (300, 50, 50)  #hidden nodes, input gran, output gran
learning = (0.01, 0.005)   #learning rate, momentum val
fileName = "FCL_files/DFES_RFdata_Tin_data(%d)_nodes(%d_%d_%d)_combErr.nwf" % (nData[:1] + nNodes)
print "Case 1-1, Data Points: %d (%.2f holdback)" % nData,
print "Hidden Nodes: %d, Input Granularity: %d, Output Granularity: %d, " % nNodes
print "Learning Rate: %.3f, Momentum: %.3f" % learning
fuzzData = random.sample(fuzzData_Tin, nData[0])
sys = DFES(inRanges, outRanges_Tin, 'sigmoid', hidNodes=nNodes[0], inGran=nNodes[1], outGran=nNodes[2])
sys.train(fuzzData, holdback=nData[1], LR=learning[0], M=learning[1], maxIterations=maxIter, xConverge=xConverge, combError=True)
sys.write_weights(fileName) #write network weight foil
print "~~~~~~~~~~~~~~~ Optimization Complete ~~~~~~~~~~~~~~~"


## TRAINING VH:
nData = (630, 0.2)      #data to use, holdback rate
nNodes = (300, 50, 50)  #hidden nodes, input gran, output gran
learning = (0.01, 0.005)  #learning rate, momentum val
fileName = "FCL_files/DFES_RFdata_VH_data(%d)_nodes(%d_%d_%d)_combErr.nwf" % (nData[:1] + nNodes)
print "Case 1-1, Data Points: %d (%.2f holdback)" % nData,
print "Hidden Nodes: %d, Input Granularity: %d, Output Granularity: %d, " % nNodes
print "Learning Rate: %.3f, Momentum: %.3f" % learning
fuzzData = random.sample(fuzzData_VH, nData[0])
sys = DFES(inRanges, outRanges_VH, 'sigmoid', hidNodes=nNodes[0], inGran=nNodes[1], outGran=nNodes[2])
sys.train(fuzzData, holdback=nData[1], LR=learning[0], M=learning[1], maxIterations=maxIter, xConverge=xConverge, combError=True)
sys.write_weights(fileName) #write network weight foil
print "~~~~~~~~~~~~~~~ Optimization Complete ~~~~~~~~~~~~~~~"
"""
"""
## TRAINING eWT:
nData = (650, 0.2)      #data to use, holdback rate
nNodes = (260, 40, 70)  #hidden nodes, input gran, output gran
learning = (0.01, 0.002)  #learning rate, momentum val
fileName = "FCL_files/DFES_RFdata_eWT_data(%d)_nodes(%d_%d_%d)_valErr_16Sep15_short.nwf" % (nData[:1] + nNodes)
print "Case 1-1, Data Points: %d (%.2f holdback)" % nData,
print "Hidden Nodes: %d, Input Granularity: %d, Output Granularity: %d, " % nNodes
print "Learning Rate: %.3f, Momentum: %.3f" % learning
fuzzData = random.sample(fuzzData_eWT, nData[0])
sys = DFES(inRanges, outRanges_eWT, 'sigmoid', hidNodes=nNodes[0], inGran=nNodes[1], outGran=nNodes[2])
sys.train(fuzzData, holdback=nData[1], LR=learning[0], M=learning[1], maxIterations=maxIter, xConverge=xConverge, combError=False)
sys.write_weights(fileName) #write network weight foil
print "~~~~~~~~~~~~~~~ Optimization Complete ~~~~~~~~~~~~~~~"
"""