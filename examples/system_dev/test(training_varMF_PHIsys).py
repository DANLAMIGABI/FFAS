"""
Testing Phi System training
author: Frank Patterson - 4Apr2015
Testing training modules
"""
import copy, random

import skfuzzy as fuzz
from training import *
from systems import *
import fuzzy_operations as fuzzyOps
from timer import Timer

import numpy as np

import matplotlib.pyplot as plt

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


##
def createMFs(ranges, n, m):
    """
    create evenly spaced MFs for inputs over ranges
    ranges : dict
        {input:[min,max]... } for inputs that MFs are needed for
    n = number of MFs
    m = MF type (3,4)#
    """
    dictX = {}
    for inp in ranges:
        if m == 3 or m == 4: #triangular and trapezoidal MFs
            mi, ma = ranges[inp][0], ranges[inp][1] #get min and max
            half_width = (ma-mi)/float(n-1)
            step_width = 2*half_width/(m-1)
            MFs = []
            for i in range(n):
                range_start = mi+(i-1)*half_width
                MFparams = [range_start + i*step_width for i in range(m)]
                MFs.append(MFparams)
            MFdict = {'A'+str(i): MFs[i] for i in range(len(MFs))}
            dictX[inp] = MFdict
        elif m == 2:   #gaussian functions
            mi, ma = ranges[inp][0], ranges[inp][1] #get min and max
            half_width = (ma-mi)/float(n-1)
            std = half_width/2.0
            MFs = []
            for i in range(n):
                mean = mi + i*half_width
                MFparams = [mean, std]
                MFs.append(MFparams)
            MFdict = {'A'+str(i): MFs[i] for i in range(len(MFs))}
            dictX[inp] = MFdict
    return dictX
    



#get average system empty weight ratio
#operations1 = {('SYS_PHI_AVGofUNIONS', 'phi'):  ( [('VL_SYS_UNION', 'phi'), ('FWD_SYS_UNION', 'phi'), ('WING_SYS_TYPE', 'phi'), ('ENG_SYS_TYPE', 'phi')], 'AVERAGE' ),
#                }
#combinedData = combine_inputs(combinedData, operations1)

#write_expert_data(data, 'data/POC_combinedPhiData.csv')

# Create Input Triangular MFs
input_3gaussMFs = createMFs(input_ranges, 3, 2)
input_5gaussMFs = createMFs(input_ranges, 5, 2)
input_7gaussMFs = createMFs(input_ranges, 7, 2)
input_9gaussMFs = createMFs(input_ranges, 9, 2)
input_5triMFs = createMFs(input_ranges, 5, 3)
input_7triMFs = createMFs(input_ranges, 7, 3)
input_9triMFs = createMFs(input_ranges, 9, 3)
input_5trapMFs = createMFs(input_ranges, 5, 3)
input_7trapMFs = createMFs(input_ranges, 7, 3)

#Create Output Triangular MFs
output_5gaussMFs = createMFs(output_ranges, 5, 2)
output_7gaussMFs = createMFs(output_ranges, 7, 2)
output_9gaussMFs = createMFs(output_ranges, 9, 2)
output_7triMFs = createMFs(output_ranges, 7, 3)
output_9triMFs = createMFs(output_ranges, 9, 3)
output_7trapMFs = createMFs(output_ranges, 7, 4)
output_9trapMFs = createMFs(output_ranges, 7, 4)

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

""" 
LAST UNOPT
[   ('VL_SYS_TECH', 'phi'),
                ('FWD_SYS_PROP', 'eta_p'),
                ('FWD_SYS_TYPE', 'TP'),
                ('VL_SYS_TECH', 'LD'),
                ('WING_SYS_TYPE', 'LD'),
                ('WING_SYS_TYPE', 'WS'),
                ('VL_SYS_TECH', 'w'),
                ('VL_SYS_TYPE', 'w'),
                ('VL_SYS_PROP', 'w'),
                ('VL_SYS_TYPE', 'phi'),]


LESS OLD
('VL_SYS_TECH', 'phi'), ('VL_SYS_DRV', 'phi'),
('FWD_SYS_TYPE', 'TP'), ('VL_SYS_TECH', 'LD'),
('WING_SYS_TYPE', 'LD'), ('WING_SYS_TYPE', 'WS'),
('VL_SYS_TYPE', 'w'), ('ENG_SYS_TYPE', 'phi'),
('FWD_SYS_PROP', 'phi'), ('VL_SYS_TYPE', 'TP'),
('ENG_SYS_TYPE', 'SFC'), ('VL_SYS_TYPE', 'phi'),
('VL_SYS_PROP', 'phi'),]
 OLD
('FWD_SYS_PROP', 'eta_p'),
('VL_SYS_TECH', 'w'),
('WING_SYS_TYPE', 'LD'),
('WING_SYS_TYPE', 'WS'),
('FWD_SYS_TYPE', 'TP'),
('VL_SYS_TYPE', 'w'),
('ENG_SYS_TYPE', 'phi'),
('FWD_SYS_PROP', 'phi'),
('WING_SYS_TYPE', 'phi'),
('VL_SYS_TYPE', 'e_d'),
('VL_SYS_PROP', 'w'),
('ENG_SYS_TYPE', 'SFC')
"""
                

## TESTS
def testMFs(tData, vData, test_name, inMFs, outMFs, outForm, inDataForm, outDataForm, 
            defuzz=None, sysOutType="fuzzy", errType="dist", plot=True):
    
    print "*************************************"
    print "TESTING:  ", test_name
    print len(tData), "training points", len(vData), "validation points."
    #inMFs = input_5gaussMFs       #system in
    #outMFs = output_7gaussMFs
    #defuzz = None
    
    #outForm = 'gauss'
    #inDataForm = 'gauss'
    #outDataForm = 'gauss'
    #errType = 'fuzzy'
    
    input_arrays, output_arrays = generate_MFs(inMFs, outMFs)
    
    inputMFs = { inp: copy.deepcopy(input_arrays[inp[1]])  for inp in inputList }
    inputPARAMs = {inp: copy.deepcopy(inMFs[inp[1]])  for inp in inputList }
    
    outputMFs = {'sys_phi' : copy.deepcopy(output_arrays['sys_phi'])}
    outputPARAMs = {'sys_phi' : copy.deepcopy(outMFs['sys_phi'])}
        
    combinedData = copy.deepcopy(tData)
    
    #generate rules
    with Timer() as t:
        rule_grid = train_system(inputMFs, outputMFs, combinedData, inDataMFs=inDataForm, outDataMFs=outDataForm)
    
    #write out FCL
    write_fcl_file_FRBS(inputPARAMs, outputPARAMs, rule_grid, defuzz, 'test_sys_phi.fcl')
    
    #get system
    inputs, outputs, rulebase, AND_operator, OR_operator, aggregator, implication, \
        defuzz = build_fuzz_system('test_sys_phi.fcl')
    sys = Fuzzy_System(inputs, outputs, rulebase, AND_operator, OR_operator, aggregator, 
                    implication, defuzz)
    
    print '=> ', t.secs, 'secs to build', len(sys.rulebase), 'rules'
    
    #test system
    testData = copy.deepcopy(vData)
    
    with Timer() as t:
        error = getError(testData, sys, inMF=inDataForm, outMF=outDataForm, sysOutType=sysOutType, errType=errType)
        
    print '=> ', t.secs, 'secs to check error'
    print 'Total System Error:', sum([err[2] for err in error if err[2] <> None])
    print 'Mean Square System Error:', sum([err[2]**2 for err in error if err[2] <> None]) / len([err for err in error if err[2] <> None]) 
    print 'Root Mean Square System Error:', ( sum([err[2]**2 for err in error if err[2] <> None]) / len([err for err in error if err[2] <> None])  )**0.5
    
    
    #check random data points for time
    check = 15
    t_tot = 0.0
    for j in range(check):
        i = random.randrange(0, len(testData))
        inputs = {key[0]+"_"+key[1]:testData[i][0][key] for key in testData[i][0]}
        with Timer() as t:
            sysOut = sys.run(inputs)
        t_tot = t_tot + float(t.secs)
    print 'Average system time of %d points => %.5f s' % (check, t_tot/check)
    print '                                 => %.5f s per rule' % ((t_tot/check)/len(sys.rulebase))
    
    
    #actual vs. predicted plot
    alpha = 0.7
    plt.figure()
    plt.title('Actual vs. Predicted at Max Alpha Cut'+test_name)
    for err in error:
        if outDataForm == 'gauss': AC_actual = fuzzyOps.alpha_cut(alpha, [err[0][0],err[0][1]])
        else: AC_actual = fuzzyOps.alpha_at_val(err[0][0],err[0][1])
        if outForm == 'gauss': AC_pred = fuzzyOps.alpha_cut(alpha, (err[1][0],err[1][1]))
        else: AC_pred = fuzzyOps.alpha_at_val(err[1][0],err[1][1])
        
        if AC_actual <> None and AC_pred <> None: 
            plt.scatter(AC_actual[0], AC_pred[0], marker='o', c='r')
            plt.scatter(AC_actual[1], AC_pred[1], marker='x', c='b')
    
    plt.plot([1,9],[1,9], '--k')     
    plt.xlim([1,9])
    plt.ylim([1,9])
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    
    #visuzlize system with random data point
    #i = random.randrange(0, len(combinedData))
    #inputs = {key[0]+"_"+key[1]:combinedData[i][0][key] for key in combinedData[i][0]}
    #sys.run(inputs, TESTMODE=True)
    
    #check random data points for time
    """
    plt.figure()
    plt.title('Random Tests:'+test_name)
    for j in range(9):
        i = random.randrange(0, len(combinedData))
        inputs = {key[0]+"_"+key[1]:combinedData[i][0][key] for key in combinedData[i][0]}
        sysOut = sys.run(inputs)
        sysOut = sysOut[sysOut.keys()[0]]
        plt.subplot(3,3,j+1)
        plt.plot(sysOut[0], sysOut[1], '-r')
        plt.plot(combinedData[i][2][0], combinedData[i][2][1], '--k')
        plt.ylim([0,1.1])
        plt.xlim([1,9])
    """

    #actual vs. error plot
    """
    plt.figure()
    plt.title('Actual (Centroid) vs. Error'+test_name)
    cents = [fuzz.defuzz(err[0][0], err[0][1], 'centroid') for err in error]
    plt.scatter(cents, [err[2] for err in error])
    plt.xlabel('Actual (Centroid)')
    plt.ylabel('Fuzzy Error')
    """
    return len(sys.rulebase), sum([err[2]**2 for err in error if err[2] <> None]) / len([err for err in error if err[2] <> None]) 

#Testing plot:
#plot_rule_grid(rule_grid, inputMFs, outputMFs, combinedData, ('VL_SYS_UNION',   'phi'), ('FWD_SYS_UNION', 'phi'), ('WING_SYS_TYPE',   'phi'))
#plot_parallel(rule_grid, inputMFs, outputMFs, combinedData, None)
if __name__ == "__main__":

    
    # Read in Input Data for Morph
    dataIn = readFuzzyInputData('data/morphInputs_13Jun15.csv')
    
    # Get data linked to inputs
    data = buildInputs(ASPECT_list, dataIn, 'data/phiData_300pts.csv', True)
    
    plt.ioff()
    """
    #TEST NUmber of points
    inMFs = input_5gaussMFs       
    outMFs = output_9gaussMFs
    test_name = 'TEST IN DATA - SYS: In:5-2 Out:9-2   DATA: In:2 Out:2'
    outForm = 'gauss'
    inDataForm = 'gauss'
    outDataForm = 'gauss'
    nRand = 10
    
    MSE = [[] for i in range(nRand)]
    nRs = [[] for i in range(nRand)]
    dCounts = [int(x) for x in np.arange(100, len(data)+1, 20.)]
    print dCounts
    MSE = [[] for i in range(len(dCounts))]
    nRs = [[] for i in range(len(dCounts))]
    
    validationData = copy.deepcopy(data[:100])
    
    for i in range(len(dCounts)):
        n = dCounts[i]
        for j in range(nRand):
            random.shuffle(data)
            tDat = copy.deepcopy(data[:n])
            vDat = copy.deepcopy(validationData)
            nR, e = testMFs(tDat, vDat, test_name, inMFs, outMFs, outForm, inDataForm, outDataForm)
            MSE[i].append(e)
            nRs[i].append(nR)
        
        plt.close('all')
        plt.clf()
    
    
    MSEmin = [min(x) for x in MSE]
    MSEmax = [max(x) for x in MSE]
    MSEavg = [np.average(x) for x in MSE]
    
    nRsmin = [min(x) for x in nRs]
    nRsmax = [max(x) for x in nRs]
    nRsavg = [np.average(x) for x in nRs]
    
    
    MpRmin = [MSEmin[i]/nRsmax[i] for i in range(len(MSEmin))]
    MpRmax = [MSEmax[i]/nRsmin[i] for i in range(len(MSEmax))]
    MpRavg = [MSEavg[i]/nRsavg[i] for i in range(len(MSEavg))]
    
    plt.figure()
    ax = plt.subplot(3,1,1)
    ax.plot(dCounts, MSEmin, '--k')
    ax.plot(dCounts, MSEmax, '--k')
    ax.plot(dCounts, MSEavg, 'b', lw=2.0)
    ax.set_xlabel('Training Points')
    ax.set_ylabel('Mean Squared Error (MSE)')
    ax.grid(True)

    ax = plt.subplot(3,1,2)
    ax.plot(dCounts, nRsmin, '--k')
    ax.plot(dCounts, nRsmax, '--k')
    ax.plot(dCounts, nRsavg, 'b', lw=2.0)
    ax.set_xlabel('Training Points')
    ax.set_ylabel('Number of Rules')
    ax.grid(True)

    ax = plt.subplot(3,1,3)
    ax.plot(dCounts, MpRmin, '--k')
    ax.plot(dCounts, MpRmax, '--k')
    ax.plot(dCounts, MpRavg, 'b', lw=2.0)
    ax.set_xlabel('Training Points')
    ax.set_ylabel('Error (MSE) per Rule')
    ax.grid(True)

    plt.figure()
    ax = plt.subplot(111)
    ax.plot(dCounts, MpRmin, '--k')
    ax.plot(dCounts, MpRmax, '--k')
    ax.plot(dCounts, MpRavg, 'b', lw=2.0)
    ax.set_xlabel('Training Points')
    ax.set_ylabel('Error (MSE) per Rule')
    ax.grid(True)
    plt.show()
    
    """
    trainData = copy.deepcopy(data[50:])
    valData = copy.deepcopy(data[:150])

    
    # TEST GAUSSIAN MFs w/ various data types
    inMFs = input_7gaussMFs       
    outMFs = output_9gaussMFs
    
    test_name = 'SYS: In:7-2 Out:9-2   DATA: In:2 Out:2'
    outForm = 'gauss'
    inDataForm = 'gauss'
    outDataForm = 'gauss'
    testMFs(trainData, valData, test_name, inMFs, outMFs, outForm, inDataForm, outDataForm)
    
    
    test_name = 'SYS: In:7-2 Out:9-2   DATA: In:3 Out:3'
    outForm = 'gauss'
    inDataForm = 'tri'
    outDataForm = 'tri'
    testMFs(trainData, valData, test_name, inMFs, outMFs, outForm, inDataForm, outDataForm)


    test_name = 'SYS: In:7-2 Out:9-2   DATA: In:4 Out:4'
    outForm = 'gauss'
    inDataForm = 'trap'
    outDataForm = 'trap'
    testMFs(trainData, valData, test_name, inMFs, outMFs, outForm, inDataForm, outDataForm)    
    
    # TEST TRI MFs w/ various data types
    inMFs = input_7triMFs       
    outMFs = output_9triMFs
    
    test_name = 'SYS: In:7-3 Out:9-3   DATA: In:2 Out:2'
    outForm = 'tri'
    inDataForm = 'gauss'
    outDataForm = 'gauss'
    testMFs(trainData, valData, test_name, inMFs, outMFs, outForm, inDataForm, outDataForm)
    
    test_name = 'SYS: In:7-3 Out:9-3   DATA: In:3 Out:3'
    outForm = 'tri'
    inDataForm = 'tri'
    outDataForm = 'tri'
    testMFs(trainData, valData, test_name, inMFs, outMFs, outForm, inDataForm, outDataForm)
    
    test_name = 'SYS: In:7-3 Out:9-3   DATA: In:4 Out:4'
    outForm = 'tri'
    inDataForm = 'trap'
    outDataForm = 'trap'
    testMFs(trainData, valData, test_name, inMFs, outMFs, outForm, inDataForm, outDataForm)    
    
    # TEST TRAP MFs w/ various data types
    inMFs = input_7trapMFs       
    outMFs = output_9trapMFs
    
    test_name = 'SYS: In:7-4 Out:9-4   DATA: In:2 Out:2'
    outForm = 'trap'
    inDataForm = 'gauss'
    outDataForm = 'gauss'
    testMFs(trainData, valData, test_name, inMFs, outMFs, outForm, inDataForm, outDataForm)
    
    test_name = 'SYS: In:7-4 Out:9-4   DATA: In:3 Out:3'
    outForm = 'trap'
    inDataForm = 'tri'
    outDataForm = 'tri'
    testMFs(trainData, valData, test_name, inMFs, outMFs, outForm, inDataForm, outDataForm)
    
    test_name = 'SYS: In:7-4 Out:9-4   DATA: In:4 Out:4'
    outForm = 'trap'
    inDataForm = 'trap'
    outDataForm = 'trap'
    testMFs(trainData, valData, test_name, inMFs, outMFs, outForm, inDataForm, outDataForm)          
    plt.show()
    