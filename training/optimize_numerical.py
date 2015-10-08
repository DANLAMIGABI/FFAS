"""
author: Frank Patterson - 4Apr2015
- optimize fuzzy system with train_numerical.py
- optimizes numerically trained style fuzzy rule-based system 

"""
import sys
import copy

import fuzzy_operations as fuzzyOps
from systems import *

from . import train_numerical
from . import read_data

from fuzzy_error import getError, fuzErrorInt

import numpy as np
import scipy as sp
import scipy.optimize as opt
import skfuzzy as fuzz

import random
import math

import matplotlib.pyplot as plt
plt.ioff()

from datetime import datetime
from timer import Timer


#####################
def parameterize(MFstructIn, MFstructOut, inMFs, outMFs):
    """
    Parameterizes the membership functions for inputs and outputs based on MFstruct.
    Assumes that for each input/output all the MFs are of the same type. 
    
    ----- INPUTS -----
    MFstructIn : dict
        structure of inputs in dict form: [ ['in_name', [MF1name, MF2name, ... ], ['type', 'type', ...] ],
                                            ['in_name', [MF1name, MF2name, ... ], ['type', 'type', ...] ], ... ]

    MFstructOut : dict
        structure of outputs in dict form: [ ['out_name', [MF1name, MF2name, ... ], ['type', 'type', ...] ], ... ]
    
    inMFs : dict
        input MF parameters
        
    outMFs : dict        
        output MF parameters inform:
        MFs = { 'In1' : {'A1':[x1,x2,x3], 'A2':[x1,x2,x3],   'A3':[x1,x2,x3]},
                'In2' : {'A1':[x1,x2,x3], 'A2':[x1,x2,x3],   'A3':[x1,x2,x3]}, ...}
                
    ----- OUTPUTS -----            
    
    varList : list
        list of parameters... parameterization of MFs
       
    """
    varList = []
    
    for inX in MFstructIn: #for each input
        
        if all([type == 'gauss' for type in inX[2]]): #gaussian MFs
            x=0
            for mf in inX[1]:
                varList.append(inMFs[inX[0]][mf][0] - x)  #append distance to mean (starting at 0)
                varList.append(inMFs[inX[0]][mf][1])      #append std_dev
                x = inMFs[inX[0]][mf][0]
                
        if all([type == 'tri' for type in inX[2]]): #triangular MFs (a, b, c)
            x=0
            for mf in inX[1]:
                varList.append(x - inMFs[inX[0]][mf][0])  #append distance to a (starting at 0)
                varList.append(inMFs[inX[0]][mf][1]-inMFs[inX[0]][mf][0])      #b-a
                varList.append(inMFs[inX[0]][mf][2]-inMFs[inX[0]][mf][1])      #c-b
                x = inMFs[inX[0]][mf][2]        

    for outX in MFstructOut: #for each input
        
        if all([type == 'gauss' for type in outX[2]]): #gaussian MFs
            x=0
            for mf in outX[1]:
                varList.append(outMFs[outX[0]][mf][0] - x)  #append distance to mean (starting at 0)
                varList.append(outMFs[outX[0]][mf][1])      #append std_dev
                x = outMFs[outX[0]][mf][0]
                
        if all([type == 'tri' for type in outX[2]]): #triangular MFs (a, b, c)
            x=0
            for mf in outX[1]:
                varList.append(x - outMFs[outX[0]][mf][0])  #append distance to a (starting at 0)
                varList.append(outMFs[outX[0]][mf][1]-outMFs[outX[0]][mf][0])      #b-a
                varList.append(outMFs[outX[0]][mf][2]-outMFs[outX[0]][mf][1])      #c-b
                x = outMFs[outX[0]][mf][2]      
                      
    return varList
                     
#####################
def deparameterize(MFstructIn, MFstructOut, varList):
    """
    Takes in the parameterization of the membership functions for inputs and 
    outputs based on MFstruct. Returns the membership functions
    Assumes that for each input/output all the MFs are of the same type. Assumes 
    single output.
    
    ----- INPUTS -----
    MFstructIn : dict
        structure of inputs in dict form: [ ['in_name', [MF1name, MF2name, ... ], ['type', 'type', ...] ],
                                            ['in_name', [MF1name, MF2name, ... ], ['type', 'type', ...] ], ... ]

    MFstructOut : dict
        structure of outputs in dict form: [ ['out_name', [MF1name, MF2name, ... ], ['type', 'type', ...] ], ... ]
    
    varList : list
        list of parameters... parameterization of MFs
                
    ----- OUTPUTS -----                
    inMFs : dict
        input MF parameters
    outMFs : dict        
        output MF parameters inform:
        MFs = { 'In1' : {'A1':[x1,x2,x3], 'A2':[x1,x2,x3],   'A3':[x1,x2,x3]},
                'In2' : {'A1':[x1,x2,x3], 'A2':[x1,x2,x3],   'A3':[x1,x2,x3]}, ...}   
    """
    # combine input and output dicts
    inMFs  = {x[0]:{} for x in MFstructIn}
    outMFs = {x[0]:{} for x in MFstructOut}
        
    for inX in MFstructIn: #for each input
        
        if all([type == 'gauss' for type in inX[2]]): #gaussian MFs
            x=0
            for mf in inX[1]:
                a = varList.pop(0)  #distance to mean (starting at 0)
                b = varList.pop(0)  #append std_dev
                inMFs[inX[0]][mf] = [x + a, b]
                x = x + a
        
        if all([type == 'tri' for type in inX[2]]): #triangular MFs (a, b, c)
            x=0
            for mf in inX[1]:
                a = varList.pop(0)  
                b = varList.pop(0)
                c = varList.pop(0)  
                inMFs[inX[0]][mf] = [x-a, x-a+b, x-a+b+c]
                x = x-a+b+c



    for outX in MFstructOut: #for each input
        
        if all([type == 'gauss' for type in outX[2]]): #gaussian MFs
            x=0
            for mf in outX[1]:
                a = varList.pop(0)  #distance to mean (starting at 0)
                b = varList.pop(0)  #append std_dev
                outMFs[outX[0]][mf] = [x + a, b]
                x = x + a
        
        if all([type == 'tri' for type in outX[2]]): #triangular MFs (a, b, c)
            x=0
            for mf in outX[1]:
                a = varList.pop(0)  
                b = varList.pop(0)
                c = varList.pop(0)  
                outMFs[outX[0]][mf] = [x-a, x-a+b, x-a+b+c]
                x = x-a+b+c   
                
                
    return inMFs, outMFs

#####################
def getBounds(MFstructIn, MFstructOut, inRanges, outRanges):
    """
    Returns bounds for each parameter.
    
    ----- INPUTS -----
    MFstructIn : dict
        structure of inputs in dict form: [ ['in_name', [MF1name, MF2name, ... ], ['type', 'type', ...] ],
                                            ['in_name', [MF1name, MF2name, ... ], ['type', 'type', ...] ], ... ]
    MFstructOut : dict
        structure of outputs in dict form: [ ['out_name', [MF1name, MF2name, ... ], ['type', 'type', ...] ], ... ]
    
    inRanges : dict
        {'in_name': [min, max], ... }
    outRanges : dict
        {'in_name': [min, max], ... }
    
    
    
    ----- OUTPUTS -----
    varMins : list
    
    varMaxs : list
    
    """
    k_MEANin = 2.0
    k_STDin = 0.8
    k_MEANout = 2.0
    k_STDout = 1.0
    k_TRIout = 1.3
    #Determine Bounds #adjusted for 2nd parameterization (MF_to_varList2/varList_to_MFs2)
    varMins = []
    varMaxs = []
    
    for inX in MFstructIn: #for each input
        vrange = inRanges[inX[0]]
        rdist = float(max(vrange)) - float(min(vrange))
        print "INPUT", inX, "RDIST:", rdist, "VRANGE:", vrange
        for i in range( len(inX[2]) ): #for each MF
            if inX[2][i] == 'gauss':
                if i == 0: varMins.append(min(vrange)-0.15*rdist)           #dist to mean - within 15% range of min
                else:      varMins.append(0.05*rdist)                       #dist to mean - positive
                if i == 0: varMaxs.append(min(vrange)+0.15*rdist)           #dist to mean - within 15% range of min
                else:      varMaxs.append(k_MEANin*rdist/len(inX[2]))       #dist to mean - max of 0.3 range 
                varMins.append(0.02*rdist)                  #std          - min of 0.02 range
                varMaxs.append(k_STDin*rdist/len(inX[2]))   #std          - max of 0.6 range 
                
            elif inX[2][i] == 'tri':
                if i == 0: varMins.append(-1*min(vrange))  #dist 0 to a - neg min of range 
                else:      varMins.append(0.0)             #dist c to a -  positive
                if i == 0: varMaxs.append(-1*min(vrange)+0.5*rdist)   #dist c to a -  max of 0.3 range and 0.7 of min
                else:      varMaxs.append(0.3*rdist)                         #dist c to a -  max of 0.3 range  
                varMins.append(0.05*rdist)         #dist a to b -  positive
                varMaxs.append(0.3*rdist)   #dist a to b -  max of 0.3 range 
                varMins.append(0.05*rdist)         #dist b to c -  positive
                varMaxs.append(0.3*rdist)   #dist b to c -  max of 0.3 range 
                
    for outX in MFstructOut: #for each input
        vrange = outRanges[outX[0]]
        rdist = max(vrange) - min(vrange)
        print "OUTPUT", outX, "RDIST:", rdist, "VRANGE:", vrange
        for i in range( len(outX[2]) ): #for each MF    
            if outX[2][i] == 'gauss':
                if i == 0: varMins.append(min(vrange)-0.1*rdist)    #dist to mean - within 10% range of min
                else:      varMins.append(0.0)                       #dist to mean - positive
                if i == 0: varMaxs.append(min(vrange)+0.1*rdist)    #dist to mean - within 10% range of min
                else:      varMaxs.append(k_MEANout*rdist/(len(outX[2])-1)) #dist to mean - max of 0.3 range 
                varMins.append(0.06*rdist)                   #std - min
                varMaxs.append(k_STDout*rdist/(len(outX[2])-1))  #std - max
                
            elif outX[2][i] == 'tri':
                if i == 0: varMins.append(-1*min(vrange))  #dist 0 to a - neg min of range 
                else:      varMins.append(0.0)             #dist c to a -  positive
                if i == 0: varMaxs.append(-1*min(vrange)+0.5*rdist)   #dist c to a -  max of 0.3 range and 0.7 of min
                else:      varMaxs.append(0.3*rdist)                  #dist c to a -  max of 0.3 range  
                varMins.append(0.02*rdist)                        #dist a to b -  positive
                varMaxs.append(k_TRIout*rdist/(len(outX[2])-1))   #dist a to b -  max of 0.3 range 
                varMins.append(0.02*rdist)                        #dist b to c -  positive
                varMaxs.append(k_TRIout*rdist/(len(outX[2])-1))   #dist b to c -  max of 0.3 range 
                
  
    return varMins, varMaxs
    
#####################
def createMFs(vrange, names, m):
    """
    Create evenly spaced MFs for inputs over ranges
    
    ------INPUTS------
    ranges : iterable
        (min, max) range for MFs
    n : iterable
        names of MFs
    m : int
        MF type (3,4)#
    -----OUTPUTS-----
    MFdict : dict
        membership function dictionary {MF: params, MF: params, ... }
    """
    n = len(names) #number of MFs
    MFs = {}

    if m == 3 or m == 4: #triangular and trapezoidal MFs
         #get min and max
        half_width = (max(vrange) - min(vrange))/float(n-1)
        step_width = 2*half_width/(m-1)
        MFs = []
        for i in range(n):
            range_start = min(vrange)+(i-1)*half_width
            MFparams = [range_start + i*step_width for i in range(m)]
            MFs.append(MFparams)

    elif m == 2:   #gaussian functions
        half_width = (max(vrange)-min(vrange))/float(n-1)
        std = half_width/2.0
        MFs = []
        for i in range(n):
            mean = min(vrange) + i*half_width
            MFparams = [mean, std]
            MFs.append(MFparams)
    
    MFdict = {names[i]: MFs[i] for i in range(len(MFs))}
            
    return MFdict
        
#####################   
def get_system_error(FCLfile, valData, Nmax=None, inDataMFs='tri', outDataMFs='tri', errorType='crispSoS'):
    """
    determines the system error as calculated by the validation data
    FCLfile - FCL file path/name to build FRBS
    valData is validation data in format: [quant_inputs, qual_inputs, outputData]
                                with each data item {['function', 'var'] : [min,max]} (or [val])
    inDataMFs and outDataMFs - type of MFs for inputs and outputs in data
    errorType is type of error
    nMax is the max number of points from the data to use
    """
    q = 0 #0 for quant data, 1 or qual data
        
    allErrors_comb = []
        
    #load fuzzy system
    inputs, outputs, rulebase, AND_operator, OR_operator, aggregator, implication, defuzz = build_fuzz_system(FCLfile)
    sys = Fuzzy_System(inputs, outputs, rulebase, AND_operator, OR_operator, aggregator, implication, defuzz)
        
    allErrors = [] #list of all errors
    i = 0 #counter
    
    for data_item in valData: #for each data item
        valIns = {}
        valOuts = {} 
        
        for inKey in data_item[q]: #for each input key build selected MF
            [inXs, inYs] = fuzzyOps.rangeToMF(data_item[q][inKey], inDataMFs)
            valIns['_'.join(inKey)] = [inXs, inYs]   #addinput MF to input dict to build inputs for system                                              
        
        if outDataMFs <> 'sing':
            [outXs, outYs] = fuzzyOps.rangeToMF(data_item[2], outDataMFs)
            valOuts = [outXs, outYs]  #data outputs
        elif outDataMFs =='sing': #singleton output MF: [avg]
            dAvg = sum(data_item[2])/len(data_item[2])
            valOuts = dAvg  #data outputs
        
        valOuts = [outXs, outYs]  #data outputs
        sysOuts = sys.run(valIns) #get system output
        
        if errorType == 'crispSoS': #capture crisp errors
            allErrors.append(dAvg - sysOuts.itervalues().next())
        if errorType == 'fuzzy': #capture fuzzy error
            allErrors.append( fuzErrorInt([outXs, outYs], sysOuts.itervalues().next())**2 )
        
        i = i+1
        
        if Nmax <> None: #check for Nmax exceeded
            if i > Nmax: break

    if errorType == 'crispSoS': #sum squares of errors and divide by 2N (Simon 2002)
        allErrors = [x**2 for x in allErrors]
        error = (sum(allErrors)/(len(allErrors)))**0.5
    elif errorType == 'fuzzy': #get a fuzzy error measure
        error = (sum(allErrors)/(len(allErrors)))**0.5
        
    return error

##
def run_optimization(train_data, val_data, MFstructIn, inRanges, MFstructOut, 
                     outRanges, fclName, trainMax, valMax, inMFform = 'sing', 
                     outMFform = 'sing', defuzz=None, optMethod='diffEv', 
                     popX=1.5):
    """
    Run optimization
    
    ------INPUTS------
    train_data : list
        data to train fuzzy sys in form: [ [quant_inputs, qual_inputs, outputData], 
                                           [quant_inputs, qual_inputs, outputData], ...], 
        with each data item {['function', 'var'] : [min,max]} (or [val])
    val_data: list
        data to validate fuzzy sys (same form as train_data)
    inMFs : dict
    
    inRanges : dict
    
    outMFs : dict
    
    outRanges : dict
    
    fclName : string
    
    trainMax : int
        max number of training data points to use
    valMax : int
        max number of validation data points to use
    inMFform : string
        MF form of input data ('sing', 'tri', 'trap')
    outMFform : string
        MF form of input data ('sing', 'tri', 'trap')
    defuzz : string
        defuzzification form to use
    """
    
    plt.ioff()
    print "Using Input Data MF:", inMFform, "  and Output Data MF:", outMFform
    
    if outMFform == 'sing' and defuzz <> None: errorType = "crispSoS"
    else:                                      errorType = "fuzzy"
    
    if defuzz == None: defuzz = None
    else:              defuzz = defuzz
    
    types = [None, None, 'gauss', 'tri', 'trap'] #potential types of MFs
    
    #initialize MFs
    inMFs = {}
    outMFs = {}
    for line in MFstructIn:
        inMFs[line[0]] = createMFs( inRanges[line[0]], line[1], types.index(line[2][0]) )
    for line in MFstructOut:
        outMFs[line[0]] = createMFs( outRanges[line[0]], line[1], types.index(line[2][0]) )
        
    #Determine Bounds
    varMins, varMaxs = getBounds(MFstructIn, MFstructOut, inRanges, outRanges)
    varBounds = zip(varMins, varMaxs) #turn lists into list of [(min,max), (min,max), ... ]
    
    #Initialize system
    inMF_funcs, outMF_funcs = read_data.generate_MFs(inMFs, outMFs)
    rule_grid = train_numerical.train_system(inMF_funcs, outMF_funcs, train_data, 
                                            inDataMFs=inMFform, outDataMFs=outMFform, maxDataN=trainMax)
    train_numerical.write_fcl_file_FRBS(inMFs, outMFs, rule_grid, defuzz, fclName)
    
    inList  = [(input, len(inMFs[input])) for input in inMFs]        #constant order list of inputs and num of MFs
    outList = [(output, len(outMFs[output])) for output in outMFs]   #constant order list of output and num of MFs
    
    #get varList based on initial MF guesses
    varList = parameterize(list(MFstructIn), list(MFstructOut), inMFs, outMFs)
    
    tempBESTfit = [10**10]
    
    #FITNESS FUNCTION:
    def fitness(varList, penalty=True):#, tBESTfit=tempBESTfit):
        defuzz = None
        #try:
        with Timer() as t:
            #translate varlist to MFs (MFs in list form)
            inMFs, outMFs = deparameterize(list(MFstructIn), list(MFstructOut), list(varList))
            
            #translate list MFs into MF functions (MF in array form)
            inMF_funcs, outMF_funcs = read_data.generate_MFs(inMFs, outMFs)
            
            #train system
            rule_grid = train_numerical.train_system(inMF_funcs, outMF_funcs, train_data, 
                                                    inDataMFs=inMFform, outDataMFs=outMFform, 
                                                    maxDataN=trainMax)
            
            #write out FCL
            train_numerical.write_fcl_file_FRBS(inMFs, outMFs, rule_grid, defuzz, fclName)
            
            #build test system
            inputs, outputs, rulebase, AND_operator, OR_operator, aggregator, \
                implication, defuzz = build_fuzz_system(fclName)
            testSys = Fuzzy_System(inputs, outputs, rulebase, AND_operator, \
                OR_operator, aggregator, implication, defuzz)
            
            #test system
            error = getError(copy.deepcopy(val_data), testSys, inMF=inMFform, 
                                outMF=outMFform, sysOutType=errorType, errType='dist')
            MSE = (1.0/len([err for err in error if err[2] <> None]))*sum([err[2]**2 for err in error if err[2] <> None])
            if penalty:
                k = float(len([e for e in error if e[2] == None]))/len(error)
                MSE = MSE*math.exp(2.0*k) #penalty for lots of "Nones" from error function
            else: k = 0
            
        print " ==> system fitness %.4f in %.1f sec (k=%.2f)" % (MSE, t.secs,k)    
        
        if MSE < tempBESTfit[0]: 
            train_numerical.write_fcl_file_FRBS(inMFs, outMFs, rule_grid, defuzz, fclName[:-4]+'_tempBEST_'+str(MSE)+'.fcl')
            tempBESTfit[0] = MSE
            print "--- new temp best written ---"
            
        return MSE
            
        #except:
        #    print " ==> FITNESS CALC ERROR!", sys.exc_info()[0]
        #    return 9999.
    
    #iterCount = 0
    #CALLBACK FUNCTION:
    def callbackF(xk, convergence=1.0):
        f = fitness(xk)
        print "Optimization Iteration Complete... "
        print "Best Fitness = %.4f ... plotting" % f
        #plotMF_OPT(MFstructIn, MFstructOut, xk, inRanges, outRanges)
        #plt.show()
        
    
    #PRINT OUT INFO ON OPTIMIZATION PROBLEM            
    print 'Optimizing', len(varList), 'membership function variables...'
    for i in range(len(varList)):
        print '=> var: %.3f < x%d < %.3f ; init: %.3f' % (varMins[i], i, varMaxs[i], varList[i]), 
        if varList[i] < varMins[i] or varList[i] > varMaxs[i]: print "   error!"
        else: print "   nominal"
        
    i = 0
    print "**************"
    for line in MFstructIn:
        print "Input", line[0]
        for j in range(len(line[1])):
            print "  MF:", line[1][j], " Type:", line[2][j], " InitParams:", inMFs[line[0]][line[1][j]]

            print "      Vars:", varList[ i : i+types.index(line[2][j]) ],
            print "Mins:", varMins[ i : i+types.index(line[2][j])], 
            print "Maxs:", varMaxs[ i : i+types.index(line[2][j])]
            
            i = i + types.index(line[2][j])
    for line in MFstructOut:
        print "Output", line[0]
        for j in range(len(line[1])):
            print "   MF:", line[1][j], " Type:", line[2][j], " InitParams:", outMFs[line[0]][line[1][j]]
            print "      Vars:", varList[ i : i+types.index(line[2][j]) ],
            print "Mins:", varMins[ i : i+types.index(line[2][j])], 
            print "Maxs:", varMaxs[ i : i+types.index(line[2][j])]
            i = i + types.index(line[2][j])
            
    #CHECK BASELINE FITNESS
    print ""
    print "BASELINE ",
    baseFit = fitness(varList)
    
    plotMF_OPT(MFstructIn, MFstructOut, varList, inRanges, outRanges, varMins=varMins, varMaxs=varMaxs) #varMins=varMins, varMaxs=varMaxs)
    #plt.draw()
    plt.show()

    # RUN OPTIMIZATION: 
    if optMethod == 'diffEv':
        results = opt.differential_evolution(fitness, varBounds, #init='latinhypercube',
                                            popsize=popX, polish=False, #8.0/len(varBounds), #mutation=(0.05, 0.2), 
                                            maxiter=50, disp=True, callback=callbackF)
        varList = list(results.x) #get optimized result
        
    elif optMethod == 'GA':
        from ga import GenAlg
        optimizer = GenAlg(fitness, varList, varBounds, crossover=0.8, mutation=0.06, 
                           popMult=popX, bestRepeat=15, bitsPerGene=10,
                           elite_flag=True, direction='min')
        results = optimizer.run()
        varList = list(results[1]) #get varlist 
    
    elif optMethod == 'L-BFGS-B':
        x, f, results = opt.fmin_l_bfgs_b(fitness, varList, approx_grad=True, bounds=varBounds,
                                    factr=1e8, iprint=0, maxfun=10000, maxiter=500,
                                    callback=callbackF)
        varList = list(x)

    #translate varlist to MFs (MFs in list form)
    inMFs, outMFs = deparameterize(list(MFstructIn), list(MFstructOut), list(varList))
    
    #translate list MFs into MF functions (MF in array form)
    inMF_funcs, outMF_funcs = read_data.generate_MFs(inMFs, outMFs)
    
    #train system
    rule_grid = train_numerical.train_system(inMF_funcs, outMF_funcs, train_data, 
                                             inDataMFs=inMFform, outDataMFs=outMFform, 
                                             maxDataN=1000)
    
    #write out FCL
    train_numerical.write_fcl_file_FRBS(inMFs, outMFs, rule_grid, defuzz, fclName[:-4]+'BEST.fcl')

    #test system
    sysError = get_system_error(fclName, val_data, Nmax=valMax, inDataMFs=inMFform, outDataMFs=outMFform, errorType=errorType)
    print "OPTIMIZED SYSERROR:", sysError, "RESULTS:"
    print results
    #print results.message
    
def plotMF_OPT(MFstructIn, MFstructOut, varList, inRanges, outRanges, varMins=None, varMaxs=None):
    """
    Plot membership functions in optimization problem
    
    """
    inMFsMin, outMFsMin = None, None
    inMFsMax, outMFsMax = None, None
    inMFs, outMFs = deparameterize(list(MFstructIn), list(MFstructOut), list(varList))
    if not varMins == None:
        inMFsMin, outMFsMin = deparameterize(list(MFstructIn), list(MFstructOut), list(varMins))
    if not varMaxs == None:
        inMFsMax, outMFsMax = deparameterize(list(MFstructIn), list(MFstructOut), list(varMaxs))

    #plot
    #plt.close('all')
    #plt.clf() #start a fresh figure if one is open
    
    plt.figure(figsize=(10, 6))
    i = 1
    for inp in inMFs:
        plt.subplot(len(inMFs) + len(outMFs), 1, i)
        for mf in inMFs[inp]:
            vals = fuzzyOps.paramsToMF(inMFs[inp][mf])
            plt.plot(vals[0], vals[1], lw=2.0)
            if inMFsMin <> None:
                vals = fuzzyOps.paramsToMF(inMFsMin[inp][mf])
                plt.plot(vals[0], vals[1], '--', lw=0.8)
            if inMFsMax <> None:
                vals = fuzzyOps.paramsToMF(inMFsMax[inp][mf])
                plt.plot(vals[0], vals[1], ':', lw=0.8)
            plt.plot([inRanges[inp][0], inRanges[inp][0]], [0.,1.], '-k', lw=3.0)
            plt.plot([inRanges[inp][1], inRanges[inp][1]], [0.,1.], '-k', lw=3.0)
            plt.ylabel('INPUT'+ str(inRanges[inp]))
        i = i+1
        plt.yticks([0.0,1.0])
    for otp in outMFs:
        plt.subplot(len(inMFs) + len(outMFs), 1, i)
        for mf in outMFs[otp]:
            vals = fuzzyOps.paramsToMF(outMFs[otp][mf])
            plt.plot(vals[0], vals[1], lw=2.0)
            if outMFsMin <> None:
                vals = fuzzyOps.paramsToMF(outMFsMin[otp][mf])
                plt.plot(vals[0], vals[1], '--', lw=0.8, )
            if outMFsMax <> None:
                vals = fuzzyOps.paramsToMF(outMFsMax[otp][mf])
                plt.plot(vals[0], vals[1], ':', lw=0.8, )
            plt.plot([outRanges[otp][0], outRanges[otp][0]], [0.,1.], '-k', lw=3.0)
            plt.plot([outRanges[otp][1], outRanges[otp][1]], [0.,1.], '-k', lw=3.0)
            
            plt.ylabel('OUTPUT')
        i = i+1  
    #plt.draw()
    
