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
from fuzzy_error import getError

import matplotlib.pyplot as plt

import scipy.optimize as opt

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
    'sigma': [0.05, 0.4],
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
    

# Read in Input Data for Morph
dataIn = readFuzzyInputData('data/morphInputs_13Jun15.csv')

# Get data linked to inputs
combData = buildInputs(ASPECT_list, dataIn, 'data/phiData_300pts.csv', True)

# Get union of FWD and VL system empty weight ratio and average wing loading
#operations1 = { ('VL_SYS_UNION', 'phi'):  ( [('VL_SYS_TYPE', 'phi'), ('VL_SYS_PROP', 'phi'), ('VL_SYS_DRV', 'phi'), ('VL_SYS_TECH', 'phi')], 'UNION' ),
#                ('FWD_SYS_UNION', 'phi'): ( [('FWD_SYS_TYPE', 'phi'), ('FWD_SYS_PROP', 'phi'), ('FWD_SYS_DRV', 'phi')], 'UNION' ),
#                ('VL_SYS_UNION', 'w'):  ( [('VL_SYS_TYPE', 'w'), ('VL_SYS_PROP', 'w'), ('VL_SYS_TECH', 'w')], 'AVERAGE' ),
#                }
#data = combine_inputs(copy.deepcopy(combData), operations1)


# Create Input Triangular MFs
input_5gaussMFs = createMFs(input_ranges, 5, 2)
input_7gaussMFs = createMFs(input_ranges, 7, 2)
input_9gaussMFs = createMFs(input_ranges, 9, 2)
input_5triMFs = createMFs(input_ranges, 5, 3)
input_7triMFs = createMFs(input_ranges, 7, 3)
input_9triMFs = createMFs(input_ranges, 9, 3)

#Create Output Triangular MFs
output_7gaussMFs = createMFs(output_ranges, 7, 2)
output_9gaussMFs = createMFs(output_ranges, 9, 2)
output_7triMFs = createMFs(output_ranges, 7, 3)
output_9triMFs = createMFs(output_ranges, 9, 3)
output_7trapMFs = createMFs(output_ranges, 7, 4)
output_9trapMFs = createMFs(output_ranges, 7, 4)

#import pdb; pdb.set_trace()

## TESTS
def input_test(baselineMSE=None, inMFlist=[]):
    test_name = 'input'
    print "*************************************"
    print "TESTING:  ", test_name
    inMFs = input_5gaussMFs       #system in
    outMFs = output_9gaussMFs
    defuzz = None
    
    outForm = 'gauss'
    inDataForm = 'gauss'
    outDataForm = 'gauss'
    outType = 'fuzzy'
    errCalc='dist'
        
    input_arrays, output_arrays = generate_MFs(inMFs, outMFs)
    inputMFs    = { (d[1], d[3]): copy.deepcopy(input_arrays[d[3]]) for d in dataIn }
    inputPARAMs = { (d[1], d[3]): copy.deepcopy(inMFs[d[3]]) for d in dataIn }
    outputMFs = {'sys_phi' : copy.deepcopy(output_arrays['sys_phi'])}
    outputPARAMs = {'sys_phi' : copy.deepcopy(outMFs['sys_phi'])}
    
    t_base = 0.05 #threshold for 
    t_rem = 0.01 #threshold for removing a parameter (as percent of base MSE)

    #GET BASELINE ERROR:
    print "Getting baseline error for ", len(inputMFs), "inputs ... "
    #random.shuffle(combData)
    #
    
    #random.shuffle(combData)
    combDataVal = combData[:] 
    print len(combDataVal), "data points for validation."
    #random.shuffle(combData)
    combDataTrain = combData[:]
    print len(combDataTrain), "data points for training."
    
    combinedData = copy.deepcopy(combDataTrain) #copy data to preserve raw data
    valData = copy.deepcopy(combDataVal)
    
    #if starting from scratch
    if baselineMSE == None: 
        rule_grid = train_system(inputMFs, outputMFs, combinedData, inDataMFs=inDataForm, outDataMFs=outDataForm) #train rule grid
        write_fcl_file_FRBS(inputPARAMs, outputPARAMs, rule_grid, defuzz, 'test_sys_phi.fcl')     #write out FCL    
        inputs, outputs, rulebase, AND_operator, OR_operator, aggregator, implication, defuzz = build_fuzz_system('test_sys_phi.fcl')    #get system
        
        sys = Fuzzy_System(inputs, outputs, rulebase, AND_operator, OR_operator, aggregator, implication, defuzz)
        
        error = getError(valData, sys, inMF=inDataForm, outMF=outDataForm, sysOutType=outType, errType=errCalc)     #test system
        #print "TOTAL ERROR:", sum([x[2] for x in error])
        base_mse = (1.0/len([err for err in error if err[2] <> None]))*sum([err[2]**2 for err in error if err[2] <> None])
        
        for inp in copy.deepcopy(inputMFs): #for each input, check if all antecedents are same MF and remove
            input_antMFs = []
            for rule in rule_grid: 
                for ant in rule: 
                    if (ant[0], ant[1]) == inp and (not ant[2] in input_antMFs): 
                        input_antMFs.append(ant[2])
            
            if len(input_antMFs) == 1: 
                print inp, "removed. No effect on consequent from antecedent MFs."
                inputMFs.pop(inp)
                inputPARAMs.pop(inp)
    #if already baselined
    else: 
        base_mse = baselineMSE #for 
        
    if len(inMFlist) > 0: #if inputs limited, reduce input list
        print "Using reduced input set..."
        for inp in copy.deepcopy(inputMFs):
            if not inp in inMFlist:
                inputMFs.pop(inp)
                inputPARAMs.pop(inp)
    
    print 'Baseline Mean Square System Error:', base_mse

    inputMFs_rem = {} #store removed parameters
    inputPARAMs_rem = {}
    input_deltas = [] #track inputs and effects
    input_count = len(inputMFs)
    
    while True:
        print "_________________________________________________________________"
        print "CHECKING MODEL EFFECTS OF ALL INPUTS..."
        for inp in copy.deepcopy(inputMFs): #for each input get delta error for input removeal
                        
            combinedData = copy.deepcopy(combDataTrain) #copy data to preserve raw data
            valData = copy.deepcopy(combDataVal)
            
            print "Checking Input", inp, 
            #REMOVE INPUT:
            inputMFs_rem[inp] = inputMFs.pop(inp)
            inputPARAMs_rem[inp] = inputPARAMs.pop(inp)
                    
            #GET NEW ERROR
            
            rule_grid = train_system(inputMFs, outputMFs, combinedData, inDataMFs=inDataForm, outDataMFs=outDataForm) #train rule grid
            write_fcl_file_FRBS(inputPARAMs, outputPARAMs, rule_grid, defuzz, 'test_sys_phi.fcl')     #write out FCL
            inputs, outputs, rulebase, AND_operator, OR_operator, aggregator, implication, defuzz = build_fuzz_system('test_sys_phi.fcl')    #get system
            sys1 = Fuzzy_System(inputs, outputs, rulebase, AND_operator, OR_operator, aggregator, implication, defuzz)
            print len(sys1.inputs), 'remaining', len(rule_grid), 'rules.'
            
            error = getError(valData, sys1, inMF=inDataForm, outMF=outDataForm, sysOutType=outType, errType=errCalc)     #test system
            #print "TOTAL ERROR:", sum([x[2] for x in error])
            new_mse = (1.0/len([err for err in error if err[2] <> None]))*sum([err[2]**2 for err in error if err[2] <> None])
            del_mse = (new_mse - base_mse)/base_mse
            print '  ==> Adjusted MSE: %.4f; normalized delta_MSE: %.6f' % (new_mse, del_mse)
            input_deltas.append([inp, del_mse])
        
            #ADD INPUT BACK:
            inputMFs[inp] = inputMFs_rem.pop(inp) #remove from removed lists
            inputPARAMs[inp] = inputPARAMs_rem.pop(inp)
            
        t_rem_ = copy.copy(t_rem)
        
        print "---------------------------------------"
        print "REMOVING INPUTS WITH LITTLE EFFECT..."
        while True:
            
            for pair in input_deltas:         #make new model without inputs below t_rem threshold
                if pair[1] < t_rem_:
                    inputMFs_rem[pair[0]]    = inputMFs.pop(pair[0]) #move inputs to removal list
                    inputPARAMs_rem[pair[0]] = inputPARAMs.pop(pair[0])
            print "%d inputs out of %d remaining at removal delta of %.4f" % (len(inputMFs), len(dataIn), t_rem_)
            
            combinedData = copy.deepcopy(combDataTrain) #copy data to preserve raw data
            valData = copy.deepcopy(combDataVal)
                        
            #GET NEW SYSTEM ERROR
            rule_grid = train_system(inputMFs, outputMFs, combinedData, inDataMFs=inDataForm, outDataMFs=outDataForm)
            write_fcl_file_FRBS(inputPARAMs, outputPARAMs, rule_grid, defuzz, 'test_sys_phi.fcl')     #write out FCL
            inputs, outputs, rulebase, AND_operator, OR_operator, aggregator, implication, \
                defuzz = build_fuzz_system('test_sys_phi.fcl')    #get system
            sys = Fuzzy_System(inputs, outputs, rulebase, AND_operator, OR_operator, aggregator, implication, defuzz)
            print '%d inputs remaining %d rules at removal delta of %.4f' % (len(sys.inputs), len(rule_grid), t_rem_)
            
            error = getError(valData, sys, inMF=inDataForm, outMF=outDataForm, sysOutType=outType, errType=errCalc)     #test system
            newbase_mse = (1.0/len([err for err in error if err[2] <> None]))*sum([err[2]**2 for err in error if err[2] <> None])
            del_mse = (newbase_mse - base_mse)/base_mse
            print 'New System MSE: %.4f; normalized delta_MSE: %.4f' % ((1.0/len(error))*sum([err[2]**2 for err in error]), del_mse)
            
            
            if (newbase_mse - base_mse)/base_mse < t_base or t_rem_ < 0.0001: break #if model within t_base, break
            
            else: #move all inputs back and adjust t_rem
                for inp in copy.deepcopy(inputMFs_rem):
                    inputMFs[inp]    = inputMFs_rem.pop(inp) 
                    inputPARAMs[inp] = inputPARAMs_rem.pop(inp)
                t_rem_ = t_rem_ - 0.5*t_rem_
            
        print "%d inputs out of %d remaining:" % (len(inputMFs), len(dataIn))
        for inp in inputMFs: print inp
        if input_count == len(inputMFs):
            print "No inputs removed... COMPLETE!"
            break
        else:
            input_count = len(inputMFs)
            #base_mse = newbase_mse
            input_deltas = [] #track inputs and effects
            inputMFs_rem = {} #clear out removed parameters
            inputPARAMs_rem = {} #clear out removed parameters
            print "REEVALUATING"
        
    print "%d inputs out of %d3 remaining:" % (len(inputMFs), len(dataIn))
    for inp in inputMFs: print inp
    
## TESTS

def input_optimize():

    print "*************************************"
    print "OPTIMIZING:  "
    inMFs = input_5gaussMFs
    outMFs = output_9gaussMFs
    defuzz = None
    
    outForm = 'gauss'
    inDataForm = 'gauss'
    outDataForm = 'gauss'
    outType = 'fuzzy'
    errType = 'fuzzy'
        
    input_arrays, output_arrays = generate_MFs(inMFs, outMFs)
    inputMFs    = { (d[1], d[3]): copy.deepcopy(input_arrays[d[3]]) for d in dataIn } #dict of all input MFs
    inputPARAMs = { (d[1], d[3]): copy.deepcopy(inMFs[d[3]]) for d in dataIn }        #list of input parameters
    outputMFs = {'sys_phi' : copy.deepcopy(output_arrays['sys_phi'])}
    outputPARAMs = {'sys_phi' : copy.deepcopy(outMFs['sys_phi'])}
    
    combDataVal = combData[:150] 
    print len(combDataVal), "data points for validation."
    combDataTrain = combData[:300]
    print len(combDataTrain), "data points for training."
    
    input_list = [key for key in inputMFs] #constant order list of inputMFs, index=ID
    
    for inp in input_list: print inp
    
    x = [random.choice([0,1]) for i in range(len(input_list))] #1 or 0 for each inputMF

    def fitness(inputIDs):
        
        inputIDs = [int(round(x,0)) for x in inputIDs]
        print "Inputs:", inputIDs,
        
        if not 1 in inputIDs: return 9999.0
        
        inMFs = {}
        inPARAMs = {} 
        for i in range(len(input_list)):
            if inputIDs[i] == 1: 
                inMFs[input_list[i]] = inputMFs[input_list[i]]
                inPARAMs[input_list[i]] = inputPARAMs[input_list[i]]
            
        
        dataVal = copy.deepcopy(combDataVal)
        dataTrain = copy.deepcopy(combDataTrain)
        
        rule_grid = train_system(inMFs, outputMFs, dataTrain, inDataMFs=inDataForm, outDataMFs=outDataForm)
        write_fcl_file_FRBS(inPARAMs, outputPARAMs, rule_grid, None, 'opt_sys_phi.fcl')     #write out FCL
        inputs, outputs, rulebase, AND_operator, OR_operator, aggregator, implication, defuzz = build_fuzz_system('opt_sys_phi.fcl')    #get system
        sys = Fuzzy_System(inputs, outputs, rulebase, AND_operator, OR_operator, aggregator, implication, defuzz)
        #sys.run(None, TESTMODE=True)
        error = getError(dataVal, sys, inMF=inDataForm, outMF=outDataForm, sysOutType=errType)     #test system
        base_mse = (1.0/len(error))*sum([err[2]**2 for err in error if err <> None])
        print 'MSE:', base_mse
    
        return base_mse
    
    bounds = [(0.0,1.0) for i in input_list]
    
    results = opt.differential_evolution(fitness, bounds, popsize=1.5, 
                                         polish=False, disp=True)
        
    print "FINISHED!"
    print results
    IDs = [int(i) for i in results.x]
    for i in range(len(input_list)):
        if IDs[i] == 1: print input_list[i] 





 
##
def test0():
    test_name = '12 Inputs'
    
    print "*************************************"
    print "TESTING:  ", test_name
    inMFs = input_7gaussMFs       #system in
    outMFs = output_9gaussMFs
    defuzz = None
    
    outForm = 'tri'
    inDataForm = 'tri'
    outDataForm = 'tri'
    errType = 'fuzzy'
    
    input_arrays, output_arrays = generate_MFs(inMFs, outMFs)
    
    inputMFs = {    ('VL_SYS_UNION', 'phi'):        copy.deepcopy(input_arrays['phi']),
                    ('FWD_SYS_UNION', 'phi'):       copy.deepcopy(input_arrays['phi']), 
                    ('WING_SYS_TYPE', 'phi'):       copy.deepcopy(input_arrays['phi']),
                    ('ENG_SYS_TYPE', 'phi'):        copy.deepcopy(input_arrays['phi']),
                    ('VL_SYS_UNION', 'w'):          copy.deepcopy(input_arrays['w']),
                    ('VL_SYS_TYPE', 'TP'):          copy.deepcopy(input_arrays['TP']),
                    ('WING_SYS_TYPE', 'WS'):        copy.deepcopy(input_arrays['WS']),
                    ('VL_SYS_PROP', 'sigma'):       copy.deepcopy(input_arrays['sigma']),    
                    ('VL_SYS_TYPE', 'e_d'):         copy.deepcopy(input_arrays['e_d']),
                    ('VL_SYS_DRV', 'eta_d'):        copy.deepcopy(input_arrays['eta_d']),
                    ('FWD_SYS_DRV', 'eta_d'):       copy.deepcopy(input_arrays['eta_d']),
                    ('FWD_SYS_PROP', 'eta_p'):      copy.deepcopy(input_arrays['eta_p']),
                }
    inputPARAMs = { ('VL_SYS_UNION', 'phi'):        copy.deepcopy(inMFs['phi']), 
                    ('FWD_SYS_UNION', 'phi'):       copy.deepcopy(inMFs['phi']),
                    ('WING_SYS_TYPE', 'phi'):       copy.deepcopy(inMFs['phi']),
                    ('ENG_SYS_TYPE', 'phi'):        copy.deepcopy(inMFs['phi']),    
                    ('VL_SYS_UNION', 'w'):          copy.deepcopy(inMFs['w']), 
                    ('VL_SYS_TYPE', 'TP'):          copy.deepcopy(inMFs['TP']),
                    ('WING_SYS_TYPE', 'WS'):        copy.deepcopy(inMFs['WS']), 
                    ('VL_SYS_PROP', 'sigma'):       copy.deepcopy(inMFs['sigma']),    
                    ('VL_SYS_TYPE', 'e_d'):         copy.deepcopy(inMFs['e_d']),
                    ('VL_SYS_DRV', 'eta_d'):        copy.deepcopy(inMFs['eta_d']),
                    ('FWD_SYS_DRV', 'eta_d'):       copy.deepcopy(inMFs['eta_d']),
                    ('FWD_SYS_PROP', 'eta_p'):      copy.deepcopy(inMFs['eta_p']),
                }
    
    outputMFs = {'sys_phi' : copy.deepcopy(output_arrays['sys_phi'])}
    outputPARAMs = {'sys_phi' : copy.deepcopy(outMFs['sys_phi'])}
    
    combinedData = copy.deepcopy(data)
    
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
    with Timer() as t:
        error = getError(combinedData, sys, inMF=inDataForm, outMF=outDataForm, sysOutType=errType)
    print '=> ', t.secs, 'secs to check error'
    print 'Total System Error:', sum([err[2] for err in error])
    print 'Mean Square System Error:', (1.0/len(error))*sum([err[2]**2 for err in error])
    print 'Root Mean Square System Error:', ( (1.0/len(error)) * sum([err[2]**2 for err in error]) )**0.5
    
    #actual vs. predicted plot
    plt.figure()
    plt.title('Actual vs. Predicted at Max Alpha Cut'+test_name)
    for err in error:
        if outDataForm == 'gauss': AC_actual = fuzzyOps.alpha_cut(0.8, [err[0][0],err[0][1]])
        else: AC_actual = fuzzyOps.alpha_at_val(err[0][0],err[0][1])
        if outForm == 'gauss': AC_pred = fuzzyOps.alpha_cut(0.8, (err[1][0],err[1][1]))
        else: AC_pred = fuzzyOps.alpha_at_val(err[1][0],err[1][1])
        
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
    
    #check random data points (9)
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
    
    #actual vs. error plot
    plt.figure()
    plt.title('Actual (Centroid) vs. Error'+test_name)
    cents = [fuzz.defuzz(err[0][0], err[0][1], 'centroid') for err in error]
    plt.scatter(cents, [err[2] for err in error])
    plt.xlabel('Actual (Centroid)')
    plt.ylabel('Fuzzy Error')   
    
##
def test1():
    test_name = '9 Inputs'
    
    print "*************************************"
    print "TESTING:  ", test_name
    inMFs = input_7gaussMFs       #system in
    outMFs = output_9gaussMFs
    defuzz = None
    
    outForm = 'tri'
    inDataForm = 'tri'
    outDataForm = 'tri'
    errType = 'fuzzy'
    
    input_arrays, output_arrays = generate_MFs(inMFs, outMFs)
    
    inputMFs = {    ('VL_SYS_UNION', 'phi'):        copy.deepcopy(input_arrays['phi']),
                    ('FWD_SYS_UNION', 'phi'):       copy.deepcopy(input_arrays['phi']), 
                    ('WING_SYS_TYPE', 'phi'):       copy.deepcopy(input_arrays['phi']),
                    ('ENG_SYS_TYPE', 'phi'):        copy.deepcopy(input_arrays['phi']),
                    ('VL_SYS_UNION', 'w'):          copy.deepcopy(input_arrays['w']),
                    ('VL_SYS_TYPE', 'TP'):          copy.deepcopy(input_arrays['TP']),
                    ('WING_SYS_TYPE', 'WS'):        copy.deepcopy(input_arrays['WS']),
                    ('VL_SYS_PROP', 'sigma'):       copy.deepcopy(input_arrays['sigma']),    
                    ('VL_SYS_TYPE', 'e_d'):         copy.deepcopy(input_arrays['e_d']),
                }
    inputPARAMs = { ('VL_SYS_UNION', 'phi'):        copy.deepcopy(inMFs['phi']), 
                    ('FWD_SYS_UNION', 'phi'):       copy.deepcopy(inMFs['phi']),
                    ('WING_SYS_TYPE', 'phi'):       copy.deepcopy(inMFs['phi']),
                    ('ENG_SYS_TYPE', 'phi'):        copy.deepcopy(inMFs['phi']),    
                    ('VL_SYS_UNION', 'w'):          copy.deepcopy(inMFs['w']), 
                    ('VL_SYS_TYPE', 'TP'):          copy.deepcopy(inMFs['TP']),
                    ('WING_SYS_TYPE', 'WS'):        copy.deepcopy(inMFs['WS']), 
                    ('VL_SYS_PROP', 'sigma'):       copy.deepcopy(inMFs['sigma']),    
                    ('VL_SYS_TYPE', 'e_d'):         copy.deepcopy(inMFs['e_d']),
                }
    
    outputMFs = {'sys_phi' : copy.deepcopy(output_arrays['sys_phi'])}
    outputPARAMs = {'sys_phi' : copy.deepcopy(outMFs['sys_phi'])}
    
    combinedData = copy.deepcopy(data)
    
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
    with Timer() as t:
        error = getError(combinedData, sys, inMF=inDataForm, outMF=outDataForm, sysOutType=errType)
    print '=> ', t.secs, 'secs to check error'
    print 'Total System Error:', sum([err[2] for err in error])
    print 'Mean Square System Error:', (1.0/len(error))*sum([err[2]**2 for err in error])
    print 'Root Mean Square System Error:', ( (1.0/len(error)) * sum([err[2]**2 for err in error]) )**0.5
    
    #actual vs. predicted plot
    plt.figure()
    plt.title('Actual vs. Predicted at Max Alpha Cut'+test_name)
    for err in error:
        if outDataForm == 'gauss': AC_actual = fuzzyOps.alpha_cut(0.8, [err[0][0],err[0][1]])
        else: AC_actual = fuzzyOps.alpha_at_val(err[0][0],err[0][1])
        if outForm == 'gauss': AC_pred = fuzzyOps.alpha_cut(0.8, (err[1][0],err[1][1]))
        else: AC_pred = fuzzyOps.alpha_at_val(err[1][0],err[1][1])
        
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
    
    #check random data points (9)
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
    
    #actual vs. error plot
    plt.figure()
    plt.title('Actual (Centroid) vs. Error'+test_name)
    cents = [fuzz.defuzz(err[0][0], err[0][1], 'centroid') for err in error]
    plt.scatter(cents, [err[2] for err in error])
    plt.xlabel('Actual (Centroid)')
    plt.ylabel('Fuzzy Error')   
    

##
def test2():
    test_name = '7 Inputs'
    
    print "*************************************"
    print "TESTING:  ", test_name
    inMFs = input_7gaussMFs       #system in
    outMFs = output_9gaussMFs
    defuzz = None
    
    outForm = 'tri'
    inDataForm = 'tri'
    outDataForm = 'tri'
    errType = 'fuzzy'
    
    input_arrays, output_arrays = generate_MFs(inMFs, outMFs)
    
    inputMFs = {    ('VL_SYS_UNION', 'phi'):        copy.deepcopy(input_arrays['phi']),
                    ('FWD_SYS_UNION', 'phi'):       copy.deepcopy(input_arrays['phi']), 
                    ('WING_SYS_TYPE', 'phi'):       copy.deepcopy(input_arrays['phi']),
                    ('ENG_SYS_TYPE', 'phi'):        copy.deepcopy(input_arrays['phi']),
                    ('VL_SYS_UNION', 'w'):          copy.deepcopy(input_arrays['w']),
                    ('VL_SYS_TYPE', 'TP'):          copy.deepcopy(input_arrays['TP']),
                    ('WING_SYS_TYPE', 'WS'):        copy.deepcopy(input_arrays['WS']),
                }
    inputPARAMs = { ('VL_SYS_UNION', 'phi'):        copy.deepcopy(inMFs['phi']), 
                    ('FWD_SYS_UNION', 'phi'):       copy.deepcopy(inMFs['phi']),
                    ('WING_SYS_TYPE', 'phi'):       copy.deepcopy(inMFs['phi']),
                    ('ENG_SYS_TYPE', 'phi'):        copy.deepcopy(inMFs['phi']),    
                    ('VL_SYS_UNION', 'w'):          copy.deepcopy(inMFs['w']), 
                    ('VL_SYS_TYPE', 'TP'):          copy.deepcopy(inMFs['TP']),
                    ('WING_SYS_TYPE', 'WS'):        copy.deepcopy(inMFs['WS']), 
                }
    
    outputMFs = {'sys_phi' : copy.deepcopy(output_arrays['sys_phi'])}
    outputPARAMs = {'sys_phi' : copy.deepcopy(outMFs['sys_phi'])}
    
    combinedData = copy.deepcopy(data)
    
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
    with Timer() as t:
        error = getError(combinedData, sys, inMF=inDataForm, outMF=outDataForm, sysOutType=errType)
    print '=> ', t.secs, 'secs to check error'
    print 'Total System Error:', sum([err[2] for err in error])
    print 'Mean Square System Error:', (1.0/len(error))*sum([err[2]**2 for err in error])
    print 'Root Mean Square System Error:', ( (1.0/len(error)) * sum([err[2]**2 for err in error]) )**0.5
    
    #actual vs. predicted plot
    plt.figure()
    plt.title('Actual vs. Predicted at Max Alpha Cut'+test_name)
    for err in error:
        if outDataForm == 'gauss': AC_actual = fuzzyOps.alpha_cut(0.8, [err[0][0],err[0][1]])
        else: AC_actual = fuzzyOps.alpha_at_val(err[0][0],err[0][1])
        if outForm == 'gauss': AC_pred = fuzzyOps.alpha_cut(0.8, (err[1][0],err[1][1]))
        else: AC_pred = fuzzyOps.alpha_at_val(err[1][0],err[1][1])
        
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
    
    #check random data points (9)
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
    
    #actual vs. error plot
    plt.figure()
    plt.title('Actual (Centroid) vs. Error'+test_name)
    cents = [fuzz.defuzz(err[0][0], err[0][1], 'centroid') for err in error]
    plt.scatter(cents, [err[2] for err in error])
    plt.xlabel('Actual (Centroid)')
    plt.ylabel('Fuzzy Error')   
    
    
    
##
def test3():
    test_name = 'System phi Inputs'
    
    print "*************************************"
    print "TESTING:  ", test_name
    inMFs = input_7gaussMFs       #system in
    outMFs = output_9gaussMFs
    defuzz = None
    
    outForm = 'tri'
    inDataForm = 'tri'
    outDataForm = 'tri'
    errType = 'fuzzy'
    
    input_arrays, output_arrays = generate_MFs(inMFs, outMFs)
    
    inputMFs = {    ('VL_SYS_UNION', 'phi'):        copy.deepcopy(input_arrays['phi']),
                    ('FWD_SYS_UNION', 'phi'):       copy.deepcopy(input_arrays['phi']), 
                    ('WING_SYS_TYPE', 'phi'):       copy.deepcopy(input_arrays['phi']),
                    ('ENG_SYS_TYPE', 'phi'):        copy.deepcopy(input_arrays['phi']),
                }
    inputPARAMs = { ('VL_SYS_UNION', 'phi'):        copy.deepcopy(inMFs['phi']), 
                    ('FWD_SYS_UNION', 'phi'):       copy.deepcopy(inMFs['phi']),
                    ('WING_SYS_TYPE', 'phi'):       copy.deepcopy(inMFs['phi']),
                    ('ENG_SYS_TYPE', 'phi'):        copy.deepcopy(inMFs['phi']),    
                }
    
    outputMFs = {'sys_phi' : copy.deepcopy(output_arrays['sys_phi'])}
    outputPARAMs = {'sys_phi' : copy.deepcopy(outMFs['sys_phi'])}
    
    combinedData = copy.deepcopy(data)
    
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
    with Timer() as t:
        error = getError(combinedData, sys, inMF=inDataForm, outMF=outDataForm, sysOutType=errType)
    print '=> ', t.secs, 'secs to check error'
    print 'Total System Error:', sum([err[2] for err in error])
    print 'Mean Square System Error:', (1.0/len(error))*sum([err[2]**2 for err in error])
    print 'Root Mean Square System Error:', ( (1.0/len(error)) * sum([err[2]**2 for err in error]) )**0.5
    
    #actual vs. predicted plot
    plt.figure()
    plt.title('Actual vs. Predicted at Max Alpha Cut'+test_name)
    for err in error:
        if outDataForm == 'gauss': AC_actual = fuzzyOps.alpha_cut(0.8, [err[0][0],err[0][1]])
        else: AC_actual = fuzzyOps.alpha_at_val(err[0][0],err[0][1])
        if outForm == 'gauss': AC_pred = fuzzyOps.alpha_cut(0.8, (err[1][0],err[1][1]))
        else: AC_pred = fuzzyOps.alpha_at_val(err[1][0],err[1][1])
        
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
    
    #check random data points (9)
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
    
    #actual vs. error plot
    plt.figure()
    plt.title('Actual (Centroid) vs. Error'+test_name)
    cents = [fuzz.defuzz(err[0][0], err[0][1], 'centroid') for err in error]
    plt.scatter(cents, [err[2] for err in error])
    plt.xlabel('Actual (Centroid)')
    plt.ylabel('Fuzzy Error')   
  





    
##
def test4():
    test_name = 'Full phis and majors Inputs'
    
    print "*************************************"
    print "TESTING:  ", test_name
    inMFs = input_7gaussMFs       #system in
    outMFs = output_9gaussMFs
    defuzz = None
    
    outForm = 'tri'
    inDataForm = 'tri'
    outDataForm = 'tri'
    errType = 'fuzzy'
    
    input_arrays, output_arrays = generate_MFs(inMFs, outMFs)
    
    inputMFs = {    ('VL_SYS_TYPE', 'phi'):         copy.deepcopy(input_arrays['phi']), 
                    ('VL_SYS_PROP', 'phi'):         copy.deepcopy(input_arrays['phi']), 
                    ('VL_SYS_DRV', 'phi'):          copy.deepcopy(input_arrays['phi']), 
                    ('VL_SYS_TECH', 'phi'):         copy.deepcopy(input_arrays['phi']),
                    ('FWD_SYS_TYPE', 'phi'):        copy.deepcopy(input_arrays['phi']), 
                    ('FWD_SYS_PROP', 'phi'):        copy.deepcopy(input_arrays['phi']), 
                    ('FWD_SYS_DRV', 'phi'):         copy.deepcopy(input_arrays['phi']),
                    ('WING_SYS_TYPE', 'phi'):       copy.deepcopy(input_arrays['phi']),
                    ('ENG_SYS_TYPE', 'phi'):        copy.deepcopy(input_arrays['phi']),
                    ('VL_SYS_UNION', 'w'):          copy.deepcopy(input_arrays['w']),
                    ('VL_SYS_TYPE', 'TP'):          copy.deepcopy(input_arrays['TP']),
                    ('WING_SYS_TYPE', 'WS'):        copy.deepcopy(input_arrays['WS']),                
                }
    inputPARAMs = { ('VL_SYS_TYPE', 'phi'):         copy.deepcopy(inMFs['phi']), 
                    ('VL_SYS_PROP', 'phi'):         copy.deepcopy(inMFs['phi']), 
                    ('VL_SYS_DRV', 'phi'):          copy.deepcopy(inMFs['phi']), 
                    ('VL_SYS_TECH', 'phi'):         copy.deepcopy(inMFs['phi']),
                    ('FWD_SYS_TYPE', 'phi'):        copy.deepcopy(inMFs['phi']), 
                    ('FWD_SYS_PROP', 'phi'):        copy.deepcopy(inMFs['phi']), 
                    ('FWD_SYS_DRV', 'phi'):         copy.deepcopy(inMFs['phi']),
                    ('WING_SYS_TYPE', 'phi'):       copy.deepcopy(inMFs['phi']),
                    ('ENG_SYS_TYPE', 'phi'):        copy.deepcopy(inMFs['phi']),  
                    ('VL_SYS_UNION', 'w'):          copy.deepcopy(inMFs['w']),
                    ('VL_SYS_TYPE', 'TP'):          copy.deepcopy(inMFs['TP']),
                    ('WING_SYS_TYPE', 'WS'):        copy.deepcopy(inMFs['WS']),
                }
    
    outputMFs = {'sys_phi' : copy.deepcopy(output_arrays['sys_phi'])}
    outputPARAMs = {'sys_phi' : copy.deepcopy(outMFs['sys_phi'])}
    
    combinedData = copy.deepcopy(combData)
    
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
    with Timer() as t:
        error = getError(combinedData, sys, inMF=inDataForm, outMF=outDataForm, sysOutType=errType)
    print '=> ', t.secs, 'secs to check error'
    print 'Total System Error:', sum([err[2] for err in error])
    print 'Mean Square System Error:', (1.0/len(error))*sum([err[2]**2 for err in error])
    print 'Root Mean Square System Error:', ( (1.0/len(error)) * sum([err[2]**2 for err in error]) )**0.5
    
    #actual vs. predicted plot
    plt.figure()
    plt.title('Actual vs. Predicted at Max Alpha Cut'+test_name)
    for err in error:
        if outDataForm == 'gauss': AC_actual = fuzzyOps.alpha_cut(0.8, [err[0][0],err[0][1]])
        else: AC_actual = fuzzyOps.alpha_at_val(err[0][0],err[0][1])
        if outForm == 'gauss': AC_pred = fuzzyOps.alpha_cut(0.8, (err[1][0],err[1][1]))
        else: AC_pred = fuzzyOps.alpha_at_val(err[1][0],err[1][1])
        
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
    
    #check random data points (9)
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
    
    #actual vs. error plot
    plt.figure()
    plt.title('Actual (Centroid) vs. Error'+test_name)
    cents = [fuzz.defuzz(err[0][0], err[0][1], 'centroid') for err in error]
    plt.scatter(cents, [err[2] for err in error])
    plt.xlabel('Actual (Centroid)')
    plt.ylabel('Fuzzy Error') 
            


##
def test5():
    test_name = 'Full phi Inputs'
    
    print "*************************************"
    print "TESTING:  ", test_name
    inMFs = input_7gaussMFs       #system in
    outMFs = output_9gaussMFs
    defuzz = None
    
    outForm = 'tri'
    inDataForm = 'tri'
    outDataForm = 'tri'
    errType = 'fuzzy'
    
    input_arrays, output_arrays = generate_MFs(inMFs, outMFs)
    
    inputMFs = {    ('VL_SYS_TYPE', 'phi'):         copy.deepcopy(input_arrays['phi']), 
                    ('VL_SYS_PROP', 'phi'):         copy.deepcopy(input_arrays['phi']), 
                    ('VL_SYS_DRV', 'phi'):          copy.deepcopy(input_arrays['phi']), 
                    ('VL_SYS_TECH', 'phi'):         copy.deepcopy(input_arrays['phi']),
                    ('FWD_SYS_TYPE', 'phi'):        copy.deepcopy(input_arrays['phi']), 
                    ('FWD_SYS_PROP', 'phi'):        copy.deepcopy(input_arrays['phi']), 
                    ('FWD_SYS_DRV', 'phi'):         copy.deepcopy(input_arrays['phi']),
                    ('WING_SYS_TYPE', 'phi'):       copy.deepcopy(input_arrays['phi']),
                    ('ENG_SYS_TYPE', 'phi'):        copy.deepcopy(input_arrays['phi']),                
                }
    inputPARAMs = { ('VL_SYS_TYPE', 'phi'):         copy.deepcopy(inMFs['phi']), 
                    ('VL_SYS_PROP', 'phi'):         copy.deepcopy(inMFs['phi']), 
                    ('VL_SYS_DRV', 'phi'):          copy.deepcopy(inMFs['phi']), 
                    ('VL_SYS_TECH', 'phi'):         copy.deepcopy(inMFs['phi']),
                    ('FWD_SYS_TYPE', 'phi'):        copy.deepcopy(inMFs['phi']), 
                    ('FWD_SYS_PROP', 'phi'):        copy.deepcopy(inMFs['phi']), 
                    ('FWD_SYS_DRV', 'phi'):         copy.deepcopy(inMFs['phi']),
                    ('WING_SYS_TYPE', 'phi'):       copy.deepcopy(inMFs['phi']),
                    ('ENG_SYS_TYPE', 'phi'):        copy.deepcopy(inMFs['phi']),  
                }
    
    outputMFs = {'sys_phi' : copy.deepcopy(output_arrays['sys_phi'])}
    outputPARAMs = {'sys_phi' : copy.deepcopy(outMFs['sys_phi'])}
    
    combinedData = copy.deepcopy(combData)
    
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
    with Timer() as t:
        error = getError(combinedData, sys, inMF=inDataForm, outMF=outDataForm, sysOutType=errType)
    print '=> ', t.secs, 'secs to check error'
    print 'Total System Error:', sum([err[2] for err in error])
    print 'Mean Square System Error:', (1.0/len(error))*sum([err[2]**2 for err in error])
    print 'Root Mean Square System Error:', ( (1.0/len(error)) * sum([err[2]**2 for err in error]) )**0.5
    
    #actual vs. predicted plot
    plt.figure()
    plt.title('Actual vs. Predicted at Max Alpha Cut'+test_name)
    for err in error:
        if outDataForm == 'gauss': AC_actual = fuzzyOps.alpha_cut(0.8, [err[0][0],err[0][1]])
        else: AC_actual = fuzzyOps.alpha_at_val(err[0][0],err[0][1])
        if outForm == 'gauss': AC_pred = fuzzyOps.alpha_cut(0.8, (err[1][0],err[1][1]))
        else: AC_pred = fuzzyOps.alpha_at_val(err[1][0],err[1][1])
        
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
    
    #check random data points (9)
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
    
    #actual vs. error plot
    plt.figure()
    plt.title('Actual (Centroid) vs. Error'+test_name)
    cents = [fuzz.defuzz(err[0][0], err[0][1], 'centroid') for err in error]
    plt.scatter(cents, [err[2] for err in error])
    plt.xlabel('Actual (Centroid)')
    plt.ylabel('Fuzzy Error')       
    
    
    
if __name__ == "__main__":
    #input_test()
    input_optimize()
        #2.5*(2.569771982)
    #input_test(baselineMSE=None, inMFlist=[])
    #test0()
    #test1()
    #test2()
    #test3()
    
    #operations2 = { ('VL_SYS_UNION', 'w'):  ( [('VL_SYS_TYPE', 'w'), ('VL_SYS_PROP', 'w'), ('VL_SYS_TECH', 'w')], 'AVERAGE' ),}
    #combData = combine_inputs(copy.deepcopy(combData), operations2)
    
    #test4()
    #test5()
    #plt.show()