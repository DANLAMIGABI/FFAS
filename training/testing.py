"""
author: Frank Patterson - 4Apr2015
Testing training modules
"""

### TESTING TRAIN NUMERICAL
import copy
import random

from training import *
from systems import *
import matplotlib.pyplot as plt

SYS_list = ['VL_SYS', 'FWD_SYS', 'WING_SYS', 'ENG_SYS'] #a list of systems
ASPECT_list = ['VL_SYS_TYPE', 'VL_SYS_PROP', 'VL_SYS_DRV', 'VL_SYS_TECH', \
            'FWD_SYS_PROP', 'FWD_SYS_DRV', 'FWD_SYS_TYPE',\
            'WING_SYS_TYPE', \
            'ENG_SYS_TYPE']    #list of system functional aspects 

#INPUT MFs
#baseline input MFs
input_trapMFs = \
{   'LD'   : {'low':[5,5,8,10],        'med':[6,8,15,17],          'high':[13,15,25,25]},
    'w'    : {'low':[3,3,15,20],       'med':[10,15,50,55],        'high':[45,50,150,150]},
    'e_d'  : {'low':[0.,0.,0.05,0.07], 'med':[0.03,0.05,0.15,0.17],'high':[0.13,0.15,0.3,0.3]},
    'phi'  : {'low':[5.5,6.5,9,9],     'med':[2.5,3.5,6.5,7.5],    'high':[1,1,3.5,4.5]},
    'FM'   : {'low':[.5,.5,.6,.65],    'med':[0.55,0.6,0.7,0.75],  'high':[0.68,0.7,0.9,0.9]},
    'f'    : {'low':[1,1,3.5,4.5],     'med':[2.5,3.5,6.5,7.5],    'high':[5.5,6.5,9,9]},
    'V_max': {'low':[150,150,190,210], 'med':[180,190,260,290],    'high':[250,260,550,550]},
    'eta_p': {'low':[0.6,0.6,0.7,0.75],'med':[0.67,0.7,0.83,0.88], 'high':[0.8,0.83,1.0,1.0]},
    'sigma': {'low':[.05,.05,.09,.10], 'med':[0.08,0.09,0.15,0.18],'high':[0.13,0.15,0.4,0.4]},
    'TP'   : {'low':[0.1,0.1,1,1.5],   'med':[0.9,1.0,10,11],      'high':[9,10,20,20]},
    'PW'   : {'low':[.01,.01,.1,.15],  'med':[0.08,0.1,1.0,1.5],   'high':[0.7,1.0,5.0,5.0]},
    'eta'  : {'low':[.5,.5,.75,.85],   'med':[0.8,0.85,0.93,0.95], 'high':[0.9,0.93,1.0,1.0]},
    'WS'   : {'low':[15,15,40,50],     'med':[20,40,120,150],      'high':[100,120,300,300]},
    'SFC'  : {'low':[0.,0.,.45,.5],    'med':[0.4,0.45,0.58,0.6],  'high':[0.55,0.58,1.0,1.0]},
    }
   
input_triMFs = copy.deepcopy(input_trapMFs) #triangular  MFs
for var in input_triMFs:
    for ling in input_triMFs[var]:
        input_triMFs[var][ling] = [input_triMFs[var][ling][0], 0.5*(input_triMFs[var][ling][1]+input_triMFs[var][ling][2]), input_triMFs[var][ling][3]]

input_mixedMFs = copy.deepcopy(input_trapMFs) #1st and last MFs are trapezoidal, middle are triangular
for var in input_mixedMFs:
    for ling in input_mixedMFs[var]:
        if ling == 'med':
            input_mixedMFs[var][ling] = [input_mixedMFs[var][ling][0], 
                                            0.5*(input_mixedMFs[var][ling][1]+input_mixedMFs[var][ling][2]), 
                                            input_mixedMFs[var][ling][3]]        
##
def test_reading_training_PHI():
    # Training system for System Empty Weight Ratio ##

    #test reading fuzzy input data
    dataIn = readFuzzyInputData('data/morphInputs_13Jun15.csv')
    combinedData = buildInputs(ASPECT_list, dataIn, 'data/phiData_251pts.csv', True)
    
    
    #get union of FWD and VL system empty weight ratio and average wing loading
    operations1 = {('VL_SYS_UNION', 'phi'):  ( [('VL_SYS_TYPE', 'phi'), ('VL_SYS_PROP', 'phi'), ('VL_SYS_DRV', 'phi'), ('VL_SYS_TECH', 'phi')], 'UNION' ),
                    ('FWD_SYS_UNION', 'phi'): ( [('FWD_SYS_TYPE', 'phi'), ('FWD_SYS_PROP', 'phi'), ('FWD_SYS_DRV', 'phi')], 'UNION' ),
                    ('VL_SYS_UNION', 'w'):  ( [('VL_SYS_TYPE', 'w'), ('VL_SYS_PROP', 'w'), ('VL_SYS_TECH', 'w')], 'AVERAGE' ),
                    }
    combinedData = combine_inputs(combinedData, operations1)

    #get average system empty weight ratio
    operations1 = {('SYS_PHI', 'phi'):  ( [('VL_SYS_UNION', 'phi'), ('FWD_SYS_UNION', 'phi'), ('WING_SYS_TYPE', 'phi'), ('ENG_SYS_TYPE', 'phi')], 'AVERAGE' ),
                    }
    combinedData = combine_inputs(combinedData, operations1)
    
    
    output_trapMFs = {'sys_phi': { 'verylow' :[1,1,2,3], 'low'     :[1,2,4,5], 'med'     :[3,4,6,7], 'high'    :[5,6,8,9], 'veryhigh':[7,8,9,9]} }
    
    input_arrays, output_arrays = generate_MFs(input_mixedMFs, output_trapMFs)
    
    #define actual MFs
    inMFs = \
        {('SYS_PHI',       'phi'):     input_arrays['phi'],
        ('VL_SYS_UNION',  'w'):        input_arrays['w'],
        ('VL_SYS_TYPE',   'TP'):       input_arrays['TP'],
        ('WING_SYS_TYPE', 'WS'):       input_arrays['WS'],
        ('VL_SYS_PROP',   'sigma'):    input_arrays['sigma'],
        }
    
    #OUTPUT MFs:
    outMFs = {'sys_phi': output_arrays['sys_phi']}
    
    rule_grid = train_system(inMFs, outMFs, combinedData)

    #Testing plot:
    #plot_rule_grid(rule_grid, inMFs, outMFs, combinedData, ('SYS_PHI',   'phi'), ('VL_SYS_UNION', 'w'), ('WING_SYS_TYPE',   'WS'))
    
    #plot_parallel(rule_grid, inMFs, outMFs, combinedData, None)
    
    #inMFs_write = inMFs
    #for var in inMFs_write: 
    #    inMFs_write[var] = input_mixedMFs[var[1]] #get piecewise list of input_MFs
    
    write_fcl_file_FRBS(inMFs_write, output_trapMFs, rule_grid, None, 'FCL_files/test_sys_phi.fcl')

##
def test_reading_training_FoM():
    # Training system for System Figure of Merit    ##
                                                    
    dataIn = readFuzzyInputData('data/POC_morphInputs_01Mar15.csv')
    combinedData = buildInputs(ASPECT_list, dataIn, 'data/FoMdata_13Apr15.csv', False,
                                inputCols={'w':4, 'sigma':8, 'e_d':9, 'eta':11,},
                                outputCols={'sysFoM':12})
    
    output_trapMFs = {'sys_FoM': {  'verylow' :[0.00, 0.00, 0.45, 0.55],
                                    'low'     :[0.40, 0.45, 0.60, 0.65],
                                    'med'     :[0.55, 0.60, 0.70, 0.75],
                                    'high'    :[0.65, 0.70, 0.85, 0.90],
                                    'veryhigh':[0.80, 0.85, 1.00, 1.00]} }
    
    inputs, output = read_data.generate_MFs(input_mixedMFs, output_trapMFs)
    
    inMFs = \
    {   ('DATA',   'e_d'):     inputs['e_d'],
        ('DATA',   'sigma'):   inputs['sigma'],
        ('DATA' ,   'w'):      inputs['w'],
        ('DATA' ,   'eta'):    inputs['eta'],
    }
    
    #OUTPUT MF
    outMFs = \
    {'sys_FoM': output['sys_FoM']}
    
    rule_grid = train_system(inMFs, outMFs, combinedData, inDataMFs='sing', outDataMFs='sing')
    
    #Show plot:
    plot_rule_grid(rule_grid, inMFs, outMFs, combinedData, ('DATA',   'w'), ('DATA',   'sigma'), ('DATA',   'eta'))
    
    plot_parallel(rule_grid, inMFs, outMFs, combinedData, None)
    
    inMFs_write = inMFs
    for var in inMFs_write: 
        inMFs_write[var] = input_mixedMFs[var[1]] #get piecewise list of input_MFs
    
    write_fcl_file_FRBS(inMFs_write, output_trapMFs, rule_grid, 'FCL_files/FOMsys_16Apr15.fcl')

##
def test_reading_training_FoM_mp():
    # Training system for System Figure of Merit    ##
                                                    
    dataIn = readFuzzyInputData('data/POC_morphInputs_01Mar15.csv')
    combinedData = buildInputs(ASPECT_list, dataIn, 'data/FoMdata_13Apr15.csv', False,
                                inputCols={'w':4, 'sigma':8, 'e_d':9, 'eta':11,},
                                outputCols={'sysFoM':12})
    
    output_trapMFs = {'sys_FoM': {  'verylow' :[0.00, 0.00, 0.45, 0.55],
                                    'low'     :[0.40, 0.45, 0.60, 0.65],
                                    'med'     :[0.55, 0.60, 0.70, 0.75],
                                    'high'    :[0.65, 0.70, 0.85, 0.90],
                                    'veryhigh':[0.80, 0.85, 1.00, 1.00]} }
    
    inputs, output = generate_MFs(input_mixedMFs, output_trapMFs)
    
    inMFs = \
    {   ('DATA',   'e_d'):     inputs['e_d'],
        ('DATA',   'sigma'):   inputs['sigma'],
        ('DATA' ,   'w'):      inputs['w'],
        ('DATA' ,   'eta'):    inputs['eta'],
    }
    
    #OUTPUT MF
    outMFs = \
    {'sys_FoM': output['sys_FoM']}
    
    rule_grid = train_system_mp(inMFs, outMFs, combinedData, inDataMFs='sing', outDataMFs='sing')
    
    #Show plot:
    plot_rule_grid(rule_grid, inMFs, outMFs, combinedData, ('DATA',   'w'), ('DATA',   'sigma'), ('DATA',   'eta'))
    
    plot_parallel(rule_grid, inMFs, outMFs, combinedData, None)
    
    inMFs_write = inMFs
    for var in inMFs_write: 
        inMFs_write[var] = input_mixedMFs[var[1]] #get piecewise list of input_MFs
    
    write_fcl_file_FRBS(inMFs_write, output_trapMFs, rule_grid, 'FCL_files/FOMsys_16Apr15.fcl')

    
    
    ## TEST OPTIMIZATION (SYS_FoM)
def test_opt_FoM():
    
    testFCL = 'FCL_files/FOMsys_16Apr15.fcl'
    
    dataIn = readFuzzyInputData('data/POC_morphInputs_01Mar15.csv')
    combTrainData = buildInputs(ASPECT_list, dataIn, 'data/FoMdata_13Apr15.csv', False,
                                inputCols={'w':4, 'sigma':8, 'e_d':9, 'eta':11,},
                                outputCols={'sysFoM':12})
    
    combValData = buildInputs(ASPECT_list, dataIn, 'data/FoMdata_val_19Apr15.csv', False,
                              inputCols={'w':4, 'sigma':8, 'e_d':9, 'eta':11,},
                              outputCols={'sysFoM':12})                                      
        
    input_MFs = {   ('DATA',   'e_d')   : {'A0':[0.,0.025,0.07],    'A1':[0.03,0.10,0.17],  'A2':[0.13,0.22,0.3]},
                    ('DATA',   'sigma') : {'A0':[.05,.07,.10],      'A1':[0.08,0.12,0.18],  'A2':[0.13,0.27,0.4]},
                    ('DATA' ,   'w')    : {'A0':[3,9,20],           'A1':[10,32,55],        'A2':[45,100,150]},
                    ('DATA' ,   'eta')  : {'A0':[.5,.63,.85],       'A1':[0.8,0.89,0.95],   'A2':[0.9,0.96,1.0]},
    }
    
    output_MFs = {'sys_FoM': {  'A0' :[0.00, 0.00, 0.45, 0.55],
                                'A1' :[0.40, 0.45, 0.60, 0.65],
                                'A2' :[0.55, 0.60, 0.70, 0.75],
                                'A3' :[0.65, 0.70, 0.85, 0.90],
                                'A4' :[0.80, 0.85, 1.00, 1.00]} }
    
    inRanges = { ('DATA',   'e_d')  : [0.0, 0.3],
                 ('DATA',   'sigma'): [0.05,0.4],
                 ('DATA' ,   'w')   : [3.,150.],
                 ('DATA' ,   'eta') : [0.5, 1.0],}
                 
    outRanges = {'sys_FoM': [0.0, 1.0]}
                 
    nMaxTrain = 500
    nMaxVal = 300
    
    run_optimization(combTrainData, combValData, input_MFs, inRanges, output_MFs, outRanges, testFCL, nMaxTrain, nMaxVal)
    
    
    ## TEST MP OPTIMIZATION (SYS_FoM)
def test_opt_FoM_mp():
    
    testFCL = 'FCL_files/FOMsys_16Apr15.fcl'
    
    dataIn = readFuzzyInputData('data/POC_morphInputs_01Mar15.csv')
    combTrainData = buildInputs(ASPECT_list, dataIn, 'data/FoMdata_13Apr15.csv', False,
                                inputCols={'w':4, 'sigma':8, 'e_d':9, 'eta':11,},
                                outputCols={'sysFoM':12})
    
    combValData = buildInputs(ASPECT_list, dataIn, 'data/FoMdata_val_19Apr15.csv', False,
                              inputCols={'w':4, 'sigma':8, 'e_d':9, 'eta':11,},
                              outputCols={'sysFoM':12})                                      
        
    input_MFs = {   ('DATA',   'e_d')   : {'A0':[0.,0.025,0.07],    'A1':[0.03,0.10,0.17],  'A2':[0.13,0.22,0.3]},
                    ('DATA',   'sigma') : {'A0':[.05,.07,.10],      'A1':[0.08,0.12,0.18],  'A2':[0.13,0.27,0.4]},
                    ('DATA' ,   'w')    : {'A0':[3,9,20],           'A1':[10,32,55],        'A2':[45,100,150]},
                    ('DATA' ,   'eta')  : {'A0':[.5,.63,.85],       'A1':[0.8,0.89,0.95],   'A2':[0.9,0.96,1.0]},
    }
    
    output_MFs = {'sys_FoM': {  'A0' :[0.00, 0.00, 0.45, 0.55],
                                'A1' :[0.40, 0.45, 0.60, 0.65],
                                'A2' :[0.55, 0.60, 0.70, 0.75],
                                'A3' :[0.65, 0.70, 0.85, 0.90],
                                'A4' :[0.80, 0.85, 1.00, 1.00]} }
    
    inRanges = { ('DATA',   'e_d')  : [0.0, 0.3],
                 ('DATA',   'sigma'): [0.05,0.4],
                 ('DATA' ,   'w')   : [3.,150.],
                 ('DATA' ,   'eta') : [0.5, 1.0],}
                 
    outRanges = {'sys_FoM': [0.0, 1.0]}
                 
    nMaxTrain = 500
    nMaxVal = 300
    
    run_optimization_mp(combTrainData, combValData, input_MFs, inRanges, output_MFs, outRanges, testFCL, nMaxTrain, nMaxVal)
    
    
    
        
    
## TEST MFs_to_varList and varList_to_MFs 
def test_parameterization():
    tot_error = 0.0
    for i in range(500):
        
        #generate some random inputs and symmetric MFs
        inMFs_test = {}
        MFlen = random.random()
        for i in range(4): #gen 3 inputs
            lings = {}
            x = round(random.random()/random.random(),1) #get minimum to start MFs
            for j in range(random.randrange(2,6)): #gen 2-5 MFs
                if MFlen < 0.333: #tri MF
                    w = round(random.random()/random.random(),1)
                    MF = [x-w, x, x+w]
                    x = x + round(random.random()/random.random(),1)
                
                elif MFlen < 0.66: #trap MF
                    w = [round(random.random()/random.random(),1), round(random.random()/random.random(),1)]
                    MF = [x-max(w), x-min(w), x+min(w), x+max(w)]
                    x = x + round(random.random()/random.random(),1)
                
                else: #gaussian MF
                    std = random.uniform(0.5, 3)
                    MF = [x, std]
                    x = x + round(random.random()/random.random(),1)
                    
                lings['A'+str(j)] = MF
            inMFs_test['input'+str(i)] = lings
                    
        #gen an output
        outMFs_test = {}
        lings = {}
        MFlen = random.random()
        x = round(random.random()/random.random(),1) #get minimum to start MFs
        for j in range(random.randrange(3,6)): #gen 3-5 MFs
            
            if MFlen < 0.33: #tri MF
                w = round(random.random()/random.random(),1)
                MF = [x-w, x, x+w]
                x = x + round(random.random()/random.random(),1)
            elif MFlen < 0.66: #trap MF
                w = [round(random.random()/random.random(),1), round(random.random()/random.random(),1)]
                MF = [x-max(w), x-min(w), x+min(w), x+max(w)]
                x = x + round(random.random()/random.random(),1)
            else: #gaussian MF
                std = random.uniform(0.5, 3)
                MF = [x, std]
                x = x + round(random.random()/random.random(),1)
                
            lings['A'+str(j)] = MF
        outMFs_test['output'+str(i)] = lings
    
        inList_test = [(key, len(inMFs_test[key])) for key in inMFs_test]
        outList_test = [(key, len(outMFs_test[key])) for key in outMFs_test]
        
        varList1, typeList1 = MF_to_varList2(inList_test[:], outList_test[:], inMFs_test, outMFs_test)
        #print 'inList:', inList_test
        #print 'varList:', varList1
        #print 'typeList:', typeList1
        
        inMFs_test1, outMFs_test1 = varList_to_MFs2(inList_test[:], outList_test[:], varList1[:], typeList1[:])
    
        error = 0
        
        #print 'INPUT MFS COMPARE!'
        for inp in inMFs_test1: 
            #print 'COMPARE INPUT:', inp, ' : ', len(inMFs_test[inp]), 'MFs'
            for mf in inMFs_test[inp]: 
                #print '     IN: ', mf, ':', inMFs_test[inp][mf]
                #print '     OUT:', mf, ':', inMFs_test1[inp][mf]
                error = error + sum([a-b for a,b in zip(inMFs_test[inp][mf], inMFs_test1[inp][mf])])
                #print '         Error =', error
                
        #print 'OUTPUT MFS COMPARE!'
        for otp in outMFs_test: 
            #print 'COMPARE OUTPUT:', otp, ' : ', len(outMFs_test[otp]), 'MFs'
            for mf in outMFs_test[otp]: 
                #print '     IN: ', mf, ':', outMFs_test[otp][mf]    
                #print '     OUT:', mf, ':', outMFs_test1[otp][mf]  
                error = error + sum([a-b for a,b in zip(outMFs_test[otp][mf], outMFs_test1[otp][mf])])
                #print '         Error =', error
        tot_error = tot_error + error
        print 'Error =', round(error,4)
        
    print 'Total error =', round(tot_error,4)
    
### TEST FUZZY ERROR
    
    
## FRBS TESTING:
def test_FRBS_error_funcs():
    #get test data
    dataIn = None #readFuzzyInputData('data/POC_morphInputs_01Mar15.csv')
    data = buildInputs(ASPECT_list, dataIn, 'data/FoMdata_13Apr15.csv', False,        #training data set
                       inputCols={'w':4, 'sigma':8, 'e_d':9, 'eta':11,}, 
                       outputCols={'sysFoM':12})
    for d in data:
        d[0] = {(inp[0] + '_' + inp[1]):d[0][inp] for inp in d[0]}
    
    # BUILD TEST SYSTEM
    longfile = 'FCL_files/FOMsys_16Apr15.fcl'    
    inputs, outputs, rulebase, AND_operator, OR_operator, aggregator, implication, defuzz = build_fuzz_system(longfile)
    sys = Fuzzy_System(inputs, outputs, rulebase, AND_operator, OR_operator, aggregator, implication, defuzz)

    error = getError(data[0:500], sys, inMF='sing', outMF='sing', sysOutType='crisp')
    
    error = sorted(error, key=lambda x: x[0])
    plt.figure()
    plt.scatter([x[0] for x in error],[x[1] for x in error])
    plt.show()

## NFS TESTING:    
def test_NFS_error_funcs():
    #get test data
    dataIn = None #readFuzzyInputData('data/POC_morphInputs_01Mar15.csv')
    data = buildInputs(ASPECT_list, dataIn, 'data/FoMdata_3Jun15.csv', False,        #training data set
                       inputCols={'w':4, 'sigma':8, 'e_d':9, 'eta':11,}, 
                       outputCols={'sysFoM':12})
    #for d in data:
    #    d[0] = {(inp[0] + '_' + inp[1]):d[0][inp] for inp in d[0]}
                                      
    # BUILD TEST SYSTEM
    longfile = 'FCL_files/FOMsys_trained_5In7Out_500tData_250vData.fcl'    
    inputs, outputs, rulebase, AND_operator, OR_operator, aggregator, implication, defuzz = build_fuzz_system(longfile)
    sys = NEFPROX(inputs, outputs, rulebase, None)
    
    error = getRangeError(data[0 : 100], sys, inMF='sing', outMF='sing')
    
    plt.figure()
    for err in error:
        plt.plot([err[0], err[0]], err[1], '-o', c='#666666', lw=1.5, ms=0.5)
    plt.plot([0.0,5.0],[0.0,5.0],'-r')
    plt.xlim([min([x[0] for x in error]),max([x[0] for x in error])])
    plt.ylim([min([x[0] for x in error]),max([x[0] for x in error])])
    plt.show()   

    
    
    ### TEST_FCL FILE WRITER
def test_fcl_file_writer():
    testFCL = 'FCL_files/FOMsys_16Apr15.fcl'
    inputs, outputs, rulebase, AND_operator, OR_operator, aggregator, implication, defuzz = build_fuzz_system(testFCL)
    NFS = NEFPROX(inputs, outputs, rulebase, defuzz) #system built
        
    write_fcl_file_NFS(NFS, 'FCL_files/test_file.fcl' )
    
    
    ### TEST TRAINING WITH MACKEY GLASS FUNCTION  ####
    #Mackey-Glass differential equation
    #dx/dt = beta*(x_tau/(1+x_tau^n)) - gamma*x
    #dx/dt(t) = ( 0.2x(t-r) ) / ( 1+x^10(t-r) ) - 0.1x(t)
def test_NEFPROX_MackeyGlass():    
    t_max = 3000
    dt = 1.0
    
    beta = 0.2
    tau = 17
    n = 10.0
    gamma = 0.1
    
    t = [0.0]
    x = [1.2]
    
    while t[-1] < t_max:
        x_tau = x[-1*min(len(x), tau+1)]
        x_dot = beta*(x_tau/(1+x_tau**n)) - gamma*x[-1]
        x.append(x[-1] + x_dot*dt)
        t.append(t[-1]+dt)
        
    plt.figure()
    plt.plot( t[1118:2117], x[1118:2117] )
    plt.show()
        
    #collect data
    inputData = []
    for i in range(1118, 2117):
        i1 = {  ('DATA', 't_18'): [x[i-18]],
                ('DATA', 't_12'): [x[i-12]],
                ('DATA', 't_6'):  [x[i-6]],
                ('DATA', 't_0'):  [x[i-0]], }
        o1 = [x[i+6]]
        inputData.append([i1, None, o1])
    print 'min:', min([id[2][0] for id in inputData])
    print 'max:', max([id[2][0] for id in inputData])
    minx = min([id[2][0] for id in inputData])
    maxx = max([id[2][0] for id in inputData])
    
    n = 7 #number of MFs
    m = 3 #MF type (3,4)#
    
    half_width = (maxx-minx)/float(n-1)
    step_width = 2*half_width/(m-1)
    MFs = []
    for i in range(n):
        range_start = minx+(i-1)*half_width
        MFparams = [range_start + i*step_width for i in range(m)]
        MFs.append(MFparams)
    MFdict = {'A'+str(i): MFs[i] for i in range(len(MFs))}
    
    triInputMFs = { ('DATA_t_18'): copy.deepcopy(MFdict),
                    ('DATA_t_12'): copy.deepcopy(MFdict),
                    ('DATA_t_6'):  copy.deepcopy(MFdict),
                    ('DATA_t_0'):  copy.deepcopy(MFdict), }
                 
    triOutputMFs = {('t_plus_6'):  MFdict }
    
    inputMFs, outputMFs = generate_MFs(triInputMFs, triOutputMFs)
    
    #append MFparams (just for neuro-fuzzy systems)
    for inp in inputMFs:
        for ling in inputMFs[inp]: 
            inputMFs[inp][ling] = inputMFs[inp][ling] + [triInputMFs[inp][ling]]
    
    #append MFparams (just for neuro-fuzzy systems)
    for otp in outputMFs:
        for ling in outputMFs[otp]:
            outputMFs[otp][ling] = outputMFs[otp][ling] + [triOutputMFs[otp][ling]]
    
    #ranges for constraints
    inLimits = {k:[minx - half_width - 0.1*(maxx-minx), maxx + half_width + 0.1*(maxx-minx)] for k in inputMFs}
    outLimits = {k:[minx - half_width - 0.1*(maxx-minx), maxx + half_width + 0.1*(maxx-minx)] for k in outputMFs}
    
    NFS = NEFPROX({}, {}, [], 'centroid') #system built
    NFS, optData = train_NEFPROX(NFS, inputData[:700], inputData[700:], inputMFs, outputMFs, 
                                    inLimits, outLimits, sigma=0.0001, maxIterations=20)
    write_fcl_file_NFS(NFS, 'mackey_glass_test.fcl')
    
    #write report 
    f = open( "test_report.txt", 'w' )
    for k in vars(optData):
        f.write(str(k) + "=" + str(vars(optData)[k])+"\n")
    f.close()

    NFSoutput = []
    for x1 in inputData:
        inData = {inp[0]+'_'+inp[1]: sum(x1[0][inp])/len(x1[0][inp]) for inp in x1[0]}  
        output = NFS.run( inData )
        NFSoutput.append(output.itervalues().next())
    
    plt.figure()
    plt.plot(t[1118:2117], x[1118:2117])
    plt.plot(t[1118+6:2117+6], NFSoutput)
    plt.show()

    print "----- INPUT MFS -----"
    for mf in NFS.inputMFs:
        print mf, NFS.inputMFs[mf][2],
    print "----- OUTPUT MFS -----"
    for mf in NFS.outputMFs:
        print mf, NFS.outputMFs[mf][2],
    print "----- RULES -----"
    for rule in NFS.layer2:
        for inp in NFS.connect1to2[rule]: print inp,
        print NFS.connect2to3[rule].keys()

    NFS.plot()
    
    import fuzzy_error
    
    error = fuzzy_error.getError(copy.deepcopy(inputData[300:800]), NFS, inMF='sing', outMF='sing', sysOutType='crisp')
    error = sorted(error, key=lambda x: x[0])
    plt.figure()
    plt.scatter([x[0] for x in error],[x[1] for x in error])
    plt.plot([0.0,5.0],[0.0,5.0],'-r')
    plt.xlim([min([x[0] for x in error]),max([x[0] for x in error])])
    plt.ylim([min([x[0] for x in error]),max([x[0] for x in error])])
    
    NFS.defuzz = None
    error2 = fuzzy_error.getRangeError(copy.deepcopy(inputData[0:500]), NFS, inMF='sing', outMF='sing')
    plt.figure()
    for err in error2:
        plt.plot([err[0], err[0]], err[1], '-o', c='#666666', lw=1.5, ms=0.5)
    plt.plot([0.0,5.0],[0.0,5.0],'-r')
    plt.xlim([min([x[0] for x in error]),max([x[0] for x in error])])
    plt.ylim([min([x[0] for x in error]),max([x[0] for x in error])])
    plt.show()
    
    
if __name__=="__main__":
    
    #test_reading_training_PHI()
    #test_reading_training_FoM()
    #test_opt_FoM()
    test_parameterization()
    #test_FRBS_error_funcs()
    #test_NFS_error_funcs()
    #test_fcl_file_writer()
    #test_NEFPROX_MackeyGlass()
    #test_reading_training_FoM_mp()]\
    #test_opt_FoM_mp()