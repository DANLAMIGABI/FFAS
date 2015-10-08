# -*- coding: utf-8 -*-
"""
Created on Mon Mar 30 08:33:12 2015

@author: frankpatterson
"""

import numpy as np
import skfuzzy as fuzz
import copy
import random

import fuzzy_operations as fuzzOps
import matplotlib.pyplot as plt

from datetime import datetime

###
############################
def write_fcl_file_NFS(system, filename):
    """
    For Neuro Fuzzy System (NEFPROX)
    takes in the NFS system, and writes a fuzzy control language file
    
    ------ INPUTS ------
    NFsys : nefprox instance
        optimized neuro fuzzy system
    filename : string
        name of file to write
    """
    
    #print 'Writing', filename, 'fcl file...'        
    
    c = open(filename, 'w')
    c.truncate()
    
    with open(filename, 'w') as fclfile:   #with the file open
    
        fclfile.seek(0) #start at beginning
        fclfile.write('FUNCTION_BLOCK ' + filename + '\n') #start fcl block
        fclfile.write('     #written with write_fcl_file_NFS' + '\n ')
        fclfile.write('\n')
        
        #get input MFS
        mm = {nrn1:{} for nrn1 in system.layer1}
        for input in mm:              #for each input
            for key in system.inputMFs:
                if key[0] == input: 
                    mm[input][key[1]] = system.inputMFs[key][2]
        
        #write inputs (INPUT_NAME:     REAL; (* RANGE(1 .. 9) *))
        fclfile.write('VAR_INPUT ' + '\n') #start input name block
        for inp in mm:
            mi = min([min(mm[inp][x]) for x in mm[inp]])
            ma = max([max(mm[inp][x]) for x in mm[inp]])
            fclfile.write('    ' + inp + ':     REAL; (* RANGE(' + 
                          str(mi) + ' .. ' + str(ma) + ') \n')
        fclfile.write('END_VAR ' + '\n') #end block
        fclfile.write('\n')
        
        mm_out = {} #get output MFs (should only be one output)
        outputname = system.layer3.keys()[0]
        for MF in system.outputMFs:
            if not MF[1] in mm_out:
                mm_out[MF[1]] = system.outputMFs[MF][2]
            
        #write outputs (OUTPUT_NAME:     REAL; (* RANGE(1 .. 9) *)) (should only be one output)
        fclfile.write('VAR_OUTPUT ' + '\n') #start input name block
        opt = outputname
        mi = min([min(mm_out[key]) for key in mm_out])
        ma = max([max(mm_out[key]) for key in mm_out])
        fclfile.write('    '+ opt + ':     REAL; (* RANGE(' + 
                        str(mi) + ' .. ' + str(ma) + ') \n')
        fclfile.write('END_VAR ' + '\n') #end block        
        fclfile.write('\n')
        
        #write input fuzzifications 
        inputMFs = mm
        for inp in inputMFs:
            fclfile.write('FUZZIFY ' + inp + '\n')
            for ling in inputMFs[inp]:  #write linguistic terms
                fclfile.write('    TERM ' + ling + ' := ')
                if   len(inputMFs[inp][ling]) == 3: y=[0.,1.,0.] #for triangular
                elif len(inputMFs[inp][ling]) == 4: y=[0.,1.,1.,0.] #for trapezoidal
                for i in range(len(inputMFs[inp][ling])): #write out points of MF
                    fclfile.write('(' + str(inputMFs[inp][ling][i]) + ',' + str(y[i]) + ') ')
                fclfile.write(';' + '\n')
            fclfile.write('END_FUZZIFY ' + '\n')
            fclfile.write('\n')
        
        #write output defuzzifications
        outputMFs = {opt: mm_out}
        for opt in outputMFs:
            fclfile.write('DEFUZZIFY ' + opt + '\n')
            for ling in outputMFs[opt]:  #write linguistic terms
                fclfile.write('    TERM ' + ling + ' := ')
                if   len(outputMFs[opt][ling]) == 3: y=[0.,1.,0.] #for triangular
                elif len(outputMFs[opt][ling]) == 4: y=[0.,1.,1.,0.] #for trapezoidal
                for i in range(len(outputMFs[opt][ling])): #write out points of MF
                    fclfile.write('(' + str(outputMFs[opt][ling][i]) + ',' + str(y[i]) + ') ')
                fclfile.write(';' + '\n')
            fclfile.write('END_FUZZIFY ' + '\n')
            fclfile.write('\n')
                 
        #write out operators (currently hardcoded)
        fclfile.write('RULEBLOCK' + '\n')
        fclfile.write('    RULEBLOCK' + '\n')       #
        fclfile.write('     AND:MIN;' + '\n')       #and operator (PRODUCT or MIN)
        fclfile.write('    OR:MAX;' + '\n')         #or operator (MAX) (per DeMorgan, opposite of AND)
        fclfile.write('    ACT:MIN;' + '\n')        #means to apply rule firing strength to output MFs (implication)
        fclfile.write('    (*ACCU:MAX;*)' + '\n')   #means to aggregate outputs (max)
        fclfile.write('    DEFUZZ:' + system.defuzz + '\n')   #means to aggregate outputs (max)
        fclfile.write('\n')
        
        
        #write out rules
        for rule in system.layer2:
            
            ruleStr = '    RULE ' + str(int(rule)) + ':    IF '
            for MF in system.connect1to2[rule]:           #write out antecedent
                ruleStr = ruleStr + '(' + MF[0] + ' IS ' + MF[1] + ') AND '
            for MF in system.connect2to3[rule]:
                ruleStr = ruleStr[:-4] + 'THEN ' + '(' + MF[0] + ' IS ' + MF[1] + ');  '
            fclfile.write(ruleStr + '\n') #append rule to file
            
        fclfile.write('\n')
            
        fclfile.write('END_RULEBLOCK' + '\n')
        fclfile.write('END_FUNCTION_BLOCK' + '\n')
        

###
class optInfo(object): 
    """
    Class tracks information in training NFS.
    """
    trackError = [[],[]] #track training error and validation error [e_t1, e_t2, ...], [e_v1, e_v2, ...] for each iteration
    iterations = 0      #number of iterations total
    learning_rate = []  #learning rate tracked
    lenTrainData = None #len of training data
    lenValidationData = None #len of validation data
        

###
def train_NEFPROX(system, trainData, valData, inMFs, outMFs, iRange, oRange,
                  inDataMFs='sing', outDataMFs='sing', sigma=0.0005, 
                  nTrainMax=None, nValMax=None, maxIterations=500, errConvergeCount=5,
                  TESTMODE=False): 
    """"
    Trains a NEFPROX system (based on Nernberger, Nauck, Kruse - 
    "Neuro-fuzzy control based on the NEFCON-model: recent developments"
    and on Nauck, Kruse - "Neuro-Fuzzy Systems for Function Approximation")
    
    ------ INPUTS ------
    
    system : instance of nefprox
        NEFPROX system to train
    trainData : list 
        training data set
    valData : list
        separate validation data set... data in form: 
        [quant_inputs, qual_inputs, outputData]
        with each data item ['function', 'var'] : [min,max] 
        (inputs for system are named 'function var')
    inMFs : dict
        dictionary of input MFs: {inputname  : {ling1 : ([x,y], params), ling2 : ([x,y], params), ...},
                                  inputname2 : {ling1: ([x,y], params), ... }, ... }
    outMFs : dict
        dictionary of output MFs: {ling1 : [x,y], ling2 : [xy], ...} 
    iRange : dict
        dictionary of input ranges for MFs: {[input] : [x,y], ... }
    oRange : dict
            dictionary of output ranges for MFs: {[output] : [x,y], ... }
    inDataMFs : string
        type of MF for input data (given [min, max]), supports sing, trap and tri 
    outDataMFs : string
        type of MF for output data (given [min,max]), supports sing, trap and tri 
    sigma : float
        learning rate. modifies adjustments to MFs
    nTrainMax : int
        maximum number of training points to use
    nValMax : int 
        maximum number of validation points to use
    
    Note: As of 30May15 only setup to train fuzzy triangular MFs
    """

    q = 0 #0 for quant data, 1 or qual data
    
    out_name = outMFs.keys()[0]

    #limit amount of data:
    if nTrainMax <> None: trainData = trainData[:nTrainMax]
    if nValMax <> None:   valData   = valData[:nValMax]
    
    del_sysErr = 0.01 #percentage for system error to decrease
    sysErrorLast = 9999
    errDecreaseCount = 0
    errConvergeCount = 0
    errIncreaseCount = 0
    convergeCount = 5           #for this number of turns to converge    
    bestSystem = system         #track best system
    bestError = sysErrorLast    #track best error
    iteration = 1               #track iterations through data
   
    trackOptInfo = optInfo      #track of optimization information 
    trackOptInfo.lenTrainData = len(trainData)
    trackOptInfo.lenValidationData = len(valData)
    
    #print "BUILDING DATA MFS"
    #GET MFs FOR INPUT TRAINING DATA
    trainData_MFs = copy.deepcopy(trainData)
    
    for dataIt in trainData_MFs:
        
        #create input MFs for each input
        for inp in dataIt[q]: 
            if inDataMFs == 'sing':   #create singleton MF
                dataIt[q][inp] = sum(dataIt[q][inp])/len(dataIt[q][inp]) #get mean of range (or single value)
                x_range = [dataIt[q][inp]*0.9, dataIt[q][inp]*1.1, (dataIt[q][inp]*1.1 - dataIt[q][inp]*0.9)/100.] 
                dataIt[q][inp] = list(fuzzOps.singleton_to_fuzzy(dataIt[q][inp], x_range)) #turn singleton value to MF 
            elif inDataMFs == 'tri':   #create triangluar MF (min, avg, max)
                x_range = np.arange(dataIt[q][inp][0]*0.9, dataIt[q][inp][1]*1.1, (dataIt[q][inp][1]*1.1 - dataIt[q][inp][0]*0.9)/150)
                y_vals = fuzz.trimf(x_range, [dataIt[q][inp][0], sum(dataIt[q][inp])/len(dataIt[q][inp]), dataIt[q][inp][1]])
                dataIt[q][inp] = [x_range, y_vals]
            elif inDataMFs == 'trap':  #create traoeziodal MF (min, min, max, max)
                x_range = np.arange(dataIt[q][inp][0]*0.9, dataIt[q][inp][1]*1.1, (dataIt[q][inp][1]*1.1 - dataIt[q][inp][0]*0.9)/150)
                y_vals = fuzz.trimf(x_range, [dataIt[q][inp][0], dataIt[q][inp][0], dataIt[q][inp][0], dataIt[q][inp][1]])
                dataIt[q][inp] = [x_range, y_vals]

        #create output MFs
        if outDataMFs == 'sing':   #create singleton MF
            dataIt[2] = sum(dataIt[2])/len(dataIt[2]) #get average for singleton value
            x_range = [dataIt[2]*0.9, dataIt[2]*1.1, (dataIt[2]*1.1 - dataIt[2]*0.9)/100.] 
            dataIt[2] = list(fuzzOps.singleton_to_fuzzy(dataIt[2], x_range)) #turn singleton value to MF           
        elif outDataMFs == 'tri':   #create singleton MF
            x_range = np.arange(dataIt[2][0]*0.9, dataIt[2][1]*1.1, (dataIt[2][1]*1.1 - dataIt[2][0]*0.9)/150)
            y_vals = fuzz.trimf(x_range, [dataIt[2][0], sum(dataIt[2])/len(dataIt[2]), dataIt[2][1]])
            dataIt[2] = [x_range, y_vals]        
        elif outDataMFs == 'trap':   #create singleton MF
            x_range = np.arange(dataIt[2][0]*0.9, dataIt[2][1]*1.1, (dataIt[2][1]*1.1 - dataIt[2][0]*0.9)/150)
            y_vals = fuzz.trimf(x_range, [dataIt[2][0], dataIt[2][0], dataIt[2][1], dataIt[2][1]])
            dataIt[2] = [x_range, y_vals]

    #GET MFs FOR VALIDATION DATA
    valData_MFs = copy.deepcopy(valData)
    for dataIt in valData_MFs:
        
        #create input MFs for each input
        for inp in dataIt[q]: 
            if inDataMFs == 'sing':   #create singleton MF
                dataIt[q][inp] = sum(dataIt[q][inp])/len(dataIt[q][inp]) #get mean of range (or single value)
                x_range = [dataIt[q][inp]*0.9, dataIt[q][inp]*1.1, (dataIt[q][inp]*1.1 - dataIt[q][inp]*0.9)/100.] 
                dataIt[q][inp] = list(fuzzOps.singleton_to_fuzzy(dataIt[q][inp], x_range)) #turn singleton value to MF 
            elif inDataMFs == 'tri':   #create triangluar MF (min, avg, max)
                x_range = np.arange(dataIt[q][inp][0]*0.9, dataIt[q][inp][1]*1.1, (dataIt[q][inp][1]*1.1 - dataIt[q][inp][0]*0.9)/150)
                y_vals = fuzz.trimf(x_range, [dataIt[q][inp][0], sum(dataIt[q][inp])/len(dataIt[q][inp]), dataIt[q][inp][1]])
                dataIt[q][inp] = [x_range, y_vals]
            elif inDataMFs == 'trap':  #create traoeziodal MF (min, min, max, max)
                x_range = np.arange(dataIt[q][inp][0]*0.9, dataIt[q][inp][1]*1.1, (dataIt[q][inp][1]*1.1 - dataIt[q][inp][0]*0.9)/150)
                y_vals = fuzz.trimf(x_range, [dataIt[q][inp][0], dataIt[q][inp][0], dataIt[q][inp][0], dataIt[q][inp][1]])
                dataIt[q][inp] = [x_range, y_vals]

        #create output MFs
        if outDataMFs == 'sing':   #create singleton MF
            dataIt[2] = sum(dataIt[2])/len(dataIt[2]) #get average for singleton value
            x_range = [dataIt[2]*0.9, dataIt[2]*1.1, (dataIt[2]*1.1 - dataIt[2]*0.9)/100.] 
            dataIt[2] = list(fuzzOps.singleton_to_fuzzy(dataIt[2], x_range)) #turn singleton value to MF           
        elif outDataMFs == 'tri':   #create singleton MF
            x_range = np.arange(dataIt[2][0]*0.9, dataIt[2][1]*1.1, (dataIt[2][1]*1.1 - dataIt[2][0]*0.9)/150)
            y_vals = fuzz.trimf(x_range, [dataIt[2][0], sum(dataIt[2])/len(dataIt[2]), dataIt[2][1]])
            dataIt[2] = [x_range, y_vals]        
        elif outDataMFs == 'trap':   #create singleton MF
            x_range = np.arange(dataIt[2][0]*0.9, dataIt[2][1]*1.1, (dataIt[2][1]*1.1 - dataIt[2][0]*0.9)/150)
            y_vals = fuzz.trimf(x_range, [dataIt[2][0], dataIt[2][0], dataIt[2][1], dataIt[2][1]])
            dataIt[2] = [x_range, y_vals]
    
    
    #Add all inputs and input MFs
    for inp in inMFs: 
        if not inp in system.layer1:
            system.layer1[inp] = None
        for ling in inMFs[inp]:
            cMF = inMFs[inp][ling]
            system.inputMFs[(inp, ling)] = cMF
    
    #Add all outputs and output MFs
    for otp in outMFs:
        if not otp in system.layer3: 
            system.layer3[otp] = None
        for ling in outMFs[otp]:
            cMF = outMFs[otp][ling]
            system.outputMFs[(otp, ling)] = cMF
    outputname = otp #should only be one output
                            
    ###### MAIN TRAINING LOOP: 
    #iterate over data set until converged or max iterations performed
    while errConvergeCount < convergeCount and iteration <= maxIterations:
        
        #print "RANDOMIZING DATA"
        trainRef = range(len(trainData)) #get indecies for data
        random.shuffle(trainRef)         #shuffle indecies
        trainData2 = copy.deepcopy(trainData) #copy data
        trainData_MFs2 = copy.deepcopy(trainData_MFs)
        trainData = [trainData2[i] for i in trainRef] #assign new order to data
        trainData_MFs = [trainData_MFs2[i] for i in trainRef]
    
        #print "LEARNING RULES"
        #STRUCTURE LEARNING (learn rules)  
        rules = []
        for dataIt in trainData_MFs: #for each learning data pair (s_i, t_i):
        
            rule_ant = []
            #for each input create input MF
            for inp in dataIt[q]: 
                
                #find the MF (j) that returns the maximum degree of memebership for input
                maxMF = (0.0, next(k for k in inMFs[inp[0] + '_' + inp[1]])) #track firing strength and MF
                for mfx in inMFs[inp[0] + '_' + inp[1]]: 
                    union = fuzz.fuzzy_and(inMFs[inp[0] + '_' + inp[1]][mfx][0], inMFs[inp[0] + '_' + inp[1]][mfx][1],
                                        dataIt[q][inp][0], dataIt[q][inp][1]) #get union (AND) of two MFs
                    fs = max(union[1])                                           #get firing strength
                    if fs > maxMF[0]: maxMF = (fs, mfx)
                rule_ant.append((inp[0]+'_'+inp[1], maxMF[1])) #add to list of best input MFs
            
            #find output MF that output belongs to with highest degree         
            maxMF = (0.0, outMFs[out_name].itervalues().next())  #grab random MF to start
            for mfx in outMFs[out_name]:
                union = fuzz.fuzzy_and(dataIt[2][0], dataIt[2][1], 
                                    outMFs[out_name][mfx][0], outMFs[out_name][mfx][1]) #get union (AND) of two MFs
                fs = max(union[1])  #get "firing strength" (yes I know it's an output)
                if fs > maxMF[0]: maxMF = (fs, mfx)
            
            rules.append([sorted(rule_ant), maxMF[1], maxMF[0]]) #antecendent
                         
        
        #METHOD 2: for creating rule base
        #create rule grid, averaging rule "degree" for each consequent. keep the consequent
        #with the highest average "degree"
        rule_grid = {}
        for rule in rules:
            if rule[2] > 0.0 and len(rule[0]) > 0:
                antX = tuple(sorted(a for a in rule[0]))
                if not antX in rule_grid:               #if rule isn't in grid add it
                    rule_grid[antX] = {rule[1]: (rule[2], 1)} #add new dict with consequent giving (total degree, number of data points)
                else:   
                    if not rule[1] in rule_grid[antX]: #if consequent isn't already accounted for
                        rule_grid[antX][rule[1]] = (rule[2], 1) 
                    else: 
                        rule_grid[antX][rule[1]] = \
                            (rule_grid[antX][rule[1]][0] + rule[2],#add to the rule dgree and
                             rule_grid[antX][rule[1]][1] + 1)      #update the number of data points
        
        for rule in rule_grid: #for each rule grid get the average degree for each consequent
            for cons in rule_grid[rule]: 
                rule_grid[rule][cons] = (rule_grid[rule][cons][0]/ \
                                        rule_grid[rule][cons][1], rule_grid[rule][cons][1])
        
        for rule in rule_grid: #for each rule grid get the average degree for each consequent
            rule_deg = 0.0
            rule_cons = ''
            for cons in rule_grid[rule]:  
                if rule_grid[rule][cons][0] > rule_deg:  #capture consequent with hightest average "degree"
                    rule_deg = rule_grid[rule][cons][0]
                    rule_cons = cons
            rule_grid[rule] = [rule_cons, rule_deg] 
       
        #Translate Rule Grid into NEFPROX
        system.layer2 = {}
        system.connect1to2 = {}
        system.connect2to3 = {}
   
        for rule in rule_grid:    
           
            if len(system.layer2) > 0: nodeNo = max([int(ruleNo) for ruleNo in system.layer2]) + 1.0 #get new node number/name
            else: nodeNo = 0
            
            system.connect1to2[nodeNo] = {antIn:0.0 for antIn in rule} #create antecedent dict for rule
            system.layer2[nodeNo] = None
            system.connect2to3[nodeNo] = {(outputname, rule_grid[rule][0]): None}        

        
        #PARAMETER LEARNING (adjust MFs)
        trainingError = 0.0
        trainingActPre = [[],[]] #[[actuals], [prediteds]]
        time1 = datetime.now()
        for i in range(len(trainData)): #for each learning data pair (s_i, t_i) in raw training data:
            
            #progress report
            time2 = datetime.now()
            if (time2-time1).seconds > 60.0: 
                print round(100.0*(float(i)/float(len(trainData))),1), '% done learning parameters.'
                time1=datetime.now()
                #import pdb; pdb.set_trace()
                #sys.exit("Test Finish")
            
            #build input object            
            if inDataMFs == 'sing': #if crisp inputs use avg of original data
                inData = {inp[0]+'_'+inp[1]: sum(trainData[i][q][inp])/len(trainData[i][q][inp]) for inp in trainData[i][q]} 
            else: #otherwise use input
                inData = {inp[0]+'_'+inp[1]: trainData_MFs[i][q][inp] for inp in trainData_MFs[i][q]} 
            
            output = system.run(inData) #pass input through system to get result
            output = output[output.keys()[0]] #should only  be one output value, get it.
                 
            #get delta value: d = t_i - o_i for each output unit (should only be one)
            if not outDataMFs=='sing' and isinstance(output, list):  #if both output and data are fuzzy
                raise StandardError('Still have to convert data with output range to MF')
                err = fuzzy_error.fuzErrorAC(trainData_MFs[i], output)
            elif outDataMFs=='sing' and isinstance(output, float):  #if both output and data are crisp
                err = sum(trainData[i][2])/len(trainData[i][2]) - output
                trainingError = trainingError + err**2
                trainingActPre[0].append(sum(trainData[i][2])/len(trainData[i][2]))
                trainingActPre[1].append(output)
            elif not outDataMFs=='sing' and isinstance(output, float):  #if both output is fuzzy and data is crisp
                raise StandardError('You have not created a case for this yet')
            elif outDataMFs=='sing' and isinstance(output, list):  #if both output is crisp and data is fuzzy
                raise StandardError('You have not created a case for this yet')
            
            #back propagate error through system
            system.backpropagate(err, sigma, dataIt[2], iRange, oRange)
        
        trackOptInfo.trackError[0].append( (trainingError/len(trainData))**0.5 ) #save RMS training error
            
        #CHECK SYSTEM ERROR: with Validation data
        sysError = 0.0
        sysActPre = [[],[]] #[[actuals], [prediteds]]
        for i in range(len(valData)): #for each learning data pair (s_i, t_i):

            #build input object
            if inDataMFs == 'sing': #if crisp inputs use avg of original data
                inData = {inp[0]+'_'+inp[1]: sum(valData[i][q][inp])/len(valData[i][q][inp]) for inp in valData[i][q]} 
            else: #otherwise use input
                inData = {inp[0]+'_'+inp[1]: valData_MFs[i][q][inp] for inp in valData_MFs[i][q]} 
                    
            output = system.run(inData) #pass input through system to get result
            output = output[output.keys()[0]] #should only  be one output value, get it.
            
            #get delta value: d = t_i - o_i for each output unit (should only be one)
            if not outDataMFs=='sing' and isinstance(output, list):  #if both output and data are fuzzy
                raise StandardError('Still have to convert data with output range to MF')
                err = fuzzy_error.fuzErrorAC(valData_MFs[i][2], output)
            elif outDataMFs=='sing' and isinstance(output, float):  #if both output and data are crisp
                err = (sum(valData[i][2])/len(valData[i][2]) - output)**2
                sysActPre[0].append(sum(valData[i][2])/len(valData[i][2]))
                sysActPre[1].append(output)
            elif not outDataMFs=='sing' and isinstance(output, float):  #if both output is fuzzy and data is crisp
                raise StandardError('You have not created a case for this yet')
            elif outDataMFs=='sing' and isinstance(output, list):  #if both output is crisp and data is fuzzy
                raise StandardError('You have not created a case for this yet')
                
            sysError = sysError + err #sum up error
        sysError = (sysError/len(valData))**0.5 #total system error = (1/2N)sum( (t_i - o_i)^2 )
        trackOptInfo.trackError[1].append( sysError ) #track validation error
        
        #ONLY TEST MODE
        if TESTMODE:
            plt.figure()
            plt.scatter(trainingActPre[0],trainingActPre[1])
            plt.scatter(sysActPre[0], sysActPre[1])
            plt.legend(["Training", "Validation"])
            plt.show()
            
            
            
        #check error progress
        change_sysError = ((sysErrorLast - sysError)/sysErrorLast) #get normalized change in error (negative is an increase)
        
        if iteration > 1:
            if change_sysError > 0: sigma = sigma*1.03
            elif change_sysError < 0: sigma = sigma*0.5
            
        
        
        if change_sysError <= del_sysErr and \
           change_sysError >= 0: errConvergeCount = errConvergeCount + 1   #add to count if error change is small enough
        else: errConvergeCount = 0 #otherwise reset it
        
        
        
        """
        if change_sysError > 0: 
            errDecreaseCount = errDecreaseCount + 1
            errIncreaseCount = errIncreaseCount + 1   #add to count if error increases
            #errIncreaseCount = 0 #otherwise reset it
        else: 
            errIncreaseCount = errIncreaseCount + 1   #add to count if error increases
            errDecreaseCount = 0
            
        if errDecreaseCount >= 4: 
            sigma = sigma*1.1 #if error increases x times, increase the learning rate
            errIncreaseCount = 0
            errDecreaseCount = 0
        elif errIncreaseCount >= 4: 
            sigma = sigma*0.9 #if fluctuations or increases
            errIncreaseCount = 0
            errDecreaseCount = 0
        """ 
        
        if abs(sysError) < bestError:           #track best syste
            bestSystem = copy.deepcopy(system) 
            bestError = abs(sysError)

        sysErrorLast = sysError #track last error
        
        iteration = iteration + 1
        trackOptInfo.learning_rate.append(sigma) #track learnin rate
        
        print 'system error:', round(sysError,6), '  delta system error:', round(change_sysError,6), 
        print '  decrease/increase count:', errDecreaseCount, '/', errIncreaseCount, "  learning rate:", round(sigma,8)
    
    #capture opt info            
    trackOptInfo.iterations = iteration
    
    return bestSystem, trackOptInfo               


