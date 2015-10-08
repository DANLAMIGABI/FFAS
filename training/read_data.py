"""
author: Frank Patterson - 7Jun2015
- files to read in data
"""

import numpy as np
import scipy as sp
import skfuzzy as fuzz

import copy
import csv
import re
import operator
import itertools
import random

#from fuzzy_operations import *
import fuzzy_operations as fuzzyOps


###########################
def readExpertData(filename):
    """
    Read in and return Raw Expert Data
    
    ------INPUTS ------
    filename : string
        filename with data
    """
    data = []
    with open(filename, 'rU') as csvfile:   #with the file open
        input_reader = csv.reader(csvfile, delimiter=',')   #open csv reader
        
        for row in input_reader:                #read each row
            if not "INCOMPATIBLE" in row[1]:    #skip incompatible options
                outs = [float(row[-1]), float(row[-2])] #get outputs
                ins = row[1:-2]                      #get input indecies
                data.append([ins, outs])
                
    return data
    
    
    
    
###########################
def readFuzzyInputData(filename):
    """
    Read the input file and store as a list for pulling data from
    """

    print 'Reading', filename, 'data file and translating inputs.'        
    data = []
    
    with open(filename, 'rU') as csvfile:   #with the file open
        input_reader = csv.reader(csvfile, delimiter=',')   #open csv reader
        
        for row in input_reader:            #read each row
            if row[2] <> '':                #if variable is there
                data_row = []
                for x in row:               #for each item in the row
                    if '[' in x:            #check for '[min,max]' entry
                        y = x.strip(' []').split(',')                   #remove brackets from string and split at comma
                        if '.' in y: y[y.index('.')] = 0.0
                        data_row.append([ float(y[0]), float(y[1]) ])   #add data min/max as floats to list
                    else:
                        data_row.append(x)
                data.append(data_row)
                
    if len(data)>1: data.pop(0) #remove header line    
    print len(data), 'lines of input data read... inputs translated.'

    return data

###########################
def buildInputs(ASPECT_list, inputData, outputDataFile, expertFlag, inputCols={}, outputCols={}):
    """
    read in the truth data, then corrolate the morph selections with the inputs
    inputData - data correlating morph matrix functions and options to significant system inputs and qualitative judgements
    outputDataFile -  collected expert data as csv file with items in form username-output, (csv of morph options), 'minOut', 'maxOut'
                   or collected physical data in the form ?
    exertFlag - if true outputDataFile is of the first type
    """
    if expertFlag:
        #read in truth data
        print 'Reading', outputDataFile, 'expert data file and translating inputs.'        
        outputData = []
        
        with open(outputDataFile, 'rU') as csvfile:   #with the file open
            input_reader = csv.reader(csvfile, delimiter=',')   #open csv reader
            
            for row in input_reader:                #read each row
                if not "INCOMPATIBLE" in row[1]:    #skip incompatible options
                    outs = [float(row[-1]), float(row[-2])] #get outputs
                    ins = row[1:-2]                      #get input indecies
                    outputData.append([ins, outs])
                    
        print len(outputData), 'lines of input data read... inputs translated.'

        
        combinedData = []
        for i in range(len(outputData)):
            quant_inputs = {}
            qual_inputs = {}
            for j in range(len(outputData[i][0])): #for each function
                for k in range(len(inputData)): #for each input dataline
                    if inputData[k][1] == ASPECT_list[j]:   #if the function matches
                        quant_inputs[(inputData[k][1], inputData[k][3])] = inputData[k][5  + int(outputData[i][0][j])] 
                        qual_inputs[ (inputData[k][1], inputData[k][3])]  = inputData[k][11 + int(outputData[i][0][j])]
            
            combinedData.append([quant_inputs, qual_inputs, sorted(outputData[i][1])])

        
    else: #if generated data use input columns and output columns from outputDataFile
            #inputCols are in form {'var': col, 'var':col, ...}
            #only first outputCol key used
        
        print 'Reading', outputDataFile, 'data file and translating inputs.'        
        combinedData = []
        outKey = outputCols.keys()[0]
        
        with open(outputDataFile, 'rU') as csvfile:   #with the file open
            input_reader = csv.reader(csvfile, delimiter=',')   #open csv reader
            
            for row in input_reader:                #read each row
                inputs = {}
                outputs = {}
                try:
                    for inKey in inputCols:   inputs[('DATA', inKey)]   = [ round( float(row[inputCols[inKey]]), 5)]
                                        
                    if not "[" in row[outputCols[outKey]]: #for single data point in outputs
                        outputs = [round( float( row[outputCols[outKey]] ), 5)]
                    else:   #for ranged outputs 
                        out = row[outputCols[outKey]].strip("[]").split(",")
                        outputs = [float(o) for o in out]
                        
                except: 
                    None

                if len(inputs)>0: combinedData.append([inputs, None, outputs])

    return combinedData
    
###########################
def combine_inputs(inputData, operations, q=0):
    """
    takes in data and combines inputs for each data point (removes combined variables)
    inputData - data input in form [quant_inputs, qual_inputs, outputs], where inputs are in form: {('FWD_SYS_DRV', 'V_max'): [1.0, 9.0], ...}
    operations - data operations to perform in form {('OUTPUT_VAR_NAME', 'var'): ( [('INPUT_VAR1', 'var'), ('INPUT_VAR2', 'var'), ...], 'UNION') 
    - supports 'UNION', 'AVERAGE', and 'INTERSECTION' of data ranges
    q = 0 for quant data, 1 for qual
    """
    
    print "Combining some inputs in the data..."
    
    for i in range(len(inputData)):                     #for each datapoint
        for op in operations:                           #for each operation key
            inputs = [inputData[i][q][inVar] for inVar in operations[op][0]]   #pull all the input ranges
            
            #perform desired operation
            if operations[op][1] == 'UNION': 
                inVal = [min([x[0] for x in inputs]), max([x[1] for x in inputs])]
            elif operations[op][1] == 'INTERSECTION': 
                inVal = [max([x[0] for x in inputs]), min([x[1] for x in inputs])]
            elif operations[op][1] == 'AVERAGE': 
                inVal = [sum([x[0] for x in inputs])/len([x[0] for x in inputs]), 
                         sum([x[1] for x in inputs])/len([x[1] for x in inputs])]
            
            for inVar in operations[op][0]: inputData[i][q].pop(inVar) #remove old inputs
            inputData[i][q][op] = inVal                                #add new input
    
    return inputData
    

########################### 
def write_expert_data(data, filename):
    """
    Write out expert data combined with morph inputs into single file
    ------INPUTS------
    data : list
        combined data with each point in form:[quantInputs, qualInputs, outputs]
        quantInputs : dict where {('In_Sys', 'Input'), [min, max], ... }
        qualInputs  : ""
        outputs     :  [min, max]
    filename : string
        name for output file
    """
    
    q = 0 #index of data to use as inputs 0=quant, 1=qual
    
    filename = filename.split('.')[0]
    headers = [key for key in data[0][q]] #get input column names
    
    #write average file
    with open(filename+'_avg.csv', 'wb') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([k[0]+"_"+k[1] for k in headers]+['outMin', 'outMax', 'outAvg']) #write combined headers
        for di in data:
            inputs = []
            for key in headers: #grab the inputs in the right order
                inputs.append(di[q][key])   
            
            inputs = [sum(x)/len(x) for x in inputs] #average inputs
       
            writer.writerow( inputs + di[2] + [ float(sum(di[2]))/float(len(di[2])) ] ) # write everything into a row
        
    #write mins file
    with open(filename+'_min.csv', 'wb') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([k[0]+"_"+k[1] for k in headers]+['outMin', 'outMax', 'outAvg']) #write combined headers
        for di in data:
            inputs = []
            for key in headers: #grab the inputs in the right order
                inputs.append(di[q][key])   
            
            inputs = [min(x) for x in inputs] #average inputs
       
            writer.writerow( inputs + di[2] + [ float(sum(di[2]))/float(len(di[2])) ] ) # write everything into a row
            
    #write maxs file
    with open(filename+'_max.csv', 'wb') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([k[0]+"_"+k[1] for k in headers]+['outMin', 'outMax', 'outAvg']) #write combined headers
        for di in data:
            inputs = []
            for key in headers: #grab the inputs in the right order
                inputs.append(di[q][key])   
            
            inputs = [max(x) for x in inputs] #average inputs
       
            writer.writerow( inputs + di[2] + [ float(sum(di[2]))/float(len(di[2])) ] ) # write everything into a row
            
            
            
###########################
def generate_MFs(inp, outp):
    """
    Generates input MFs from input data in the form: {'var': {'ling':[Xvals_MF1], 'ling':[Xvals_MF2], ...},
                                                      'var': {'ling':[Xvals_MF1], 'ling':[Xvals_MF2], ...}, ... } 
                                                
    """
    
    inParams = copy.deepcopy(inp)
    outParams = copy.deepcopy(outp)

    for var in inParams:
        #x_min = min([min(inParams[var][ling]) for ling in inParams[var]])
        #x_max = max([max(inParams[var][ling]) for ling in inParams[var]])

        #x = np.arange(x_min,x_max,float((x_max-x_min)/100.0)) #generage x values for MF
        for ling in inParams[var]: #for each MF
        
            #if len(inParams[var][ling])   == 3: y = fuzz.trimf( x, inParams[var][ling]) #if 3 values -> triangular
            #elif len(inParams[var][ling]) == 4: y = fuzz.trapmf(x, sorted(inParams[var][ling])) #if 4 values -> trapezoidal
            inParams[var][ling] = fuzzyOps.paramsToMF(inParams[var][ling])
        
    for var in outParams:
        #x_min = min([min(outParams[var][ling]) for ling in outParams[var]])
        #x_max = max([max(outParams[var][ling]) for ling in outParams[var]])

        #x = np.arange(x_min,x_max,float((x_max-x_min)/100.0)) #generage x values for MF
        for ling in outParams[var]: #for each MF
            #if len(outParams[var][ling])   == 3: y = fuzz.trimf( x, outParams[var][ling]) #if 3 values -> triangular
            #elif len(outParams[var][ling]) == 4: y = fuzz.trapmf(x, sorted(outParams[var][ling])) #if 4 values -> trapezoidal
            outParams[var][ling] = fuzzyOps.paramsToMF(outParams[var][ling])#[x,y]        
        
    return inParams, outParams

