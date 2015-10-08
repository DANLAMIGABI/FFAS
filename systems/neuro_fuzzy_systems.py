# -*- coding: utf-8 -*-
"""
Created on Mon Mar 30 08:33:12 2015

@author: frankpatterson
"""
import numpy as np
import skfuzzy as fuzz

import fuzzy_operations as fuzzOps
from fuzzy_systems import *

from itertools import groupby
import copy

import matplotlib.pyplot as plt 

from timer import Timer


        
        
###########################
class NEFPROX:
    """
    ------INPUTS------
    """
    
    def __init__(self, inputs, outputs, rulebase, defuzz):
        self.inputMFs    = {} #dict of input MFs  {('input_name', 'ling'): [x,y,params], ...}
        self.outputMFs   = {} #dict of output MFs {('output_name', 'ling'): [x,y,params], ...}
        
        self.layer1      = {} #input node values {'input_name': value, ... }
        self.connect1to2 = {} #layer 1 to 2 connections {'rule#': {('input_name', 'ling'): value, ('input_name', 'ling'): value, ... }, ...}
        self.layer2      = {} #rule nodes {'rule#': value, ... }
        self.connect2to3 = {} #layer 2 to 3 connections {'rule#': {('output_name', 'ling'): value, ... }, ...}
        self.layer3      = {} #output nodes {'output_name': value, ... } (should only be one node for MISO system)
        
        self.defuzz = defuzz  #type of defuzzification 
        
        
        for inp in inputs: #for each input create input neuron and add MF funcs
            self.layer1[inp] = None
            for ling in inputs[inp].MFs: #get input MFs 
                self.inputMFs[(inp, ling)] = inputs[inp].MFs[ling] + [inputs[inp].MFparams[ling]]

        for otp in outputs: #for each output create neuron and add MF funcs
            self.layer3[otp] = None
            for ling in outputs[otp].MFs:
                self.outputMFs[(otp, ling)] = outputs[otp].MFs[ling] + [outputs[otp].MFparams[ling]]
                        
        for rule in rulebase: #for each rule
            ruleList = self.nestList(rule.rule_list) #get rule list as one list
            [ant, cont] = [ruleList[1: ruleList.index('THEN')], 
                           ruleList[ruleList.index('THEN')+1 :]] 
                          #split antecendent and consequent
            ant = [list(group) for k, group in groupby(ant, lambda x: x == "AND" or "OR")]
                #split up all the parts of the antecedent
            while ['AND'] in ant: ant.pop(ant.index(['AND']))
                #remove the "AND" elements from the rule list
            while ['OR'] in ant: del ant[ant.index(['OR']):ant.index(['OR'])+2] 
                #remove "OR"s and extra antecendents from one input
                
            ruleConns = {}
            for item in ant: #get inputs to the rule
                ruleConns[(str(item[0]), str(item[2]))] = 0.0
                #dict of input connections {(inputName, MFname): FS, (inputName, MFname): FS, ...}
            
            self.connect1to2[str(rule.rule_id)] = ruleConns
            self.connect2to3[str(rule.rule_id)] = {(str(cont[0]), str(cont[2])): None } #save output connection
            
            self.layer2[str(rule.rule_id)] = None #append rule Node
        
        if len(outputs) > 0:    #if given output
            output_name = next(outputs[op].name for op in outputs) #get output name (should only be one output per system)
            self.layer3[output_name] = None #append output Node
        
    
    def nestList(self, listIn):
        """
        Take nested list and build larger list without nesting
        
        ------INPTUS------
        listIn : list
            list to de-next
        """
    
        if any(isinstance(l,list) for l in listIn):                         
        #if there is an nested list in the input list
            i = listIn.index(next(l for l in listIn if isinstance(l,list)))
            #get the next instance of a list in the given list
            newList = listIn[:i] + listIn[i] + listIn[i+1:] 
            #insert list instance in larger list
            outList = self.nestList(newList) #recurse on new list
            return outList
        else: 
            return listIn #if no lists nested, return original
            
            
    def feedforward(self, inputs):
        """
        ------INPUTS------
        
        inputs : dict
            the set of outputs from previous nodes in form 
            {nodeName: value, nodeName: value, ...} (or inputs for input 
            nodes)   
        """
        #INPUT FEED FORWARD (fuzzify inputs)
        for inp in self.layer1:
            try:
                if inp in inputs: #check if input is given
                    if not isinstance(inputs[inp], list):
                        MF = fuzzOps.paramsToMF([inputs[inp]]) #get fuzzy MF for singleton
                        self.layer1[inp] = MF
                    else: 
                        self.layer1[inp] = inputs[inp]
                else: 
                    print "Not all inputs given!!!" 
                    self.layer3[self.layer3.keys()[0]] = None #set system output to None
            except:
                raise StandardError("NEFPROX input error!")
        
        #RULE FEED FORWARD (gets min (t-norm) firing strength of weights/MFterms and inputs)
        for rule in self.layer2: #for each rule
            for inp in self.connect1to2[rule]: #for each input in antecedent 
                fs = max(fuzz.fuzzy_and(self.inputMFs[inp][0], self.inputMFs[inp][1],
                                        self.layer1[inp[0]][0], self.layer1[inp[0]][1])[1])
                self.connect1to2[rule][inp] = fs
            self.layer2[rule] = min([self.connect1to2[rule][inp] for inp in self.connect1to2[rule]])
    
        #OUTPUT FEED FORWARD (apply minimum of firing strength and output MF (reduce output MF), then aggregate)
        outMFs = []
        for rule in self.connect2to3: #for each rule
            cons = self.connect2to3[rule].keys()[0]  #get consequent for single output
            if self.layer2[rule] > 0.0: #only for active rules (save time)
                outMF = copy.deepcopy(self.outputMFs[cons][:2])
                outMF[1] = np.asarray([ min(self.layer2[rule], outMF[1][i]) for i in range(len(outMF[1])) ])
                            #apply minimum of firing strength and output MF (reduce output MF)
                self.connect2to3[rule][cons] = outMF
                outMFs.append(outMF)
            else: #for inactive rules, applied MF is 0.0 for all 
                self.connect2to3[rule][cons] = [np.asarray([0.0, 0.0]), np.asarray([0.0,0.0])]
                
        #once all rules are reduced with MFs aggregate
        if len(outMFs) > 0: #check for no rules fired
            while len(outMFs) > 1: #get maximum (union) of all MFs (aggregation)
                outMFs0 = outMFs.pop(0)
                outMFs[0][0], outMFs[0][1] = fuzz.fuzzy_or(outMFs0[0], outMFs0[1], 
                                                            outMFs[0][0], outMFs[0][1])
            
            if   self.defuzz == None: pass
            elif self.defuzz == 'centroid': outMFs[0] = fuzz.defuzz(outMFs[0][0],outMFs[0][1],'centroid')
            elif self.defuzz == 'bisector': outMFs[0] = fuzz.defuzz(outMFs[0][0],outMFs[0][1],'bisector')
            elif self.defuzz == 'mom':      outMFs[0] = fuzz.defuzz(outMFs[0][0],outMFs[0][1],'mom')               #mean of maximum
            elif self.defuzz == 'som':      outMFs[0] = fuzz.defuzz(outMFs[0][0],outMFs[0][1],'som')               #min of maximum
            elif self.defuzz == 'lom':      outMFs[0] = fuzz.defuzz(outMFs[0][0],outMFs[0][1],'lom')               #max of maximum
            
            self.layer3[cons[0]] = outMFs[0]
            
        else:#if no rules fire, then output is None
        
            if self.defuzz == None:
                self.layer3[cons[0]] = [[0.0,0.0],[0.0,0.0]] #result 0.0 in fuzzy MF form
            else:
                self.layer3[cons[0]] = 0.0 #result 0.0 as crisp if some defuzz method specified
            
        return True

    def backpropagate(self, error, LR, data, inRanges, outRanges):
        """
        ------INPUTS------
        """
        
        
        #BACKPROP THROUGH OUTPUT NODE:
        if not isinstance(data, list):
            dataFuzz = fuzzOps.paramsToMF[[data]] #get fuzzy version of data for FS
        else: 
            dataFuzz = copy.deepcopy(data)
            data = fuzz.defuzz(data[0], data[1], 'centroid') #get crisp version of data
        
        for rule in self.connect2to3:    #for each rule to output connectionMF
            if self.layer2[rule] > 0.0:   #if rule is firing > 0.0
                outKey = self.connect2to3[rule].keys()[0] #get connection (outName,ling)
                fs = max(fuzz.fuzzy_and(self.outputMFs[outKey][0], self.outputMFs[outKey][1],
                                        dataFuzz[0], dataFuzz[1])[1])
                    #get "firing strength" of individual MF (result of MF for data: W(R,y_i)(t_i))

                #GET CONSTRAINTS: 
                #Triangular: MFs must overlap (or touch other MFs)
                [minP, maxP] = outRanges[outKey[0]]
                if len(self.outputMFs[outKey][2]) == 2:
                    raise StandardError("haven't programmed this")
                if len(self.outputMFs[outKey][2]) == 3:
                    all_params = [self.outputMFs[ling][2] for ling in self.outputMFs]
                    all_params.sort(key=lambda x: x[1]) #sort params by orderer of b value

                    min_ps = all_params[max(0, all_params.index(self.outputMFs[outKey][2]) - 1)] #get MF just < the one changing
                    if min_ps == self.outputMFs[outKey][2]: 
                        min_op = [minP, minP, minP]   #adjust if MF is minimum one 

                    max_ps = all_params[min(len(all_params) - 1, all_params.index(self.outputMFs[outKey][2]) + 1)]#get MF just > the one changing
                    if max_ps == self.outputMFs[outKey][2]: 
                        max_op = [maxP, maxP, maxP]  #adjust if MF is maximum one
                else: 
                    raise StandardError("haven't programmed this")

                if fs > 0: #for W(R,y_i)(t_i) > 0
                    if len(self.outputMFs[outKey][2]) == 2: #gaussian MF adjustment
                        raise StandardError("haven't programmed this")
                        
                    elif len(self.outputMFs[outKey][2]) == 3: #triangular MF adjustment
                        [a,b,c] = self.outputMFs[outKey][2][:] #get params
                        del_b = LR*error*(c - a)*self.layer2[rule]*(1-fs)
                        del_a = LR*(c - a)*self.layer2[rule] + del_b
                        del_c = -1*LR*(c - a)*self.layer2[rule] + del_b
                        b = min( max(b+del_b, min_ps[1]), max_ps[1] ) #bound b by nearest b's
                        a = min(b, min( max(a+del_a, min_ps[0]), min_ps[2] )) #bound a by nearest a and c and keep a < b
                        c = max(b, min( max(c+del_c, max_ps[0]), max_ps[2] )) #bound c by nearest a and c and keep c > b
                        self.outputMFs[outKey][2] = [a,b,c] #update params

                    elif len(self.outputMFs[outKey][2]) == 4: #trapezoidal MF adjustment
                        raise StandardError("haven't programmed this")

                else: #for W(R,y_i)(t_i) = 0
                    if len(self.outputMFs[outKey][2]) == 2: #gaussian MF adjustment
                        raise StandardError("haven't programmed this")
                        
                    elif len(self.outputMFs[outKey][2]) == 3: #triangular MF adjustment
                        [a,b,c] = self.outputMFs[outKey][2][:] #get params
                        del_b = LR*error*(c - a)*self.layer2[rule]*(1-fs)
                        del_a = np.sign(data-b)*LR*(c - a)*self.layer2[rule] + del_b
                        del_c = np.sign(data-b)*LR*(c - a)*self.layer2[rule] + del_b
                        b = min( max(b+del_b, min_ps[1]), max_ps[1] ) #bound b by nearest b's
                        a = min(b, min( max(a+del_a, min_ps[0]), min_ps[2] )) #bound a by nearest a and c and keep a < b
                        c = max(b, min( max(c+del_c, max_ps[0]), max_ps[2] )) #bound c by nearest a and c and keep c > b
                        self.outputMFs[outKey][2] = [a,b,c] #update params

                    elif len(self.outputMFs[outKey][2]) == 4: #trapezoidal MF adjustment
                        raise StandardError("haven't programmed this")
                        

                newMF = fuzzOps.paramsToMF(self.outputMFs[outKey][2]) #get updated MF
                self.outputMFs[outKey][0] = newMF[0]    #update MF
                self.outputMFs[outKey][1] = newMF[1]    #update MF
        
        #BACKPROP THROUGH RULE NODES:
        for rule in self.layer2:
            #get Rule Error: E_R = o_R(1-o_R) * sum(2*W(R,y)(t_i) - 1) * abs(error)
            # note: only one output node:
            outKey = self.connect2to3[rule].keys()[0] #get connection (outName,ling)
            fs = max(fuzz.fuzzy_and(self.outputMFs[outKey][0], self.outputMFs[outKey][1],
                                dataFuzz[0], dataFuzz[1])[1])
            E_R = self.layer2[rule]*(1-self.layer2[rule]) * (2*fs-1) * abs(error) #rule error
            
            for input in self.connect1to2[rule]:
                o_x = fuzz.defuzz(self.layer1[input[0]][0], self.layer1[input[0]][1], 'centroid') #crisp version of input
                
                if self.connect1to2[rule][input] > 0.0: #if FS from that input > 0

                    if fs > 0: #for W(R,y_i)(t_i) > 0
                        if len(self.outputMFs[outKey][2]) == 2: #gaussian MF adjustment
                            raise StandardError("haven't programmed this")
                            
                        elif len(self.outputMFs[outKey][2]) == 3: #triangular MF adjustment
                        
                            [a,b,c] = self.inputMFs[input][2][:] #get params
                            del_b = LR*E_R*(c - a)*(1-self.connect1to2[rule][input])*np.sign(o_x - b)
                            del_a = -LR*E_R*(c - a)*(1-self.connect1to2[rule][input])+del_b
                            del_c = LR*E_R*(c - a)*(1-self.connect1to2[rule][input])+del_b
                            #print 'LR', LR, 'E_R', E_R, 'c-a', c-a, '1-W(x,R)(ox)', (1-self.connect1to2[rule][input])
                            #print 'rule dels:', [del_a, del_b, del_c]
                            self.inputMFs[input][2] = [min(self.inputMFs[input][2][0]+del_a, self.inputMFs[input][2][1]+del_b), 
                                                       self.inputMFs[input][2][1]+del_b,
                                                       max(self.inputMFs[input][2][2]+del_c, self.inputMFs[input][2][1]+del_b) ] #update params                            
                        
                        elif len(self.outputMFs[outKey][2]) == 4: #trapezoidal MF adjustment
                            raise StandardError("haven't programmed this")
                            
                        newMF = fuzzOps.paramsToMF(self.inputMFs[input][2]) #get updated MF
                        self.inputMFs[input][0] = newMF[0]    #update MF
                        self.inputMFs[input][1] = newMF[1]    #update MF 
                                        
    def plot(self):
        """
        ------INPUTS------
        """
        plt.figure()
        i=1
        for inp in self.layer1: #plot each input against MFs 
            plt.subplot(len(self.layer1)+len(self.layer3), 1, i)
            plt.ylabel(inp)
            i = i + 1
           
            #plot input MFs
            for mfx in self.inputMFs:
                if mfx[0] == inp:
                    plt.plot(self.inputMFs[mfx][0], self.inputMFs[mfx][1], lw=1.0)
            #plot inputs
            if isinstance(self.layer1[inp], list): 
                plt.plot(self.layer1[inp][0],self.layer1[inp][1], lw=3.0, color='k')
            else:
                plt.plot([self.layer1[inp], self.layer1[inp]],[0, 1.0], lw=3.0, color='k')
        
        for otp in self.layer3:
            #plot output MFs
            plt.subplot(len(self.layer1)+len(self.layer3), 1, i)
            plt.ylabel(otp)
            for mfx in self.outputMFs:
                plt.plot(self.outputMFs[mfx][0], self.outputMFs[mfx][1], lw=1.0)
            #plot rule outputs
            for rule in self.connect2to3:
                MFrule = self.connect2to3[rule][self.connect2to3[rule].keys()[0]]
                plt.plot(MFrule[0], MFrule[1], '--', lw=2.0)
            #plot output (shouldonly be one)
            if isinstance(self.layer3[otp], list):
                plt.plot(self.layer3[otp][0], self.layer3[otp][1], lw=3.0, color='k')
            else:
                plt.plot([self.layer3[otp], self.layer3[otp]],[0, 1.0], lw=3.0, color='k')
            plt.xlim([min([min(self.outputMFs[mfx][2]) for mfx in self.outputMFs]), 
                      max([max(self.outputMFs[mfx][2]) for mfx in self.outputMFs])])
        plt.show()
                
        
    def run(self, inputs, TESTMODE=False):
        """
        ------INPUTS------
        """
        self.feedforward(inputs)
        
        if TESTMODE:
            self.plot()
        
        outputs = {} #get output(s) for return (should only be one)
        for key in self.layer3: 
            outputs[key] = self.layer3[key]
        return outputs
        
if __name__=="__main__": 
    pass
    