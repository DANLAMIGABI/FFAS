# -*- coding: utf-8 -*-
"""
Spyder Editor

author: Frank Patterson


DOCSTRING: 
------INPUTS------
------OUTPUTS------
"""
import numpy as np
import skfuzzy as fuzz
import itertools
import copy

import fuzzy_operations as fuzzyOps

#for testing:
import matplotlib.pyplot as plt

#define data types
class input:
    """
    An input class for the fuzzy system type. Contains name, data_type, 
    data_range ([float, float]), MFs ({'name': [[x_points],[y_points]], ... }, 
    MFparams ({'name': [a,b,...], ... }
    ------INPUTS------
    name : string
        name of input
    """
    def __init__(self,name):
        self.name = name
        self.data_type = None
        self.data_range = [None,None]
        self.MFs = {} #MF defined by ('name': [[x_points],[y_points]]) (used by fuzzy system)
        self.MFparams = {} #MF defined by ('name': [a1,a2,a3,...aN]) (used by neuro_fuzzy_system)
        
#data contains info needed for an output   
class output:
    """
    An output class for the fuzzy system type. Contains name, data_type, 
    data_range ([float, float]), MFs ({'name': [[x_points],[y_points]], ... }, 
    MFparams ({'name': [a,b,...], ... }
    ------INPUTS------
    name : string
        name of output
    """
    def __init__(self,name):
        self.name = name
        self.data_type = None
        self.data_range = [None,None]
        self.MFs = {} #MF defined by ('name': [[x_points],[y_points]])
        self.MFparams = {} #MF defined by ('name': [a1,a2,a3,...aN]) (used by neuro_fuzzy_system)

#data contains info needed for rule implementation        
class rule:
    """
    A rule class for rule objects. Contains rule_id (int), rule_list in the form
    (IF (x1 is mu1) AND (x2 is mu2) AND ... OR ... ) THEN (y1 is mu1)
    ------INPUTS------
    rule_id : int
        unique ID number for rule
    """
    def __init__(self, rule_id):
        self.rule_id = rule_id
        self.rule_list = None
        
        
#find string between two substrings
def find_substring(s,first,last):
    """
    Finds a substring in 's' between 'first' and 'last'

    ------INPUTS------
    s : string
        Larger string to search through
    first : string
        Character(s) to start substring at
    last : string
        Character(s) to stop substring at
    ------OUTPUTS------
    s : string
        string betweeen 'start' and 'last'

    """
    try:
        start = s.index( first ) + len( first )
        end = s.index ( last, start )
        return s[start:end]
    except ValueError:
        return ''

def nestList(listIn):
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
        outList = nestList(newList) #recurse on new list
        return outList
    else: 
        return listIn #if no lists nested, return original
        
###############################################################################
###############################################################################
###############################################################################    
    
#read input file and build system
def build_fuzz_system(fcl_file):
    """
    Given a Fuzzy Control Language file (FCL), builds the appropriate objects to 
    instantiate a fuzzy_system class object with.
    
    ------INPUTS------
    fcl_file : string
        text file name of FCL file to build system with
        
    ------OUTPUTS------
    inputs : dict
        dictionary of input objects in form {name: <__main__.input instance>, 
                                             name: <__main__.input instance>, ...}
    outputs : dict
        dictionary of output objects in form {name: <__main__.output instance>, 
                                              name: <__main__.output instance>, ...}    
    rulebase : list
        list of rule objects in form [<__main__.rule instance>, ... ]
    AND_operator : string
        the AND operator for the system ('PRODUCT' or 'MIN')
    OR_operator : string
        the OR operator for the system (only setup for 'MAX' operator)
    aggregator : string
        means to aggregate outputs ('MAX' only one available)
    implication : string
        means to apply rule firing strength to output MFs ('PRODUCT' or 'MIN')
    defuzz :  string
        defuzzification means ('centroid', 'bisector', 'mom', 'lom', 'som')

    """
    #print "Reading FCL File:", fcl_file
    
    f = open(fcl_file, 'r')
    lines = f.readlines()
    
    #input structures:
    inputs = {}            #dict of input structures
    outputs = {}           #dict of output structures
    rulebase = []          #dict of rule structures
    AND_operator = None    #and operator (PRODUCT or MIN)
    OR_operator = None     #or operator (MAX) (per DeMorgan, opposite of AND)
    aggregator = None      #means to aggregate outputs (max)
    implication = None     #means to apply rule firing strength to output MFs
    defuzz = None          #means for defuzzification
    
    #strip out comments
    #print "CLEANING FILE..."
    for l in lines:
        if l.strip().startswith('#'):
            lines[lines.index(l)] = ''
    
    #build input structures
    #print "INITIALIZING INPUTS..."
    flag = 0 # reading mode flag
    for l in lines:
        if l.strip().startswith('VAR_INPUT'): #search for var input block
            flag=1
            continue
        if l.strip().startswith('END_VAR'): #end search for var input block
            flag=0
            continue
        if flag == 1: #if flag set, enter reading mode
            if l.strip() == '': continue #skip empty lines
            t1 = l.rsplit(':') #split off variable name
            i = input(t1[0].strip()) #create input with variable name
            t2 = t1[1].rsplit(';')
            i.data_type = t2[0].strip() #add input variable type
            t3 = find_substring( t2[1], 'RANGE(', ')' ).rsplit("..")
            i.data_range = [ float(t3[0]), float(t3[1]) ]
            inputs[i.name] = i
            
    #build output structures
    #print "INITIALIZING OUTPUTS..."
    flag = 0 # reading mode flag
    for l in lines:
        if l.strip().startswith('VAR_OUTPUT'): #search for var input block
            flag=1
            continue
        if l.strip().startswith('END_VAR'): #end search for var input block
            flag=0
            continue
        if flag == 1: #if flag set, enter reading mode
            if l.strip() == '': continue #skip empty lines
            t1 = l.rsplit(':') #split off variable name
            i = output(t1[0].strip()) #create input with variable name
            t2 = t1[1].rsplit(';')
            i.data_type = t2[0].strip() #add input variable type
            t3 = find_substring( t2[1], 'RANGE(', ')' ).rsplit("..")
            i.data_range = [ float(t3[0]), float(t3[1]) ]
            outputs[i.name] = i

    #build input fuzzy MFs
    #print "BUILDING INPUT MFs..."
    flag = 0 # reading mode flag
    for l in lines:
        if l.strip().startswith('FUZZIFY'): #search for var input block
            flag=1
            name = l.strip().rsplit(' ')[1].strip() #get input name
            continue

        if l.strip().startswith('END_FUZZIFY'): #end search for var input block
            flag=0
            continue
            
        if flag == 1: #if flag set, enter reading mode
            if l.strip() == '': continue #skip empty lines
            t1 = l.rsplit('=')
            n = find_substring( t1[0], 'TERM', ':').strip(' \t\n\r') #pull term linguistic name
            points = [a.rsplit('(')[1] for a in t1[1].rsplit(';')[0].split(')') if ',' in a] #pull the piecewise points
            for j in range(len(points)):
                pts = []
                for p1 in points[j].rsplit(','):
                    if not ('mean' in p1 or 'std' in p1): pts.append(float(p1)) #convert point strings to float list
                    else:                                 pts.append(p1)
                points[j] = pts
           
            if len(points) <> 2: #if piecewise mf
                f_x = np.arange(min([p[0] for p in points]), max([p[0] for p in points]), 
                                (max([p[0] for p in points]) - min([p[0] for p in points]))/100.0)
            else: #for gaussian MF
                f_x = np.arange(inputs[name].data_range[0], inputs[name].data_range[1],
                                (inputs[name].data_range[1]-inputs[name].data_range[0])/100) #x_points for MF

            #determine MF function type
            if len(points) == 2:
                f_y = fuzz.gaussmf(f_x, points[0][0], points[1][0])
            elif len(points) == 3: 
                f_y = fuzz.trimf(f_x, sorted([points[0][0],points[1][0],points[2][0]]))
            elif len(points) == 4: 
                f_y = fuzz.trapmf(f_x, sorted([points[0][0],points[1][0],points[2][0],points[3][0]]))

            inputs[name].MFs[n] = [f_x, f_y] #add MF with linguistic term to input
            inputs[name].MFparams[n] = [p[0] for p in points] #add parameter values (3-tri; 4-trap)
            
    #build output fuzzy MFs
    #print "BUILDING OUTPUT MFs..."
    flag = 0 # reading mode flag
    for l in lines:
        if l.strip().startswith('DEFUZZIFY'): #search for var input block
            flag=1
            name = l.strip().rsplit(' ')[1].strip(' \t\n\r') #get input name
            continue
            
        if l.strip().startswith('END_DEFUZZIFY'): #end search for var input block
            flag=0
            continue
            
        if flag == 1: #if flag set, enter reading mode
            if l.strip() == '': continue #skip empty lines
            if not 'TERM' in l: continue
            t1 = l.rsplit('=')
            n = find_substring( t1[0], 'TERM', ':').strip() #pull term linguistic name
            points = [a.rsplit('(')[1] for a in t1[1].rsplit(';')[0].split(')') if ',' in a] #pull the piecewise points
            
            for j in range(len(points)):
                pts = []
                for p1 in points[j].rsplit(','):
                    if not ('mean' in p1 or 'std' in p1): pts.append(float(p1)) #convert point strings to float list
                    else:                                 pts.append(p1)
                points[j] = pts
            
            if len(points) <> 2: #if piecewise mf
                f_x = np.arange(min([p[0] for p in points]), max([p[0] for p in points]), 
                                ( max([p[0] for p in points]) - min([p[0] for p in points]) )/100.0)
            else: #for gaussian MF
                f_x = np.arange(outputs[name].data_range[0], outputs[name].data_range[1],
                                (outputs[name].data_range[1]-outputs[name].data_range[0])/100.0) #x_points for MF
    
            #determine MF function type
            if len(points) == 2:
                f_y = fuzz.gaussmf(f_x, points[0][0], points[1][0])
            elif len(points) == 3: 
                f_y = fuzz.trimf(f_x, sorted([points[0][0],points[1][0],points[2][0]]))
            elif len(points) == 4: 
                f_y = fuzz.trapmf(f_x, sorted([points[0][0],points[1][0],points[2][0],points[3][0]]))

            outputs[name].MFs[n] = [f_x, f_y] #add MF with linguistic term to input
            outputs[name].MFparams[n] = [p[0] for p in points] #add parameter values (3-tri; 4-trap)
            
    #build fuzzy rules
    #print "BUILDING FUZZY RULES..."
    flag = 0 # reading mode flag
    for i in range(len(lines)):
        if lines[i].strip().startswith('RULEBLOCK'): #search for var input block
            flag=1
            continue
        if lines[i].strip().startswith('END_RULEBLOCK'): #end search for var input block
            flag=0
            continue
        if flag == 1: #if flag set, enter reading mode
            if lines[i].strip() == '': continue #skip empty lines
            if lines[i].strip().startswith('RULE'):
                t1 = lines[i].strip(' \t\n\r').rsplit(':') #split off rule id from statement
                r = rule(str(t1[0].strip('RULE '))) #initialize rule with ID
                strs = t1[1]                         #init rule string
                
                #build rule string
                while not lines[i+1].strip().startswith('RULE') and not lines[i+1].strip().startswith('END_RULEBLOCK'): 
                    strs = strs + lines[i+1].strip()
                    i = i+1
                    
                #build rule_list from rules string
                s = []
                while '(' in strs or ')' in strs:
                    if strs.find('(') < strs.find(')') and strs.find('(') > -1:
                        s.append(strs.split('(',1)[0].strip('; \t\n\r'))
                        s.append('(')
                        strs = strs.split('(',1)[1]
                    else: 
                        s.append(strs.split(')',1)[0].strip('; \t\n\r'))
                        s.append(')')
                        strs = strs.split(')',1)[1]
                s.append(strs)
                s1 = []
                for j in range(len(s)):
                    s1.extend(s[j].split(' '))          #split by spaces
                while '' in s1: s1.pop(s1.index('')) #remove extra white spaces
                while '(' in s1 or ')' in s1:
                    j1,j2 = None, None
                    for j in range(len(s1)):
                        if s1[j] == '(': j1 = j
                        if s1[j] == ')' and j1 <> None: 
                            j2 = j
                            s1[j1:j2+1] = [s1[j1+1:j2]]
                            break
                            
                r.rule_list = s1                    #add rule list to rule
                rulebase.append(r)                  #append rule to rule base
                continue
            
            #get other rulebase parameters (for implication method)
            if lines[i].strip().startswith('AND'):     #pull AND operator
                t = lines[i].strip().strip(';').split(':')
                AND_operator = t[1].strip()
                #if AND_operator == 'MAX': OR_operator = 'MIN' #demorgan's law
                #if AND_operator == 'MIN': OR_operator = 'MAX' #demorgan's law
                    
            elif lines[i].strip().startswith('OR'):      #pull AND operator
                t = lines[i].strip().strip(';').split(':')
                OR_operator = t[1].strip()
                #if OR_operator == 'MAX': AND_operator = 'MIN' #demorgan's law
                #if OR_operator == 'MIN': AND_operator = 'MAX' #demorgan's law
                    
            elif 'ACCU' in lines[i].strip():
                t = find_substring(lines[i], '(*', '*)')
                t = t.strip().strip(';').split(':')
                aggregator = t[1]
            
            elif 'ACT' in lines[i].strip():
                t = lines[i].strip().strip(';').split(':')
                implication = t[1]
                
            elif 'DEFUZZ' in lines[i].strip():
                t = lines[i].strip().strip(';').split(':')
                defuzz = t[1]
                
                
    #print 'FCL Read Successfully!'        
    return (inputs, outputs, rulebase, AND_operator, OR_operator, aggregator, implication, defuzz)




###############################################################################
############################ FUZZY SYSTEM CLASS ###############################
###############################################################################

# implement the fuzzy system (get output(s))
class Fuzzy_System:
    """
    This class is a general implementation of a FUZZY RULE BASED SYSTEM (FRBS)
    the primary functionality is for a Mamdani system, but it can be updated with
    any functionality within the given framework. It has MIMO capability, but is
    intended mostly as a MISO system.
    
    ------INPUTS------
    inputs : dict
        dictionary of input objects in form {name: <__main__.input instance>, 
                                             name: <__main__.input instance>, ...}
    outputs : dict
        dictionary of output objects in form {name: <__main__.output instance>, 
                                              name: <__main__.output instance>, ...}    
    rulebase : list
        list of rule objects in form [<__main__.rule instance>, ... ]
    AND_operator : string
        the AND operator for the system ('PRODUCT' or 'MIN')
    OR_operator : string
        the OR operator for the system (only setup for 'MAX' operator)
    aggregator : string
        means to aggregate outputs ('MAX' only one available)
    implication : string
        means to apply rule firing strength to output MFs ('PRODUCT' or 'MIN')
    defuzz :  string
        defuzzification means ('centroid', 'bisector', 'mom', 'lom', 'som')
        
    """
    
    #initialize a fuzzy system
        
        #
    def __init__(self, inputs, outputs, rulebase, AND_operator, OR_operator, aggregator, implication, defuzz):
        self.inputs = inputs            #list of input structures
        self.outputs = outputs           #list of output structures
        self.rulebase = rulebase             #list of rule structures
        self.AND_operator = AND_operator    #and operator (max or product)
        self.OR_operator = OR_operator     #or operator (max) (per DeMorgan, opposite of AND)
        self.aggregator = aggregator      #means to aggregate outputs (max)
        self.implication = implication     #means to apply rule firing strength to output MFs
        self.defuzz = defuzz                #defuzzification method (None will result in fuzzy output)
                                        #supports: 'centroid': Centroid of area, 'bisector': bisector of area, 'mom' : mean of maximum,
                                        #          'som' : min of maximum, 'lom' : max of maximum
        #FOR TESTING
        self.TESTMODE = False
        #self.firedInputs = {rule.rule_id:{} for rule in self.rulebase}          #dict of {rule_id:{inputname: [x,y], ...} }for fired inputs 
        self.implicatedOutputs = {rule.rule_id:{} for rule in self.rulebase}    #dict of {rule_id:{outputname: [x,y], ...} }for fired inputs 
        
    def fuzzy_single_AND(self, singleton, n_fuzzy):
        """
        Get the minimum (AND) of singleton and fuzzy number (as numpy array)
        
        ------INPUTS------
        singleton : float 
            single value, x
        ------OUTPUTS------
        n_fuzzy : list
            fuzzy number [nparray[x], nparray[y]] (from skfuzzy package)
        """
        singleton = float(singleton)
        for i in range(1, len(n_fuzzy[0])):
            #print i, n_fuzzy[0][i], singleton, n_fuzzy[0][i+1]
            #print singleton, len(n_fuzzy[0]), i, ":", round(n_fuzzy[0][i-1],6), round(singleton,6), round(n_fuzzy[0][i],6)
            if round(n_fuzzy[0][i-1],6) <= round(singleton,6) and round(n_fuzzy[0][i],6) > round(singleton,6):  #find points around singleton              
                
                #interpolate linearly for more accurate answer
                return n_fuzzy[1][i-1] + (n_fuzzy[1][i] - n_fuzzy[1][i-1]) * \
                       ((singleton - n_fuzzy[0][i-1]) / (n_fuzzy[0][i] - n_fuzzy[0][i-1]))
        return 0.0
        
    def firing_strength(self, input_name, input_, input_sys):
        """
        Get firing stregth of an input
        
        ------INPUTS------
        input_name : string
            linguistic input name
        input_ : list or float
            corresponding to input [x,y] or singleton
        input_sys - input object
            object corresponding to system input MFs
        ------OUTPUTS------ 
        firing_strength : float
            firing strength of input and MF

        """
        if not isinstance(input_, list): #if a singleton and not a list
            fs = self.fuzzy_single_AND(input_, [input_sys.MFs[input_name][0],input_sys.MFs[input_name][1]])        
            return fs 
        x_min,y_min = fuzz.fuzzy_and(np.array(input_sys.MFs[input_name][0]),
                                     np.array(input_sys.MFs[input_name][1]),
                                     np.array(input_[0]),np.array(input_[1])) #use AND operator to get minimum of two functions
        return max(y_min)
     
     
    def rule_recurse(self, list1, input_list, TESTMODE):
        """
        Recurses through rule antecedent to get firing strength
        
        ------INPUTS------
        list1 : list
            antecedent of rule in list form
        input_list : 
            corresponding to input [x,y] or singleton

        ------OUTPUTS------ 
        list1 : list
            list with statements recursed down to firing strengths

        """
        while any(isinstance(l,list) for l in list1):

            n = next(l for l in list1 if isinstance(l,list))        #get the next instance of a list in the given list
            for j in range(len(list1)):                             #find that instance's index
                if isinstance(list1[j], list) and list1[j] == n: 
                    i = j
                    break
            list1[i] = self.rule_recurse(list1[i], input_list, TESTMODE)[0] #recurse the function on the found list        
        
        #### APPLY FUZZY RULES #### (order of operations: order of while loops)        
        ###
        while 'IS' in list1:        #get all firing stregths first
            i = list1.index('IS')
            fs = self.firing_strength(list1[i+1], input_list[list1[i-1]] , self.inputs[list1[i-1]])
            if TESTMODE == 1: 
                print "FIRING STRENGTH for", self.inputs[list1[i-1]].name, 'is', list1[i+1], 'at Input:', '=', fs
            list1[i-1:i+2] = [fs] 
        ###
        while 'ISNOT' in list1:        #get compliment firing strengths next
            i = list1.index('ISNOT')
            fs = 1 - self.firing_strength(list1[i+1], input_list[list1[i-1]] , self.inputs[list1[i-1]])
            if TESTMODE == 1: 
                print "FIRING STRENGTH for", self.inputs[list1[i-1]].name, 'is', list1[i+1], 'at Input:', '=', fs
            list1[i-1:i+2] = [fs]
        ###
        while 'OR' in list1:        #calculate ORs next
            i = list1.index('OR')
            if self.OR_operator == 'MAX':
                x = max(list1[i-1],list1[i+1])
            else: 
                raise StandardError('You havent coded this yet!')
            # other OR operators??
            if TESTMODE == 1: print "REPLACE: ", list1[i-1:i+2], 'with', x
            list1[i-1:i+2] = [x]
        ###    
        while 'AND' in list1:        #calculate ANDs next
            i = list1.index('AND')
            if self.AND_operator == 'MIN':               #use minimum operator
                x = min(list1[i-1],list1[i+1])
            elif self.AND_operator == 'PRODUCT':         #use product operator
                x = list1[i-1] * list1[i+1]    
            # other AND operators??
            if TESTMODE == 1: print "REPLACE: ", list1[i-1:i+2], 'with', x
            list1[i-1:i+2] = [x]   
        ###
        while 'ANDOR' in list1:     #calculate and/ors (means)
            i = list1.index('ANDOR')
            x = sum(list1[i-1],list1[i+1])/2.0      #take mean of two operators
            if TESTMODE == 1: print 'REPLACE: ', list1[i-1:i+2], 'with', x
            list1[i-1:i+2] = [x]
            
        return list1


    def implicate(self, fs, outputMF):
        """
        Perform implication on a given output MF for a given antecedent firing strength
        
        ------INPUTS------
        fs : float
            firing strength of rule antecedent 
        outputMF - list
            output membership function to apply implication to (form [[x_vals],[y_vals]])
        ------OUTPUTS------ 
        y_ : list
            resulting membership function
        """
        y_ = copy.deepcopy(outputMF[1])
        if self.implication == 'MIN':
            for i in range(len(y_)): 
                if y_[i] > fs: y_[i] = fs
        if self.implication == 'PRODUCT':
            for i in range(len(y_)): y_[i] = y_[i]*fs
        return y_

    def aggregate(self, MFs):
        """
        Peform aggregation on the given outputs of the rules and returns aggregated MF
        
        ------INPUTS------
        MFs : list 
            list of MF functions
        ------OUTPUTS------ 
        o1 : list
             aggregation of all MFs
        """
        o1 = MFs.pop()
        if self.aggregator == 'MAX':
            while len(MFs) > 0:
                o2 = MFs.pop()
                o1[0],o1[1] = fuzz.fuzzy_or( o1[0],o1[1],o2[0],o2[1] )
        else: 
            raise StandardError('Havent coded this yet!')
                
        return o1
        
    def run_system(self, input_list, output_key, TESTMODE=False):
        """
        Runs the fuzzy system for a single output
        ------INPUTS------
        input_list : dict
            dict of inputs {'input1':value, 'input2':value, ...} 
        output_key : string
            string key of output to calculate
        TESTMODE : int
            testmode flag (1 = On)
        ------OUTPUTS------ 
        """
        self.TESTMODE = TESTMODE
        outs = []
        for rule in self.rulebase:   #iterate over rulebase
            if TESTMODE: 
                print '------------------------------------------------------------------------'
                print 'TRANSLATING RULE: ', rule.rule_id
                  
            #break apart antecedent and consequent
            if_i = rule.rule_list.index('IF')   
            then_i = rule.rule_list.index('THEN')
            rule_ant = copy.deepcopy(rule.rule_list[if_i+1:then_i])                         #get the rule antecedent
            rule_con = copy.deepcopy(rule.rule_list[then_i+1:len(rule.rule_list)+1])[0]     #get the rule consequent
            
            if rule_con[0] == output_key:               #only follow rule if it applies to given output 

                fs = self.rule_recurse(rule_ant, input_list, TESTMODE)[0]   #get firing strength

                if TESTMODE: print 'FIRING STREGTH, RULE', rule.rule_id, ':', fs
            
                output = copy.deepcopy(self.outputs[rule_con[0]].MFs[rule_con[2]]) #get output
                output[1] = self.implicate(fs, output)#use implication to get fuzzy consequent
                
                if TESTMODE: self.implicatedOutputs[rule.rule_id][self.outputs[rule_con[0]].name] = copy.deepcopy(output)
                    
                outs.append(output)
        
        #aggregate outputs
        if len(outs) > 0: 
            output_result = self.aggregate(outs)    #aggregate outputs if there are outputs
            
            if self.defuzz <> None: #defuzzify outputs
                output_result = fuzz.defuzz(output_result[0], output_result[1], self.defuzz)
        else:
            m1 = self.outputs[output_key].data_range[0]         #get output min
            m2 = self.outputs[output_key].data_range[1]         #get output max
            x1 = np.arange(m1,m2,0.01)                          #get x range
            output_result = [x1, np.asarray([0 for i in range(len(x1))])]   #return mf function of zeros
            
            if self.defuzz <> None: #defuzzify outputs
                output_result = [0,0]
            
        return output_result
        

    def run(self, input_list, TESTMODE=0): 
        """
        Runs the fuzzy system for all inputs
         -----INPUTS-----
        input_list : dict
            dict of inputs {'input1':value, 'input2':value, ...}  #lack of name in dict assumes 0 input (0 for all possible values)
        TESTMODE : bool (optional)
            flag to output text for test mode (default is False(OFF))
        ------OUTPUTS-----
        outputs_all : dict
            dict of outputs {'output1':value, 'output2':value, ...}
        """
        
                               
        #TESTMODE = 0    #flag to indicate test mode (plots & checks)
        outputs_all = {}
        if input_list <> None:
            for key in self.outputs:
                
                output_val = self.run_system(input_list, key, TESTMODE)
                outputs_all[key] = output_val
        else:
            key = self.outputs.keys()[0]
            self.outputs = {}
            
        ###############
        if TESTMODE == 1:  #ONLY PLOT IF IN TESTMODE
            
            #plot all MFs
            plt.figure()
            i = 1
            for k in self.inputs:
                #plot each input against MFs
                plt.subplot(len(self.inputs)+len(self.outputs), 1, i)
                for k2 in self.inputs[k].MFs:
                    plt.plot(self.inputs[k].MFs[k2][0], self.inputs[k].MFs[k2][1])
                i = i + 1
                #plot input
                if False:
                    if input_list <> None:
                        if isinstance(input_list[k], list): 
                            plt.plot(input_list[k][0],input_list[k][1], lw=3.0, color='k')
                        elif input_list[k] <> None:
                            plt.plot([input_list[k],input_list[k]],[0,1.0], lw=3.0, color='k')
                        plt.ylabel(k)
                        plt.ylim([0,1.1])
            
            #plot output against MFs     
            plt.subplot(len(self.inputs)+len(self.outputs), 1, i)
            for k2 in self.outputs[key].MFs:
                plt.plot(self.outputs[key].MFs[k2][0], self.outputs[key].MFs[k2][1])
            
            if False:
                if type(outputs_all[key]) is list:
                    plt.plot(outputs_all[key][0],outputs_all[key][1], lw=3.5, color='b')
                else:
                    plt.plot([outputs_all[key],outputs_all[key]],[0.,1.], lw=3.5, color='b')
            plt.ylabel(key)
            plt.ylim([0,1.1])
            
            
            #plot rules
            """
            inps = [k for k in self.inputs]
            otps = [k for k in self.outputs]
            fig, ax = plt.subplots(nrows=len(self.rulebase),ncols=(len(inps)+len(otps)))
            
            for i in range(len(self.rulebase)):
                rID = self.rulebase[i].rule_id
                rlst = nestList(self.rulebase[i].rule_list)
                for inp in inps:
                    for k in range(len(rlst)):
                        if rlst[k] == inp: #if input is in the rule
                            ax[i,inps.index(rlst[k])].plot(self.inputs[rlst[k]].MFs[rlst[k+2]][0],
                                                        self.inputs[rlst[k]].MFs[rlst[k+2]][1], '-k', lw=0.5)
                            if isinstance(input_list[rlst[k]], list): #plot input MF or crisp val
                                ax[i,inps.index(rlst[k])].plot(input_list[rlst[k]][0], 
                                                            input_list[rlst[k]][1], '-b', lw=2.0)
                            else:
                                ax[i,inps.index(rlst[k])].plot([input_list[rlst[k]], input_list[rlst[k]]], 
                                                                [0,1.1], '-b', lw=2.0)
                                                                
                            ax[i,inps.index(rlst[k])].set_xlim([self.inputs[rlst[k]].data_range[0],
                                                                self.inputs[rlst[k]].data_range[1]])  
                            ax[i,inps.index(rlst[k])].set_ylim([0.,1.1])
                            ax[i,inps.index(rlst[k])].set_yticks([0.0, 0.5, 1.0])
                            ax[i,inps.index(rlst[k])].set_yticklabels(['',''])
                            if i < (len(self.rulebase)-1):
                                ax[i,inps.index(rlst[k])].set_xticks([])
                                ax[i,inps.index(rlst[k])].set_yticklabels([])
                            else: ax[i,inps.index(rlst[k])].set_xlabel(inp, fontsize=8)
                for otp in otps:
                    for k in range(len(rlst)):
                        if rlst[k] == otp: #if input is in the rule  
                            ax[i,len(inps)+otps.index(rlst[k])].plot(self.outputs[rlst[k]].MFs[rlst[k+2]][0],
                                                                        self.outputs[rlst[k]].MFs[rlst[k+2]][1], '-k', lw=0.5)
                            ax[i,len(inps)+otps.index(rlst[k])].plot(self.implicatedOutputs[rID][rlst[k]][0],
                                                                        self.implicatedOutputs[rID][rlst[k]][1], '-b', lw=2.0)
                            ax[i,len(inps)+otps.index(rlst[k])].set_ylim([0.,1.1])
                            ax[i,len(inps)+otps.index(rlst[k])].set_xlim([self.outputs[rlst[k]].data_range[0],
                                                                            self.outputs[rlst[k]].data_range[1]])                            
                            ax[i,len(inps)+otps.index(rlst[k])].set_yticks([0.0, 0.5, 1.0])
                            ax[i,len(inps)+otps.index(rlst[k])].set_yticklabels(['',''])
                            if i < (len(self.rulebase)-1):
                                ax[i,len(inps)+otps.index(rlst[k])].set_xticks([])
                                ax[i,len(inps)+otps.index(rlst[k])].set_yticklabels([])
                            else: ax[i,len(inps)+otps.index(rlst[k])].set_xlabel(otp, fontsize=8)
            """
            #plt.show()
                
            ###############
                
        return outputs_all

if __name__=="__main__": 
    pass

    
    