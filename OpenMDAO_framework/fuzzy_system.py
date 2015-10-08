# -*- coding: utf-8 -*-
"""
Spyder Editor

author: Frank Patterson

- version for openMDAO

"""

import numpy as np
import skfuzzy as fuzz
import itertools
import copy
import sys

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

# implement the fuzzy system (get output(s))
from openmdao.main.api import Component
from openmdao.lib.datatypes.api import Float
from openmdao.lib.datatypes.api import Int
from openmdao.lib.datatypes.api import Str
from openmdao.lib.datatypes.api import Dict

class Fuzzy_System(Component):
    
    #component inputs and outputs
    fcl_file = Str('', iotype='in', desc='File name for FCL file')
    print fcl_file
    
    #inputs
    TESTMODE = Int(0, iotype='in', desc='TestMode Flag (1 == Run in TESTMODE)')
    TESTPLOT = Int(0, iotype='in', desc='TestPlot Flag (1 == Create a TESTPLOT)')
    input_list = Dict({}, iotype='in', desc='Dict of Input Values')
    runFlag_in = Int(0, iotype='in', desc='test')
    passthrough = Int(0, iotype='in', low=0, high=1, desc='passthrough flag for incompatible options')
    
    #outputs 
    outputs_all = Dict({}, iotype='out', desc='Output Value Dict')
    input_mfs = Dict({}, iotype='out', desc='Dict of Input MFs')
    output_mfs = Dict({}, iotype='out', desc='Dict of Output MFs')
    runFlag_out = Int(0, iotype='out', desc='test')
    
    #initialize system
    def __init__(self): # inputs, outputs, rulebase, AND_operator, OR_operator, aggregator, implication):
        """ Creates a new Fuzzy System object """
        super(Fuzzy_System, self).__init__()   
        
        self.old_fcl_file = self.fcl_file #check for changing fuzzy system fcl file
        if self.fcl_file <> '':
            self.inputs, \
            self.outputs, \
            self.rulebase, \
            self.AND_operator, \
            self.OR_operator, \
            self.aggregator, \
            self.implication, \
            self.defuzz = build_fuzz_system(self.fcl_file)
            
            self.input_mfs = self.inputs        #add to MDAO outputs
            self.output_mfs = self.outputs      #add to MDAO outputs
            
            print 'New System Loaded...', len(self.inputs), 'inputs. ', \
                  len(self.outputs), 'outputs. ', len(self.rulebase), 'rules.'
            
            self.implicatedOutputs = {rule.rule_id:{} for rule in self.rulebase}    #dict of {rule_id:{outputname: [x,y], ...} }for fired inputs 



    #get minimum of fuzzy number (min y > 0)
    def fuzzy_minimum(self, n_fuzzy):
        pass
    
    #get maximum of fuzzy number (max y > 0)
    def fuzzy_maximum(self, n_fuzzy):
        pass
    
    #get the minimum (AND) of singleton and fuzzy number (as numpy array)
    # singleton - float - single value, x
    # n_fuzzy - fuzzy number [nparray[x], nparray[y]] (from skfuzzy package)
    def fuzzy_single_AND(self, singleton, n_fuzzy):
        #try:
        
        singleton = float(singleton)
        for i in range(len(n_fuzzy[0])-1):
            #print i, 'Check Range:', n_fuzzy[0][i], singleton, n_fuzzy[0][i+1]
            #print type(n_fuzzy[0][i]), type(singleton), type(n_fuzzy[0][i+1])
            if round(n_fuzzy[0][i],6) <= round(singleton,6) and round(n_fuzzy[0][i+1],6) > round(singleton,6):  #find points around singleton              
                
                #interpolate linearly for more accurate answer
                return n_fuzzy[1][i] + (n_fuzzy[1][i+1] - n_fuzzy[1][i]) * \
                       ((singleton - n_fuzzy[0][i]) / (n_fuzzy[0][i+1] - n_fuzzy[0][i]))
                       
        print 'Singleton (', singleton, ') Not Found in Fuzzy Range (',str(min(n_fuzzy[0])), '-', str(max(n_fuzzy[0])), ')!'
        return 0.0 #[ n_fuzzy[0], [0.0 for n in n_fuzzy[0]] ]
        #except TypeError as (errno, strerror):
        #    print "Type Error:".format(errno, strerror)
        #    return 0.0

    #take in float singleton value and range and return fuzzy value
    # s - single float value
    # x_range - range to build MF on ([x1,x2,step])
    def singleton_to_fuzzy(self, s, x_range):
        x = np.arange(x_range[0], x_range[1], x_range[2])
        y = np.zeros(len(x))
        print len(x), len(y)
        
        print x,y
        
        for i in range(len(x)): 
            if x[i] < s and x[i+1] >= s:
                x = np.insert(x,i+1,s)
                x = np.insert(x,i+1,s)
                x = np.insert(x,i+1,s)
                y = np.insert(y,i+1,0)
                y = np.insert(y,i+1,1)
                y = np.insert(y,i+1,0)
                break
                
        return [x,y]
        
    #get firing stregth of an input
        #input_name - linguistic input name
        #input_ - list corresponding to input [x,y] or singleton
        #input_sys - object corresponding to system input MFs
        #
    def firing_strength(self, input_name, input_, input_sys):
        if not isinstance(input_, list): #if a singleton and not a list
            fs = self.fuzzy_single_AND(input_, [input_sys.MFs[input_name][0],input_sys.MFs[input_name][1]])        
            return fs        
        x_min,y_min = fuzz.fuzzy_and(input_sys.MFs[input_name][0],input_sys.MFs[input_name][1],input_[0],input_[1]) #use AND operator to get minimum of two functions
        return max(y_min)
     
    #recurses through rule antecedent to get firing strength
    #list1 - originally the rule antecedent goes here
    def rule_recurse(self, list1, input_list, TESTMODE):
        while any(isinstance(l,list) for l in list1):

            n = next(l for l in list1 if isinstance(l,list))        #get the next instance of a list in the given list
            for j in range(len(list1)):                             #find that instance's index
                if isinstance(list1[j], list) and list1[j] == n: 
                    i = j
                    break
            list1[i] = self.rule_recurse(list1[i], input_list, TESTMODE)[0] #recurse the function on the found list
            #print 'list:', list1, 'dive deeper... '
        
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
           
        #calculate mean operators next
            
        return list1

    #perform implication on a given output MF for a given antecedent firing strength
         #fs - firing strength of rule antecedent (single float value)
         #outputMF - output membership function to apply implication to (form [[x_vals],[y_vals]])
         #returns the resulting membership function
    def implicate(self, fs, outputMF):
        y_ = copy.deepcopy(outputMF[1])
        if self.implication == 'MIN':
            for i in range(len(y_)): 
                if y_[i] > fs: y_[i] = fs
        if self.implication == 'PRODUCT':
            for i in range(len(y_)): y_[i] = y_[i]*fs
        return y_

    #peform aggregation on the given outputs of the rules
        #MFs - list of MF functions
        #returns aggregation of all MFs
    def aggregate(self, MFs):
        o1 = MFs.pop()
        if self.aggregator == 'MAX':
            while len(MFs) > 0:
                o2 = MFs.pop()
                o1[0],o1[1] = fuzz.fuzzy_or( o1[0],o1[1],o2[0],o2[1] )
                
                
        return o1
        
    #runs the fuzzy system for a single output
        #input_list - dict of inputs {'input1':value, 'input2':value, ...} 
        #output_key - key of output to calculate
        #TESTMODE   - testmode flag
        #returns    - the resulting fuzzy output (NOT Defuzzified)
    def run_system(self, input_list, output_key, TESTMODE):
                
        outs = []
        for rule in self.rulebase:   #iterate over rulebase
            if TESTMODE == 1: 
                print '------------------------------------------------------------------------'
                print 'TRANSLATING RULE: ', rule.rule_id, 'for output', output_key
                  
            #break apart antecedent and consequent
            if_i = rule.rule_list.index('IF')   
            then_i = rule.rule_list.index('THEN')
            rule_ant = copy.deepcopy(rule.rule_list[if_i+1:then_i])                         #get the rule antecedent
            rule_con = copy.deepcopy(rule.rule_list[then_i+1:len(rule.rule_list)+1])[0]     #get the rule consequent
            
            if rule_con[0] == output_key:               #only follow rule if it applies to given output 

                fs = self.rule_recurse(rule_ant, input_list, TESTMODE)[0]   #get firing strength

                if TESTMODE == 1: print 'FIRING STREGTH, RULE', rule.rule_id, ':', fs
            
                output = copy.deepcopy(self.outputs[rule_con[0]].MFs[rule_con[2]]) #get output
                output[1] = self.implicate(fs, output)#use implication to get fuzzy consequent
                outs.append(output)
        
        #aggregate outputs
        if len(outs) > 0: 
            output_result = self.aggregate(outs)    #aggregate outputs if there are outputs
        else:
            m1 = self.outputs[output_key].data_range[0]         #get output min
            m2 = self.outputs[output_key].data_range[1]         #get output max
            x1 = np.arange(m1,m2,0.01)                          #get x range
            output_result = [x1, [0 for i in range(len(x1))]]   #return mf function of zeros
            
        return output_result
        
    #runs the fuzzy system for all inputs
        #input_list          - dict of inputs {'input1':value, 'input2':value, ...}  #lack of name in dict assumes 0 input (0 for all possible values)
        #TESTMODE (optional) - flag to output text for test mode (default is OFF)
        #returns:            - dict of outputs {'output1':value, 'output2':value, ...}
    



    def execute(self): #, input_list,TESTMODE=0):  
    
       

        #try:
        TESTMODE = self.TESTMODE
        TESTPLOT = self.TESTPLOT
        input_list = self.input_list
        
        #check if for fcl file and re-read inputs
        if self.fcl_file <> self.old_fcl_file:
            print 'New FCL File:', self.fcl_file, 'detected.  Loading ....'
            self.inputs, self.outputs, self.rulebase, self.AND_operator, \
            self.OR_operator, self.aggregator, self.implication, self.defuzz = build_fuzz_system(self.fcl_file)
            
            print 'New FRBS Loaded...', len(self.inputs), 'inputs. ', \
                  len(self.outputs), 'outputs. ', len(self.rulebase), 'rules.'
            #print 'INPUTS w/ MFs:', [k for k in self.inputs]
            
            self.input_mfs = self.inputs        #add to MDAO outputs
            self.output_mfs = self.outputs      #add to MDAO outputs
            
            self.old_fcl_file = self.fcl_file   #track fcl file
        
        #----TESTMODE----:
        if TESTMODE == 1:
            print 'INPUTS PASSED:', [k for k in input_list]
            for k in input_list: 
                if isinstance(input_list[k], list): print k, 'as fuzzy input. ', len(self.inputs[k].MFs), 'MFs available.'
                else: print k, 'as', input_list[k], '.', len(self.inputs[k].MFs), 'MFs available.'

        #----------------
        
        #print 'Executing FRBS', len(input_list), 'input values read from', len(self.input_list), 'inputs.'
        #run systm for each output
        for key in self.outputs:

            if self.passthrough == 1: 
                self.outputs_all[key] = None #catch for incompatible option (does nothing if incompatible)
            else:
                output_val = self.run_system(input_list, key, TESTMODE)
                self.outputs_all[key] = output_val
            
            #----TESTMODE----:
            if TESTPLOT == 1:
                
                fig = plt.figure()
                
                i = 1
                
                for k in self.inputs:
                    #plot each input against MFs
                    plt.subplot(len(self.inputs)+len(self.outputs), 1, i)
                    for k2 in self.inputs[k].MFs:
                        plt.plot(self.inputs[k].MFs[k2][0], self.inputs[k].MFs[k2][1])
                    i = i + 1
                    
                    #plot input
                    if isinstance(input_list[k], list): 
                        plt.plot(input_list[k][0],input_list[k][1], lw=3.0, color='k')
                        plt.yticks([0.0, 0.5, 1.0])
                    else:
                        plt.plot([input_list[k],input_list[k]],[0,1.0], lw=3.0, color='k')
                        plt.yticks([0.0, 0.5, 1.0])
                    plt.ylabel(k)
                    plt.ylim([0,1.1])
                    #plt.xlim([1,9])
                    
                #plot output against MFs     
                plt.subplot(len(self.inputs)+len(self.outputs), 1, i)                
                for k in self.outputs[key].MFs:
                    plt.plot(self.outputs[key].MFs[k][0], self.outputs[key].MFs[k][1])
                    plt.yticks([0.0, 0.5, 1.0])
                plt.plot(output_val[0],output_val[1], lw=3.5, color='b')
                fig.subplots_adjust(hspace=0.5, top=0.95, bottom=0.05)
                plt.show()
            #--------------
        
        self.runFlag_out = self.runFlag_in
        
        #except Exception, err:
        #    print "!!!!!! Unexpected error !!!!!"
        #    print traceback.format_exc()
        #    #or
        #    print sys.exc_info()[0]
        #    print "!!!!!!                  !!!!!"
        #return outputs_all

"""
if __name__=="__main__": 

    #testing find_substring
    test_string = "REAL; (* RANGE(1 .. 9) *)"
    substring = find_substring(test_string, 'RANGE(', ')')
    print substring     

    #testing build_fuzz_system
    inputs, outputs, rulebase, AND_operator, OR_operator, aggregator, implication = build_fuzz_system('test.fcl')
    #PRINTING OUT SYSTEM FOR DISPLAY
    print ''
    print 'INPUTS:'    
    for key in inputs: 
        print inputs[key].name, inputs[key].data_type, inputs[key].data_range
        for key in inputs[key].MFs: print '    ', key
    print 'OUTPUTS:'
    for key in outputs:
        print outputs[key].name, outputs[key].data_type, outputs[key].data_range
        for key in outputs[key].MFs: print '    ', key
    for r in rulebase: 
        print 'RULE: ', r.rule_id
        print r.rule_list
    print 'AND operator: ', AND_operator
    print 'OR operator: ', OR_operator
    print 'AGGREGATOR: ', aggregator 
    print 'IMPLICATION: ', implication

    #testing fuzzy system
    print '------------------------------------------------------------------------'
    print '                       TEST FUZZY SYSTEM                                '
    print '------------------------------------------------------------------------'

    sys = fuzzy_system(inputs, outputs, rulebase, AND_operator, OR_operator, aggregator, implication)
    print sys.firing_strength('High', 6.3, inputs['Weight_VLsys'])
    x1 = np.arange(1.0,9.1,0.1)
    y1 = fuzz.trapmf(x1, [4,6,7,9])
    print sys.firing_strength('High', [x1,y1] , inputs['Weight_VLsys'])
    
    print 'TEST FUZZY RULE COVERAGE:'
    sys.check_rule_coverage()
    
    print 'TEST FUZZY RULE IMPLEMENTATION'
    input_list = {'Weight_VLsys': 7.0, 'Weight_VLprop': [x1,y1], 'Weight_VLdrive': 5.0}
    results = sys.run(input_list, TESTMODE=1)
    print len(results), 'results calculated.'
"""