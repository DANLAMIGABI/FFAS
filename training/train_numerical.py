"""
author: Frank Patterson - 4Apr2015
- creates a fuzzy rule base from some data set, given input and output fuzzy membership functions

v0.1.0 - 14Apr15 - basic training and visualization complete. Still need to write 
                   fuzzy data file (manually? or just print rules?). Also need verificaiton script.
v0.2.0 - 18Apr15 - completed ability to write fuzzy data file. 

"""

# set current/working directory to FFAS
import numpy as np
import skfuzzy as fuzz

import operator
import random

#from fuzzy_operations import *
import fuzzy_operations as fuzzyOps
from timer import Timer

import matplotlib.pyplot as plt 
import matplotlib.ticker as ticker
from mpl_toolkits.mplot3d.art3d import Line3DCollection

from datetime import datetime

###########################    
def train_system(inputMFs, outputMFs, data, inDataMFs='tri', outDataMFs='tri', maxDataN=None, fuzzyDataFlag=False, ruleMethod=2):
    """
    Utilizes Wang and Mendel's "Generating Fuzzy Rules by Learning from Examples"
    build a fuzzy rule set based on input data and input/output FMs
    combinedData in form: [quant_inputs, qual_inputs, outputData]
        with each data item ['function', 'var'] : [min,max]
    inDataMFs  = type of MF for input data (given [min, max]), supports sing, trap, gauss, and tri 
    outDataMFs = type of MF for output data (given [min,max]), supports sing, trap, gauss, and tri 
    threads = num of threads
    
    fuzzyDataFlag : bool
        if true data is already fuzzy
    ruleMethod : int
        Method of reducing rule base (1: keep consequent w/ highest degree
                                      2: keep consequent w/ highest average degree
                                      3: keep consequents w/ degree > k*max_degree)
    """
        
    q = 0 #0 for quant data, 1 or qual data
    
    def get_rule(data_item): 
    
        #get degrees (firing strengths) of each rule
        ants = {} #antecedents of each rule in form ['function', 'var'] : ('ling', degree) where degree is firing strengths
        for inKey in inputMFs: #for each input key

            if not fuzzyDataFlag:
                [inXs, inYs] = fuzzyOps.rangeToMF(data_item[q][inKey], inDataMFs) #build input MF
            else: 
                [inXs, inYs] = data_item[q][inKey]
            
            max_fs = ('none', 0.0) #keep track of MF with maximum firing strength
            
            for ling in inputMFs[inKey]: #for each MF in the input
                #with Timer() as t:
                fs_x,fs_y = fuzz.fuzzy_and(inputMFs[inKey][ling][0],inputMFs[inKey][ling][1], inXs, inYs)
                #print '----------> AND time:', t.msecs, 'ms for input:', inKey, 'at length', len(fs_y)
                fs = max(fs_y)  #get firing strength of data point and given antecedent MF
                if fs > max_fs[1]: max_fs = (ling,fs)

            if max_fs[0] <> 'none':
                ants[inKey + (max_fs[0],)] = max_fs[1] #append rule with highest firing strength to antecedent
                
        if len(ants) > 0:    
            rule_deg = reduce(operator.mul, [ants[k] for k in ants]) #use to calc "degree" or strength of each rule
        else: 
            rule_deg =  0.0
        
        # repeat for outputs
        conts = {} #consequents of each rule in form output : ('ling', degree)
        for outKey in outputMFs: #for each output key
        
            if not fuzzyDataFlag:
                [outXs, outYs] = fuzzyOps.rangeToMF(data_item[2], outDataMFs) #build outputMF
            else:
                [outXs, outYs] = data_item[2]
                
            max_fs = ('none', 0.0) #keep track of MF with maximum firing strength
            for ling in outputMFs[outKey]: #for each MF in the input
                #with Timer() as t:
                fs_x, fs_y = fuzz.fuzzy_and(outputMFs[outKey][ling][0], outputMFs[outKey][ling][1], outXs, outYs)
                #print '----------> AND output time:', t.msecs, 'ms'
                fs = max(fs_y)  #get firing strength of data point and given antecedent MF
                if fs > max_fs[1]: max_fs = (ling,fs)

            conts[(outKey,) + (max_fs[0],)] = max_fs[1] #append rule with highest firing strength to antecedent
        
        if len(conts) > 0:    
            rule_deg = rule_deg * reduce(operator.mul, [conts[k] for k in conts]) #use to calc "degree" or strength of each rule
        else: rule_deg = 0.0
        
        return [ants, conts, rule_deg]
   
    #Get all rules from data (one for each point)
    rules = []
    a1 = datetime.now()
    n = 0
    for di in data:                 #get each rule using above function
        #with Timer() as t:
        r = get_rule(di)
        rules.append(r)
        n = n+1
        #print '--> got rule', n, 'in', t.secs, 's'
        if maxDataN <> None and n > maxDataN: break
    a2 = datetime.now()

    #Get Rules:
    if ruleMethod == 1:
        #METHOD 1: (Wang 92) for creating rule base
        #create rule grid, keeping only consequents with the highest "degrees" or strength
        #from a single data point
        rule_grid = {}
        for rule in rules:
            if rule[2] > 0.0 and len(rule[0]) > 0:
                antX = tuple(a for a in rule[0])
                if not antX in rule_grid:               #if rule isn't in grid add it
                    rule_grid[antX] = [tuple(c for c in rule[1]), rule[2]]
                elif rule_grid[antX][1] < rule[2]:      #if new rule has higher strength/degree then replace the old one.
                    rule_grid[antX] = [tuple(c for c in rule[1]), rule[2]]
                
    elif ruleMethod == 2:
        #METHOD 2: (Patterson 15) for creating rule base
        #create rule grid, averaging rule "degree" for each consequent. keep the antecendent
        #with the highest average "degree"
        rule_grid = {}
        for rule in rules:
            if rule[2] > 0.0 and len(rule[0]) > 0:
                antX = tuple(sorted(a for a in rule[0]))
                if not antX in rule_grid:               #if rule isn't in grid add it
                    rule_grid[antX] = {tuple(c for c in rule[1]): (rule[2], 1)} #add new dict with consequent giving (total degree, number of data points)
                else:   
                    if not tuple(c for c in rule[1]) in rule_grid[antX]: #if consequent isn't already accounted for
                        rule_grid[antX][tuple(c for c in rule[1])] = (rule[2], 1) 
                    else: 
                        rule_grid[antX][tuple(c for c in rule[1])] = \
                            (rule_grid[antX][tuple(c for c in rule[1])][0] + rule[2],#add to the rule dgree and
                            rule_grid[antX][tuple(c for c in rule[1])][1] + 1)      #update the number of data points
        
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
        
    elif ruleMethod == 3:
        #METHOD 3: (Cordon, Herrera - 2000) for creating rule base
        #create rule grid, averaging rule "degree" for each consequent. 
        #keep top two consequents for each rule
        rule_grid = {}
        for rule in rules:
            if rule[2] > 0.0 and len(rule[0]) > 0:
                antX = tuple(sorted(a for a in rule[0]))
                if not antX in rule_grid:               #if rule isn't in grid add it
                    rule_grid[antX] = {tuple(c for c in rule[1]): (rule[2], 1)} #add new dict with consequent giving (total degree, number of data points)
                else:   
                    if not tuple(c for c in rule[1]) in rule_grid[antX]: #if consequent isn't already accounted for
                        rule_grid[antX][tuple(c for c in rule[1])] = (rule[2], 1) 
                    else: 
                        rule_grid[antX][tuple(c for c in rule[1])] = \
                            (rule_grid[antX][tuple(c for c in rule[1])][0] + rule[2],#add to the rule dgree and
                            rule_grid[antX][tuple(c for c in rule[1])][1] + 1)      #update the number of data points
        
        for rule in rule_grid: #for each rule grid get the average degree for each consequent
            for cons in rule_grid[rule]: 
                rule_grid[rule][cons] = (rule_grid[rule][cons][0]/ \
                                        rule_grid[rule][cons][1], rule_grid[rule][cons][1])
        
        for rule in rule_grid: #get best 2 consequents
            consequents = [ (cons, rule_grid[rule][cons][0]) for cons in rule_grid[rule] ]
            consequents.sort(key=lambda x: x[1])            #get sorted list of consequents
            con1 = consequents.pop(-1)                                  #best
            rule_grid[rule] = [con1[0]] 
            if len(consequents) > 0: 
                con2 = consequents.pop(-1)         #2nd best
                rule_grid[rule].append(con2)
  
    
    
    
    return rule_grid
  

##### PLOTTING #####

###########################
def plot_rule_grid(rule_grid, inputMFs, outputMFs, data, x_axis, y_axis, z_axis):
    """
    x_axis, y_axis, z_axis are names of input variables to plot
    #data in form: [{('DATA', 'e_d'): [0.14], 
                     ('DATA', 'sigma'): [0.21], 
                     ('DATA', 'w'): [118.24], 
                     ('DATA', 'eta'): [0.73]}, None, [0.63]]
    """
    
    q = 0 #0 for quant data, 1 or qual data
    alpha = 0.5 #alpha cut to draw rule polygons

    xs = []
    ys = []
    zs = []
    outs = []

    for d in data: #capture x,y,z locations from data
        xs.append( sum(d[q][x_axis])/len(d[q][x_axis]) )
        ys.append( sum(d[q][y_axis])/len(d[q][y_axis]) )
        zs.append( sum(d[q][z_axis])/len(d[q][z_axis]) )
        outs.append(d[2])
    
    #get colors for data based on outputs (based on average)
    mi = min([min(o) for o in outs])
    ma = max([max(o) for o in outs])
    cs = []
    for o in outs: cs.append(plt.cm.jet( (sum(o)/len(o) - mi)/(ma - mi) ))
        
    #get rule polygons - 
    #rule in form:  (('DATA', 'w', 'med'),('DATA', 'sigma', 'high'),
    #                ('DATA', 'eta', 'med'),('DATA', 'e_d', 'med')): [('sys_FoM', 'high'), 0.267753881821930
    verts = []
    ecs = []
    for rule in rule_grid:
        for ant in rule:
            if ant[0:2] == x_axis: x_ling = ant[2]
            if ant[0:2] == y_axis: y_ling = ant[2]
            if ant[0:2] == z_axis: z_ling = ant[2]
        x_verts = alpha_cut(alpha, inputMFs[x_axis][x_ling])
        y_verts = alpha_cut(alpha, inputMFs[y_axis][y_ling])
        z_verts = alpha_cut(alpha, inputMFs[z_axis][z_ling])
        s1 = [[x_verts[0], y_verts[0], z_verts[0]], [x_verts[1], y_verts[0], z_verts[0]],
              [x_verts[1], y_verts[1], z_verts[0]], [x_verts[0], y_verts[1], z_verts[0]], [x_verts[0], y_verts[0], z_verts[0]]]
        s2 = [[x_verts[0], y_verts[0], z_verts[0]], [x_verts[0], y_verts[0], z_verts[1]],
              [x_verts[0], y_verts[1], z_verts[1]], [x_verts[0], y_verts[1], z_verts[0]], [x_verts[0], y_verts[0], z_verts[0]]]
        s3 = [[x_verts[0], y_verts[0], z_verts[1]], [x_verts[1], y_verts[0], z_verts[1]],
              [x_verts[1], y_verts[1], z_verts[1]], [x_verts[0], y_verts[1], z_verts[1]], [x_verts[0], y_verts[0], z_verts[1]], [x_verts[1], y_verts[0], z_verts[1]]]
        s4 = [[x_verts[1], y_verts[0], z_verts[1]], [x_verts[1], y_verts[0], z_verts[1]],
              [x_verts[1], y_verts[1], z_verts[1]], [x_verts[1], y_verts[1], z_verts[0]], [x_verts[1], y_verts[0], z_verts[0]]]        
        verts.append(s1+s2+s3+s4)
        outMF = outputMFs[rule_grid[rule][0][0][0]][rule_grid[rule][0][0][1]]
        cent = fuzz.defuzz(outMF[0], outMF[1], 'centroid')
        ecs.append(plt.cm.jet(cent))
        
    poly = Line3DCollection(verts, colors=ecs)
    poly.set_alpha(0.5)
           
        
    #plot it out
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    ax.scatter(xs, ys, zs, c=cs, marker='o')   
    ax.add_collection3d(poly)  
        
    ax.set_xlabel(x_axis)
    ax.set_ylabel(y_axis)
    ax.set_zlabel(z_axis)

    plt.show()
    
###########################
def plot_parallel(rule_grid, inputMFs, outputMFs, data, plotside):


    #### PLOT FUNCTION ####  (ORIGINAL INPUTS: data_sets, style=None)
    # http://stackoverflow.com/questions/8230638/parallel-coordinates-plot-in-matplotlib
    def parallel_coordinates(data_sets, colors=None, lws=None, style=None):
        dims = len(data_sets[0])
        x    = range(dims)
        fig, axes = plt.subplots(1, dims-1, sharey=False)
    
        if style is None:
            style = ['r-']*len(data_sets)
        if colors is None:
            colors = ['r']*len(data_sets)
        if lws is None:
            lws = [1.0]*len(data_sets)
        # Calculate the limits on the data
        min_max_range = list()
        for m in zip(*data_sets):
            mn = min(m)
            mx = max(m)
            if mn == mx:
                mn -= 0.5
                mx = mn + 1.
            r  = float(mx - mn)
            min_max_range.append((mn, mx, r))
    
        # Normalize the data sets
        norm_data_sets = list()
        for ds in data_sets:
            nds = [(value - min_max_range[dimension][0]) / 
                    min_max_range[dimension][2] 
                    for dimension,value in enumerate(ds)]
            norm_data_sets.append(nds)
        data_sets = norm_data_sets
    
        # Plot the datasets on all the subplots
        for i, ax in enumerate(axes):
            for dsi, d in enumerate(data_sets):
                ax.plot(x, d, style[dsi], c=colors[dsi], lw=lws[dsi])
            ax.set_xlim([x[i], x[i+1]])
    
        # Set the x axis ticks 
        for dimension, (axx,xx) in enumerate(zip(axes, x[:-1])):
            axx.xaxis.set_major_locator(ticker.FixedLocator([xx]))
            ticks = len(axx.get_yticklabels())
            labels = list()
            step = min_max_range[dimension][2] / (ticks - 1)
            mn   = min_max_range[dimension][0]
            for i in xrange(ticks):
                v = mn + i*step
                labels.append('%4.2f' % v)
            axx.set_yticklabels(labels)
    
    
        # Move the final axis' ticks to the right-hand side
        axx = plt.twinx(axes[-1])
        dimension += 1
        axx.xaxis.set_major_locator(ticker.FixedLocator([x[-2], x[-1]]))
        ticks = len(axx.get_yticklabels())
        step = min_max_range[dimension][2] / (ticks - 1)
        mn   = min_max_range[dimension][0]
        labels = ['%4.2f' % (mn + i*step) for i in xrange(ticks)]
        axx.set_yticklabels(labels)
    
        # Stack the subplots 
        plt.subplots_adjust(wspace=0)

        return plt
    ####  ####  ####
    
    #Plot data
    q = 0 #0 for quant data, 1 or qual data
    n = 400 #number of data points to plot
    
    n = min(n,len(data))
    data_p = random.sample(data,n) #get random sample of data
    data_sets = []
    colorsOut = []
    lwsOut = []
    
    #get list of keys from rulebase
    keys = []
    for ant in rule_grid:
        for a in ant:
            if not a[0:2] in keys: keys.append(a[0:2])
        
    minOut = min([min(d[2]) for d in data_p]) #minimum output
    maxOut = max([max(d[2]) for d in data_p]) #maximum output
    
    for dp in data_p: 
        line = [sum(dp[q][k])/len(dp[q][k]) for k in keys] #get line of inputs in right order
        data_sets.append(line)
        colorsOut.append(plt.cm.jet( (sum(dp[2])/len(dp[2]) - minOut)/(maxOut - minOut) ))
        lwsOut.append(0.3)
        
    #plot rules
    rule_data = []
    rule_colors = []
    rule_lws = []
    for rule in rule_grid: #for each rule
        line = []
        for k in keys:      #order the inputs
            for ant in rule:    #check each element of antecendent for input
                if ant[0:2] == k:
                    line.append(fuzz.defuzz(inputMFs[k][ant[2]][0], inputMFs[k][ant[2]][1], 'centroid'))
        rule_data.append(line)
        outMF = outputMFs[rule_grid[rule][0][0][0]][rule_grid[rule][0][0][1]]
        cent = fuzz.defuzz(outMF[0], outMF[1], 'centroid')
        rule_colors.append(plt.cm.jet(cent))        
        rule_lws.append(2.0)
        
    #append rules
    data_sets = data_sets + rule_data
    colorsOut = colorsOut + rule_colors
    lwsOut = lwsOut + rule_lws 
    
    parallel_coordinates(data_sets, colors=colorsOut, lws=lwsOut).show() #plot the data

        
############################
def write_fcl_file_FRBS(inputMFs, outputMFs, rule_grid, defuzz, filename, ranges=None):
    """
    For a standard FRBS (fuzzy rule based system):
    Takes in the input/output MFs and rule grid, and writes a fuzzy control language file
    currently handles triangular and trapezoidal MFs
    """
        
    c = open(filename, 'w')
    c.truncate()
    
    with open(filename, 'w') as fclfile:   #with the file open
    
        fclfile.seek(0) #start at beginning
        fclfile.write('FUNCTION_BLOCK ' + filename + '\n') #start fcl block
        fclfile.write('     #written with write_fcl_file_FRBS' + '\n ')
        fclfile.write('\n')
        
        #write inputs (INPUT_NAME:     REAL; (* RANGE(1 .. 9) *))
        fclfile.write('VAR_INPUT ' + '\n') #start input name block
        for inp in inputMFs:
            if len(inputMFs[inp][inputMFs[inp].keys()[0]]) > 2: #for tri and trap
                mi = min([min(inputMFs[inp][key]) for key in inputMFs[inp]])
                ma = max([max(inputMFs[inp][key]) for key in inputMFs[inp]])
            else: 
                minRs = [(inputMFs[inp][key][0] - 3.0*inputMFs[inp][key][1]) for key in inputMFs[inp]]
                maxRs = [(inputMFs[inp][key][0] + 3.0*inputMFs[inp][key][1]) for key in inputMFs[inp]]
                mi = min(minRs)
                ma = max(maxRs)
            fclfile.write('    '+inp[0]+'_'+inp[1] + ':     REAL; (* RANGE(' + 
                          str(mi) + ' .. ' + str(ma) + ') \n')
        fclfile.write('END_VAR ' + '\n') #end block
        fclfile.write('\n')
        
        #write outputs (OUTPUT_NAME:     REAL; (* RANGE(1 .. 9) *))
        fclfile.write('VAR_OUTPUT ' + '\n') #start input name block
        for opt in outputMFs:
            
            
            if len(outputMFs[opt][outputMFs[opt].keys()[0]]) > 2: #if triangular or trapezoidal
                mi = min([min(outputMFs[opt][key]) for key in outputMFs[opt]])
                ma = max([max(outputMFs[opt][key]) for key in outputMFs[opt]])
            else: #for gaussian mfs
                minRs = [(outputMFs[opt][key][0] - 3.0*outputMFs[opt][key][1]) for key in outputMFs[opt]]
                maxRs = [(outputMFs[opt][key][0] + 3.0*outputMFs[opt][key][1]) for key in outputMFs[opt]]
                mi = min(minRs)
                ma = max(maxRs)
            fclfile.write('    '+ opt + ':     REAL; (* RANGE(' + 
                          str(mi) + ' .. ' + str(ma) + ') \n')
        fclfile.write('END_VAR ' + '\n') #end block        
        fclfile.write('\n')
        
        #write input fuzzifications 
        for inp in inputMFs:
            fclfile.write('FUZZIFY ' + inp[0]+'_'+ inp[1] + '\n')
            for ling in inputMFs[inp]:  #write linguistic terms
                fclfile.write('    TERM ' + ling + ' := ')
                if     len(inputMFs[inp][ling]) == 2: y=['mean','std']
                elif   len(inputMFs[inp][ling]) == 3: y=[0.,1.,0.] #for triangular
                elif len(inputMFs[inp][ling]) == 4: y=[0.,1.,1.,0.] #for trapezoidal
                for i in range(len(inputMFs[inp][ling])): #write out points of MF
                    fclfile.write('(' + str(inputMFs[inp][ling][i]) + ',' + str(y[i]) + ') ')
                fclfile.write(';' + '\n')
            fclfile.write('END_FUZZIFY ' + '\n')
            fclfile.write('\n')
        
        #write output defuzzifications
        for opt in outputMFs:
            fclfile.write('DEFUZZIFY ' + opt + '\n')
            for ling in outputMFs[opt]:  #write linguistic terms
                fclfile.write('    TERM ' + ling + ' := ')
                if     len(outputMFs[opt][ling]) == 2: y=['mean','std']
                elif   len(outputMFs[opt][ling]) == 3: y=[0.,1.,0.] #for triangular
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
        if not defuzz == None:
            fclfile.write('    DEFUZZ:' + str(defuzz) + '\n')   #means to aggregate outputs (max)
        fclfile.write('\n')
        
        
        #write out rules
        i = 0
        for rule in rule_grid:
            if type(rule_grid[rule][0]) <> 'list': #for single consequent rules
                ruleStr = '    RULE ' + str(i) + ':    IF '
                for term in rule:
                    ruleStr = ruleStr + '(' + term[0] + '_' + term[1] + ' IS ' + term[2] + ') AND '
                ruleStr = ruleStr[:-4] + 'THEN ' + '(' + rule_grid[rule][0][0][0] + ' IS ' + rule_grid[rule][0][0][1] + ');  '
                #ruleStr = ruleStr + '[' + str(rule_grid[rule][1])[:6] + '] ' eventually add rule strength?
                fclfile.write(ruleStr + '\n') #append rule to file
                i = i+1
                
            elif type(rule_grid[rule][0]) == 'list': #for multi consequent rules
                for cons in rule_grid[rule]:
                    ruleStr = '    RULE ' + str(i) + ':    IF '
                    for term in rule:
                        ruleStr = ruleStr + '(' + term[0] + '_' + term[1] + ' IS ' + term[2] + ') AND '
                    ruleStr = ruleStr[:-4] + 'THEN ' + '(' + cons[0][0] + ' IS ' + cons[0][1] + ');  '
                    #ruleStr = ruleStr + '[' + str(rule_grid[rule][1])[:6] + '] ' eventually add rule strength
                    fclfile.write(ruleStr + '\n') #append rule to file
                i = i+1
            
            
        fclfile.write('\n')
            
        fclfile.write('END_RULEBLOCK' + '\n')
        fclfile.write('END_FUNCTION_BLOCK' + '\n')
        
####################################################################################
####################################################################################
####################################################################################


if __name__ == "__main__":
    pass
    
    
    #inputs, outputs, rulebase, AND_operator, OR_operator, aggregator, implication, \
    #    defuzz = build_fuzz_system('FCL_files/FRBS_FoM/FoMsys_trained_5-2In9-2Out_sing-gauss400tData_100vData_tempBEST_0.00396757693885.fcl')
    #plot_rule_grid(rule_grid, inputMFs, outputMFs, data, x_axis, y_axis, z_axis)