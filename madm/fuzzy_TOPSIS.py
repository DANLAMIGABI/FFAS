# -*- coding: utf-8 -*-
"""
Created on Sat Dec 14 12:37:29 2013

@author: - Frank Patterson

the mcdm method for fuzzy TOPSIS

Version 1.0 (15Dec13) - 
    First release version. Handles only Triangular fuzzy numbers currently.
    Needs the fuzzy operator for ranking trapezoidal fuzzy numbers to be completed first.

Version 1.0 (1Jan14) -
    Added alternate Fuzzy TOPSIS method by Wang & Elhag that solves relative closeness
    using a compass search. Updated original TOPSIS form. Updated some of the MCDM Display
"""

import os
import copy
import matplotlib.pyplot as plt

#params = {'legend.fontsize': 10,
#          'legend.linewidth': 0.5}
#plt.rcParams.update(params)

from time import gmtime, strftime
#from operator import attrgetter
import fuzzy as fuzzy
#import scoring as scoring
import numpy

mainpath = os.path.dirname(__file__) #file absolute path
print mainpath                          
#mainpath = os.path.dirname(mainpath)
#print mainpath 

########## CONTROL MCDM METHODS ##########

#Weight/Score Types:
#   C: Crisp (reads 1 value)
#   R: Range (reads 2 values)
#   FT: Fuzzy Triangular (reads in 3 values)
#   FZ: Fuzzy Trapezoidal (reads in 4 values)
methods = [['C',1], ['R', 2], ['FT', 3], ['FZ',4]]


#get the average fuzzy score for each alterantive
#raw scores is a list of these raw scores:
# [[alt1, [critID, score], [critID, score], ... ], 
#  [alt2, [critID, score], [critID, score], ... ], ... ]
# Tri scores in the form [a1, a2, a3] where mu(a1, a2, a3) = (0,1,0)
def getAvgFuzzyScores_Alt(input_file, score_method):
    raw_scores = []
    
    f = open(input_file, 'r')
    lines = f.readlines()
    while not lines[0].startswith("Alternatives"): 
        lines.pop(0)  #read to alternatives:
    lines.pop(0)
        
    while len(lines) > 0:
        
        if lines[0].startswith("ALTERNATIVE"):
            sl = [lines[0].rsplit(',')[2].strip()]  #start list with alternative ID 
            lines.pop(0)
            
            while not lines[0].strip().startswith("ALTERNATIVE"):
                s = [x.strip().rsplit(',') for x in lines[0].rsplit(';')]   #break up line
                if len(s) < 2: 
                    lines.pop(0)                                            #skip meaningless lines
                    if len(lines) == 0: break                               #break out if no more lines
                else:
                    s_ = next(x[1:] for x in s if x[0].strip()==score_method)       #create score list
                    for i in range(len(s_)): 
                        s_[i] = float(s_[i])                               #change scores to floats
                    sl.append([s[0][0], s_])                                #append critID and score      
                    lines.pop(0)
                    if len(lines) == 0: break                               #break out if no more lines
                    
            raw_scores.append(sl)

    return raw_scores

   
#gets a list of the weights in he form:
# [[critID, fuzzy_weight], [critID, fuzzy_weight], ... ]
def getFuzzyWeights_Alt(input_file, weight_method):
    raw_weights = []
   
    f = open(input_file, 'r')
    lines = f.readlines()
    while not lines[0].startswith("Criteria"): 
        lines.pop(0)  #read to alternatives:
    lines.pop(0)

    while len(lines) > 0:
        if lines[0].startswith("Alternatives"): break #break if not listing criteria any more:
        
        crit = lines.pop(0).strip().rsplit(';')
        if len(crit) > 1:
            for i in range(len(crit)): crit[i] = crit[i].rsplit(',')        #split apart line elements
            w = next(x[1:] for x in crit if x[0].strip()==weight_method)    #create score list
            for i in range(len(w)): w[i] = float(w[i])                      #turn strings to floats in weight
            raw_weights.append([crit[0][1].strip(), w])

    return raw_weights




#get rank list from fuzzy list
#ranks from max=1 to min=n   
def getFuzzyRanks(input_list, score_method):
    listX = copy.copy(input_list)
    ranks = [None for i in input_list]
    
    for i in range(1, len(input_list)+1):
        m = listX[0]
        for j in range(1, len(listX)):
            #get dominance
            if score_method == 'FT':
                d1 = fuzzy.dominance_FuzzyTri(m, listX[j])
                d2 = fuzzy.dominance_FuzzyTri(listX[j], m)   
            if score_method == 'FZ': None
            
            #check for dominance     
            if d2 == 1.0 and d1 < 1.0: 
                m = listX[j]
        
        ranks[input_list.index(m)] = i
        listX.pop(listX.index(m))
    return ranks

#get rank list from crisp list
#ranks from max=1 to min=n
def getCrispRanks(input_list):
    listX = copy.copy(input_list)
    ranks = copy.copy(listX)
    i = 1
    while listX.count(None) < len(listX):
        x = max(listX)
        ranks[listX.index(x)] = i
        listX[listX.index(x)] = None
        i = i + 1    
    return ranks    



#####################################################################
#################### ALTERNATIVES: FUZZY TOPSIS 1####################

#perform fuzzy TOPSIS on the alternative crisp scores
#works with 1)triangular score
def alt_fuzzy_TOPSIS(dm_data, weights, weight_method, score_method, *args):
    print "Peforming Fuzzy TOPSIS"
    
    """
    #get the decision matrix  
    # [[alt1, [critID, score], [critID, score], ... ], 
    #  [alt2, [critID, score], [critID, score], ... ], ... ]
    # scores in the form [a1, a2, a3] where mu(a1, a2, a3) = (0,1,0)
    raw_scores = getAvgFuzzyScores_Alt(input_file, score_method)
    
    #get fuzzy weights
    weights = getFuzzyWeights_Alt(input_file, weight_method)
    print "WEIGHTS:"
    for wr in weights: print wr
    """
    raw_scores = dm_data    
    
    # sort raw scores by criteria ID
    for r in raw_scores: 
        r1 = [r[0]]
        r1.extend(sorted(r[1:], key=lambda x: x[0]))
    print "RAW SCORES:"
    for r in raw_scores: print r
    

    #normalize the decision matrix
    norm_scores = raw_scores
    ss = ['sum_squares']
    for i in range(1,len(norm_scores[0])):
        ss_fzy = [0 for n in norm_scores[0][i][1]]
        for j in range(len(norm_scores)): #sum the squares for each criteria
        
            if score_method == 'FT' or score_method == 'FZ':
                for k in range(len(ss_fzy)):
                    ss_fzy[k] = ss_fzy[k] + float(norm_scores[j][i][1][k])**2
            
        for f in ss_fzy: f = f**0.5 #take sqrt
        ss.append(ss_fzy)
      
    for i in range(1,len(norm_scores[0])):
        for j in range(len(norm_scores)): #divide each score by the sum of squares
            if score_method == 'FT' or score_method == 'FZ':
                for k in range(len(norm_scores[j][i][1])):
                    norm_scores[j][i][1][k] = norm_scores[j][i][1][k] / ss[i][-(k+1)]
            
    print 'Normalized DM: '
    for r in norm_scores: print r
    
    #weight the normalized decision matrix
    for i in range(1,len(norm_scores[0])):
        for j in range(len(norm_scores)):
            for k in range(len(weights)):
                if weights[k][0] == norm_scores[j][i][0]:
                    
                    if score_method == 'FT' and weight_method == 'FT':
                        norm_scores[j][i][1] = \
                        fuzzy.mult_FuzzyTri((norm_scores[j][i][1], weights[k][1]))
                        
                    elif score_method == 'FZ' and weight_method == 'FT':
                        None
                    
    print 'Weighted and Normalized DM: '    
    for r in norm_scores: print r
    
    #get positive and negative ideal
    pos_ideals = ['A+']
    neg_ideals = ['A-']
    for i in range(1,len(norm_scores[0])): 
        pos_ideals.append(norm_scores[0][i][1]) #start with first element
        neg_ideals.append(norm_scores[0][i][1]) #start with first element
        
    for i in range(1,len(norm_scores[0])):   
        for j in range(len(norm_scores)):
            
            #check positive ideal
            d1 = fuzzy.dominance_FuzzyTri(pos_ideals[i], norm_scores[j][i][1])
            d2 = fuzzy.dominance_FuzzyTri(norm_scores[j][i][1], pos_ideals[i])
            print 'A:', pos_ideals[i], ' B:', norm_scores[j][i][1], '  d(A,B):', d1,  '  d(B,A):', d2 
            if d2 == 1.0 and d1 < 0.99: pos_ideals[i] = norm_scores[j][i][1]
            
            #check negative ideal
            d1 = fuzzy.dominance_FuzzyTri(neg_ideals[i], norm_scores[j][i][1])
            d2 = fuzzy.dominance_FuzzyTri(norm_scores[j][i][1], neg_ideals[i])
            if d1 == 1.0 and d2 < 0.99: neg_ideals[i] = norm_scores[j][i][1]
            
    #for p in pos_ideals: print p
    #for p in neg_ideals: print p
    
    #get fuzzy separation distances
    # for ideal (a1_i, a2_i, a3_i) fuzzy distance is 
    # ( SUM((a1-a1_s)^2)^0.5, SUM((a2-a2_s)^2)^0.5, SUM((a3-a3_s)^2)^0.5 )
    pos_dist = []           #[alt1_dist, alt2_dist, ...]
    neg_dist = []           #[alt1_dist, alt2_dist, ...]
    
    for j in range(len(norm_scores)):
        S_pos = [0 for n in norm_scores[j][1][1]]
        S_neg = [0 for n in norm_scores[j][1][1]]
        for i in range(1,len(norm_scores[j])):

            if score_method == 'FT':
                S_pos = [S_pos[0] + (norm_scores[j][i][1][0] - pos_ideals[i][0])**2, \
                         S_pos[1] + (norm_scores[j][i][1][1] - pos_ideals[i][1])**2, \
                         S_pos[2] + (norm_scores[j][i][1][2] - pos_ideals[i][2])**2 ]
                S_neg = [S_neg[0] + (norm_scores[j][i][1][0] - neg_ideals[i][0])**2, \
                         S_neg[1] + (norm_scores[j][i][1][1] - neg_ideals[i][1])**2, \
                         S_neg[2] + (norm_scores[j][i][1][2] - neg_ideals[i][2])**2 ]
                         
            elif score_method == 'FZ': None     
            
        for a in S_pos: a = a**0.5 #take square root to get distance
        for a in S_neg: a = a**0.5 #take square root to get distance
        pos_dist.append(S_pos)
        neg_dist.append(S_neg)
        
    #print "Positive Separation:", pos_dist
    #print "Negative Separation:", neg_dist
    
    #get relative closeness
    rel_closeness = []
    for i in range(len(pos_dist)):
        rel_closeness.append([neg_dist[i][0]/(neg_dist[i][2]+pos_dist[i][2]), \
                              neg_dist[i][1]/(neg_dist[i][1]+pos_dist[i][1]), \
                              neg_dist[i][2]/(neg_dist[i][0]+pos_dist[i][0])])
    
    #print "Relative Closeness:"
    #for i in range(len(rel_closeness)): print 'Alt', i, ':', rel_closeness[i]
    
    #get rankings
    rankings = getFuzzyRanks(rel_closeness, score_method)
    #print 'Rankings:', rankings
    
    for i in range(len(rel_closeness)):
        for j in range(len(rel_closeness[i])): 
            rel_closeness[i][j] = float(str(rel_closeness[i][j])[0:6])
    
    #plot results
    #labels = ['A'+str(i+1) for i in range(len(rel_closeness))]
    labels = ['Alt ' + x[0] for x in norm_scores]  
    
    if score_method == 'FT':
        fig = plt.figure(figsize=(7, 6))
        a = fig.add_subplot(111)
        for n in rel_closeness:
            a.plot([n[0], n[1], n[2]], [0.,1.,0.], linewidth=2.5)
    if score_method == 'FZ': 
        None
        
    a.set_title('Fuzzy TOPSIS')
    a.set_ylabel('Membership')
    a.set_xlabel('Relative Closeness')
    plt.legend(labels)
        
    #path = '/temp/' + strftime("FUZZY_TOPSIS1_%Y-%m-%d %H:%M:%S", gmtime()).replace(" ", "") + '.png'
    #fig.savefig(mainpath+path)
    
    return None
    
#####################################################################
#################### ALTERNATIVES: FUZZY TOPSIS 2####################
def alt_fuzzy_TOPSIS_2(dm_data, weights, weight_method, score_method, plot_RCs=0, *args):
    

    # function for Relative Closeness
    def function(W,Rs): 
        D_m, D_p = 0, 0
        for i in range(len(W)):
            D_m = D_m + (W[i]*Rs[i])**2
            D_p = D_p + (W[i]*(Rs[i] - 1))**2
            
        return D_m**0.5 / (D_m**0.5 + D_p**0.5)
    
    # Compass Search Module
    # performs a compass search#
    def compass_search(minmax, designVars, upperBounds, lowerBounds, stepSize, startX, tolerance, functionVals):
        
        Rs = functionVals[0]    
        
        iterations = 1              #counter for iterations of search
        functionCalls = 1           #counter for function calls of search (start with one for the function call for starting point)
            
        p = startX                  #starting point for search [x1,x2,x3,x4...]    
        y = function(p, Rs)             #function value for starting location
        
        plot_y, plot_p  = [y], [p]      #plotting list for function value
        
        #search till step size is greater than some defined tolerance
        while stepSize > tolerance:
            
            improved = False        #flag to see if all directions have been searched without improvement.
            d = 0               #d is the total directions searched (so the early dimensions are searched if a move is in the later dimension)
            
            #while not all directions have been searched
            while d < 2*designVars:
            
                #in both positive and negative direction
                for m in [-1,1]:
                    i = 0   #start at 0 dimension
                    
                    #search in each design variable's dimension
                    while i < designVars:
    
                        p_temp = copy.copy(p)                   #create temporary new point
                        p_temp[i] = p_temp[i] + m*stepSize      #move it to the new desired point
                    
                        #check to see if new point is out of bounds, if so move on to next direction
                        if p_temp[i] + m*stepSize > upperBounds[i] or p_temp[i] + m*stepSize < lowerBounds[i]:
                            i = i+1
                            d = d+1
                            #print "Out of Bounds at", p_temp
                            continue
                        
                        y_temp = function(p_temp, Rs)               #get new function value at temp location
                        functionCalls = functionCalls+1
                        #print "Checking", p_temp, " Value:", y_temp
                        
                        #check for improvement
                        if minmax == 'min' and y_temp < y or \
                           minmax == 'max' and y_temp > y:
                            
                            #if there's improvement, move the point and start a new iteration
                            p,y = p_temp, y_temp
                            iterations = iterations + 1
                            plot_y.append(y)
                            plot_p.append(p)
                            improved = True
                            #print "Moved to", p, "Value:", y
                            #repeat the same direction
    
                        else:
                            d = d+1
                            i = i+1 #change direction                         
    
            if not improved: stepSize = stepSize/2
            #print "Step Size Now =", stepSize, "Point Still:", p
        
        return p, y, functionCalls, iterations
        
    def getRCvals(alphas, weights, DMn):
        RC = [[[None, None] for a in alphas] for i in DMn]
        
        for i in range(len(alphas)):        #for each alpha level
            for j in range(len(DMn)):         #for each alt
    
                #construct weight limits
                upperBounds, lowerBounds = [], []
                if score_method == 'FT': limits = [fuzzy.getAlphaCutTri(w[0],w[1],w[2],alphas[i]) for w in weights]
                elif score_method =='FZ': limits = [fuzzy.getAlphaCutTrap(w[0],w[1],w[2],w[3],alphas[i]) for w in weights]
                    
                for lim in limits:
                    upperBounds.append(lim[1])
                    lowerBounds.append(lim[0])
                
                #construct Rij Lower matrix
                if score_method =='FT':   Rs = [min(fuzzy.getAlphaCutTri( r[0],r[1],r[2],alphas[i])) for r in DMn[j]]
                elif score_method =='FZ': Rs = [min(fuzzy.getAlphaCutTrap(r[0],r[1],r[2],r[3],alphas[i])) for r in DMn[j]]
                
                designVars = len(DMn[j])
                stepSize, tolerance = 0.1, 0.00005
                
                #get RCmin
                startX = lowerBounds #[(lowerBounds[i] + upperBounds[i])/2 for i in range(len(lowerBounds))]
                Ws, RCa_min, functionCalls, iterations = \
                        compass_search('min', designVars, upperBounds, lowerBounds, \
                                       stepSize, startX, tolerance, [Rs])
                
                RC[j][i][0] = RCa_min       
                #print str(RCa_min)[0:4], 'In', iterations, 'iterations'
                
                #construct Rij Upper matrix
                if score_method == 'FT':  Rs = [max(fuzzy.getAlphaCutTri( r[0],r[1],r[2],alphas[i])) for r in DMn[j]]
                elif score_method =='FZ': Rs = [max(fuzzy.getAlphaCutTrap(r[0],r[1],r[2],r[3],alphas[i])) for r in DMn[j]]
                    
                #get RCmin
                startX = upperBounds #[(lowerBounds[i] + upperBounds[i])/2 for i in range(len(lowerBounds))]
                Ws, RCa_max, functionCalls, iterations = \
                        compass_search('max', designVars, upperBounds, lowerBounds, \
                                       stepSize, startX, tolerance, [Rs])
                
                RC[j][i][1] = RCa_max 
                #print str(RCa_max)[0:4], 'In', iterations, 'iterations'
                #print '-----'
                
        return RC
    #-------------------------------------------------------------
    #print "Peforming Fuzzy TOPSIS"
    #construct the decision matrix  
    # [[alt1, [critID, score], [critID, score], ... ], 
    #  [alt2, [critID, score], [critID, score], ... ], ... ]
    # scores in the form [a1, a2, a3] where mu(a1, a2, a3) = (0,1,0)
    raw_scores = dm_data
    
    # sort raw scores by criteria ID and sort by alt
    for r in raw_scores: 
        r1 = [r[0]]
        r1.extend(sorted(r[1:], key=lambda x: x[0]))
        r = r1
    
    #get fuzzy weights 
    weights = sorted(weights, key=lambda x: x[0])       #sort weights
    crit_IDs = [w[0] for w in weights]                  #get crit IDs
    weights = [w[1] for w in weights]                   #distill weights to just scores
    m = max(max(w) for w in weights)                    #get max weight for normalization
    weights = [[w[i]/m for i in range(len(w))] for w in weights]   #normalize weights
    
    DM, alt_IDs = [], []
    for i in range(len(raw_scores)):
        alt_IDs.append(raw_scores[i][0])
        d_ = []
        for c in crit_IDs:
            d_.append(next(raw_scores[i][j][1] for j in range(1,len(raw_scores[i])) if raw_scores[i][j][0] == c))
        DM.append(d_)
    
    levels = 21
    alphas = [float(n)/(levels-1) for n in range(levels)]
    

    #normalize the DM
    #print 'Normalizing DM'
    DMn = copy.deepcopy(DM)
    #print "NON-NORMALIZED DM"
    #for dn in DM: print dn
        
    for j in range(len(DM[0])):
        m = max(max(DM[i_][j]) for i_ in range(len(DM)) ) #get max value for each criteria to normalize
        for i in range(len(DM)):
            DMn[i][j] = [float(DM[i][j][k])/m for k in range(len(DM[i][j]))]
            
    #print "NORMALIZED DM"        
    #for dn in DMn: print dn
        
    #find minimum for each criterion (negative ideal) 
    neg_ideals = []
    for j in range(len(DM[0])):
        neg_ideals.append(min(DM, key=lambda x:x[j][0])[j][0])
    #print 'neg_ideals', neg_ideals
        
    
    #print 'Calcing Relative Closeness'
    RCs = getRCvals(alphas, weights, DMn) #solve for RC alpha cuts
    #for alt in RCs:
    #    print "------ALT------"
    #    for r in alt: print r
    minRCs = [min(alt, key=lambda x:x[0])[0] for alt in RCs]
    maxRCs = [max(alt, key=lambda x:x[1])[1] for alt in RCs]
    
    defuzz = [(sum([(x[0]+x[1])/2 for x in alt])/len(alt)) for alt in RCs] #defuzzify for ranking
    rankings = getCrispRanks(defuzz)
    
    #PLOT
    if plot_RCs == 1:
        print 'Plotting'
    
        fig = plt.figure(figsize=(7, 6))
        a = fig.add_subplot(111)  
        for alt in RCs:
            x, y = [], []    
            for i in range(len(alphas)):
                x.append(alt[i][0])
                y.append(alphas[i])
            rev = range(len(alphas))
            rev.reverse()
            for i in rev:
                x.append(alt[i][1])
                y.append(alphas[i])
            a.plot(x,y, linewidth=2.0)
        
        a.set_title('Fuzzy TOPSIS')
        a.set_ylabel('Membership')
        a.set_xlabel('Relative Closeness')
        a.axis([0,1,0,1])
        labels = ['Alt ' + x for x in alt_IDs] 
        plt.legend(labels, loc=2)
            
        #path = '/temp/' + strftime("FUZZY_TOPSIS21_%Y-%m-%d %H:%M:%S", gmtime()).replace(" ", "") + '.png'
        #fig.savefig(mainpath+path)
        
        fig2 = plt.figure(figsize=(7,6))
        a2 = plt.subplot(111)
        a2.bar(range(len(labels)), defuzz, 0.4)
        a2.set_title('Defuzzified TOPSIS RCs')
        a2.set_ylabel('Relative Closeness')
        a2.set_xlabel('Alternatives')
        ind = numpy.arange(len(labels))
        a2.set_xticks(ind + 0.5*0.4)
        a2.set_xticklabels(labels, rotation=90)    
        
        #path2 = '/temp/' + strftime("FUZZY_TOPSIS22_%Y-%m-%d %H:%M:%S", gmtime()).replace(" ", "") + '.png'
        #fig2.savefig(mainpath+path2)
        plt.show()
        
    return alt_IDs, alphas, RCs

#------------------------------------------------------------------------
#-----------------------------TESTING------------------------------------
#------------------------------------------------------------------------
if __name__=="__main__": 
    #alt_fuzzy_TOPSIS("alternative_data.txt", 'FT', 'FT')
    weights_FT = [['001', [0.09, 0.51, 1.00]], \
                  ['002', [0.08, 0.44, 0.90]], \
                  ['003', [0.04, 0.26, 0.60]], \
                  ['004', [0.04, 0.22, 0.56]], \
                  ['005', [0.04, 0.22, 0.55]] ]
    weights_FZ =   [ ['001', [0.09, 0.26, 0.77, 1.00]], \
                   ['002', [0.08, 0.21, 0.68, 0.90]], \
                   ['003', [0.04, 0.13, 0.43, 0.60]], \
                   ['004', [0.04, 0.10, 0.38, 0.56]], \
                   ['005', [0.04, 0.10, 0.37, 0.55]] ]

    #TESTING:
    import alt_data_reader as data_reader
    print "Testing... GO"
    qual_data, quant_data = data_reader.read_data('alt_evals_data.csv', 10)    
    data1 = data_reader.reduce_data(qual_data, 'FUZZ_TRI_1')
    data2 = data_reader.reduce_data(qual_data, 'FUZZ_TRAP_1')
    
    RCs = alt_fuzzy_TOPSIS(data1, weights_FT, 'FT', 'FT')

    alt_IDs, alphas, RCs = alt_fuzzy_TOPSIS_2(data1, weights_FT, 'FT', 'FT', plot_RCs=1)
    alt_IDs, alphas, RCs = alt_fuzzy_TOPSIS_2(data2, weights_FZ, 'FZ', 'FZ', plot_RCs=1)
    
    #RCs = alt_fuzzy_TOPSIS_2("alternative_data.txt", 'FZ', 'FZ', plot_RCs=1)
    