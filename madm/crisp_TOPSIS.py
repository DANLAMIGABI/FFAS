# -*- coding: utf-8 -*-
"""
Created on Fri Dec 13 16:15:28 2013

@author: - Frank Patterson

the mcdm methods for crisp numbers

Version 1.0 (15Dec13) - 
    First release version. Only averages scores from different users. Only has 
    crisp TOPSIS.
    
"""

import os
import copy
import matplotlib.pyplot as plt

import numpy as np
from scipy.stats import norm

import random
from time import gmtime, strftime
import time

#from operator import attrgetter
#import scoring

mainpath = os.path.dirname(__file__)                 #file absolute path
mainpath = os.path.dirname(mainpath)

########## CONTROL MCDM METHODS ##########
#Weight/Score Types:
#   C: Crisp (reads 1 value)
#   R: Range (reads 2 values)
#   FT: Fuzzy Triangular (reads in 3 values)
#   FZ: Fuzzy Trapezoidal (reads in 4 values)
methods = [['C',1], ['R', 2], ['FT', 3], ['FZ',4]]


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
    
    
####################################################################
#################### ALTERNATIVES: CRISP TOPSIS ####################

#perform crisp TOPSIS on the alternative crisp scores
def alt_crisp_TOPSIS(data, weights, plot, *args):
    #print "Peforming Crisp TOPSIS"
    
    #raw scores is a list of raw scores:
    # [[alt1, [critID, score], [critID, score], ... ], 
    #  [alt2, [critID, score], [critID, score], ... ], ... ]
    raw_scores = data

    weights = sorted(weights, key=lambda x: x[0])       #sort weights by crit ID
    #don't normalize weights!
    #m = max(w[1] for w in weights)
    #weights = [[w[0], w[1]/m] for w in weights]   #normalize weights
    
    #get list of criteria IDs
    critIDs = []
    for i in range(1, len(raw_scores[0])): critIDs.append(raw_scores[0][i][0])  
    
    #get normalized scores: divide by the square root of sum of squares
    #print 'Normalizing Scores'
    norm_scores = raw_scores
    ssqs = []                       #sum of squares list corresponds to critIDs
    for id_ in critIDs:             #calc each sum of squares
        s = 0.0
        for row in raw_scores:
            for j in range(1, len(row)): 
                if row[j][0] == id_: 
                    s = s + row[j][1]**2.0
        ssqs.append(s)

    for r in range(len(norm_scores)): 
        for i in range(1, len(norm_scores[r])):    #divide each score by the sqrt of sum of squares
        
            for j in range(len(critIDs)):
                if norm_scores[r][i][0] == critIDs[j]: 
                    norm_scores[r][i][1] = norm_scores[r][i][1]/(ssqs[j]**0.5)        
    
    #apply weights:
    #print 'Applying Weights'
    for row in norm_scores:
        for i in range(1, len(row)):
            wt = next((w[1] for w in weights if w[0] == row[i][0]), 0.0)
            
            row[i][1] = row[i][1] * wt
            
    
    #get positive and negative ideal
    #print 'Getting Ideals'
    pos_ideals = [0 for i in range(len(critIDs))]
    neg_ideals = [1 for i in range(len(critIDs))]
    for i in range(len(critIDs)):
        for row in norm_scores:
            for j in range(1, len(row)):
                if row[j][0] == critIDs[i]:
                    if row[j][1] > pos_ideals[i]: pos_ideals[i] = row[j][1]
                    if row[j][1] < neg_ideals[i]: neg_ideals[i] = row[j][1]
                    
    #get distances from positive and negative ideal and relative closeness
    #distance = sqrt(sum((score-idealscore^2))
    #closeness = neg_dist / (neg_dist + pos_dist)
    #print 'Calculating Distances'
    pos_dist = []           #[alt1_dist, alt2_dist, ...]
    neg_dist = []           #[alt1_dist, alt2_dist, ...]
    rel_closeness = []      #[alt1_dist, alt2_dist, ...]

    for row in norm_scores:
        pd = 0
        nd = 0
        for i in range(1,len(row)):
            for j in range(len(critIDs)):
                if critIDs[j] == row[i][0]:
                    p_ideal = pos_ideals[j]
                    n_ideal = neg_ideals[j]
                    
            pd = pd + (row[i][1] - p_ideal)**2
            nd = nd + (row[i][1] - n_ideal)**2
            
        pd = pd**0.5
        nd = nd**0.5
        rc = nd/(nd+pd)
        
        pos_dist.append(pd)
        neg_dist.append(nd)
        rel_closeness.append(rc)
    
    rankings = getCrispRanks(rel_closeness)
    
    #plot results
    if plot == 1:
        #print 'Plotting Results'
        try:
            labels = []
            for i in range(len(rel_closeness)): labels.append('A'+str(i))
            
            fig = plt.figure(figsize=(7,6))
            a = plt.subplot(111)
            width = 0.4
            a.bar(range(len(labels)), rel_closeness, width)
            a.set_title('Crisp TOPSIS')
            a.set_ylabel('Relative Closeness')
            a.set_xlabel('Alternatives')
            ind = np.arange(len(labels))
            a.set_xticks(ind + 0.5*width)
            a.set_xticklabels(labels, rotation=90)    
                
            path = '/temp/' + strftime("FUZZY_TOPSIS21_%Y-%m-%d %H:%M:%S", gmtime()).replace(" ", "") + '.png'
            fig.savefig(mainpath+path)
        except: 
            path = ''
 
    return rel_closeness, rankings
   
#runs n random cases in the crisp ranges, uniformlly distributed
def prob_TOPSIS_uniform(data_ranges, weight_ranges, n, alts, rc_quantiles, dist_type):
        
    #sort and normalize  weight ranges
    weight_ranges = sorted(weight_ranges, key=lambda x: x[0])       #sort weights
    m = max(max(w[1]) for w in weight_ranges)
    weight_ranges = [[w[0], [w[1][0]/m, w[1][1]/m]] for w in weight_ranges]   #normalize weights
    
    #generate result structures
    quantiles = range(rc_quantiles)
    mq = float(max(quantiles) + 1)
    quantiles = [(q + 1) / mq for q in quantiles]        #quantiles for relative closeness
    q_counter = [[0 for i in quantiles] for i in data_ranges]    #counters for RC quantiles
    rank_counter = [[0 for i in data_ranges] for i in data_ranges] #counters for alt ranks [count ranks 1-n for 1-n alternatives]
    full_ranks = []    
    full_RCs = [[] for i in data_ranges] #full data for RCs

    #perform n iterations
    start_time = time.clock()  
    for ni in range(n):
        
        if time.clock()-start_time > 5.0:
            print ni, '/', n, 'iterations performed... ', str(100*float(ni)/float(n))[0:4], '%'
            start_time = time.clock() 
        
        #create random scores/weights 
        ####### UNIFORM #######
        if dist_type == 'U':
            raw_scores = copy.deepcopy(data_ranges)
            for i in range(len(raw_scores)): #for each alternative
                for j in range(1,len(raw_scores[i])): #for each input
                    raw_scores[i][j][1] = random.uniform(data_ranges[i][j][1][0], \
                                                         data_ranges[i][j][1][1])
                                                           
            #create input weights
            weights = copy.deepcopy(weight_ranges)
            for i in range(len(weights)): #for each weight
                weights[i][1] = random.uniform(weight_ranges[i][1][0], \
                                               weight_ranges[i][1][1])
        
        ####### TRIANGULAR #######
        if dist_type == 'T':
            raw_scores = copy.deepcopy(data_ranges)
            for i in range(len(raw_scores)): #for each alternative
                for j in range(1,len(raw_scores[i])): #for each input
                    if len(raw_scores[i][j][1]) < 3:
                        raw_scores[i][j][1] = random.triangular(data_ranges[i][j][1][0], \
                                                             data_ranges[i][j][1][1])
                    else:
                        raw_scores[i][j][1] = random.triangular(data_ranges[i][j][1][0], \
                                                             data_ranges[i][j][1][2], \
                                                             data_ranges[i][j][1][1])                                                               
            #create input weights
            weights = copy.deepcopy(weight_ranges)
            for i in range(len(weights)): #for each weight
                if len(weights[i][1]) < 3:
                    weights[i][1] = random.triangular(weight_ranges[i][1][0], \
                                                   weight_ranges[i][1][1])  
                else:
                    weights[i][1] = random.triangular(weight_ranges[i][1][0], \
                                                   weight_ranges[i][1][2], \
                                                   weight_ranges[i][1][1])   
        # ============================ RUN TOPSIS ==============================
        rel_closeness, rankings = alt_crisp_TOPSIS(raw_scores, weights, 0)        
        
        full_ranks.append(rankings)
        #rankings = getCrispRanks(rel_closeness)    

        # =====================================================================
    
        #update rank counter
        for i in range(len(rankings)): rank_counter[i][rankings[i]-1] = \
                                       rank_counter[i][rankings[i]-1] + 1
                                       
        """                               
        #update RC quantile counters
        for i in range(len(rel_closeness)):
            flag = 0
            for j in range(len(quantiles)):
                if rel_closeness[i] <= quantiles[j]: 
                    flag = 1
                    q_counter[i][j] = q_counter[i][j] + 1
                    break
            if flag == 0: print 'NOT FOUND:', rel_closeness[i], quantiles[j]
        """        
        
        #capture RCs
        for i in range(len(rel_closeness)):
            full_RCs[i].append(rel_closeness[i])

    #get historgrams
    for i in range(len(q_counter)):        
        hist, q_bins = np.histogram( full_RCs[i], bins=quantiles, \
                                     range=(min(quantiles),max(quantiles)) )
        q_counter[i] = np.ndarray.tolist(hist)        
        
    #turn quantiles into PDF
    norm_fits = [[None, None] for i in q_counter]
    for i in range(len(full_RCs)):
        norm_fits[i][0], norm_fits[i][1] = norm.fit(full_RCs[i])

    return rank_counter, full_ranks, quantiles, q_counter, norm_fits, full_RCs



    
#------------------------------------------------------------------------
#-----------------------------TESTING------------------------------------
#------------------------------------------------------------------------
if __name__=="__main__": 
    #rel_closeness, rankings = alt_crisp_TOPSIS("alternative_data.txt", 'C', 'C', 0)
    #print rel_closeness
    #print rankings

    weights = [['001', 3378.],['002', 2881.],['003', 1751.],['004', 1472.],['005', 1436.]]
    weight_ranges = [ ['001', [0.09, 1.00]], \
                      ['002', [0.08, 0.90]], \
                      ['003', [0.04, 0.60]], \
                      ['004', [0.04, 0.56]], \
                      ['005', [0.04, 0.55]] ]

    #TESTING:
    import alt_data_reader as data_reader
    print "Testing... GO"
    qual_data, quant_data = data_reader.read_data('alt_evals_data.csv', 10)    
    data1 = data_reader.reduce_data(qual_data, 'AVG')
    data2 = data_reader.reduce_data(qual_data, 'AVG_RANGE')

    results = alt_crisp_TOPSIS(data1, weights, 1)


    #TEST MONTECARLO
    from itertools import cycle
    
    n = 3000
    quants = 50
    alts = 10
    rank_counter, full_ranks, quantiles, q_counter, norm_fits, full_RCs = prob_TOPSIS_uniform(data2, weight_ranges, n, quants, alts, 'U')
    
    print "RANK COUNTER"
    for r in rank_counter: print r, sum(r)
    print ""
    print "QUANTILES:"
    print quantiles
    print "QUANTILE COUNTER"
    for q in q_counter: print q, sum(q)
    print "Sample PDFs"
    for i in range(len(norm_fits)): print 'Alternative', i, ': mu =',  norm_fits[i][0], ' std =', norm_fits[i][1]
         
    colors = plt.get_cmap('Set1')(np.linspace(0, 1.0, len(q_counter)))
    lines = ['-']#["-","--","-.",":"]
    linecycler = cycle(lines)

    plt.figure()
    plt.xlabel('Relative Closeness')
    plt.ylabel('Frequency')
    plt.title('Relative Closeness Histogram')
    for i in range(len(q_counter)): plt.plot(quantiles[1:], q_counter[i], next(linecycler), \
                                             linewidth=2, label='ALT ' + str(i+1), color=colors[i] )
    plt.legend()
    
    plt.figure()
    plt.xlabel('Priority')
    plt.ylabel('Sample Probability')
    plt.title('Priority Distribution')
    x = np.linspace(0.0, 1.0, 500)
    for i in range(len(norm_fits)): 
        p = norm.pdf(x, norm_fits[i][0], norm_fits[i][1])
        plt.plot(x, p, next(linecycler), linewidth=2, \
                 label='ALT ' + str(i+1), color=colors[i] )
    plt.legend(loc=2)    
    #plt.xlim([0.0, 1.0])    
    
    plt.figure()
    plt.xlabel('Priority')
    plt.ylabel('Sample Probability')
    plt.title('Cumulative Distribution')
    for i in range(len(norm_fits)): 
        p = norm.cdf(x, norm_fits[i][0], norm_fits[i][1])
        plt.plot(x, p, next(linecycler), linewidth=2, \
                 label='ALT ' + str(i+1), color=colors[i] )
    plt.legend(loc=2)    
    #plt.xlim([0.0, 1.0])   
    
    plt.figure()
    plt.xlabel('Rank')
    plt.ylabel('Frequency')
    plt.title('Rank Histogram')
    ranks = [i+1 for i in range(len(rank_counter[0]))]
    for i in range(len(rank_counter)): plt.plot(ranks, rank_counter[i], next(linecycler), \
                                                linewidth=2, label='ALT ' + str(i+1), color=colors[i] )
    plt.legend()    
    
    
    

    
    #a = range(1,10)
    #b = range(4,13)
    #ind = np.arange(len(a))
    
    
    plt.figure()
    plt.xlabel('Relative Closeness')
    plt.ylabel('Frequency')
    plt.title('Relative Closeness Histogram')
    plt.hist(q_counter[0],quantiles)
    lines = ['-','-','-','-','-','-','-','-','-','-']
    plt.hist(full_RCs,15,color=colors,)#, next(linecycler), \
                                             #linewidth=2, label='ALT ' + str(i+1), color=colors[i] )
    plt.show()

        