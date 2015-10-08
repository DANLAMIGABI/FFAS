# -*- coding: utf-8 -*-
"""
Created on Mon Feb  2 16:40:35 2015

@author: frankpatterson
"""

import copy, time, random
import numpy as np
from scipy.stats import norm
from numpy import linalg as LA


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
#################### ALTERNATIVES: CRISP AHP ####################

#perform crisp Analytical Heirarchy Process on the alternative crisp scores
#uses normalized row averages from comparison matrices
def alt_crisp_AHP1(data, weights, *args):
    # takes in inputs:
    #     data = [[critID, [alt1#, alt2#, alt3#, ...], 
    #                     [alt1/alt1, alt1/alt2, alt1/alt3, ...],
    #                     [alt2/alt1, alt2/alt2, alt2/alt3, ...],
    #                     ...]
    #             [critID, [alt1#, alt2#, alt3#, ...], 
    #                     [alt1/alt1, alt1/alt2, alt1/alt3, ...],
    #                     [alt2/alt1, alt2/alt2, alt2/alt3, ...],
    #                     ...]
    #           ]
    #     weights =  [[critID1, critWeight1], [critID2, critWeight2], ...]
    #
    #

    # normalize alternative matrices by sum of columns
    #print 'Begininng Analytical Heirarchy Process...'
    #print 'Normalizing each Criterion comparison matrix...'
    alts = data[0][1]
    
    norm_data = []
    for i in range(len(data)):                  #parse each criteria matrix
        norm_matrix = [data[i][0], data[i][1]]  #init norm data matrix for criteria
        
        for j in range(2,len(data[i])):         #parse rows of comparison matrix
            norm_row = []                       #init norm data 
            
            for k in range(len(data[i][j])):    #parse columns of comparison matrix
                row_sum = sum(data[i][j1][k] for j1 in range(2,len(data[i])) ) 
                norm_row.append( data[i][j][k] / row_sum )                                                #divide by sum of column
            norm_matrix.append(norm_row)        #append row to matrix
        norm_data.append(norm_matrix)
    
    #sum rows of normalized matrix into total score
    sums = []
    for i in range(len(norm_data)):             #for each criteria
        col = [norm_data[i][0]]                 #add criteria ID
        for j in range(len(alts)):              #for each alternative in same order
            col.append( sum( norm_data[i][ norm_data[i][1].index(alts[j])+2] ) / \
                        len( norm_data[i][ norm_data[i][1].index(alts[j])+2] ))
        sums.append(col)
    
    #sort but don't normalize weights
    total = sum(w[1] for w in weights)
    sorted_weights = []
    for s in sums: 
        for w in weights:
            if w[0] == s[0]: 
                sorted_weights.append(w[1]/total) 
    
    #multiply columns by criteria weights and sum
    scores = []
    for j in range(len(alts)):         
        score = sum( sorted_weights[i]*sums[i][j+1] for i in range(len(sums))) # / \
                #len([sorted_weights[i]*sums[i][j+1] for i in range(len(sums))])
        scores.append([alts[j], score])

    #for s in scores: print s   
    return scores    
    


        

#perform crisp Analytical Heirarchy Process on the alternative crisp scores
#uses eigenvector method?
#uses geometric means
def alt_crisp_AHP2(data, weights, *args):
    # takes in inputs:
    #     data = [[critID, [alt1#, alt2#, alt3#, ...], 
    #                     [alt1/alt1, alt1/alt2, alt1/alt3, ...],
    #                     [alt2/alt1, alt2/alt2, alt2/alt3, ...],
    #                     ...]
    #             [critID, [alt1#, alt2#, alt3#, ...], 
    #                     [alt1/alt1, alt1/alt2, alt1/alt3, ...],
    #                     [alt2/alt1, alt2/alt2, alt2/alt3, ...],
    #                     ...]
    #           ]
    #     weights =  [[critID1, critID2, critID3, ...],
    #                   [crit1/crit1 crit1/crit2, ...],
    #                   crit2/crit1 crit2/crit2, ...],
    #                   ... 
    #                   ]
    #

    #sort the data matrices by criteria ID
    data.sort(key = lambda row: row[0])
    alts = data[0][1]   #get alt IDs
    
    #get only the pairwise comparison matrices 
    comp_matrices = []
    for i in range(len(data)):
        matrix = []
        for j in range(2,len(data[i])): 
            matrix.append(data[i][j])
        comp_matrices.append(matrix)

    # calculate priority vectors by normalizing the geometric means of the rows.    
    p_vectors = []
    for i in range(len(data)):
        priority_vector = [None for x in comp_matrices[i]]
        matrix_sum = 0. #sum row geometric means
        for j in range(len(comp_matrices[i])):
            row_product = reduce(lambda x, y: x * y, comp_matrices[i][j], 1) 
                                                    #multiply all the row elements
            geom_mean = row_product**(1.0/len(comp_matrices[i][j]))
                                                    #get geometric mean
            priority_vector[j] = geom_mean          #create priority vector of row geometric means
            matrix_sum = matrix_sum + geom_mean     #add to matrix sum
        priority_vector = [pv/matrix_sum for pv in priority_vector]
        p_vectors.append(priority_vector)    
      
    #sort and normalize weights
    weights.sort(key = lambda row: row[0])
   
    #normalize weights
    q = max(w[1] for w in weights)
    for i in range(len(weights)):
        weights[i][1] = weights[i][1] / q
    
    #multiply columns by criteria weights and sum
    scores = []
    for j in range(len(alts)): 
        score = sum( weights[i][1]*p_vectors[i][j] for i in range(len(p_vectors)))
            #sum priority vector values * corresponding weights.
        scores.append([alts[j], score])

    #for s in scores: print s   
    return scores    

        
#runs n random cases in the crisp ranges, uniformlly distributed
def prob_AHP_uniform(data_ranges, weight_ranges, n, alts, priority_quantiles, dist_type):
    # use AHP input structure:    
    #     data = [[critID, [alt1#, alt2#, alt3#, ...], 
    #                     [alt1/alt1, alt1/alt2, alt1/alt3, ...],
    #                     [alt2/alt1, alt2/alt2, alt2/alt3, ...],
    #                     ...]
    #             [critID, [alt1#, alt2#, alt3#, ...], 
    #                     [alt1/alt1, alt1/alt2, alt1/alt3, ...],
    #                     [alt2/alt1, alt2/alt2, alt2/alt3, ...],
    #                     ...]
    #           ]
    #     weights =  [[critID1, critWeight1], [critID2, critWeight2], ...]
    #
    # n - number of runs
    # alts - number of alts
    # priority_quantiles - number of quantiles
    # dist_type - distribution type: 'U' - uniform
    #                                'T' - triangular (uses data/weight ranges of min-mode-max or assumes midpoint if no mode)

    #generate random inputs
 
    quantiles = range(priority_quantiles)
    mq = float(max(quantiles) + 1)
    quantiles = [(q) / mq for q in quantiles]        #quantiles for priority vector (0 to 1)
    bin_width = quantiles[1] - quantiles[0]
    q_counter = [[0 for i in quantiles] for i in range(alts)]   #counters for priority quantiles
    rank_counter = [[0 for i in range(alts)] for i in range(alts)] #counters for alt ranks [count ranks 1-n for 1-n alternatives]
    full_ranks = []      
    full_results = [[] for i in range(alts)] #full data for RCs

    #perform n iterations
    start_time = time.clock()  
    for ni in range(n):
        
        if time.clock()-start_time > 5.0:
            print ni, '/', n, 'iterations performed... ', str(100*float(ni)/float(n))[0:4], '%'
            start_time = time.clock() 
        
        #create random scores/weights 
        ####### UNIFORM #######
        if dist_type == 'U':
            rand_comparison = copy.deepcopy(data_ranges)
            for i in range(len(data_ranges)): #for each criteria matrix
                for j in range(2,len(data_ranges[i])): #for each row of matrix
                    for k in range(len(data_ranges[i][j])):
                        rand_comparison[i][j][k] = random.uniform(data_ranges[i][j][k][0], data_ranges[i][j][k][1])
                            #get random ratio in range
                                            
            weights = copy.deepcopy(weight_ranges)
            for i in range(len(weights)): #for each weight
                weights[i][1] = random.uniform(weight_ranges[i][1][0], weight_ranges[i][1][1])
                    #get random weight

        ####### TRIANGULAR #######
        if dist_type == 'T':
            rand_comparison = copy.deepcopy(data_ranges)
            for i in range(len(data_ranges)): #for each criteria matrix
                for j in range(2,len(data_ranges[i])): #for each row of matrix
                    for k in range(len(data_ranges[i][j])):
                        if len(data_ranges[i][j][k]) < 3:
                            if data_ranges[i][j][k][0] == data_ranges[i][j][k][1]: rand_comparison[i][j][k] = data_ranges[i][j][k][1]
                            else: rand_comparison[i][j][k] = random.triangular(data_ranges[i][j][k][0], 
                                                                         data_ranges[i][j][k][1])
                        else:
                            if data_ranges[i][j][k][0] == data_ranges[i][j][k][1]: rand_comparison[i][j][k] = data_ranges[i][j][k][1]
                            else: rand_comparison[i][j][k] = random.triangular(data_ranges[i][j][k][0], \
                                                                         data_ranges[i][j][k][2], \
                                                                         data_ranges[i][j][k][1])
                            #get random ratio in range
                                            
            weights = copy.deepcopy(weight_ranges)
            for i in range(len(weights)): #for each weight
                if len(weight_ranges[i][1]) < 3:
                    weights[i][1] = random.triangular(weight_ranges[i][1][0], \
                                                      weight_ranges[i][1][1])
                else:
                    weights[i][1] = random.triangular(weight_ranges[i][1][0], \
                                                      weight_ranges[i][1][2], \
                                                      weight_ranges[i][1][1])
                    #get random weight

        # ============================ RUN AHP ==============================
        #priorities = alt_crisp_AHP1(rand_comparison, weights)
        priorities = alt_crisp_AHP2(rand_comparison, weights)
        
        rankings = getCrispRanks([x[1] for x in priorities])
        priorities = [x[1] for x in priorities]
        full_ranks.append(rankings)
        # =====================================================================
    
        #update rank counter
        for i in range(len(rankings)): rank_counter[i][rankings[i]-1] = \
                                       rank_counter[i][rankings[i]-1] + 1
        
        """        
        #update RC quantile counters
        for i in range(len(priorities)):
            flag = 0
            for j in range(len(quantiles)):
                if priorities[i] <= quantiles[j]: 
                    flag = 1
                    q_counter[i][j] = q_counter[i][j] + 1
                    break
            if flag == 0: print 'NOT FOUND:', priorities[i], quantiles[j]
        """    
        
        #capture full results
        for i in range(len(priorities)):
            full_results[i].append(priorities[i])
            
    #get histograms            
    for i in range(len(q_counter)):        
        hist, q_bins = np.histogram( full_results[i], bins=quantiles, \
                                     range=(min(quantiles),max(quantiles)) )
        q_counter[i] = np.ndarray.tolist(hist)
        #q_counter[i].append(0)
            #change to list and append 0 to make same length as bins are inclusive:
            #[a,b), so 0 items greater than last bin
        
    #normal fits for given data [mu, std] for each alternative
    norm_fits = [[None, None] for i in q_counter]
    for i in range(len(full_results)):
        norm_fits[i][0], norm_fits[i][1] = norm.fit(full_results[i])
    
    
    return rank_counter, full_ranks, quantiles, q_counter, norm_fits, full_results

if __name__ == "__main__":
    
    import matplotlib.pyplot as plt
    from itertools import cycle
    
    weights = [['001', 3378.],['002', 2881.],['003', 1751.],['004', 1472.],['005', 1436.]];
    weight_ranges = [ ['001', [0.09, 1.00]], \
                      ['002', [0.08, 0.90]], \
                      ['003', [0.04, 0.60]], \
                      ['004', [0.04, 0.56]], \
                      ['005', [0.04, 0.55]] ]
                      
    #TESTING:
    import alt_data_reader as data_reader
    print "Testing... GO"
    qual_data, quant_data = data_reader.read_data('alt_evals_data_thesis_final.csv', 10)    
    data1 = data_reader.reduce_data(quant_data, 'AHP_CRISP')
    data2 = data_reader.reduce_data(quant_data, 'AHP_RANGE')
    data22= data_reader.reduce_data(quant_data, 'AHP_RANGE_LARGE')
    data3 = data_reader.reduce_data(quant_data, 'AHP_FUZZ_TRI_1')
    
    results = alt_crisp_AHP1(data1, weights)
    results = alt_crisp_AHP2(data1, weights)
    for r in results: print r  
    
    
    n = 2000
    alts = 10
    priority_quantiles = 70 #high number due to limited range of priority vector
    
    rank_counter, full_ranks, quantiles, q_counter, norm_fits, full_results = \
        prob_AHP_uniform(data2, weight_ranges, n, alts, priority_quantiles, 'T')

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
    plt.xlabel('Priority')
    plt.ylabel('Frequency')
    plt.title('Priority Histogram')
    for i in range(len(q_counter)): plt.plot(quantiles[1:], q_counter[i], next(linecycler), \
                                             linewidth=2, label='ALT ' + str(i+1), color=colors[i] )
    plt.legend(loc=2)
    #plt.xlim([0.05, 0.15])        
    
    plt.figure()
    plt.xlabel('Priority')
    plt.ylabel('Sample Probability')
    plt.title('Priority Distribution')
    x = np.linspace(0.00, 1.0, 500)
    for i in range(len(norm_fits)): 
        p = norm.pdf(x, norm_fits[i][0], norm_fits[i][1])
        plt.plot(x, p, next(linecycler), linewidth=2, \
                 label='ALT ' + str(i+1), color=colors[i] )
    plt.legend(loc=2)    
    #plt.xlim([0.05, 0.15])    
    
    plt.figure()
    plt.xlabel('Priority')
    plt.ylabel('Sample Probability')
    plt.title('Cumulative Distribution')
    for i in range(len(norm_fits)): 
        p = norm.cdf(x, norm_fits[i][0], norm_fits[i][1])
        plt.plot(x, p, next(linecycler), linewidth=2, \
                 label='ALT ' + str(i+1), color=colors[i] )
    plt.legend(loc=2)    
    #plt.xlim([0.05, 0.15])    
    
    plt.figure()
    plt.xlabel('Rank')
    plt.ylabel('Frequency')
    plt.title('Rank Histogram')
    ranks = [i+1 for i in range(len(rank_counter[0]))]
    print ranks
    for i in range(len(rank_counter)): plt.plot(ranks, rank_counter[i], next(linecycler), \
                                                linewidth=2, label='ALT ' + str(i+1), color=colors[i] )
    plt.legend(loc=5)    
    
    plt.figure()
    plt.xlabel('Relative Closeness')
    plt.ylabel('Frequency')
    plt.title('Relative Closeness Histogram')
    plt.hist(q_counter[0],quantiles)
    lines = ['-','-','-','-','-','-','-','-','-','-']
    plt.hist(full_results,15,color=colors,)#, next(linecycler), \
                                             #linewidth=2, label='ALT ' + str(i+1), color=colors[i] )
    #plt.xlim([0, 0.2])    
    plt.show()
    
    
"""  LARGE AHP RANGES:
Sample PDFs
Alternative 0 : mu = 0.0983824496557  std = 0.00371373565999
Alternative 1 : mu = 0.104641173251  std = 0.00446641702642
Alternative 2 : mu = 0.109733352285  std = 0.00433903686628
Alternative 3 : mu = 0.0881423189729  std = 0.00395901884186
Alternative 4 : mu = 0.0918630119063  std = 0.00397857198232
Alternative 5 : mu = 0.0956559270258  std = 0.00470710112349
Alternative 6 : mu = 0.104232101349  std = 0.00381597518076
Alternative 7 : mu = 0.112513189185  std = 0.00315644232808
Alternative 8 : mu = 0.0867742130446  std = 0.00480366960328
Alternative 9 : mu = 0.108062263325  std = 0.00390462060837

     AVG AHP RANGES:
Sample PDFs 
Alternative 0 : mu = 0.0967562993661  std = 0.00320031462046
Alternative 1 : mu = 0.102949432541  std = 0.00382817149469
Alternative 2 : mu = 0.112644511053  std = 0.00460979600696
Alternative 3 : mu = 0.0859933616297  std = 0.00368902582642
Alternative 4 : mu = 0.0913407390175  std = 0.00354388089778
Alternative 5 : mu = 0.0969078698925  std = 0.00504655147823
Alternative 6 : mu = 0.102952674364  std = 0.00328643212595
Alternative 7 : mu = 0.113101175635  std = 0.00247279333947
Alternative 8 : mu = 0.0880745337215  std = 0.00499967066786
Alternative 9 : mu = 0.109279402779  std = 0.0042076212699
"""