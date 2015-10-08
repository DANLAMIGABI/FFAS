# -*- coding: utf-8 -*-
"""
Created on Thu Feb  5 08:55:25 2015

@author: frankpatterson
"""

import numpy as np
from numpy import linalg as LA

import fuzzy

####################################################################
#################### ALTERNATIVES: CRISP AHP ####################

#perform fuzzy Analytical Heirarchy Process on the alternative crisp scores
#uses normalized row averages from comparison matrices (straight translation of AHP to fuzzy arithmetic)
def alt_fuzzy_AHP1(data, weights, score_method, *args):
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
    #     score_method = 'FT': fuzzy triangular
    #                    'FZ': fuzzy trapezoidal

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
                if score_method == 'FT':
                    row_sum = fuzzy.add_FuzzyTri(data[i][j1][k] for j1 in range(2,len(data[i])) ) 
                    norm_row.append( fuzzy.divide_FuzzyTri(data[i][j][k], row_sum ))  
                                                           #divide by sum of column
                if score_method == 'FZ':
                    pass
                
            norm_matrix.append(norm_row)        #append row to matrix
        norm_data.append(norm_matrix)
        
    #sum rows of normalized matrix into total score
    sums = []
    for i in range(len(norm_data)):             #for each criteria
        col = [norm_data[i][0]]                 #add criteria ID
        for j in range(len(alts)):              #for each alternative in same order
            total = fuzzy.add_FuzzyTri( norm_data[i][ norm_data[i][1].index(alts[j])+2] )
            count = len( norm_data[i][ norm_data[i][1].index(alts[j])+2] )
            col.append([t/count for t in total])
        sums.append(col)
    
    #sort weights (assume already normalized)
    #total = sum(w[1] for w in weights)# removed
    sorted_weights = []
    for s in sums: 
        for w in weights:
            if w[0] == s[0]: 
                sorted_weights.append(w[1]) #/total)#removed
                    
    #multiply columns by criteria weights and sum
    scores = []
    for j in range(len(alts)):
        weighted_scores = [] 
        for i in range(len(sums)):
            weighted_scores.append( fuzzy.mult_FuzzyTri([sorted_weights[i], sums[i][(j+1)]]) )
                #multiply weight * score
        score = fuzzy.add_FuzzyTri(weighted_scores)
        scores.append([alts[j], score])
        
    return scores    

#perform fuzzy Analytical Heirarchy Process on the alternative crisp scores
#uses Buckley's Method (1985): 
#Buckley, J.J., 1985, Fuzzy hierarchical analysis, Fuzzy Sets and Systems, 17(3): 233–247.
def alt_fuzzy_AHP2(data, weights, score_method, *args):
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
    #     score_method = NOT USED!

    #sort the data matrices by criteria ID
    data.sort(key = lambda row: row[0])

    # modify matrix to create trapezoidal numbers:
    #accepts fuzzy numbers up to lenth of 4 (crisp, range, triangular, trapezoidal)
    #remove criteria ids and alt IDs and just retain data
    crits = [d[0] for d in data] #saving in case needed
    alt_IDs = data[0][1]    
    comp_matrices = [] 
    for i in range(len(data)):                  #parse each criteria matrix 
        matrix = []       
        for j in range(2,len(data[i])):         #parse rows of comparison matrix
            mat_row = [[1,1,1,1] for j1 in range(2,len(data[i]))]
            for k in range(len(data[i][j])):    #parse columns of comparison matrix
                if len(data[i][j][k]) == 4: 
                    mat_row[k] = data[i][j][k]
                elif len(data[i][j][k]) == 3: 
                    mat_row[k] = [data[i][j][k][0], data[i][j][k][1], \
                                  data[i][j][k][1], data[i][j][k][2]]
                elif len(data[i][j][k]) == 2: 
                    mat_row[k] = [data[i][j][k][0], data[i][j][k][0], \
                                  data[i][j][k][1], data[i][j][k][1]]
                elif len(data[i][j][k]) == 1: 
                    mat_row[k] = [data[i][j][k][0], data[i][j][k][0], \
                                  data[i][j][k][0], data[i][j][k][0]]    
            matrix.append(mat_row)
        comp_matrices.append(matrix)                                     

    #for c in comp_matrices: 
    #    print "NEW CRIT"
    #    for c1 in c: print c1                                 
                                     
    #get geometric mean of each row in each matrix
    p_vectors = []
    for i in range(len(comp_matrices)):
        priority_vector = [None for x in comp_matrices[i]]
        matrix_sum = [0.,0.,0.,0.] #sum row geometric means
        for j in range(len(comp_matrices[i])):
            row_product = fuzzy.mult_FuzzyTrap(comp_matrices[i][j]) 
                #multiply all the row elements
            geom_mean = [x**(1.0/len(comp_matrices[i][j])) for x in row_product]
                #get geometric mean
            priority_vector[j] = geom_mean #create priority vector of row geometric means
            matrix_sum = fuzzy.add_FuzzyTrap([matrix_sum, geom_mean]) #add to matrix sum
        priority_vector = [fuzzy.divide_FuzzyTrap(pv, matrix_sum) \
                           for pv in priority_vector]
        p_vectors.append(priority_vector)
    
    #sort and normalize weights
    weights.sort(key = lambda row: row[0])
    #change weights to fuzzy trapezoidal numbers
    for i in range(len(weights)):
        if len(weights[i][1]) == 4: pass
        elif len(weights[i][1]) == 3:
            weights[i][1] = [weights[i][1][0], weights[i][1][1],
                             weights[i][1][1], weights[i][1][2]]
        elif len(weights[i][1]) == 2:
            weights[i][1] = [weights[i][1][0], weights[i][1][0],
                             weights[i][1][1], weights[i][1][1]]    
        elif len(weights[i][1]) == 1:
            weights[i][1] = [weights[i][1], weights[i][1],
                             weights[i][1], weights[i][1]]
    
    #don't normalize weights
    #q = max(max(w[1]) for w in weights)
    #for i in range(len(weights)):
    #    for j in range(len(weights[i][1])):
    #        weights[i][1][j] = weights[i][1][j] / q
            
    #multiply prority vectors by criteria weights and sum
    #return in structure [[altID1, score1], [altID2, score2], ...]
    #where each score if a fuzzy number (a,b,c,d)
    scores = []
    for j in range(len(alt_IDs)):
        scores.append([alt_IDs[j], fuzzy.add_FuzzyTrap(fuzzy.mult_FuzzyTrap([weights[i][1], p_vectors[i][j]]) \
                                          for i in range(len(weights)))])
    
    return scores  
    
#perform fuzzy Analytical Heirarchy Process on the alternative crisp scores
#uses Chang's Extant Analysis
#Chang, D.Y., 1992, Extent Analysis and Synthetic Decision, Optimization Techniques 
#   and Applications, World Scientific, Singapore, 1: 352.
def alt_fuzzy_AHP3(data, weights, score_method, *args):
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
    #     score_method = NOT USED!

    #sort the data matrices by criteria ID
    data.sort(key = lambda row: row[0])

    # modify matrix to create trapezoidal numbers:
    #accepts fuzzy numbers up to lenth of 4 (crisp, range, triangular, trapezoidal)
    #remove criteria ids and alt IDs and just retain data
    crits = [d[0] for d in data] #saving in case needed
    alt_IDs = data[0][1]    
    comp_matrices = [] 
    for i in range(len(data)):                  #parse each criteria matrix 
        matrix = []       
        for j in range(2,len(data[i])):         #parse rows of comparison matrix
            mat_row = [[1,1,1,1] for j1 in range(2,len(data[i]))]
            for k in range(len(data[i][j])):    #parse columns of comparison matrix
                if len(data[i][j][k]) == 4: 
                    mat_row[k] = data[i][j][k]
                elif len(data[i][j][k]) == 3: 
                    mat_row[k] = [data[i][j][k][0], data[i][j][k][1], \
                                  data[i][j][k][1], data[i][j][k][2]]
                elif len(data[i][j][k]) == 2: 
                    mat_row[k] = [data[i][j][k][0], data[i][j][k][0], \
                                  data[i][j][k][1], data[i][j][k][1]]
                elif len(data[i][j][k]) == 1: 
                    mat_row[k] = [data[i][j][k][0], data[i][j][k][0], \
                                  data[i][j][k][0], data[i][j][k][0]]    
            matrix.append(mat_row)
        comp_matrices.append(matrix)                                     

    #for c in comp_matrices: 
    #    print "NEW CRIT"
    #    for c1 in c: print c1                                 
                                     
    #get  computing the normalized value of row sums (i.e. fuzzy synthetic extent) 
    #by fuzzy arithmetic operations: 
    for i in range(len(comp_matrices)):
        S_i = [None for r in comp_matrices[i]]  #row sums
        for j in range(len(comp_matrices[i])):
            S_i[j] = fuzzy.add_FuzzyTrap(comp_matrices[i][j])
        mat_total = fuzzy.add_FuzzyTrap(S_i) #matrix total (for normalization)
        S_i = [fuzzy.divide_FuzzyTrap(s,mat_total) for s in S_i] #normalize row sums

        
    #get degrees of possibility S_i > all other S  
        
    
    
    
    
    """
    #sort and normalize weights
    weights.sort(key = lambda row: row[0])
    #change weights to fuzzy trapezoidal numbers
    for i in range(len(weights)):
        if len(weights[i][1]) == 4: pass
        elif len(weights[i][1]) == 3:
            weights[i][1] = [weights[i][1][0], weights[i][1][1],
                             weights[i][1][1], weights[i][1][2]]
        elif len(weights[i][1]) == 2:
            weights[i][1] = [weights[i][1][0], weights[i][1][0],
                             weights[i][1][1], weights[i][1][1]]    
        elif len(weights[i][1]) == 1:
            weights[i][1] = [weights[i][1], weights[i][1],
                             weights[i][1], weights[i][1]]
    
    #don't normalize weights
    #q = max(max(w[1]) for w in weights)
    #for i in range(len(weights)):
    #    for j in range(len(weights[i][1])):
    #        weights[i][1][j] = weights[i][1][j] / q
            
    #multiply prority vectors by criteria weights and sum
    #return in structure [[altID1, score1], [altID2, score2], ...]
    #where each score if a fuzzy number (a,b,c,d)
    scores = []
    for j in range(len(alt_IDs)):
        scores.append([alt_IDs[j], fuzzy.add_FuzzyTrap(fuzzy.mult_FuzzyTrap([weights[i][1], p_vectors[i][j]]) \
                                          for i in range(len(weights)))])
    """
    
    return None  
    
 
if __name__ == "__main__":
    weight_labels = ['Empty Wt.', 'Max AS', 'Hover Eff.', 'L/D', 'Prop. Eff.']
    weights =       [ ['001', 0.51],['002', 0.44],['003', 0.26], \
                      ['004', 0.22],['005', 0.22]]
    weights_ranges = [ ['001', [0.09, 1.00]], \
                       ['002', [0.08, 0.90]], \
                       ['003', [0.04, 0.60]], \
                       ['004', [0.04, 0.56]], \
                       ['005', [0.04, 0.55]] ]
    weights_tri =   [ ['001', [0.09, 0.51, 1.00]], \
                      ['002', [0.08, 0.44, 0.90]], \
                      ['003', [0.04, 0.26, 0.60]], \
                      ['004', [0.04, 0.22, 0.56]], \
                      ['005', [0.04, 0.22, 0.55]] ]
    weights_trap =   [ ['001', [0.09, 0.26, 0.77, 1.00]], \
                       ['002', [0.08, 0.21, 0.68, 0.90]], \
                       ['003', [0.04, 0.13, 0.43, 0.60]], \
                       ['004', [0.04, 0.10, 0.38, 0.56]], \
                       ['005', [0.04, 0.10, 0.37, 0.55]] ]
    #TESTING:
    import alt_data_reader as data_reader
    print "Testing... GO"
    qual_data, quant_data = data_reader.read_data('alt_evals_data.csv', 10)    
    data1 = data_reader.reduce_data(qual_data, 'AHP_RANGE')
    data_tri = data_reader.reduce_data(qual_data, 'AHP_FUZZ_TRI_1')
    data_trap = data_reader.reduce_data(qual_data, 'AHP_FUZZ_TRAP_1')
    
    results = alt_fuzzy_AHP3(data_tri, weights_tri, 'FT', 0)
    
    for r in results: print r