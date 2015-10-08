# -*- coding: utf-8 -*-
"""
Created on Sun Feb  1 18:17:14 2015

@author: frankpatterson
"""
import csv
import fuzzy as fuzzy

#input_ranges = [[1,9],[150.,550.],[0.3,0.9],[4.,25.],[0.3, 0.9]] # input ranges
#in_rrs = [[x[0]/x[1], x[1]/x[0]] for x in input_ranges]
#out_rr = [1.0/9.0, 9.0] #output ratio range

def read_data(input_file, max_experts):
    """ 
    Reads in the raw scores from each expert for each alternative by criteria for each expert
    data format:
    data = [ [crit#, [alt#, [min1, max1], [min2, max2], [min3, max3] ... ], 
                        [alt#, [min1, max1], [min2, max2], [min3, max3] ... ],
                        ... ], 
                [crit#, [alt#, [min1, max1], [min2, max2], [min3, max3] ... ], 
                        [alt#, [min1, max1], [min2, max2], [min3, max3] ... ],
                        ... ], 
            ]
    both a  qual data and quant data are created
    max experts determines the split between qualitative expert evals and quantitative
    """ 
    #get qualitative data first    
    with open(input_file, 'rU') as csvfile:
        
        input_reader = csv.reader(csvfile, delimiter=',')
        data = []
        crit_data = []        
        for row in input_reader:        #read each row
        
            if 'CHAR' in row[0]:                        #if starting a new criteria
                if len(crit_data) > 1: data.append(crit_data)
                    
                crit_data = [row[0].rsplit('_')[1]]     #start new criteria data structrure
                next
            else:
                alt_data = [row[0].strip(' .,')]
                for i in range(1,len(row)):                             #for each score in the row
                    if i < max_experts:                                 #get qualitative data
                        if '[' in row[i]:
                            score = row[i].strip(' []').rsplit(',')     #separate score string into list [min, max]
                            alt_data.append([float(s) for s in score])  #convert scores to floats
                crit_data.append(alt_data)
        if len(crit_data) > 1: data.append(crit_data) #append last criteria data
        
        qual_data = data        #save as qualitative data
        
    #repeat for quantitative data  
    with open(input_file, 'rU') as csvfile:
        input_reader = csv.reader(csvfile, delimiter=',')
        data = []
        crit_data = []
        
        for row in input_reader:        #read each row
            if 'CHAR' in row[0]:                        #if starting a new criteria
                if len(crit_data) > 1: data.append(crit_data)
                    
                crit_data = [row[0].rsplit('_')[1]]     #start new criteria data structrure
                next
            else:
                alt_data = [row[0].strip(' .,')]
                for i in range(1,len(row)):                             #for each score in the row
                    if i > max_experts:                                 #get qualitative data
                        if '[' in row[i]:
                            score = row[i].strip(' []').rsplit(',')     #separate score string into list [min, max]
                            alt_data.append([float(s) for s in score])  #convert scores to floats
                crit_data.append(alt_data)
        if len(crit_data) > 1: data.append(crit_data) #append last criteria data
        
        quant_data = data        #save as qualitative data
            
    return qual_data, quant_data
    
def reduce_data(data_in, out_type):
    """
    reduce the list of expert alternative data (min, max) to some usable scores
    in a decision matrix (except the AHS data reductions)
        [[alt1, [critID, score], [critID, score], ... ], 
         [alt2, [critID, score], [critID, score], ... ], ... ]
        Example: Fuzzy tri scores in the form [a1, a2, a3] where mu(a1, a2, a3) = (0,1,0)
    
    out_types: 
        'AVG' - average all mins and maxs and take the average (return 1 value)
        'AVG_RANGE' - returns the average min and averae max
        'FUZZ_TRI_1' - return fuzzy triangular (avg min, avg, avg max) for (0, 1, 0)
        'FUZZ_TRAP_1' - returns a fuzzy trapezoidal number (min, avg min, avg max, max) for (0,1,1,0)
        'FUZZ_TRAP_2' - returns a fuzzy trapezoidal number (avg min, max(min), min(max), avg max) for (0,1,1,0)
        'FUZZ_TRAP_UNI' -  returns a fuzzy trapezoidal number (avg min, avg min, avg max, avg max) for (0,1,1,0)       
        'AHP_CRISP' - get Analytical Hierarchy Process (comparison) matricies from expert's scores (ratio of avgerages)
        'AHP_RANGE' - get Analytical Hierarchy Process (comparison) matricies from expert's scores (range by avg_min/avg_max - avg_max/avg_min)
        'AHP_RANGE_LARGE' - get Analytical Hierarchy Process (comparison) matricies from expert's scores (range by min(min)/max(max) - max(max)/min(min))
        'AHP_FUZZY_TRI_1' - get Analytical Hierarchy Process (comparison) matricies from expert's scores (same as FUZZ_TRI_1)    
        'AHP_FUZZY_TRAP_1' - get Analytical Hierarchy Process (comparison) matricies from expert's scores (same as FUZZ_TRAP_1)    
        'AHP_FUZZY_TRAP_2' - get Analytical Hierarchy Process (comparison) matricies from expert's scores (same as FUZZ_TRAP_2)  
        'AHP_FUZZY_TRAP_UNI' - get Analytical Hierarchy Process (comparison) matricies from expert's scores (same as FUZZ_TRAP_3)    


    """
    
    #get number of alts:
    crit_count = len(data_in)
    alt_count = len(data_in[0]) #actually number of alts + 1 as they start at index 1
    print crit_count, 'Criteria found; ', alt_count-1, 'Alternatives found'
    
    #maximum ranges from existing ratios
    mm_ranges = []
    for i in range(crit_count):
        max1 = max([max([x3[1] for x3 in x2[1:]]) for x2 in data_in[i][1:]]) #get max eval for criteria
        min1 = min([min([x3[0] for x3 in x2[1:]]) for x2 in data_in[i][1:]]) #get min eval for criteria
        #range1 = max1/min1 - min1/max1
        mm_ranges.append( [min1/max1, max1/min1] ) #get minimum and maximum ratios

    #translate to this range
    out_mm = [1./9., 9.0]
        
    def scale_ratio(inratio, in_range, out_range=out_mm):
        if inratio < 1:
            out = min(out_range) + (inratio-min(in_range)) * \
                   (1.0 - min(out_range)) / (1.0 - min(in_range))
            #print 'in:', inratio, '  in_R:', in_range, '  out_R:', out_range, '  min:', out
            return out
        if inratio > 1:
            out = 1.0 + (inratio - 1.0) * (max(out_range) - 1.0) / (max(in_range) - 1.0)
            #print 'in:', inratio, '  in_R:', in_range, '  out_R:', out_range, '  max:', out
            return out
        else: return 1.0
            
    data_out = []
    #--------------------------------------------------------------------------
    # average all mins and maxs and take the average (return 1 value)
    if out_type == 'AVG':
        
        for i in range(1,alt_count):
            alt_scores = [data_in[0][i][0]]
            for j in range(crit_count):
                all_scores = data_in[j][i][1:len(data_in[j][i])]
                avg = sum([sum(x)/2.0 for x in all_scores])/ \
                          len([sum(x) for x in all_scores])
                alt_scores.append([data_in[j][0], avg])
            data_out.append(alt_scores)

    #--------------------------------------------------------------------------
    # returns the average min and averae max
    if out_type == 'AVG_RANGE':
        
        for i in range(1,alt_count):
            alt_scores = [data_in[0][i][0]]
            for j in range(crit_count):
                all_scores = data_in[j][i][1:len(data_in[j][i])]
                avgs = [sum([x[0] for x in all_scores])/len(all_scores), \
                        sum([x[1] for x in all_scores])/len(all_scores)]
                alt_scores.append([data_in[j][0], avgs])
            data_out.append(alt_scores)
    
    #--------------------------------------------------------------------------
    # return fuzzy triangular (avg min, avg, avg max) for (0, 1, 0)
    if out_type == 'FUZZ_TRI_1':
        
        for i in range(1,alt_count):
            alt_scores = [data_in[0][i][0]]
            for j in range(crit_count):
                all_scores = data_in[j][i][1:len(data_in[j][i])]
                avgs = [sum([x[0] for x in all_scores])/len(all_scores), \
                        sum([sum(x)/2.0 for x in all_scores])/ \
                          len([sum(x) for x in all_scores]), \
                        sum([x[1] for x in all_scores])/len(all_scores)]
                alt_scores.append([data_in[j][0], avgs])
            data_out.append(alt_scores)
                
    #--------------------------------------------------------------------------
    # returns a fuzzy trapezoidal number (min, avg min, avg max, max) for (0,1,1,0)
    if out_type == 'FUZZ_TRAP_1':
        
        for i in range(1,alt_count):
            alt_scores = [data_in[0][i][0]]
            for j in range(crit_count):
                all_scores = data_in[j][i][1:len(data_in[j][i])]
                avgs = [min([x[0] for x in all_scores]), 
                        sum([x[0] for x in all_scores])/len(all_scores), 
                        sum([x[1] for x in all_scores])/len(all_scores),
                        max([x[1] for x in all_scores])]
                alt_scores.append([data_in[j][0], avgs])
            data_out.append(alt_scores)

    #--------------------------------------------------------------------------
    # returns a fuzzy trapezoidal number (avg min, max(min), min(max), avg max) for (0,1,1,0)
    if out_type == 'FUZZ_TRAP_2':
        
        for i in range(1,alt_count):
            alt_scores = [data_in[0][i][0]]
            for j in range(crit_count):
                all_scores = data_in[j][i][1:len(data_in[j][i])]
                avgs = [sum([x[0] for x in all_scores])/len(all_scores), 
                        max([x[0] for x in all_scores]),
                        min([x[1] for x in all_scores]),
                        sum([x[1] for x in all_scores])/len(all_scores)]
                avgs.sort()
                alt_scores.append([data_in[j][0], avgs])
            data_out.append(alt_scores)
                        
    #--------------------------------------------------------------------------
    # returns a fuzzy trapezoidal number (avg min, avg min, avg max, avg max) for (0,1,1,0)
    if out_type == 'FUZZ_TRAP_UNI':
        
        for i in range(1,alt_count):
            alt_scores = [data_in[0][i][0]]
            for j in range(crit_count):
                all_scores = data_in[j][i][1:len(data_in[j][i])]
                avgs = [sum([x[0] for x in all_scores])/len(all_scores), 
                        sum([x[0] for x in all_scores])/len(all_scores), 
                        sum([x[1] for x in all_scores])/len(all_scores),
                        sum([x[1] for x in all_scores])/len(all_scores)]
                alt_scores.append([data_in[j][0], avgs])
            data_out.append(alt_scores)
            
    #--------------------------------------------------------------------------
    # get Analytical Hierarchy Process (comparison) matricies from expert's scores (ratio of avgerages)
    #     data = [[critID, [alt1#, alt2#, alt3#, ...], 
    #                     [alt1/alt1, alt1/alt2, alt1/alt3, ...],
    #                     [alt2/alt1, alt2/alt2, alt2/alt3, ...],
    #                     ...]
    #             [critID, [alt1#, alt2#, alt3#, ...], 
    #                     [alt1/alt1, alt1/alt2, alt1/alt3, ...],
    #                     [alt2/alt1, alt2/alt2, alt2/alt3, ...],
    #                     ...]
    #           ]
    if out_type == 'AHP_CRISP':
        
        data_out = []
        for i in range(crit_count):             #for each criteria

            comp_matrix = [data_in[i][0]] #init comparison matrix with critID
            
            alt_nums = []
            for j in range(1,alt_count): alt_nums.append(data_in[i][j][0])     
            comp_matrix.append(alt_nums)
            
            for j in range(1,alt_count):        #for each alternative
                comp_row = []
                all_scores1 = data_in[i][j][1:len(data_in[i][j])]
                avg1 = sum([sum(x)/2.0 for x in all_scores1])/ \
                           len([sum(x) for x in all_scores1])    #get avg score of 1st alt
                for k in range(1,alt_count):
                    all_scores2 = data_in[i][k][1:len(data_in[i][k])]
                    avg2 = sum([sum(x)/2.0 for x in all_scores2])/ \
                              len([sum(x) for x in all_scores2])    #get avg score of 2nd alt
                              
                    if j == k: #catch for alt1 = alt2
                        comp_row.append(1.)
                    else:
                        ratio = scale_ratio((avg1/avg2), mm_ranges[i], out_range=out_mm)
                        comp_row.append(ratio)
                        
                comp_matrix.append(comp_row)
            data_out.append(comp_matrix)

    #--------------------------------------------------------------------------
    # get Analytical Hierarchy Process (comparison) matricies from expert's scores (ratio of avgerages)
    #     data = [[critID, [alt1#, alt2#, alt3#, ...], 
    #                     [alt1/alt1, alt1/alt2, alt1/alt3, ...],
    #                     [alt2/alt1, alt2/alt2, alt2/alt3, ...],
    #                     ...]
    #             [critID, [alt1#, alt2#, alt3#, ...], 
    #                     [alt1/alt1, alt1/alt2, alt1/alt3, ...],
    #                     [alt2/alt1, alt2/alt2, alt2/alt3, ...],
    #                     ...]
    #           ]
    if out_type == 'AHP_RANGE':
        data_out = []
        for i in range(crit_count):
            comp_matrix = [data_in[i][0]] #init comparison matrix with critID
            
            alt_nums = []
            for j in range(1,alt_count): alt_nums.append(data_in[i][j][0])     
            comp_matrix.append(alt_nums)
            
            for j in range(1,alt_count):
                comp_row = []
                all_scores1 = data_in[i][j][1:len(data_in[i][j])]
                avg_min1 = sum([x[0] for x in all_scores1])/ \
                           len([x[0] for x in all_scores1])  #get average minimum score
                avg_max1 = sum([x[1] for x in all_scores1])/ \
                           len([x[1] for x in all_scores1])  #get average maximum score
                for k in range(1,alt_count):
                    all_scores2 = data_in[i][k][1:len(data_in[i][k])]
                    avg_min2 = sum([x[0] for x in all_scores2])/ \
                               len([x[0] for x in all_scores2])  #get average minimum score                 
                    avg_max2 = sum([x[1] for x in all_scores2])/ \
                               len([x[1] for x in all_scores2])  #get average maximum score                      
                               
                    if j == k: #catch for alt1 = alt2
                        comp_row.append([1.,1.])
                    else:
                        ratios = [avg_min1/avg_max2, avg_max1/avg_min2]
                        for l in range(len(ratios)):
                            ratios[l] = scale_ratio(ratios[l], mm_ranges[i], out_range=out_mm)
                        comp_row.append(ratios)
                                   
                comp_matrix.append(comp_row)
            data_out.append(comp_matrix)

    #--------------------------------------------------------------------------
    # get Analytical Hierarchy Process (comparison) matricies from expert's scores (ratio of avgerages)
    #     data = [[critID, [alt1#, alt2#, alt3#, ...], 
    #                     [alt1/alt1, alt1/alt2, alt1/alt3, ...],
    #                     [alt2/alt1, alt2/alt2, alt2/alt3, ...],
    #                     ...]
    #             [critID, [alt1#, alt2#, alt3#, ...], 
    #                     [alt1/alt1, alt1/alt2, alt1/alt3, ...],
    #                     [alt2/alt1, alt2/alt2, alt2/alt3, ...],
    #                     ...]
    #           ]
    if out_type == 'AHP_RANGE_LARGE':
        data_out = []
        for i in range(crit_count):
            comp_matrix = [data_in[i][0]] #init comparison matrix with critID
            
            alt_nums = []
            for j in range(1,alt_count): alt_nums.append(data_in[i][j][0])     
            comp_matrix.append(alt_nums)
            
            for j in range(1,alt_count):
                comp_row = []
                all_scores1 = data_in[i][j][1:len(data_in[i][j])]
                avg_min1 = min([x[0] for x in all_scores1]) #get minimum of minimum scores
                avg_max1 = max([x[1] for x in all_scores1]) #get maximum of maximum score
                for k in range(1,alt_count):
                    all_scores2 = data_in[i][k][1:len(data_in[i][k])]
                    avg_min2 = min([x[0] for x in all_scores2]) #get average minimum score                 
                    avg_max2 = max([x[1] for x in all_scores2]) #get average maximum score                      
                               
                    if j == k: #catch for alt1 = alt2
                        comp_row.append([1.,1.])
                    else: 
                        ratios = [avg_min1/avg_max2, avg_max1/avg_min2]
                        for l in range(len(ratios)):
                            ratios[l] = scale_ratio(ratios[l], mm_ranges[i], out_range=out_mm)
                        comp_row.append(ratios)
                                   
                comp_matrix.append(comp_row)
            data_out.append(comp_matrix)


    #--------------------------------------------------------------------------
    # get Analytical Hierarchy Process (comparison) matricies from expert's scores (ratio of avgerages)
    #     data = [[critID, [alt1#, alt2#, alt3#, ...], 
    #                     [alt1/alt1, alt1/alt2, alt1/alt3, ...],
    #                     [alt2/alt1, alt2/alt2, alt2/alt3, ...],
    #                     ...]
    #             [critID, [alt1#, alt2#, alt3#, ...], 
    #                     [alt1/alt1, alt1/alt2, alt1/alt3, ...],
    #                     [alt2/alt1, alt2/alt2, alt2/alt3, ...],
    #                     ...]
    #           ]
    if out_type == 'AHP_FUZZ_TRI_1':
        data_out = []
        for i in range(crit_count):
            comp_matrix = [data_in[i][0]] #init comparison matrix with critID
            
            alt_nums = []
            for j in range(1,alt_count): alt_nums.append(data_in[i][j][0])     
            comp_matrix.append(alt_nums)
            
            for j in range(1,alt_count):
                comp_row = [] 
                all_scores1 = data_in[i][j][1:len(data_in[i][j])]
                scores1 = [sum([x[0] for x in all_scores1])/len(all_scores1), \
                           sum([sum(x)/2.0 for x in all_scores1])/ \
                           len([sum(x) for x in all_scores1]), \
                           sum([x[1] for x in all_scores1])/len(all_scores1)] 
                            #get fuzzy tri scores (min, avg, max) for 1st alt
                for k in range(1,alt_count):
                    all_scores2 = data_in[i][k][1:len(data_in[i][k])]
                    scores2 = [sum([x[0] for x in all_scores2])/len(all_scores2), \
                               sum([sum(x)/2.0 for x in all_scores2])/ \
                               len([sum(x) for x in all_scores2]), \
                               sum([x[1] for x in all_scores2])/len(all_scores2)] 
                                #get fuzzy tri scores (min, avg, max) for 2nd alt
                               
                    if j == k: #catch for alt1 = alt2
                        comp_row.append([1.,1.,1.])
                    else: 
                        ratios = fuzzy.divide_FuzzyTri(scores1,scores2)
                        for l in range(len(ratios)):
                            ratios[l] = scale_ratio(ratios[l], mm_ranges[i], out_range=out_mm)
                        comp_row.append(ratios)
                        
                        
                comp_matrix.append(comp_row)
            data_out.append(comp_matrix)
                        
    #--------------------------------------------------------------------------
    # get Analytical Hierarchy Process (comparison) matricies from expert's scores (ratio of avgerages)
    #     data = [[critID, [alt1#, alt2#, alt3#, ...], 
    #                     [alt1/alt1, alt1/alt2, alt1/alt3, ...],
    #                     [alt2/alt1, alt2/alt2, alt2/alt3, ...],
    #                     ...]
    #             [critID, [alt1#, alt2#, alt3#, ...], 
    #                     [alt1/alt1, alt1/alt2, alt1/alt3, ...],
    #                     [alt2/alt1, alt2/alt2, alt2/alt3, ...],
    #                     ...]
    #           ]
    if out_type == 'AHP_FUZZ_TRAP_1':
        data_out = []
        for i in range(crit_count):
            comp_matrix = [data_in[i][0]] #init comparison matrix with critID
            
            alt_nums = []
            for j in range(1,alt_count): alt_nums.append(data_in[i][j][0])     
            comp_matrix.append(alt_nums)
            
            for j in range(1,alt_count):
                comp_row = [] 
                all_scores1 = data_in[i][j][1:len(data_in[i][j])]
                scores1 = [min([x[0] for x in all_scores1]), 
                           sum([x[0] for x in all_scores1])/len(all_scores1), 
                           sum([x[1] for x in all_scores1])/len(all_scores1),
                           max([x[1] for x in all_scores1])]
                            #get fuzzy tri scores (min, avg, max) for 1st alt
                for k in range(1,alt_count):
                    all_scores2 = data_in[i][k][1:len(data_in[i][k])]
                    scores2 = [min([x[0] for x in all_scores2]), 
                               sum([x[0] for x in all_scores2])/len(all_scores2), 
                               sum([x[1] for x in all_scores2])/len(all_scores2),
                               max([x[1] for x in all_scores2])]
                                #get fuzzy tri scores (min, avg, max) for 2nd alt
                               
                    if j == k: #catch for alt1 = alt2
                        comp_row.append([1.,1.,1.])
                    else: 
                        ratios = fuzzy.divide_FuzzyTrap(scores1,scores2)
                        for l in range(len(ratios)):
                            ratios[l] = scale_ratio(ratios[l], mm_ranges[i], out_range=out_mm)
                        comp_row.append(ratios)                        
                        
                comp_matrix.append(comp_row)
            data_out.append(comp_matrix)
            
    #--------------------------------------------------------------------------
    # get Analytical Hierarchy Process (comparison) matricies from expert's scores (ratio of avgerages)
    #     data = [[critID, [alt1#, alt2#, alt3#, ...], 
    #                     [alt1/alt1, alt1/alt2, alt1/alt3, ...],
    #                     [alt2/alt1, alt2/alt2, alt2/alt3, ...],
    #                     ...]
    #             [critID, [alt1#, alt2#, alt3#, ...], 
    #                     [alt1/alt1, alt1/alt2, alt1/alt3, ...],
    #                     [alt2/alt1, alt2/alt2, alt2/alt3, ...],
    #                     ...]
    #           ]
    if out_type == 'AHP_FUZZ_TRAP_2':
        data_out = []
        for i in range(crit_count):
            comp_matrix = [data_in[i][0]] #init comparison matrix with critID
            
            alt_nums = []
            for j in range(1,alt_count): alt_nums.append(data_in[i][j][0])     
            comp_matrix.append(alt_nums)
            
            for j in range(1,alt_count):
                comp_row = [] 
                all_scores1 = data_in[i][j][1:len(data_in[i][j])]
                scores1 = [sum([x[0] for x in all_scores1])/len(all_scores1), 
                           max([x[0] for x in all_scores1]),
                           min([x[1] for x in all_scores1]),
                           sum([x[1] for x in all_scores1])/len(all_scores1)]
                scores1.sort()
                            #get fuzzy tri scores (min, avg, max) for 1st alt
                for k in range(1,alt_count):
                    all_scores2 = data_in[i][k][1:len(data_in[i][k])]
                    scores2 = [sum([x[0] for x in all_scores2])/len(all_scores2), 
                               max([x[0] for x in all_scores2]),
                               min([x[1] for x in all_scores2]),
                               sum([x[1] for x in all_scores2])/len(all_scores2)]
                    scores2.sort()
                                #get fuzzy tri scores (min, avg, max) for 2nd alt
                               
                    if j == k: #catch for alt1 = alt2
                        comp_row.append([1.,1.,1.])
                    else: 
                        ratios = fuzzy.divide_FuzzyTrap(scores1,scores2)
                        for l in range(len(ratios)):
                            ratios[l] = scale_ratio(ratios[l], mm_ranges[i], out_range=out_mm)
                        comp_row.append(ratios)  
                        
                comp_matrix.append(comp_row)
            data_out.append(comp_matrix)
            
    #--------------------------------------------------------------------------
    # get Analytical Hierarchy Process (comparison) matricies from expert's scores (ratio of avgerages)
    #     data = [[critID, [alt1#, alt2#, alt3#, ...], 
    #                     [alt1/alt1, alt1/alt2, alt1/alt3, ...],
    #                     [alt2/alt1, alt2/alt2, alt2/alt3, ...],
    #                     ...]
    #             [critID, [alt1#, alt2#, alt3#, ...], 
    #                     [alt1/alt1, alt1/alt2, alt1/alt3, ...],
    #                     [alt2/alt1, alt2/alt2, alt2/alt3, ...],
    #                     ...]
    #           ]
    if out_type == 'AHP_FUZZ_TRAP_UNI': #(use a box trapezoidal score [avg min, avg min, avg max, avg, max] to mimic the uniform probabilistic
        data_out = []
        for i in range(crit_count):
            comp_matrix = [data_in[i][0]] #init comparison matrix with critID
            
            alt_nums = []
            for j in range(1,alt_count): alt_nums.append(data_in[i][j][0])     
            comp_matrix.append(alt_nums)
            
            for j in range(1,alt_count):
                comp_row = [] 
                all_scores1 = data_in[i][j][1:len(data_in[i][j])]
                scores1 = [sum([x[0] for x in all_scores1])/len(all_scores1), 
                           sum([x[0] for x in all_scores1])/len(all_scores1),
                           sum([x[1] for x in all_scores1])/len(all_scores1),
                           sum([x[1] for x in all_scores1])/len(all_scores1)]
                scores1.sort()
                            #get fuzzy tri scores (min, avg, max) for 1st alt
                for k in range(1,alt_count):
                    all_scores2 = data_in[i][k][1:len(data_in[i][k])]
                    scores2 = [sum([x[0] for x in all_scores2])/len(all_scores2), 
                               sum([x[0] for x in all_scores2])/len(all_scores2),
                               sum([x[1] for x in all_scores2])/len(all_scores2),
                               sum([x[1] for x in all_scores2])/len(all_scores2)]
                    scores2.sort()
                                #get fuzzy tri scores (min, avg, max) for 2nd alt
                               
                    if j == k: #catch for alt1 = alt2
                        comp_row.append([1.,1.,1.])
                    else: 
                        ratios = fuzzy.divide_FuzzyTrap(scores1,scores2)
                        for l in range(len(ratios)):
                            ratios[l] = scale_ratio(ratios[l], mm_ranges[i], out_range=out_mm)
                        comp_row.append(ratios)  
                        
                comp_matrix.append(comp_row)
            data_out.append(comp_matrix)
    #--------------------------------------------------------------------------
    #--------------------------------------------------------------------------
    #--------------------------------------------------------------------------    
    
    return data_out    
    
if __name__ == "__main__":
    
    #TESTING:
    print "Testing... GO"
    qual_data, quant_data = read_data('alt_evals_data_thesis_final.csv', 10)
    
    data1 = reduce_data(qual_data,'AVG')
    data2 = reduce_data(quant_data,'AVG_RANGE')
    data3 = reduce_data(qual_data,'FUZZ_TRI_1')
    data4 = reduce_data(qual_data,'FUZZ_TRAP_1')  
    data41 = reduce_data(qual_data,'FUZZ_TRAP_2')  
    data5 = reduce_data(quant_data, 'AHP_CRISP')
    data51 = reduce_data(quant_data, 'AHP_RANGE')
    data52 = reduce_data(quant_data, 'AHP_RANGE_LARGE')
    data6 = reduce_data(qual_data, 'AHP_FUZZ_TRI_1')
    data6 = reduce_data(qual_data, 'AHP_FUZZ_TRAP_1')
    data61 = reduce_data(qual_data, 'AHP_FUZZ_TRAP_2')
    
    for d in data2: 
        for d1 in d: print d1
    

    
