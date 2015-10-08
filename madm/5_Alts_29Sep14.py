# -*- coding: utf-8 -*-
"""
Created on Tue Sep 23 21:58:33 2014

@author: frankpatterson
"""
#!/usr/bin/env python
import numpy as np
import scipy as sp
import scipy.stats

import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
from matplotlib.legend_handler import HandlerLine2D

from itertools import cycle
import crisp_TOPSIS as crisp_methods
import fuzzy_topsis


def mean_confidence_interval(data, confidence=0.90):
    a = 1.0*np.array(data)
    n = len(a)
    m, se = np.mean(a), np.std(a)
    h = se * sp.stats.t._ppf((1+confidence)/2., n-1)
    return m, m-h, m+h

############## 100K Runs of TOPSIS ###############
'''
#100K Iterations
#RANK COUNTER
rank_counter = [[7717, 11954, 15417, 25475, 39437],[1396, 6346, 15249, 32763, 44246],[50237, 25104, 15828, 7156, 1675],[20410, 25218, 24690, 19673, 10009],[20240, 31378, 28816, 14933, 4633] ]

#QUANTILES:
quantiles = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0]
#QUANTILE COUNTER
q_counter = [[344, 1191, 3152, 5728, 8046, 9416, 10348, 10972, 10645, 9459, 8097, 6662, 5590, 4526, 3066, 1748, 725, 237, 42, 6],[486, 1054, 2444, 4839, 7945, 11041, 13178, 13727, 13463, 12615, 10089, 5964, 2461, 581, 92, 20, 1, 0, 0, 0],[1, 0, 5, 45, 141, 495, 1060, 2112, 3463, 5603, 8029, 10694, 12954, 13680, 14259, 13779, 9366, 3518, 715, 81],[0, 0, 0, 33, 232, 1141, 3559, 7939, 11889, 13723, 13226, 11963, 10684, 9091, 6962, 4671, 2787, 1285, 565, 250],[0, 3, 46, 135, 407, 1120, 2437, 4772, 8292, 11620, 14353, 15199, 14160, 11457, 7876, 4523, 2308, 961, 284, 47]]
#Sample PDFs
PDFs = [[0.00344, 0.01191, 0.03152, 0.05728, 0.08046, 0.09416, 0.10348, 0.10972, 0.10645, 0.09459, 0.08097, 0.06662, 0.0559, 0.04526, 0.03066, 0.01748, 0.00725, 0.00237, 0.00042, 6e-05],[0.00486, 0.01054, 0.02444, 0.04839, 0.07945, 0.11041, 0.13178, 0.13727, 0.13463, 0.12615, 0.10089, 0.05964, 0.02461, 0.00581, 0.00092, 0.0002, 1e-05, 0.0, 0.0, 0.0],[1e-05, 0.0, 5e-05, 0.00045, 0.00141, 0.00495, 0.0106, 0.02112, 0.03463, 0.05603, 0.08029, 0.10694, 0.12954, 0.1368, 0.14259, 0.13779, 0.09366, 0.03518, 0.00715, 0.00081],[0.0, 0.0, 0.0, 0.00033, 0.00232, 0.01141, 0.03559, 0.07939, 0.11889, 0.13723, 0.13226, 0.11963, 0.10684, 0.09091, 0.06962, 0.04671, 0.02787, 0.01285, 0.00565, 0.0025],[0.0, 3e-05, 0.00046, 0.00135, 0.00407, 0.0112, 0.02437, 0.04772, 0.08292, 0.1162, 0.14353, 0.15199, 0.1416, 0.11457, 0.07876, 0.04523, 0.02308, 0.00961, 0.00284, 0.00047]]
'''

#rel_closeness, rankings = alt_crisp_TOPSIS("alternative_data.txt", 'C', 'C', 0)
#print rel_closeness
#print rankings


n = 50000
quants = 50
rank_counter, quantiles, q_counter, PDFs, full_RCs = crisp_methods.prob_TOPSIS_uniform("alternative_data.txt", n, quants)

#run fuzzy TOPSIS
alt_IDs, alphas, FT_RCs = fuzzy_topsis.alt_fuzzy_TOPSIS_2("alternative_data.txt", 'FT', 'FT')
alt_IDs, alphas, FZ_RCs = fuzzy_topsis.alt_fuzzy_TOPSIS_2("alternative_data.txt", 'FZ', 'FZ')
defuzz_FT = [(sum([(x[0]+x[1])/2 for x in alt])/len(alt)) for alt in FT_RCs]
defuzz_FZ = [(sum([(x[0]+x[1])/2 for x in alt])/len(alt)) for alt in FZ_RCs]

"""
normalize the PDFs
"""
norm_PDFs = []
for i in range(len(PDFs)):
    m = max(PDFs[i])
    n = [x/m for x in PDFs[i]]
    norm_PDFs.append(n)


#plotting parameters
labels = ['Alt ' + x for x in alt_IDs] 
params = {'legend.fontsize': 12, 'axes.labelsize': 12}
plt.rcParams.update(params)

#compare intervals at:    
alpha = 0.75        #confidence interval
alpha_c = 0.75      #alpha cuts

CIs = []
for i in range(len(full_RCs)):
    m, cl, cu = mean_confidence_interval(full_RCs[i], confidence=alpha)
    CIs.append([m, cl, cu])
print '% CIs:'
for CI in CIs: print CI
 

colors = plt.get_cmap('Set1')(np.linspace(0, 1.0, len(q_counter)))
lines = ['-']#["-","--","-.",":"]
linecycler = cycle(lines)
   
#compare 90% CIs with 0.9 alpha cuts
plt.figure()
for i in range(len(CIs)): 
    dat1, = plt.plot(CIs[i][1:], [i+0.8,i+0.8], '-|', color='blue')
    #plt.text(CIs[i][2]+0.03, i+0.83, 'Range:' + str(CIs[i][2] - CIs[i][1])[0:4], fontsize=9 )
    dat2, = plt.plot(CIs[i][0], [i+0.8], 'o', color='blue')
    plt.text(CIs[i][0]-0.02, i+0.88, str(CIs[i][0])[0:4], fontsize=12 )


print 'Alpha Cuts'
for j in range(len(alphas)): 
    if alphas[j] == alpha_c: break #get index for 0.9 alpha cut
for i in range(len(FT_RCs)):
    print FT_RCs[i][j]
    dat3, = plt.plot(FT_RCs[i][j],[i+1.2,i+1.2], '-h', color='red')
    #plt.text(FT_RCs[i][j][0]-0.13, i+1.13, 'Range:' + str(FT_RCs[i][j][1]-FT_RCs[i][j][0])[0:4], fontsize=9 )
    dat4, = plt.plot(defuzz_FT[i],i+1.2, 'x', color='red')
    plt.text(defuzz_FT[i]-0.02, i+1.28, str(defuzz_FT[i])[0:4], fontsize=12 )
    
    
    
#plt.title('Comparison of ' + str(alpha*100.0) + '% confidence intervals with ' + str(alpha_c) + ' alpha cuts')
plt.xlabel('Relative Closeness')
plt.ylabel('Alternatives')
#plt.grid(color='k', linestyle='-')
plt.axis([0.0,1.0,0,(len(CIs)+1)])
labels1 = [str(alpha*100.0) + '% CI', 'Mean', str(alpha_c) + ' Alpha-cut', 'Defuzzified Centriod']
handles=[dat1, dat2, dat3, dat4]
plt.legend(handles, labels1, loc=2, numpoints=1)


    
    
'''
print "RANK COUNTER"
for r in rank_counter: print r, sum(r)
print ""
print "QUANTILES:"
print quantiles
print "QUANTILE COUNTER"
for q in q_counter: print q, sum(q)
print "Sample PDFs"
for p in PDFs: print p, sum(p)
''' 


plt.figure()
plt.xlabel('Relative Closeness')
plt.ylabel('Frequency')
#plt.title('Relative Closeness Histogram')
for i in range(len(q_counter)): plt.plot(quantiles, q_counter[i], next(linecycler), \
                                         linewidth=2, label='ALT ' + str(i+1), color=colors[i] )
plt.legend(labels, loc=2)


plt.figure()
plt.xlabel('Relative Closeness')
plt.ylabel('Sample Probability')
plt.title('Relative Closeness Distribution')
for i in range(len(PDFs)): plt.plot(quantiles, PDFs[i], next(linecycler), \
                                    linewidth=2, label='ALT ' + str(i+1), color=colors[i] )
plt.legend(labels, loc=2)


plt.figure()
plt.xlabel('Rank')
plt.ylabel('Frequency')
plt.title('Rank Histogram')
ranks = [i+1 for i in range(len(rank_counter[0]))]
#print ranks
for i in range(len(rank_counter)): plt.plot(ranks, rank_counter[i], next(linecycler), \
                                            linewidth=2, label='ALT ' + str(i+1), color=colors[i] )
plt.legend(labels, loc=2)




fig = plt.figure()
a = fig.add_subplot(111)  
for alt in FZ_RCs:
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

fig = plt.figure()
a = fig.add_subplot(111)  
for alt in FT_RCs:
    x, y = [], []    
    for i in range(len(alphas)):
        x.append(alt[i][0])
        y.append(alphas[i])
    rev = range(len(alphas))
    rev.reverse()
    for i in rev:
        x.append(alt[i][1])
        y.append(alphas[i])
    a.plot(x,y, linewidth=2.0, color=colors[FT_RCs.index(alt)])

#a.set_title('Fuzzy TOPSIS')
a.set_ylabel('Membership')
a.set_xlabel('Relative Closeness')
a.axis([0,1,0,1])
labels = ['Alt ' + x for x in alt_IDs] 
plt.legend(labels, loc=2)


plt.figure()
plt.xlabel('Relative Closeness')
plt.ylabel('Sample Probability')
plt.title('Relative Closeness Distribution')
for i in range(len(PDFs)): plt.plot(quantiles, norm_PDFs[i], next(linecycler), \
                                    linewidth=2, label='ALT ' + str(i+1), color=colors[i] )
plt.legend(labels, loc=2)




plt.show()
    
        