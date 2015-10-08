# -*- coding: utf-8 -*-
"""
Created on Sun Feb  8 10:53:42 2015

@author: frankpatterson
"""

import alt_data_reader as data_reader
import crisp_AHP as AHP
import fuzzy_AHP as fAHP
import fuzzy as fuzz

import numpy as np
import scipy as sp
#import scipy.stats
from scipy.stats import norm


import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib as mpl
from pylab import *
plt.ioff()


from itertools import cycle

########### Change Defaults for Plots ###########
#font = {'family':'sans-serif'}
#mpl.rc('font', **font)
mpl.rc('text', usetex=False)

########### CI Function ###########
print '##################### RUNNING AHP COMPARISON #####################'

def confidence_interval(data, confidence=0.90):
    a = 1.0*np.array(data)
    n = len(a)
    m, se = np.mean(a), np.std(a)
    h = se * sp.stats.t._ppf((1+confidence)/2., n-1)
    return m, m-h, m+h

alt_IDs = ['Tlt_Rot', 'Tlt_Wing', 'FiW_TJ', 'Tlt_FiW', 'Stop_Rot', 'Auto_Gy', 'Tail_Sit', 'FiW_Push', 'Helipln', 'FiB_TD' ]
alt_IDs = ['Alt. 1','Alt. 2','Alt. 3','Alt. 4','Alt. 5','Alt. 6','Alt. 7','Alt. 8', 'Alt. 9', 'Alt. 10']

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
########### Get Data ###########
#qual_data, quant_data = data_reader.read_data('alt_evals_data.csv', 10)  
#qual_data2, quant_data2 = data_reader.read_data('alt_evals_data_real.csv', 10)
qual_data2, quant_data2 = data_reader.read_data('alt_evals_data_thesis_final_31aug15.csv', 10)  
data_avgRange = data_reader.reduce_data(quant_data2, 'AHP_RANGE')
data_tri = data_reader.reduce_data(quant_data2, 'AHP_FUZZ_TRI_1')
data_trap = data_reader.reduce_data(quant_data2, 'AHP_FUZZ_TRAP_UNI')

########### Run Monte Carlo TOPSIS ###########
alts = 10                   #number of alternatives
n = 10000                    #number of iterations
quants = 70                 #quantiles
top_N = 4                   #top alts to plot and consider
create_plots = True        #flag to create plots

rank_counter, full_ranks, quantiles, q_counter, norm_fits, full_Ps = \
    AHP.prob_AHP_uniform(data_tri, weights_tri, n, alts, quants, 'T')

########### Run Fuzzy AHP ###########
#compare intervals at:    
alpha = 0.9        #confidence interval
alpha_c = 0.9      #alpha cuts

#alt_IDs, alphas, P_cuts = fTOPSIS.alt_fuzzy_TOPSIS_2(data_tri, weights_tri, 'FT', 'FT', plot_RCs=0)
results = fAHP.alt_fuzzy_AHP2(data_tri, weights_tri, 'FT', 0)

suffix = '_tri'

########### Data Reduction ###########
alphas = [float(n1)/(21-1) for n1 in range(21)]       #create 21 alpha cuts
P_cuts = [[fuzz.getAlphaCut(r[1],a) for a in alphas] for r in results]

#create fuzzy RC plots 
P_plots = []  #organize RC plots as [[alt1_x, alt1_y], [alt2_x, alt2_y], ...]
for alt in P_cuts:
    x1 = [a[0] for a in alt]
    x2 = [alt[len(alphas)-1-i][1] for i in range(len(alphas))]
    y1 = alphas
    y2 = [alphas[len(alphas)-1-i] for i in range(len(alphas))]
    P_plots.append([x1+x2, y1+y2])

#get fuzzy centroids and rank them
P_centroids = [[ID] for ID in alt_IDs]
for i in range(len(P_cuts)):
    C_area = sum([(c[1]-c[0])*0.5*(c[1] + c[0]) for c in P_cuts[i]])
    P_centroids[i].append(C_area/sum([(c[1]-c[0]) for c in P_cuts[i]]))
fuzzCent_rankings = AHP.getCrispRanks([P[1] for P in P_centroids])
for i in range(len(P_centroids)): P_centroids[i].append(fuzzCent_rankings[i])
    
#get mean of alpha cut at 1.0 and rank them
i1 = alphas.index(1.0)
fuzz1means = [sum(cuts[i1])/len(cuts[i1]) for cuts in P_cuts]
fuzz1_ranks = AHP.getCrispRanks(fuzz1means)
    
P_UC_meas1 = [[ID] for ID in alt_IDs] #capture uncertainty as measured by ratio of 0.1 level support to value at m=1.0
for i in range(len(P_UC_meas1)):
    P_UC_meas1[i].append((P_cuts[i][alphas.index(0.1)][1] -  - P_cuts[i][alphas.index(0.1)][0])/ \
                         P_cuts[i][alphas.index(1.0)][1])

#get & diplay dominance (possibility) matrix
DM = [[a] for a in alt_IDs];
for i in range(alts):
    DM_row = []
    for j in range(alts):
        d = fuzz.dominance_AlphaCut(alphas, P_cuts[i], alphas, P_cuts[j]) 
            #get dominance of alt i over alt j
        DM_row.append(d)
    DM[i].append(DM_row)

print '\n'
print 'Fuzzy Dominance Matrix'
for i in range(len(DM)): 
    print DM[i][0], ':', [str(x)[0:5] for x in DM[i][1]], ':',  str(min(DM[i][1]))[0:5], ':', str(sum(DM[i][1])/len(DM[i][1]))[0:5]

#rank monte carlo fit means
prob_ranks = AHP.getCrispRanks([nf[0] for nf in norm_fits])
#get monte carlo CIs
CIs = []
for i in range(len(full_Ps)):
    m, cl, cu = confidence_interval(full_Ps[i], confidence=alpha)
    CIs.append([m, cl, cu])

#get probability of top N
prob_rankedN = [[a] for a in alt_IDs] 
    #list of alt lists, each item is probability of alt being ranked <= i+1
for i in range(alts):
    prob_rankedn_row = [0.0 for a in range(alts)]
    for j in range(alts):
        prob_rankedn_row[j] = \
            float(len([x[i] for x in full_ranks if x[i] <= j+1]))/len(full_ranks)
                #get probability that alt i rank is <= j+1
    prob_rankedN[i].append(prob_rankedn_row)
    
########### DISPLAY AGGREGATE RESULTS ###########
print '\n'
print "ALTERNATIVE: FUZZY CENTROID : RANKING"    
for i in range(len(P_centroids)): 
    print 'Alternative', P_centroids[i][0], ': centroid =',  P_centroids[i][1], \
          'rank =', P_centroids[i][2], '(',fuzz1_ranks[i],')', ' UC_meas1 =', P_UC_meas1[i][1]

print "\n"  
print "ALTERNATIVE: MEAN : STDDEV : RANK :", alpha, "CI"
for i in range(len(norm_fits)): 
    print 'Alternative', alt_IDs[i], ': mu =',  round(norm_fits[i][0],4), ' std =', \
          round(norm_fits[i][1],4), 'rank =', prob_ranks[i], ' CI =', \
          round(CIs[i][1],3), '-', round(CIs[i][2],3)

print "\n"
print "Prob of each Alt being Ranked n"
for i in range(len(prob_rankedN)):
    print 'Alt', prob_rankedN[i][0], ':', prob_rankedN[i][1:]
    
########### REDUCE TOP N ALTS ###########
#select top N alternatives by mu
top_Alts = [i for i in range(len(prob_ranks)) if prob_ranks[i] <= top_N]
    #get indecies for top N altneratives
print top_Alts

#get results for top 5 alts
TOP_P_plots = [P_plots[i] for i in range(len(P_plots)) if i in top_Alts]
TOP_norm_fits = [norm_fits[i] for i in range(len(norm_fits)) if i in top_Alts]


########### CREATE NORMALIZED PDFS ###########
PDF_x = np.linspace(0.0, 1.0, 500)
norm_PDFs = []
for i in range(len(TOP_norm_fits)):
    PDF_y = sp.stats.norm.pdf(PDF_x, TOP_norm_fits[i][0], TOP_norm_fits[i][1])
    norm_PDFs.append([y/max(PDF_y) for y in PDF_y])


########### PLOT RESULTS ###########
#setup colors for plots
colors =  ['b','g','r', 'k', 'm', 'c', 'darkgreen', 'pink', 'wheat', 'violet']#['k' for i in range(alts)] 
greys =   ['#000000', '#303030', '#606060', '#909090', '#C0C0C0',]# '#F0F0F0']
greycycler = cycle(greys)
lines =   ['-' for i in range(alts)]#["-","--","-.",":"]
linecycler = cycle(lines)
markers = ["o", "^", "s", "x", "d", "+"]
markercycler = cycle(markers)
hatches = ['+' , 'x' , 'o', '.', '\\' , '+' , '*' , 'O' , '.' , '*', '|']                                                                                        
hatchcycler = cycle(hatches)

#PLOT FUZZY RESULTS
if create_plots:
    with PdfPages('results_AHP.pdf') as pdf:
        
        #FUZZY RESULTS
        linecycler = cycle(lines)
        markercycler = cycle(markers)  
        plt.figure(figsize=(8, 2.5), dpi=160,)
        ax = plt.subplot(1,1,1)
        plt.xlabel('Relative Closeness')
        plt.ylabel('Membership')
        axlines = [[],[]]
        for i in range(len(TOP_P_plots)): 
            p1, = plt.plot(TOP_P_plots[i][0], TOP_P_plots[i][1], next(linecycler), linewidth=2, \
                     label=str(alt_IDs[top_Alts[i]]), color=colors[i] )
            axlines[0].append(p1)
            p2, = plt.plot(P_centroids[top_Alts[i]][1], 0.0, marker=next(markercycler), ms=8.0,
                           lw=0.1, clip_on=False, color='k')
            axlines[1].append(p2)
        plt.legend([(axlines[0][i], axlines[1][i]) for i in range(len(axlines[0]))],
                   [alt_IDs[tA] for tA in top_Alts], loc=2, handlelength=4)
        plt.savefig('saved_figures/fuzz_AHPbest'+suffix+'.png', bbox_inches='tight')
        pdf.savefig()
        
        #PLOT CRISP RESULTS (PDF & CDF)
        linecycler = cycle(lines)
        markercycler = cycle(markers)  
        plt.figure(figsize=(8, 2.5), dpi=160,)
        ax = plt.subplot(1,1,1)
        plt.xlabel('Priority')
        plt.ylabel('f(x)')
        axlines = [[],[]]
        for i in range(len(TOP_norm_fits)): 
            p = sp.stats.norm.pdf(PDF_x, TOP_norm_fits[i][0], TOP_norm_fits[i][1])
            p1, = plt.plot(PDF_x, p, next(linecycler), linewidth=2, \
                     label=str(alt_IDs[top_Alts[i]]), color=colors[i] )
            axlines[0].append(p1)
            p2, = ax.plot(TOP_norm_fits[i][0], 0.0, marker=next(markercycler), 
                          lw=0.1, ms=8.0, clip_on=False, color='k')
            axlines[1].append(p2)
        plt.legend([(axlines[0][i], axlines[1][i]) for i in range(len(axlines[0]))], 
                   [alt_IDs[tA] for tA in top_Alts], loc=2, handlelength=4, prop={'size':10})  
        plt.xlim([0.1, 0.7])    
        plt.savefig('saved_figures/pdf_AHPbest'+suffix+'.png', bbox_inches='tight')
        pdf.savefig()    
                
            
        plt.figure()
        linecycler = cycle(lines)
        markercycler = cycle(markers)
        plt.xlabel('Priority')
        plt.ylabel('F(x)')
        #plt.title('MC: Cumulative Distribution')
        for i in range(len(TOP_norm_fits)): 
            p = sp.stats.norm.cdf(PDF_x, TOP_norm_fits[i][0], TOP_norm_fits[i][1])
            plt.plot(PDF_x, p, next(linecycler), linewidth=2, \
                     label=str(alt_IDs[top_Alts[i]]), color=colors[i] )
        plt.legend(loc=0, prop={'size':10})    
        plt.xlim([0.1, 0.7])   
        plt.savefig('saved_figures/cdf_AHPbest'+suffix+'.png', bbox_inches='tight')
        pdf.savefig()
            
            
        #PLOT NORMALIZED PDFs
        plt.figure()
        plt.xlabel('Priority')
        plt.ylabel('Sample Probability')
        plt.title('MC: Priority Distribution')
        for i in range(len(TOP_norm_fits)): 
            plt.plot(PDF_x, norm_PDFs[i], next(linecycler), linewidth=1.5, \
                     label=str(alt_IDs[top_Alts[i]])+'(PDF)', color='#757575' )
        for i in range(len(TOP_P_plots)): 
            plt.plot(TOP_P_plots[i][0], TOP_P_plots[i][1], next(linecycler), linewidth=1.5, \
                     label=str(alt_IDs[top_Alts[i]])+'(MF)', color=colors[i] )
        plt.legend(loc=0, prop={'size':10})    
        #plt.xlim([0.0, 0.4]) 
        pdf.savefig()
    
        #Compare CIs and Alpha-Cuts
        #compare 90% CIs with 0.9 alpha cuts
        fig, ax = plt.subplots()
        j = 0
        for i in top_Alts: 
            dat1, = ax.plot(CIs[i][1:], [j+0.8,j+0.8], '--|', color='k', lw=2.0)
            dat2, = ax.plot(CIs[i][0], [j+0.8], 'o', color='k')
            plt.text(CIs[i][0]-0.02, j+0.88, str(CIs[i][0])[0:4], fontsize=12 )
            j = j+1
        for ac_i in range(len(alphas)): 
            if alphas[ac_i] == alpha_c: break #get index for 0.9 alpha cut
        j = 0
        for i in top_Alts:
            dat3, = ax.plot(P_cuts[i][ac_i], [j+1.2,j+1.2], '-|', color='k', lw=2.0)
            dat4, = ax.plot(P_centroids[i][1], j+1.2, 'x', color='k')
            ax.text(P_centroids[i][1]-0.02, j+1.28, str(P_centroids[i][1])[0:4], fontsize=12 )
            j = j+1
        ax.set_xlabel('Priority')
        ax.set_ylabel('Alternatives')
        ax.set_yticklabels(['']+[i+1 for i in top_Alts])
        #ax.set_xlim([0.0,0.7])
        ax.set_ylim([0,(len(top_Alts)+1)])
        labels1 = [str(alpha*100.0) + '% CI', 'Mean', str(alpha_c) + ' Alpha-cut', 'Defuzzified Centriod']
        handles=[dat1, dat2, dat3, dat4]
        plt.legend(handles, labels1, loc='upper center', bbox_to_anchor=(0.5, 1.05),
          ncol=2, fancybox=True, shadow=True, numpoints=1, prop={'size':10})
        ax.grid(b=True, which='major', axis='x', color='k', linestyle='--', lw='0.5')
        plt.savefig('saved_figures/compare_AHPbest'+suffix+'.png', bbox_inches='tight')
        pdf.savefig()
        
        #PLOT RANK HISTOGRAMS
        fig = plt.figure()
        ax = fig.add_subplot(111)
        top_Ranks = [rank_counter[i] for i in range(len(rank_counter)) if i in top_Alts]
        ind = np.arange(len(top_Ranks[0]))     #x locations for groups
        width = 0.15                             #wdith of bars
        for i in range(len(top_Ranks)):
            rects = ax.bar(ind+(width*i), top_Ranks[i], width, color=colors[i], hatch=next(hatchcycler))
            lines = ax.plot(ind+(width*i), top_Ranks[i], next(linecycler), color=colors[i], lw=2.0)
        ax.set_xlim(-width, len(ind)+width)
        ax.set_xticks(ind+3*width)
        ax.set_xticklabels([str(i) for i in range(1,len(top_Ranks[0])+1)])
        #for tick in ax.xaxis.get_major_ticks(): tick.label.set_fontsize(10)
        plt.xlabel('Ranks')
        plt.ylabel('Frequency')
        #plt.title('Rank Histogram (Top 5 Alternatives)')
        plt.legend([alt_IDs[i] for i in range(len(alt_IDs)) if i in top_Alts])
        plt.savefig('saved_figures/rankHist_AHPbest'+suffix+'.png', bbox_inches='tight')
        pdf.savefig() 

        #Plot Probability of top N with N        
        plt.figure()
        for i in range(len(prob_rankedN)):
            plt.plot([N+1 for N in range(len(prob_rankedN[i][1]))], prob_rankedN[i][1], 
                     next(linecycler),marker=next(markercycler), color=colors[i], lw=2.0)
        plt.xlabel('N')
        plt.ylabel('Probability of Being Ranked in Top N')
        plt.legend(['Alt.' + str(int(alt_IDs.index(pt[0])+1)) for pt in prob_rankedN], loc='center right', 
                   bbox_to_anchor=(1.25, 0.5), ncol=1, handlelength=3)   
        plt.savefig('saved_figures/probTopN_AHP'+suffix+'.png', bbox_inches='tight')
        
        #PLOT DOMINANCE
        DM_min = min([min(x[1]) for x in DM])-0.1
        plt.figure()
        ax = plt.subplot(1,1,1)
        for i in range(len(DM)):
            #pp = ax.plot([N+1 for N in range(len(DM[i][1]))], DM[i][1], next(linecycler), 
            #             clip_on=False, marker=next(markercycler), color='k', lw=2.0)
            for j in range(len(DM[i][1])):
                fc = plt.get_cmap('gray')((DM[i][1][j] - DM_min)/(1.0 - DM_min))   #get a facecolor
                ax.add_patch(Rectangle((j+1, alts-i),1,1,facecolor=fc, lw=1.0)) #color block
                if i <> j: ax.text(j+1.5, alts-i+0.5, str(DM[i][1][j])[0:4], {'ha':'center', 'va':'center'}, fontsize=13)
                else: ax.text(j+1.5, alts-i+0.5, '-', {'ha':'center', 'va':'center'}, fontsize=13)
            ax.text(alts+1.5, alts-i+0.5, str(min(DM[i][1]))[0:4], {'ha':'center', 'va':'center'}, fontsize=13, weight='bold')
            ax.add_patch(Rectangle((alts+1, alts-i),1,1,facecolor=plt.get_cmap('gray')((min(DM[i][1]) - DM_min)/(1.0 - DM_min)), lw=1.0))          
            ax.text(alts+2.5, alts-i+0.5, str(sum(DM[i][1])/len(DM[i][1]))[0:4], {'ha':'center', 'va':'center'}, fontsize=13, weight='bold')
            ax.add_patch(Rectangle((alts+2, alts-i),1,1,facecolor=plt.get_cmap('gray')((sum(DM[i][1])/len(DM[i][1]) - DM_min)/(1.0 - DM_min)), lw=1.0))
            ax.add_patch(Rectangle((11,1),2,10, facecolor='none', lw=3.0))
        ax.set_xlim([1.,alts+3.])
        ax.set_ylim([1.,alts+1.])
        ax.xaxis.set_label_position('top')
        ax.xaxis.tick_top()
        ax.set_xticks([x+0.5 for x in range(1,alts+3)])
        ax.set_xticklabels(([x+1 for x in range(len(alt_IDs))]+['Min.', 'Avg.']), fontsize=11)
        ax.xaxis.set_tick_params(size=0)
        ax.set_yticklabels([x+1 for x in range(len(alt_IDs))[::-1]], fontsize=11)
        ax.set_yticks([x+0.5 for x in range(1,alts+1)])
        ax.yaxis.set_tick_params(size=0)
        ax.set_xlabel('Alternative')
        ax.set_ylabel('Alternative')
        #p = ax.legend([pt[0] for pt in prob_rankedN], loc='center right',  
        #              bbox_to_anchor=(1.25, 0.5), ncol=1, handlelength=3)
        plt.savefig('saved_figures/dominance_AHP'+suffix+'.png', bbox_inches='tight')


################## CHECK SENSITIVITY/RANK REVERSAL ###################
print "Performing Sensitivity Study..."
org_alts = alts
N_max = alts - top_N                 #max number of designs to omit

print alt_IDs
fuzzy_dom_top_N = [[a] for a in alt_IDs]     
        #get total fuzzy dominance of each alternative 
        #(possiblitity of being the best)
for i in range(len(fuzzy_dom_top_N)): 
    for j in range(len(DM)): 
       if fuzzy_dom_top_N[i][0] == DM[j][0]: fuzzy_dom_top_N[i].append(min(DM[j][1]))
           #append each original dominance

fuzzy_rank_top_N = [[alt_IDs[i], fuzzCent_rankings[i]] for i in range(len(fuzzCent_rankings))]
        #get rank at each iteration

prob_top_N = [[a] for a in alt_IDs]
for i1 in range(len(prob_top_N)): prob_top_N[i1].append(prob_rankedN[i1][1][top_N-1])
    #each list (index i) is a list of probabilities (index j) that alt i is in 
    #top N alternatives with j+1 of the worst alternatives removed

prob_ranks_org = prob_ranks
for k in range(N_max):
    print '\n'
    print 'Removing worst Alt,', k+1, 'total removed'
    worst_index = prob_ranks_org.index(max(prob_ranks_org)) #get worst ranked (prob) remaining design of original 10
    print 'Alternative', alt_IDs[worst_index], 'removed'    
    #remove instances of worst from ranks and data
    prob_ranks_org.pop(worst_index)
    
    """
    for i in range(len(qual_data)): #remove worst alt from original data
        qual_data[i].pop(worst_index)
        quant_data[i].pop(worst_index)
        qual_data2[i].pop(worst_index)
        quant_data2[i].pop(worst_index)
    #rebuild AHP matrices        
    data_avgRange = data_reader.reduce_data(quant_data2, 'AHP_RANGE')
    data_tri = data_reader.reduce_data(quant_data2, 'AHP_FUZZ_TRI_1')
    data_trap = data_reader.reduce_data(quant_data2, 'AHP_FUZZ_TRAP_2')
    """
    for i1 in range(len(data_avgRange)):
        data_avgRange[i1].pop(worst_index+2)
        for j1 in range(1, len(data_avgRange[i1])): data_avgRange[i1][j1].pop(worst_index)
        data_tri[i1].pop(worst_index+2)
        for j1 in range(1, len(data_tri[i1])): data_tri[i1][j1].pop(worst_index)
        data_trap[i1].pop(worst_index+2)
        for j1 in range(1, len(data_trap[i1])): data_trap[i1][j1].pop(worst_index)
        
    
    alt_IDs.pop(worst_index)
    alts = alts - 1

    ########### Run Monte Carlo TOPSIS ###########
    rank_counter, full_ranks, quantiles, q_counter, norm_fits, full_Ps = \
        AHP.prob_AHP_uniform(data_avgRange, weights_ranges, n, alts, quants, 'U')
    
    ########### Run Fuzzy AHP ###########    
    results = fAHP.alt_fuzzy_AHP2(data_tri, weights_tri, 'FZ', 0)
    
    
    ########### Data Reduction ###########
    alphas = [float(n1)/(21-1) for n1 in range(21)]       #create 21 alpha cuts
    P_cuts = [[fuzz.getAlphaCut(r[1],a) for a in alphas] for r in results]
    
    #create fuzzy RC plots 
    P_plots = []  #organize RC plots as [[alt1_x, alt1_y], [alt2_x, alt2_y], ...]
    for alt in P_cuts:
        x1 = [a[0] for a in alt]
        x2 = [alt[len(alphas)-1-i][1] for i in range(len(alphas))]
        y1 = alphas
        y2 = [alphas[len(alphas)-1-i] for i in range(len(alphas))]
        P_plots.append([x1+x2, y1+y2])
    
    #get fuzzy centroids and rank them
    P_centroids = [[ID] for ID in alt_IDs]
    for i in range(len(P_cuts)):
        C_area = sum([(c[1]-c[0])*0.5*(c[1] + c[0]) for c in P_cuts[i]])
        P_centroids[i].append(C_area/sum([(c[1]-c[0]) for c in P_cuts[i]]))
    fuzzCent_rankings = AHP.getCrispRanks([P[1] for P in P_centroids])
    for i in range(len(P_centroids)): P_centroids[i].append(fuzzCent_rankings[i])
        
    for i in range(len(fuzzy_rank_top_N)): 
        for j in range(len(P_centroids)): 
           if fuzzy_rank_top_N[i][0] == P_centroids[j][0]: fuzzy_rank_top_N[i].append(fuzzCent_rankings[j])
               #append new rank             
        
    P_UC_meas1 = [[ID] for ID in alt_IDs] #capture uncertainty as measured by ratio of 0.1 level support to value at m=1.0
    for i in range(len(P_UC_meas1)):
        P_UC_meas1[i].append((P_cuts[i][alphas.index(0.1)][1] -  - P_cuts[i][alphas.index(0.1)][0])/ \
                             P_cuts[i][alphas.index(1.0)][1])
    
    #get & diplay dominance (possibility) matrix
    DM = [[a] for a in alt_IDs];
    for i in range(alts):
        DM_row = []
        for j in range(alts):
            d = fuzz.dominance_AlphaCut(alphas, P_cuts[i], alphas, P_cuts[j]) 
                #get dominance of alt i over alt j
            DM_row.append(d)
        DM[i].append(DM_row)

    for i in range(len(fuzzy_dom_top_N)): 
        for j in range(len(DM)): 
           if fuzzy_dom_top_N[i][0] == DM[j][0]: fuzzy_dom_top_N[i].append(min(DM[j][1]))
               #append new dominance        
    
    #rank monte carlo fit means
    prob_ranks = AHP.getCrispRanks([nf[0] for nf in norm_fits])
    #get monte carlo CIs
    CIs = []
    for i in range(len(full_Ps)):
        m, cl, cu = confidence_interval(full_Ps[i], confidence=alpha)
        CIs.append([m, cl, cu])
    
    #get probability of top N
    prob_rankedN = [[a] for a in alt_IDs] 
        #list of alt lists, each item is probability of alt being ranked <= i+1
    for i in range(alts):
        prob_rankedn_row = [0.0 for a in range(alts)]
        for j in range(alts):
            prob_rankedn_row[j] = \
                float(len([x[i] for x in full_ranks if x[i] <= j+1]))/len(full_ranks)
                    #get probability that alt i rank is <= j+1
        prob_rankedN[i].append(prob_rankedn_row)

    for i1 in range(len(prob_top_N)):
        for j1 in range(len(prob_rankedN)):
            if prob_rankedN[j1][0] == prob_top_N[i1][0]:
                prob_top_N[i1].append(prob_rankedN[j1][1][top_N - 1])
                    #append the probability that alternative i1 is in top N
        
        
#add a probability of 0 once alternative removed
for i1 in range(len(prob_top_N)):
    if len(prob_top_N[i1]) < max(len(pn) for pn in prob_top_N): prob_top_N[i1].append(0.0)

#add a dominance of 0 once alterantive removed
#for i1 in range(len(fuzzy_dom_top_N)):
#    if len(fuzzy_dom_top_N[i1]) < max(len(pn) for pn in fuzzy_dom_top_N): fuzzy_dom_top_N[i1].append(0.0)

print '\n'
print 'Probability of Alternative being ranked in top N with i worst alts removed'
for pn in prob_top_N: print pn
    
print '\n'
print 'Dominance of Alternative with i worst alts removed'
for fa in fuzzy_dom_top_N: print fa

print '\n'
print 'Ranks of Alternatives with i worst alts removed'
for fc in fuzzy_rank_top_N: print fc

    
if create_plots:
    plt.figure()
    plt.ylabel('Probability of Being in Top ' + str(top_N))
    plt.xlabel('Alternatives Removed (N)')
    for i in range(len(prob_top_N)):
        plt.plot(range(len(prob_top_N[i][1:])), prob_top_N[i][1:], 
                 next(linecycler), marker=next(markercycler), color=colors[i], lw=2)
    plt.legend([pt[0] for pt in prob_top_N], loc='center right', 
               bbox_to_anchor=(1.25, 0.5), ncol=1, handlelength=3)   
    plt.savefig('saved_figures/sensitivity_AHP'+suffix+'.png', bbox_inches='tight')
    
    #plot fuzzy sensitivity (rank)
    plt.figure()
    plt.ylabel('Fuzzy Ranks')
    plt.xlabel('Alternatives Removed (N)')
    for i in range(len(fuzzy_rank_top_N)):
        plt.plot(range(len(fuzzy_rank_top_N[i][1:])), fuzzy_rank_top_N[i][1:], 
                 next(linecycler),marker=next(markercycler), color=colors[i], lw=2)
    plt.legend([pt[0] for pt in fuzzy_rank_top_N], loc='center right', 
               bbox_to_anchor=(1.25, 0.5), ncol=1, handlelength=3)
    plt.ylim([1,10])
    plt.gca().invert_yaxis()
    plt.savefig('saved_figures/sensitivity_AHP_fuzzRank'+suffix+'.png', bbox_inches='tight')

    
    #plot fuzzy sensitivity
    plt.figure()
    plt.ylabel('Fuzzy Dominance')
    plt.xlabel('Alternatives Removed (N)')
    for i in range(len(fuzzy_dom_top_N)):
        plt.plot(range(len(fuzzy_dom_top_N[i][1:])), fuzzy_dom_top_N[i][1:], 
                 next(linecycler),marker=next(markercycler), color=colors[i], lw=2)
    plt.legend([pt[0] for pt in fuzzy_dom_top_N], loc='center right', 
               bbox_to_anchor=(1.25, 0.5), ncol=1, handlelength=3)
    plt.ylim([0.92, 1.01])
    plt.savefig('saved_figures/sensitivity_AHP_fuzzy'+suffix+'.png', bbox_inches='tight')
           
#plt.show()    
