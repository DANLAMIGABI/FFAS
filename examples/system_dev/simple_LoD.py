# -*- coding: utf-8 -*-
"""
@author: frankpatterson
"""

import numpy as np
import skfuzzy as fuzz

from training import *
from systems import *

import fuzzy_operations as fuzzyOps

import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
plt.ioff()

##
dataIn = readFuzzyInputData('data/morphInputs_13Jun15.csv')

inputs, outputs, rulebase, AND_operator, OR_operator, aggregator, implication, defuzz = \
    build_fuzz_system('FCL_files/LoDsys_simple_13Jun15.fcl')
sys = Fuzzy_System(inputs, outputs, rulebase, AND_operator, OR_operator, aggregator, implication, defuzz)

               #name          #VL_SYS_f   #VL_TECH_f  #FWD_TYPE_f #WING_f  #VL_TECH_LoD #WING_LoD    #IND L0D
ps_inputs = [ ['Alt 1: BaseTR',   [2,7],      [1,9],      [5,8],      [3,7],  [5,25],     [8,20],      [8.,17.]], #8-18 
              ['Alt 2: BaseTW',   [2,7],      [1,9],      [5,8],      [3,7],  [5,25],     [8,20],      [10.,17.]], #10-17
              ['Alt 3: FIW_TJ',   [5,8],      [1,9],      [1,9],      [6,9],  [5,25],     [5,10],      [6.,13.]], #6-13
              ['Alt 4: Tilt_FIW', [5,8],      [1,9],      [1,9],      [5,9],  [5,25],     [9,22],      [8.,16.]],#8-16
              ['Alt 5: StopRot',  [2,5],      [1,6],      [1,9],      [2,6],  [5,25],     [5,10],      [7.,12.]], #4-10
              ['Alt 6: AutoGyro', [2,5],      [5,7],      [1,9],      [3,7],  [5,25],     [8,20],      [9.,15.]], #10-20
              ['Alt 7: TwinTS',   [6,9],      [1,9],      [1,9],      [3,7],  [5,25],     [8,20],      [9.,16.]], #9-16
              ['Alt 8: FixedFIW', [5,8],      [1,9],      [1,9],      [5,9],  [5,25],     [9,22],      [11.,17.]], #11-14
              ['Alt 9: HeliPL',   [2,5],      [4,7],      [1,9],      [3,7],  [5,25],     [8,20],      [8.,16.]], #8-17
              ['Alt 10: FIB-TD',  [6,8],      [1,9],      [1,8],      [3,7],  [5,25],     [8,20],      [10.,17.]], #10-18
            ]



#build inputs 
fig1, ax1 = plt.subplots(figsize=(7, 7))
fig2, ax2 = plt.subplots(nrows=len(ps_inputs)/2, ncols=2)
for alt in ps_inputs:
    
    
    #union of drag evals
    f = [max(alt[1][0], alt[2][0], alt[3][0], alt[4][0]),
         min(alt[1][1], alt[2][1], alt[3][1], alt[4][1])]
    print alt[0], f
    #SYSTEM_f_x = np.arange(0.9*f[0], 1.1*f[1], (1.1*f[1]-0.9*f[0])/100.0)
    #SYSTEM_f = fuzz.trapmf(SYSTEM_f_x, [f[0],f[0],f[1],f[1]]) #trap input MFS
    #SYSTEM_f = fuzz.trimf(SYSTEM_f_x, [f[0],0.5*(f[0]+f[1]),f[1]]) #tri input MFS
    SYSTEM_f = fuzzyOps.rangeToMF(f, 'trap')
    
    LoD = [min(alt[6]), max(alt[6])]
    #WING_LoD_x = np.arange(0.9*LoD[0], 1.1*LoD[1], (1.1*LoD[1]-0.8*LoD[0])/100.0)
    #WING_LoD = fuzz.trapmf(WING_LoD_x, [LoD[0],LoD[0],LoD[1],LoD[1]]) #trap input MFS
    #WING_LoD = fuzz.trimf(WING_LoD_x, [LoD[0],0.5*(LoD[0]+LoD[1]),LoD[1]]) #tri input MFS
    WING_LoD = fuzzyOps.rangeToMF(LoD, 'trap')
    output = sys.run({'SYSTEM_f': SYSTEM_f, 
                      'WING_LoD': WING_LoD }, TESTMODE=True)
    output = output['sys_LoD']  


    #plot ranges from max alpha cuts
    outRange = fuzzyOps.alpha_at_val(output[0], output[1])
    
    #calculate overlap
    overlap = 100.*float(min(max(outRange), max(alt[7])) - max(min(outRange), min(alt[7])))/ \
              max( max(outRange)-min(outRange) , max(alt[7])-min(alt[7]) )
    print alt[0], overlap, '% overlap'
    
    ax1.bar(outRange[0], 0.3, width=outRange[1]-outRange[0], bottom=ps_inputs.index(alt)+0.7, color='r')
    ax1.bar(alt[7][0],   0.3, width=alt[7][1]-alt[7][0],     bottom=ps_inputs.index(alt)+1.0, color='b')
    ax1.text(max(max(outRange),max(alt[7]))+0.5, ps_inputs.index(alt)+0.9, str(round(overlap,1))+'%',
             fontsize=10)
    
    #plot actual outputs
    i = ps_inputs.index(alt)
    ax2[(i-(i%2))/2, i%2].plot(output[0], output[1], '-r')
    exp_x = np.arange(0.9*min(alt[7]), 1.1*max(alt[7]), (max(alt[7])-min(alt[7]))/50.)
    exp_y = fuzz.trapmf(exp_x, [min(alt[7]), min(alt[7]), max(alt[7]), max(alt[7])]) 
    ax2[(i-(i%2))/2, i%2].plot(exp_x, exp_y, '-b')


ax1.set_ylim([0,11])
ax1.set_xlim([5,25])
ax1.yaxis.grid(True)
ax1.set_yticks(range(1,11))
ax1.set_yticklabels([alt[0] for alt in ps_inputs])
ax1.set_xlabel('System L/D')

fontP = FontProperties()
fontP.set_size('medium')
ax1.legend(['Fuzzy System Output', 'Expert System Evaluations'], bbox_to_anchor=(1.0, 1.06), prop=fontP)




plt.draw()
plt.show()



    