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
dataIn = readFuzzyInputData('data/POC_morphInputs_01Mar15.csv')

inputs, outputs, rulebase, AND_operator, OR_operator, aggregator, implication, defuzz = \
    build_fuzz_system('FCL_files/etaPsys_simple_14Aug15.fcl') #/Users/frankpatterson/Google Drive/Thesis/FFAS/FCL_files/etaPsys_simple_14Aug15.py
sys = Fuzzy_System(inputs, outputs, rulebase, AND_operator, OR_operator, aggregator, implication, defuzz)

               #name          #FWD_PROP_eta_p   #FWD_TYPE_eta_p  #FWD_DRV_eta_d   #VL_TYPE_TECH #FWD_SYS_TYPE #IND etaP
ps_inputs = [ ['Alt 1: BaseTR',   [0.6,0.8],    [0.6,0.8],      [0.85,0.96],   2,       2,    [0.67,0.85], ], 
              ['Alt 2: BaseTW',   [0.8,0.98],   [0.6,0.8],      [0.85,0.96],   2,       2,    [0.75,0.89], ],
              ['Alt 3: FIW_TJ',   [0.5,0.7],    [0.6,1.0],      [0.9,1.0],     1,       1,     [0.30,0.72], ],
              ['Alt 4: Tilt_FIW', [0.6,0.9],    [0.6,0.8],      [0.85,0.96],   2,       2,     [0.65,0.85], ],
              ['Alt 5: StopRot',  [0.5,0.7],    [0.6,1.0],      [0.9,1.0],     3,       1,     [0.4,0.72], ],
              ['Alt 6: AutoGyro', [0.8,0.98],   [0.6,1.0],      [0.85,0.96],   5,       1,     [0.88,0.95], ],
              ['Alt 7: TwinTS',   [0.6,0.8],    [0.6,0.8],      [0.85,0.96],   2,       2,     [0.65,0.90], ],
              ['Alt 8: FixedFIW', [0.8,0.98],   [0.6,1.0],      [0.85,0.96],   1,       1,     [0.90,0.95], ],
              ['Alt 9: HeliPL',   [0.5, 0.7],   [0.6,1.0],      [0.9,1.0],     5,       1,     [0.35,0.72], ],
              ['Alt 10: FIB-TD',  [0.6,0.9],    [0.6,1.0],      [0.85, 0.96],  1,       4,     [0.60,0.95], ],
            ]

#TEST INDIVIDUAL
"""
alt = ps_inputs[2]
etaP = [max([alt[1][0], alt[2][0]]), min([alt[1][1], alt[2][1]])]
FWD_SYS_eta_d = fuzzyOps.rangeToMF(etaP, 'trap')
FWD_DRV_eta_d = fuzzyOps.rangeToMF(alt[3], 'trap')

output = sys.run({'FWD_SYS_eta_p': FWD_SYS_eta_d, 
                  'FWD_DRV_eta_d' : FWD_DRV_eta_d, },
                  TESTMODE=True)
output = output['sys_etaP']  
"""

#build inputs 
fig1, ax1 = plt.subplots(figsize=(7, 7))
fig2, ax2 = plt.subplots(nrows=len(ps_inputs)/2, ncols=2)
for alt in ps_inputs:
    
    
    #get data
    etaP = [max([alt[1][0], alt[2][0]]), min([alt[1][1], alt[2][1]])]
    FWD_SYS_eta_p = fuzzyOps.rangeToMF(etaP, 'trap')
    FWD_DRV_eta_d = fuzzyOps.rangeToMF(alt[3], 'trap')
    
    output = sys.run({'FWD_SYS_eta_p': FWD_SYS_eta_p, 
                    'FWD_DRV_eta_d' : FWD_DRV_eta_d, },
                    TESTMODE=False)
    output = output['sys_etaP']  


    #plot ranges from max alpha cuts
    outRange = fuzzyOps.alpha_at_val(output[0], output[1])
    
    IND_asses = alt[6]
    #calculate overlap
    overlap = 100.*float(min(max(outRange), max(IND_asses)) - max(min(outRange), min(IND_asses)))/ \
              max( max(outRange)-min(outRange) , max(IND_asses)-min(IND_asses) )
    print alt[0], overlap, '% overlap'
    
    ax1.bar(outRange[0], 0.3, width=outRange[1]-outRange[0], bottom=ps_inputs.index(alt)+0.7, color='r')
    ax1.bar(IND_asses[0],   0.3, width=IND_asses[1]-IND_asses[0],     bottom=ps_inputs.index(alt)+1.0, color='b')
    ax1.text(1.01, ps_inputs.index(alt)+0.9, str(round(overlap,1))+'%',
             fontsize=10)
    
    #plot actual outputs
    i = ps_inputs.index(alt)
    ax2[(i-(i%2))/2, i%2].plot(output[0], output[1], '-r')
    exp_x = np.arange(0.9*min(IND_asses), 1.1*max(IND_asses), (max(IND_asses)-min(IND_asses))/50.)
    exp_y = fuzz.trapmf(exp_x, [min(IND_asses), min(IND_asses), max(IND_asses), max(IND_asses)]) 
    ax2[(i-(i%2))/2, i%2].plot(exp_x, exp_y, '-b')


ax1.set_ylim([0,11])
ax1.set_xlim([0.6,1.0])
ax1.yaxis.grid(True)
ax1.set_yticks(range(1,11))
ax1.set_yticklabels([alt[0] for alt in ps_inputs])
ax1.set_xlabel('System Propulsive Efficiency')

fontP = FontProperties()
fontP.set_size('medium')
ax1.legend(['Fuzzy System Output', 'Expert System Evaluations'], bbox_to_anchor=(1.0, 1.06), prop=fontP)




plt.draw()
plt.show()



    