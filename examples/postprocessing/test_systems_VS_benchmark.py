# TEST Fuzzy Systems vs. System evaluations


import numpy as np
import skfuzzy as fuzz

from training import *
from systems import *
import fuzzy_operations as fuzzyOps

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Ellipse
from matplotlib import ticker

plt.ioff()

ASPECT_list = ['VL_SYS_TYPE', 'VL_SYS_PROP', 'VL_SYS_DRV', 'VL_SYS_TECH', \
            'FWD_SYS_PROP', 'FWD_SYS_DRV', 'FWD_SYS_TYPE',\
            'WING_SYS_TYPE', \
            'ENG_SYS_TYPE']   
                
                
system_names = [ 'Alt 1: BaseTR', 'Alt 2: BaseTW','Alt 3: FIW_TJ', 'Alt 4: Tilt_FIW', 'Alt 5: StopRot',
                 'Alt 6: AutoGyro', 'Alt 7: TwinTS', 'Alt 8: FixedFIW', 'Alt 9: HeliPL', 'Alt 10: FIB-TD']

system_options = [  [2,2,1,2,2,1,2,1,1], 
                    [2,1,1,2,1,1,1,5,1],
                    [4,3,3,1,4,3,1,2,2],
                    [4,3,1,2,3,1,2,4,1],
                    [1,2,2,3,4,3,1,6,3],
                    [1,2,1,5,1,1,1,1,1],
                    [6,2,1,2,2,1,1,1,1],
                    [4,3,1,1,1,1,1,4,1],
                    [1,2,2,5,4,3,1,1,3],
                    [5,3,1,1,3,1,4,1,4]]
                    
                    #phi ,  maxAS   ,   FoM    , L/D  ,   eta_p
system_evals = [    [[4.,6.],[300.,330.],[0.65,0.75],[8.,17.],[0.67,0.85]],
                    [[3.,5.],[325.,370.],[0.67,0.77],[10.,17.],[0.75,0.90]],
                    [[4.,8.],[380.,450.],[0.25,0.6], [6.,13.],[0.3,0.72]],
                    [[3.,7.],[310.,340.],[0.6,0.8],  [8.,16.],[0.65,0.85]],
                    [[3.,6.],[355.,400.],[0.3,0.55], [6.,11.],[0.4,0.72]],
                    [[5.,7.],[305.,355.],[0.65,0.82],[9.,14.],[0.88,0.95]],
                    [[7.,8.],[315.,365.],[0.62,0.78],[9.,16.],[0.65,0.9]],
                    [[4.,7.],[330.,390.],[0.77,0.83],[11.,17.],[0.9,0.95]],
                    [[5.,8.],[350.,410.],[0.3,0.55], [8.,16.],[0.35,0.72]],
                    [[3.,7.],[330.,375.],[0.55,0.85],[10.,18.],[0.6,0.95]],]               
                    
 #system_evals = [    [[4.,6.],[230.,290.],[0.65,0.75],[8.,17.],[0.67,0.85]],
 #                   [[3.,5.],[250.,450.],[0.67,0.77],[10.,17.],[0.75,0.90]],
 #                   [[4.,8.],[300.,550.],[0.25,0.6],[6.,13.],[0.3,0.72]],
 #                   [[3.,7.],[180.,330.],[0.6,0.8],[8.,16.],[0.65,0.85]],
 #                   [[3.,6.],[200.,550.],[0.3,0.55],[6.,11.],[0.4,0.72]],
 #                   [[5.,7.],[175.,350.],[0.65,0.82],[9.,14.],[0.88,0.95]],
 #                   [[7.,8.],[230.,450.],[0.62,0.78],[9.,16.],[0.65,0.9]],
 #                   [[4.,7.],[350.,375.],[0.77,0.83],[11.,17.],[0.9,0.95]],
 #                   [[5.,8.],[300.,450.],[0.3,0.55],[8.,16.],[0.35,0.72]],
 #                   [[3.,7.],[200.,375.],[0.55,0.85],[10.,18.],[0.6,0.95]],]
                                                   
#setup data:
dataIn = readFuzzyInputData('data/morphInputs_13Jun15.csv')

def getInputs(system, function, input, options):
    var_list = [] #list of [system, function, var, range, quant inputs, out
    for line in dataIn:
        var_list.append([line[0], line[1], line[3], line[4], \
                         line[options[ASPECT_list.index(line[1])] + 4], \
                         line[options[ASPECT_list.index(line[1])] + 10] ])
    inputs = []
    for line in var_list:
        if system == None and function == None:
            if line[2] == input: inputs.append(line[4])
        if function == None:
            if line[0] == system and line[2] == input: inputs.append(line[4])
        elif system == None: 
            if line[1] == function and line[2] == input: inputs.append(line[4])
        else: 
            if line[0] == system and line[1] == function and line[2] == input: inputs.append(line[4])
    
    return inputs


def quantify(inFuz, inR, outR):
    """quantify through linear interpolation
    """
    x0, x1 = inR[0], inR[1]
    y0, y1 = outR[0], outR[1] 

    for x in inFuz[0]:
        outXs = [(y0 + (y1-y0)*((x-x0)/(x1-x0))) for x in inFuz[0]]

    return [outXs,inFuz[1]]

## CREATE SYSTEMS:


# Empty Weight:
inputs, outputs, rulebase, AND_operator, OR_operator, aggregator, implication, defuzz = \
    build_fuzz_system('FCL_files/FRBS_phi/PHIsys_trained_5-2In9-2Out_gauss250tData_50vData_optInputsBEST_GA.fcl') #PHIsys_trained_5-2In9-2Out_gauss250-50Data_diffEvBEST.fcl
phi_sys = Fuzzy_System(inputs, outputs, rulebase, AND_operator, OR_operator, aggregator, implication, defuzz)

# FoM
inRanges = {    'DATA_e_d':     [0,     0.3],
                'DATA_sigma':   [0.05,  0.4],
                'DATA_w':       [0.,    150.],
                'DATA_eta':     [0.5,   1.0], }
outRanges = {'sys_FoM' : [0.4, 1.0] }
FM_sys = DFES(inRanges, outRanges, 'sigmoid', 160, 50, 50)
FM_sys.read_weights('FCL_files/DFES_FoM/DFES_FOMdata_data(500)_nodes(160_50_50).nwf') #read network weight foil

# L/D
inputs, outputs, rulebase, AND_operator, OR_operator, aggregator, implication, defuzz = \
    build_fuzz_system('FCL_files/LoDsys_simple_13Jun15.fcl')
LoD_sys = Fuzzy_System(inputs, outputs, rulebase, AND_operator, OR_operator, aggregator, implication, defuzz)

# etaP
inputs, outputs, rulebase, AND_operator, OR_operator, aggregator, implication, defuzz = \
    build_fuzz_system('FCL_files/etaPsys_simple_14Aug15.fcl') #/Users/frankpatterson/Google Drive/Thesis/FFAS/FCL_files/etaPsys_simple_14Aug15.py
etaP_sys = Fuzzy_System(inputs, outputs, rulebase, AND_operator, OR_operator, aggregator, implication, defuzz)

# RF Systems:

inRanges = {    'SYSTEM_QUANT_PHI':  [0.5,0.85],#[0.5, 0.95],
                'VL_SYS_w':          [1, 150],
                'WING_SYS_TYPE_WS':  [15.,300.],
                'sys_etaP':          [0.6, 1.0],
                'VL_SYS_DRV_eta_d':  [0.7, 1.0],#[0.4, 1.0],
                'sys_FoM':           [0.3, 1.0],
                'VL_SYS_e_d':        [0.0, 0.3],
                'ENG_SYS_TYPE_SFC':  [0.35, 0.75],#[0.45, 1.05],
               # 'SYS_dragX':        [,],#[0.6, 1.15],
                'SYS_type':          [0.5, 3.5],#[0.5, 3.5],
               #'SYS_tech':          [-0.5, 4.5],#[-0.5, 4.5],
               # 'SYS_jet':          [,],#[0.5, 2.5],
            }
            
outRanges_GWT = {'sys_GWT'   : [5000.,50000.] }
outRanges_Pin = {'sys_Pinst' : [1000, 15000] }
outRanges_VH  = {'sys_VH'    : [200, 500] }

inOrder = ['VL_SYS_e_d', 'WING_SYS_TYPE_WS', 'SYSTEM_QUANT_PHI', 'SYS_type', 'VL_SYS_DRV_eta_d', 'sys_FoM', 'VL_SYS_w', 'sys_etaP', 'ENG_SYS_TYPE_SFC']

GWT_sys = DFES(inRanges, outRanges_GWT, 'sigmoid', 250, 30, 50, inputOrder=inOrder)
GWT_sys.read_weights('FCL_files/DFES_RF/BEST/RFdata20Aug_DFES_GWT_30_250_50.nwf') #read network weight foil

Pin_sys = DFES(inRanges, outRanges_Pin, 'sigmoid', 250, 30, 50, inputOrder=inOrder)
Pin_sys.read_weights('FCL_files/DFES_RF/BEST/RFdata20Aug_DFES_Pin_30_250_50.nwf')#'FCL_files/DFES_RF/BEST/RFdata20Aug_DFES_Pin_30_250_50.nwf') #read network weight foil

VH_sys = DFES(inRanges, outRanges_VH, 'sigmoid', 250, 40, 50, inputOrder=inOrder)
VH_sys.read_weights('FCL_files/DFES_RF/BEST/RFdata20Aug_DFES_VH_40_250_50.nwf') #read network weight foil

results = [[] for opt in system_options] #save results



## RUN SYSTEMS:
for opts in system_options: #for each option

    #Evaluate phi
    _VLT_w = getInputs(None, 'VL_SYS_TYPE', 'w', opts)[0] #VL_SYS_TYPE_w : 
    VLT_w = fuzzyOps.rangeToMF(_VLT_w, 'gauss')
    
    _VLT_ed = getInputs(None, 'VL_SYS_TYPE', 'e_d', opts)[0] #VL_SYS_TYPE_e_d    
    VLT_ed = fuzzyOps.rangeToMF(_VLT_ed, 'gauss')

    _VLP_w = getInputs(None, 'VL_SYS_PROP', 'w', opts)[0] #VL_SYS_PROP_w      
    VLP_w= fuzzyOps.rangeToMF(_VLP_w, 'gauss')

    _VLTe_w = getInputs(None, 'VL_SYS_TECH', 'w', opts)[0] #VL_SYS_TECH_w  
    VLTe_w = fuzzyOps.rangeToMF(_VLTe_w, 'gauss')
    
    _FSP_phi = getInputs(None, 'FWD_SYS_PROP', 'phi', opts)[0] #FWD_SYS_PROP_phi
    FSP_phi = fuzzyOps.rangeToMF(_FSP_phi, 'gauss')

    _FSP_etap = getInputs(None, 'FWD_SYS_PROP', 'eta_p', opts)[0]#FWD_SYS_PROP_eta_p  : FWD system prop efficiency
    FSP_etap = fuzzyOps.rangeToMF(_FSP_etap, 'gauss')

    _FST_TP = getInputs(None, 'FWD_SYS_TYPE', 'TP', opts)[0]    #FWD_SYS_TYPE_TP    
    FST_TP = fuzzyOps.rangeToMF(_FST_TP, 'gauss')

    _WST_phi = getInputs(None, 'WING_SYS_TYPE', 'phi', opts)[0] #WING_SYS_TYPE_phi  
    WST_phi = fuzzyOps.rangeToMF(_WST_phi, 'gauss')

    _WST_WS = getInputs(None, 'WING_SYS_TYPE', 'WS', opts)[0]   #WING_SYS_TYPE_WS  : WING system wing loading 
    WST_WS = fuzzyOps.rangeToMF(_WST_WS, 'gauss')

    _WST_LD = getInputs(None, 'WING_SYS_TYPE', 'LD', opts)[0]   #WING_SYS_TYPE_LD   
    WST_LD = fuzzyOps.rangeToMF(_WST_LD, 'gauss')

    _EST_phi = getInputs(None, 'ENG_SYS_TYPE', 'phi', opts)[0]   #ENG_SYS_TYPE_phi   
    EST_phi = fuzzyOps.rangeToMF(_EST_phi, 'gauss')

    _EST_SFC = getInputs(None, 'ENG_SYS_TYPE', 'SFC', opts)[0]  #ENG_SYS_TYPE_SFC   
    EST_SFC = fuzzyOps.rangeToMF(_EST_SFC, 'gauss')

    _VLTe_phi = getInputs(None, 'VL_SYS_TECH', 'phi', opts)[0]
    VLTe_phi = fuzzyOps.rangeToMF(_VLTe_phi, 'gauss')#'VL_SYS_TECH_phi'

    _VLD_phi = getInputs(None, 'VL_SYS_DRV', 'phi', opts)[0]
    VLD_phi = fuzzyOps.rangeToMF(_VLD_phi, 'gauss')#'VL_SYS_DRV_phi'
    
    _VLT_TP = getInputs(None, 'VL_SYS_TYPE', 'TP', opts)[0]
    VLT_TP = fuzzyOps.rangeToMF(_VLT_TP, 'gauss')#'VL_SYS_TYPE_TP'
    
    _VLT_phi = getInputs(None, 'VL_SYS_TYPE', 'phi', opts)[0]
    VLT_phi = fuzzyOps.rangeToMF(_VLT_phi, 'gauss')#'VL_SYS_TYPE_phi'
    
    _VLP_phi = getInputs(None, 'VL_SYS_PROP', 'phi', opts)[0]
    VLP_phi = fuzzyOps.rangeToMF(_VLP_phi, 'gauss')#'VL_SYS_PROP_phi'

    _FSD_etad = getInputs(None, 'FWD_SYS_DRV', 'eta_d', opts)[0]
    FSD_etad = fuzzyOps.rangeToMF(_FSD_etad, 'gauss')#FWD_SYS_DRV', 'eta_d'

    _FST_phi = getInputs(None, 'FWD_SYS_TYPE', 'phi', opts)[0]    
    FST_phi = fuzzyOps.rangeToMF(_FST_phi, 'gauss')#FWD_SYS_TYPE', 'phi'
    
    _VLTe_f = getInputs(None, 'VL_SYS_TECH', 'f', opts)[0]
    VLTe_f = fuzzyOps.rangeToMF(_VLTe_f, 'gauss')#VL_SYS_TECH', 'f'    
    
    _VLT_f = getInputs(None, 'VL_SYS_TYPE', 'f', opts)[0]
    VLT_f = fuzzyOps.rangeToMF(_VLT_f, 'gauss') #VL_SYS_TYPE', 'f'

    _VLTe_LD = getInputs(None, 'VL_SYS_TECH', 'LD', opts)[0]
    VLTe_LD = fuzzyOps.rangeToMF(_VLTe_LD, 'gauss')#VL_SYS_TECH', 'LD'

    _WST_f = getInputs(None, 'WING_SYS_TYPE', 'f', opts)[0]
    WST_f = fuzzyOps.rangeToMF(_WST_f, 'gauss') #WING_SYS_TYPE', 'f'


    output_phi = phi_sys.run( { 'VL_SYS_TECH_phi':  VLTe_phi,
                                'FWD_SYS_DRV_eta_d':  FSD_etad,
                                'FWD_SYS_TYPE_phi':  FST_phi,
                                'VL_SYS_TECH_f':  VLTe_f,
                                'FWD_SYS_PROP_eta_p':  FSP_etap,
                                'VL_SYS_TECH_w':  VLTe_w,
                                'VL_SYS_TYPE_f':  VLT_f,
                                'VL_SYS_TECH_LD':  VLTe_LD,
                                'WING_SYS_TYPE_LD':  WST_LD,
                                'FWD_SYS_TYPE_TP':  FST_TP,
                                'VL_SYS_TYPE_w':  VLT_w,
                                'WING_SYS_TYPE_f':  WST_f,
                                'VL_SYS_PROP_w':  VLP_w,
                                'VL_SYS_TYPE_phi':  VLT_phi,
                                'VL_SYS_PROP_phi':  VLP_phi, })

    """
    {'VL_SYS_TECH_phi': VLTe_phi,
    'FWD_SYS_PROP_eta_p': FSP_etap,
    'VL_SYS_DRV_phi': VLD_phi,
    'VL_SYS_TECH_w': VLTe_w,
    'WING_SYS_TYPE_WS': WST_WS,
    'FWD_SYS_TYPE_TP': FST_TP,
    'VL_SYS_TYPE_w': VLT_w,
    'VL_SYS_TYPE_e_d': VLT_ed,
    'VL_SYS_TYPE_TP': VLT_TP,
    'VL_SYS_TYPE_phi': VLT_phi,
    'VL_SYS_PROP_phi': VLP_phi,}
    """
    results[system_options.index(opts)].append(output_phi['sys_phi'])
    
    #Evaluate FoM
    _ed = getInputs(None, 'VL_SYS_TYPE', 'e_d', opts)[0] #download
    ed = fuzzyOps.rangeToMF(_ed, 'gauss')
    
    _sigma = getInputs(None, 'VL_SYS_PROP', 'sigma', opts)[0] #solidity
    sigma = fuzzyOps.rangeToMF(_sigma, 'gauss')
    
    #_w = getInputs('VL_SYS', None, 'w', opts) #diskloading (average)
    #_w = [np.average([x[0] for x in _w]), np.average([x[1] for x in _w])]
    _w1 = getInputs(None, 'VL_SYS_TYPE', 'w', opts)[0] #solidity
    _w2 = getInputs(None, 'VL_SYS_PROP', 'w', opts)[0] #solidity
    _w3 = getInputs(None, 'VL_SYS_TECH', 'w', opts)[0] #solidity
    _w = [max(_w3[0], np.average([_w1[0], _w2[0]])), min(_w3[1], np.average([_w1[1], _w2[1]])) ]
    sigma = fuzzyOps.rangeToMF(_sigma, 'gauss')
   
    w = fuzzyOps.rangeToMF(_w, 'gauss')
    
    _eta = getInputs(None, 'VL_SYS_DRV', 'eta_d', opts)[0] #drive efficiency
    eta = fuzzyOps.rangeToMF(_eta, 'gauss')
    
    output_FM = FM_sys.run({'DATA_e_d': ed, 'DATA_sigma': sigma, 'DATA_w': w, 'DATA_eta': eta})
    results[system_options.index(opts)].append(output_FM['sys_FoM'])

    #Evaluate L/D
    _f = getInputs(None, None, 'f', opts)  #get drag (intersection)
    _f = [np.average([x[0] for x in _f]), np.average([x[1] for x in _f])]
    f = fuzzyOps.rangeToMF(_f, 'trap')
    
    _LDw   = getInputs(None, 'WING_SYS_TYPE', 'LD', opts)[0]
    _LDvt = getInputs(None, 'VL_SYS_TECH', 'LD', opts)[0]
    _LD = [max([_LDw[0], _LDvt[0]]), min([_LDw[1], _LDvt[1]])]
    LD = fuzzyOps.rangeToMF(_LD, 'trap')
    
    output_LD = LoD_sys.run({'SYSTEM_f': f, 'WING_LoD': LD})
    results[system_options.index(opts)].append(output_LD['sys_LoD']) 

    #Evaluate eta_P
    _FST_etap = getInputs(None, 'FWD_SYS_TYPE', 'eta_p', opts)[0]#  : FWD system type efficiency
    _FSD_etap = getInputs(None, 'FWD_SYS_DRV', 'eta_p', opts)[0]#  : FWD system type efficiency
    _FWD_etap = [max( [_FST_etap[0], _FSP_etap[0], _FSD_etap[0]] ), min( [_FST_etap[1], _FSP_etap[1], _FSD_etap[1]] )]
    FWD_etap = fuzzyOps.rangeToMF(_FWD_etap, 'trap')
    
    _FSD_etad = getInputs(None, 'FWD_SYS_DRV', 'eta_d', opts)[0]#  : FWD drive efficiency
    FSD_etad = fuzzyOps.rangeToMF(_FSD_etad, 'trap')
    
    output_etaP = etaP_sys.run({'FWD_SYS_eta_p': FWD_etap, 
                                'FWD_DRV_eta_d' : FSD_etad, })
    results[system_options.index(opts)].append(output_etaP['sys_etaP']) 

    # Evaluate RF Methods
    inR = [1,9]
    outR = [0.85, 0.45]
    sysPHI = quantify(output_phi['sys_phi'], inR, outR) #quantify phi

    outR = [0.75,0.35]
    quantSFC = quantify(EST_SFC, inR, outR) #quantify phi
    
    VL_SYS_TYPE = opts[0]
    FWD_SYS_TYPE = opts[6]
    if FWD_SYS_TYPE == 2: 
            T = 1 #tiling VL
    else:
        if VL_SYS_TYPE < 4: 
            T = 2 #compound
        if VL_SYS_TYPE == 4 or VL_SYS_TYPE == 5: 
            T = 3 #other
        if VL_SYS_TYPE == 6:
            T = 1 #tilting tailsitter

    SYSTYPE = fuzzyOps.rangeToMF([T,T], 'gauss')
    
    VL_SYS_TECH = opts[3]
    if   VL_SYS_TECH == 2: T = 3
    elif VL_SYS_TECH == 3: T = 2
    else:                  T = VL_SYS_TECH
    SYSTECH = fuzzyOps.rangeToMF([T,T], 'gauss')
    
    ENG_TYPE = opts[8] #('DATA', 'jet')        (1-false, 2-true)
    SYSJET = 1
    if ENG_TYPE == 2 or ENG_TYPE == 3: SYSJET=2

    output_GWT = GWT_sys.run({  'SYSTEM_QUANT_PHI': sysPHI , 'VL_SYS_w': w, 'WING_SYS_TYPE_WS': WST_WS, 
                                'sys_etaP': output_etaP['sys_etaP'], 'VL_SYS_DRV_eta_d': eta, 'sys_FoM': output_FM['sys_FoM'],
                                'VL_SYS_e_d': ed, 'ENG_SYS_TYPE_SFC': quantSFC, #'SYS_dragX': f,
                                'SYS_type': SYSTYPE,})# 'SYS_tech': SYSTECH,}) #'SYS_jet': SYSJET})
    results[system_options.index(opts)].append(output_GWT['sys_GWT'])

    
    output_Pin = Pin_sys.run({  'SYSTEM_QUANT_PHI': sysPHI , 'VL_SYS_w': w, 'WING_SYS_TYPE_WS': WST_WS, 
                                'sys_etaP': output_etaP['sys_etaP'], 'VL_SYS_DRV_eta_d': eta, 'sys_FoM': output_FM['sys_FoM'],
                                'VL_SYS_e_d': ed, 'ENG_SYS_TYPE_SFC': quantSFC, #'SYS_dragX': f,
                                'SYS_type': SYSTYPE,})# 'SYS_tech': SYSTECH,}) #'SYS_jet': SYSJET})
    results[system_options.index(opts)].append(output_Pin['sys_Pinst'])
    
    output_VH = VH_sys.run({  'SYSTEM_QUANT_PHI': sysPHI , 'VL_SYS_w': w, 'WING_SYS_TYPE_WS': WST_WS, 
                                'sys_etaP': output_etaP['sys_etaP'], 'VL_SYS_DRV_eta_d': eta, 'sys_FoM': output_FM['sys_FoM'],
                                'VL_SYS_e_d': ed, 'ENG_SYS_TYPE_SFC': quantSFC, #'SYS_dragX': f,
                                'SYS_type': SYSTYPE,})# 'SYS_tech': SYSTECH,})# 'SYS_jet': SYSJET})
    results[system_options.index(opts)].append(output_VH['sys_VH'])

    #output_eWT = eWT_sys.run({  'SYSTEM_QUANT_PHI': sysPHI , 'VL_SYS_w': w, 'WING_SYS_TYPE_WS': WST_WS, 
    #                            'sys_etaP': output_etaP['sys_etaP'], 'VL_SYS_DRV_eta_d': eta, 'sys_FoM': output_FM['sys_FoM'],
    #                            'VL_SYS_e_d': ed, 'ENG_SYS_TYPE_SFC': quantSFC, 'SYS_dragX': f,
    #                            'SYS_type': SYSTYPE, 'SYS_tech': SYSTECH, 'SYS_jet': SYSJET})
    #results[system_options.index(opts)].append(output_eWT['sys_eWT'])    

### PLOT phi results
fig = plt.figure(figsize=(10,8))
res_ind = 0
alpha = 1.0
for i in range(len(results)):
    ax = plt.subplot(5,2,i+1)
    ax.plot(results[i][res_ind][0],results[i][res_ind][1], lw=2.0)
    
    ax.add_patch( patches.Rectangle( (system_evals[i][0][0], 0.0),
                                     float(system_evals[i][0][1] - system_evals[i][0][0]),# width
                                     1.0, fill=True, alpha=0.15, hatch='-', color='r')) #height, etc.          
    eMF = fuzzyOps.rangeToMF(system_evals[i][0], 'gauss')
    ax.plot(eMF[0], eMF[1], '-r', lw=2.0)
    ax.set_ylim([0.0, 1.01])
    ax.set_xlim([1.0, 9.0])
    ax.set_yticks([0.0, 0.5, 1.0])
    ax.set_ylabel(r' $\mu(x)$')
    if i > 7: ax.set_xlabel(r'$\phi$ (1-Poor, 9-Excellent)') 
    if i < 8: ax.set_xticks([1,2,3,4,5,6,7,8,9])
    ax.text(np.average(system_evals[i][0]), 0.8, system_names[i], rotation=90, fontsize=10)
    ax.xaxis.grid(True)
    if i == 0: ax.legend(['DFES Output', 'Benchmark MF', 'Benchmark Range'],
                         bbox_to_anchor=(1.85, 1.38), ncol=3, fontsize=12)

plt.subplots_adjust(left=0.09, bottom=0.08, right=0.94, top=0.94, wspace=None, hspace=0.18)
   
   
   
    
### PLOT FoM results
fig = plt.figure(figsize=(10,8))
res_ind = 1
alpha = 1.0
for i in range(len(results)):
    ax = plt.subplot(5,2,i+1)
    ax.plot(results[i][res_ind][0],results[i][res_ind][1], lw=2.0)
    ax.add_patch( patches.Rectangle( (system_evals[i][2][0], 0.0),
                                     float(system_evals[i][2][1] - system_evals[i][2][0]),# width
                                     1.0, fill=True, alpha=0.15, hatch='-', color='r')) #height, etc.          
    eMF = fuzzyOps.rangeToMF(system_evals[i][2], 'gauss')
    ax.plot(eMF[0], eMF[1], '-r', lw=2.0)    
    ax.set_ylim([0.0, 1.01])
    ax.set_yticks([0.0, 0.5, 1.0])
    ax.set_ylabel(r' $\mu(x)$')
    ax.set_xlim([0.4, 1.0])
    if i > 7: ax.set_xlabel('FoM')
    if i < 8: ax.set_xticklabels([])
    ax.text(np.average(system_evals[i][2]), 0.8, system_names[i], rotation=90, fontsize=10)
    ax.xaxis.grid(True)
    if i == 0: ax.legend(['DFES Output', 'Benchmark MF', 'Benchmark Range'],
                         bbox_to_anchor=(1.85, 1.38), ncol=3, fontsize=12)

plt.subplots_adjust(left=0.09, bottom=0.08, right=0.94, top=0.94, wspace=None, hspace=0.18)

### PLOT L/D results
fig = plt.figure(figsize=(10,8))
res_ind = 2
alpha = 1.0
for i in range(len(results)):
    ax = plt.subplot(5,2,i+1)
    ax.plot(results[i][res_ind][0],results[i][res_ind][1], lw=2.0)
    
    ax.add_patch( patches.Rectangle( (system_evals[i][3][0], 0.0),
                                     float(system_evals[i][3][1] - system_evals[i][3][0]),# width
                                     1.0, fill=True, alpha=0.15, hatch='-', color='r')) #height, etc.          
    eMF = fuzzyOps.rangeToMF(system_evals[i][3], 'trap')
    ax.plot(eMF[0], eMF[1], '-r', lw=2.0)
    ax.set_ylim([0.0, 1.01])
    ax.set_yticks([0.0, 0.5, 1.0])
    ax.set_ylabel(r' $\mu(x)$')
    ax.set_xlim([5., 25.])
    if i > 7: ax.set_xlabel('L/D')
    if i < 8: ax.set_xticklabels([])
    ax.text(np.average(system_evals[i][3]), 0.8, system_names[i], rotation=90, fontsize=10)
    ax.xaxis.grid(True)
    if i == 0: ax.legend(['DFES Output', 'Benchmark MF', 'Benchmark Range'],
                         bbox_to_anchor=(1.85, 1.38), ncol=3, fontsize=12)

plt.subplots_adjust(left=0.09, bottom=0.08, right=0.94, top=0.94, wspace=None, hspace=0.18)
    
### PLOT etaP results
fig = plt.figure(figsize=(10,8))
res_ind = 3
alpha = 1.0
for i in range(len(results)):
    ax = plt.subplot(5,2,i+1)
    ax.plot(results[i][res_ind][0],results[i][res_ind][1], lw=2.0)
    
    ax.add_patch( patches.Rectangle( (system_evals[i][4][0], 0.0),
                                     float(system_evals[i][4][1] - system_evals[i][4][0]),# width
                                     1.0, fill=True, alpha=0.15, hatch='-', color='r')) #height, etc.          
    eMF = fuzzyOps.rangeToMF(system_evals[i][4], 'trap')
    ax.plot(eMF[0], eMF[1], '-r', lw=2.0)
    ax.set_ylim([0.0, 1.01])
    ax.set_yticks([0.0, 0.5, 1.0])
    ax.set_ylabel(r' $\mu(x)$')
    ax.set_xlim([0.5, 1.0])
    if i > 7: ax.set_xlabel(r'System $\eta_P$')
    if i < 8: ax.set_xticklabels([])
    ax.text(np.average(system_evals[i][4]), 0.8, system_names[i], rotation=90, fontsize=10)
    ax.xaxis.grid(True)
    if i == 0: ax.legend(['DFES Output', 'Benchmark MF', 'Benchmark Range'],
                         bbox_to_anchor=(1.85, 1.38), ncol=3, fontsize=12)

plt.subplots_adjust(left=0.09, bottom=0.08, right=0.94, top=0.94, wspace=None, hspace=0.18)
    
### PLOT GWT results
fig = plt.figure(figsize=(10,8))
res_ind = 4
alpha = 1.0
for i in range(len(results)):
    ax = plt.subplot(5,2,i+1)
    ax.plot(results[i][res_ind][0],results[i][res_ind][1], lw=2.0)
    cent = fuzz.defuzz(np.array(results[i][res_ind][0]), np.array(results[i][res_ind][1]), 'centroid')
    plt.plot([cent, cent], [0.0,2.0], '-r', lw=2.5)
    ax.set_ylim([0.0, 1.01])
    ax.set_yticks([0.0, 0.5, 1.0])
    ax.set_ylabel(r' $\mu(x)$')
    #ax.set_xticks([5000, 10000, 15000, 20000])
    ax.set_xticks([5000, 10000, 15000, 20000, 25000, 30000, 35000, 40000, 45000])
    ax.set_xticklabels([5, 10, 15, 20, 25, 30, 35, 40, 45])
    #ax.set_xlim([1000,25000])
    if i > 7: ax.set_xlabel('Gross Weight (x1000 lbs)')
    if i < 8: ax.set_xticklabels([])
    ax.grid(True)
plt.subplots_adjust(left=0.09, bottom=0.08, right=0.94, top=0.94, wspace=None, hspace=0.18)

### PLOT Pinst results
fig = plt.figure(figsize=(10,8))
res_ind = 5
alpha = 1.0
for i in range(len(results)):
    ax = plt.subplot(5,2,i+1)
    ax.plot(results[i][res_ind][0],results[i][res_ind][1], lw=2.0)
    cent = fuzz.defuzz(np.array(results[i][res_ind][0]), np.array(results[i][res_ind][1]), 'centroid')
    plt.plot([cent, cent], [0.0,2.0], '-r', lw=2.5)
    ax.set_ylim([0.0, 1.01])
    ax.set_yticks([0.0, 0.5, 1.0])
    ax.set_ylabel(r' $\mu(x)$')
    #ax.set_xlim([1000,9000])
    if i > 7: ax.set_xlabel('Power Installed (hp)')
    if i < 8: ax.set_xticklabels([])
    ax.grid(True)
plt.subplots_adjust(left=0.09, bottom=0.08, right=0.94, top=0.94, wspace=None, hspace=0.18)

### PLOT MaxAS results
fig = plt.figure(figsize=(10,8))
res_ind = 6
alpha = 1.0
for i in range(len(results)):
    ax = plt.subplot(5,2,i+1)
    ax.plot(results[i][res_ind][0],results[i][res_ind][1], lw=2.0)
    
    ax.add_patch( patches.Rectangle( (system_evals[i][1][0], 0.0),
                                     float(system_evals[i][1][1] - system_evals[i][1][0]),# width
                                     1.0, fill=True, alpha=0.15, hatch='-', color='r')) #height, etc.          
    eMF = fuzzyOps.rangeToMF(system_evals[i][1], 'gauss')
    ax.plot(eMF[0], eMF[1], '-r', lw=2.0)
    ax.set_ylim([0.0, 1.01])
    ax.set_xlim([250, 500])
    if i > 7: ax.set_xlabel('Max Airspeed (kts)')
    if i < 8: ax.set_xticklabels([])
    ax.set_yticks([0.0, 0.5, 1.0])
    ax.set_ylabel(r' $\mu(x)$')
    ax.text(np.average(system_evals[i][1]), 0.8, system_names[i], rotation=90, fontsize=10)
    ax.xaxis.grid(True)
plt.subplots_adjust(left=0.09, bottom=0.08, right=0.94, top=0.94, wspace=None, hspace=0.18)


#### PLOT FUNCTIONS
"""
x_ind = 0 # Empty Weight
y_ind = 4 # Prop Eff.
pts = 80

colors = ['grey', 'r', 'orange', 'coral', 'orange', 'lawngreen', 'c', 'b', 'purple', 'pink']
fig = plt.figure(figsize=(10,8))

for i in range(len(results)):#range(len(results)):
    ax = plt.subplot(5,2,i+1)
    a_ix = fuzzyOps.alpha_cut(0.05, (results[i][x_ind][0],results[i][x_ind][1]))
    a_iy = fuzzyOps.alpha_cut(0.05, (results[i][y_ind][0],results[i][y_ind][1]))
    if a_ix <> None and a_iy <> None:
        try:
        xs = np.arange(a_ix[0], a_ix[1], (a_ix[1]-a_ix[0])/pts)
        ys = np.arange(a_iy[0], a_iy[1], (a_iy[1]-a_iy[0])/pts)
        mxs = np.interp(xs, results[i][x_ind][0], results[i][x_ind][1])
        mys = np.interp(ys, results[i][y_ind][0], results[i][y_ind][1])
        ms = np.ones((len(mxs), len(mys)))
        for j in range(len(mxs)):
            for k in range(len(mys)):
                ms[j,k] = min(mxs[j],mys[k])
        #ms = np.outer(mxs, mys)
        plt.pcolormesh(xs,ys,ms, cmap='Blues', vmin=0.0, vmax=1.0)
        cb = plt.colorbar()
        tick_locator = ticker.MaxNLocator(nbins=5)
        cb.locator = tick_locator
        cb.update_ticks()
        #cb.label
    ax.set_xlim([1.,9.])
    #ax.set_ylim([5,25])
    ax.set_ylim([0.0,1.0])
    if i%2 == 0: ax.set_ylabel('Propulsive Efficiency') #ax.set_ylabel('Lift/Drag Ratio')
    if i > 7: ax.set_xlabel(r'$\phi$ (1-Poor, 9-Excellent)')
    if i < 8: ax.set_xticklabels([])
plt.subplots_adjust(left=0.09, bottom=0.08, right=0.94, top=0.94, wspace=None, hspace=0.18)
"""

##Compare Alpha Cuts
alpha = 0.5
"""

#phi system
res_ind = 0
print "Empty Weight: Compare Ranges at alpha = %.2d" % alpha
for i in range(len(results)):
    ac = fuzzyOps.alpha_cut(alpha, results[i][res_ind])
    print "     ", system_names[i], "-     Expert:", system_evals[i][0],          
    if ac <> None: print "     System:", "[", round(ac[0],1), ",", round(ac[1],1), "]"
    else: print "     System:", "[None]"

#FoM system
res_ind = 1
print "Figure of Merit: Compare Ranges at alpha = %.2d" % alpha
for i in range(len(results)):
    ac = fuzzyOps.alpha_cut(alpha, results[i][res_ind])
    print "     ", system_names[i], "-     Expert:", system_evals[i][2],          
    if ac <> None: print "     System:", "[", round(ac[0],1), ",", round(ac[1],1), "]"
    else: print "     System:", "[None]"

#L/D system
res_ind = 2
print "Lift/Drag Ratio: Compare Ranges at alpha = %.2d" % alpha
for i in range(len(results)):
    ac = fuzzyOps.alpha_cut(alpha, results[i][res_ind])
    print "     ", system_names[i], "-     Expert:", system_evals[i][3],          
    if ac <> None: print "     System:", "[", round(ac[0],1), ",", round(ac[1],1), "]"
    else: print "     System:", "[None]"

#etaP system
res_ind = 3
print "System Propulsive Efficiency: Compare Ranges at alpha = %.2d" % alpha
for i in range(len(results)):
    ac = fuzzyOps.alpha_cut(alpha, results[i][res_ind])
    print "     ", system_names[i], "-     Expert:", system_evals[i][4],          
    if ac <> None: print "     System:", "[", round(ac[0],1), ",", round(ac[1],1), "]"
    else: print "     System:", "[None]"


#VH system
res_ind = 6
print "Maximum Airspeed: Compare Ranges at alpha = %.2d" % alpha
for i in range(len(results)):
    ac = fuzzyOps.alpha_cut(alpha, results[i][res_ind])
    print "     ", system_names[i], "-     Expert:", system_evals[i][1],          
    if ac <> None: print "     System:", "[", round(ac[0],1), ",", round(ac[1],1), "]"
    else: print "     System:", "[None]"
"""

plt.show()
#Test inputs
#for opt in system_options:
#    w = getInputs('VL_SYS', None, 'w', opt)
#    print w, '... union: ', [np.average([x[0] for x in w]), np.average([x[1] for x in w])]