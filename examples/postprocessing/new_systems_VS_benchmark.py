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

goalVals = {'sys_phi'  :6.7, 
            'sys_FoM'  :0.775, 
            'sys_LoD'  :12.5, 
            'sys_etaP' :0.875, 
            'sys_Pin'  :3500.0,
            'sys_GWT'  :10000,
            'sys_VH'   :325}

ASPECT_list = ['VL_SYS_TYPE', 'VL_SYS_PROP', 'VL_SYS_DRV', 'VL_SYS_TECH', \
            'FWD_SYS_PROP', 'FWD_SYS_DRV', 'FWD_SYS_TYPE',\
            'WING_SYS_TYPE', \
            'ENG_SYS_TYPE']   
                
                
system_names = [ #'Alt 1: BaseTR', 
                 #'Alt 2: BaseTW',
                 #'Alt 3: FIW_TJ', 
                 #'Alt 4: Tilt_FIW', 
                 #'Alt 5: StopRot',
                 'Alt 6: AutoGyro', 
                 'Alt 7: TwinTS', 
                 'Alt 8: FixedFIW', 
                 #'Alt 9: HeliPL', 
                 #'Alt 10: FIB-TD'
                ]
system_options = [  #[2,2,1,2,2,1,2,1,1], 
                    #[2,1,1,2,1,1,2,5,1],
                    #[4,3,3,1,4,3,1,2,2],
                    #[4,3,1,2,3,1,2,4,1],
                    #[1,2,2,3,4,3,1,6,3],
                    [1,2,1,5,1,1,1,1,1],
                    [6,2,1,2,2,1,1,1,1],
                    [4,3,1,1,1,1,1,4,1],
                    #[1,2,2,5,4,3,1,1,3],
                    #[5,3,1,1,3,1,4,1,4]
                ]
new_sys_names = [ 'Alt N1: Compound', 'Alt N2: TandemComp', 'Alt N1: FlyingFIW', 'Alt N1: CompTS', 'Alt X1: Test5', 'Alt X1: Test6' ]

newsys_options = [ [1, 2, 1, 2, 1, 1, 1, 1, 4],
                   [3, 1, 1, 2, 1, 1, 3, 1, 4],
                   [4, 3, 1, 1, 1, 1, 4, 3, 4],
                   [6, 1, 1, 2, 1, 1, 3, 1, 4]
                 ]
                    
                    #phi ,  maxAS   ,   FoM    , L/D  ,   eta_p
#system_evals = [    [[4.,6.],[300.,330.],[0.65,0.75],[8.,17.],[0.67,0.85]],
#                    [[3.,5.],[325.,370.],[0.67,0.77],[10.,17.],[0.75,0.90]],
#                    [[4.,8.],[380.,450.],[0.25,0.6], [6.,13.],[0.3,0.72]],
#                    [[3.,7.],[310.,340.],[0.6,0.8],  [8.,16.],[0.65,0.85]],
#                    [[3.,6.],[355.,400.],[0.3,0.55], [6.,11.],[0.4,0.72]],
#                    [[5.,7.],[305.,355.],[0.65,0.82],[9.,14.],[0.88,0.95]],
#                    [[7.,8.],[315.,365.],[0.62,0.78],[9.,16.],[0.65,0.9]],
#                    [[4.,7.],[330.,390.],[0.77,0.83],[11.,17.],[0.9,0.95]],
#                    [[5.,8.],[350.,410.],[0.3,0.55], [8.,16.],[0.35,0.72]],
#                    [[3.,7.],[330.,375.],[0.55,0.85],[10.,18.],[0.6,0.95]],]               
                    

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
results_new = [[] for opt in newsys_options]


## RUN SYSTEMS:

def runFramework(opts): #run each option set
    #print opts

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

    sysOut_phi = output_phi['sys_phi']
    
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
    
    sysOut_FoM = output_FM['sys_FoM']

    #Evaluate L/D
    _f = getInputs(None, None, 'f', opts)  #get drag (intersection)
    _f = [np.average([x[0] for x in _f]), np.average([x[1] for x in _f])]
    f = fuzzyOps.rangeToMF(_f, 'trap')
    
    _LDw   = getInputs(None, 'WING_SYS_TYPE', 'LD', opts)[0]
    _LDvt = getInputs(None, 'VL_SYS_TECH', 'LD', opts)[0]
    _LD = [max([_LDw[0], _LDvt[0]]), min([_LDw[1], _LDvt[1]])]
    LD = fuzzyOps.rangeToMF(_LD, 'trap')
    output_LD = LoD_sys.run({'SYSTEM_f': f, 'WING_LoD': LD})
    sysOut_LD = output_LD['sys_LoD']

    #Evaluate eta_P
    _FST_etap = getInputs(None, 'FWD_SYS_TYPE', 'eta_p', opts)[0]#  : FWD system type efficiency
    _FSD_etap = getInputs(None, 'FWD_SYS_DRV', 'eta_p', opts)[0]#  : FWD system type efficiency
    _FWD_etap = [max( [_FST_etap[0], _FSP_etap[0], _FSD_etap[0]] ), min( [_FST_etap[1], _FSP_etap[1], _FSD_etap[1]] )]
    FWD_etap = fuzzyOps.rangeToMF(_FWD_etap, 'trap')
    
    _FSD_etad = getInputs(None, 'FWD_SYS_DRV', 'eta_d', opts)[0]#  : FWD drive efficiency
    FSD_etad = fuzzyOps.rangeToMF(_FSD_etad, 'trap')
    output_etaP = etaP_sys.run({'FWD_SYS_eta_p': FWD_etap, 
                                'FWD_DRV_eta_d' : FSD_etad, })
    sysOut_etaP = output_etaP['sys_etaP']

    # Evaluate RF Methods
    inR = [1,9]
    outR = [0.85, 0.5]
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
    
    ###
    #print "w:", _w, "  WS:", _WST_WS, "  etad:", _eta, "  e_d:", _ed, "  type:", T
    ###
    output_GWT = GWT_sys.run({  'SYSTEM_QUANT_PHI': sysPHI , 'VL_SYS_w': w, 'WING_SYS_TYPE_WS': WST_WS, 
                                'sys_etaP': output_etaP['sys_etaP'], 'VL_SYS_DRV_eta_d': eta, 'sys_FoM': output_FM['sys_FoM'],
                                'VL_SYS_e_d': ed, 'ENG_SYS_TYPE_SFC': quantSFC, #'SYS_dragX': f,
                                'SYS_type': SYSTYPE,})# 'SYS_tech': SYSTECH,}) #'SYS_jet': SYSJET})
    sysOut_GWT = output_GWT['sys_GWT']

    
    output_Pin = Pin_sys.run({  'SYSTEM_QUANT_PHI': sysPHI , 'VL_SYS_w': w, 'WING_SYS_TYPE_WS': WST_WS, 
                                'sys_etaP': output_etaP['sys_etaP'], 'VL_SYS_DRV_eta_d': eta, 'sys_FoM': output_FM['sys_FoM'],
                                'VL_SYS_e_d': ed, 'ENG_SYS_TYPE_SFC': quantSFC, #'SYS_dragX': f,
                                'SYS_type': SYSTYPE,})# 'SYS_tech': SYSTECH,}) #'SYS_jet': SYSJET})
    sysOut_Pin = output_Pin['sys_Pinst']
    
    #Pin_sys.test([({  'SYSTEM_QUANT_PHI': sysPHI , 'VL_SYS_w': w, 'WING_SYS_TYPE_WS': WST_WS, 
    #                            'sys_etaP': output_etaP['sys_etaP'], 'VL_SYS_DRV_eta_d': eta, 'sys_FoM': output_FM['sys_FoM'],
    #                            'VL_SYS_e_d': ed, 'ENG_SYS_TYPE_SFC': quantSFC, #'SYS_dragX': f,
    #                            'SYS_type': SYSTYPE,}, output_Pin) ], plotPoints=1)
    
    output_VH = VH_sys.run({  'SYSTEM_QUANT_PHI': sysPHI , 'VL_SYS_w': w, 'WING_SYS_TYPE_WS': WST_WS, 
                                'sys_etaP': output_etaP['sys_etaP'], 'VL_SYS_DRV_eta_d': eta, 'sys_FoM': output_FM['sys_FoM'],
                                'VL_SYS_e_d': ed, 'ENG_SYS_TYPE_SFC': quantSFC, #'SYS_dragX': f,
                                'SYS_type': SYSTYPE,})# 'SYS_tech': SYSTECH,})# 'SYS_jet': SYSJET})
    sysOut_VH = output_VH['sys_VH']

    #output_eWT = eWT_sys.run({  'SYSTEM_QUANT_PHI': sysPHI , 'VL_SYS_w': w, 'WING_SYS_TYPE_WS': WST_WS, 
    #                            'sys_etaP': output_etaP['sys_etaP'], 'VL_SYS_DRV_eta_d': eta, 'sys_FoM': output_FM['sys_FoM'],
    #                            'VL_SYS_e_d': ed, 'ENG_SYS_TYPE_SFC': quantSFC, 'SYS_dragX': f,
    #                            'SYS_type': SYSTYPE, 'SYS_tech': SYSTECH, 'SYS_jet': SYSJET})
    #results[system_options.index(opts)].append(output_eWT['sys_eWT'])    

    return sysOut_phi, sysOut_FoM, sysOut_LD, sysOut_etaP, sysOut_GWT, sysOut_Pin, sysOut_VH


for opts in system_options: #for each option
    sysOut_phi, sysOut_FoM, sysOut_LD, sysOut_etaP, sysOut_GWT, sysOut_Pin, sysOut_VH = runFramework(opts)
    
    results[system_options.index(opts)].append(sysOut_phi)
    results[system_options.index(opts)].append(sysOut_FoM)
    results[system_options.index(opts)].append(sysOut_LD)
    results[system_options.index(opts)].append(sysOut_etaP)
    results[system_options.index(opts)].append(sysOut_GWT)
    results[system_options.index(opts)].append(sysOut_Pin)
    results[system_options.index(opts)].append(sysOut_VH)

for opts in newsys_options: #for each option
    sysOut_phi, sysOut_FoM, sysOut_LD, sysOut_etaP, sysOut_GWT, sysOut_Pin, sysOut_VH = runFramework(opts)
    
    results_new[newsys_options.index(opts)].append(sysOut_phi)
    results_new[newsys_options.index(opts)].append(sysOut_FoM)
    results_new[newsys_options.index(opts)].append(sysOut_LD)
    results_new[newsys_options.index(opts)].append(sysOut_etaP)
    results_new[newsys_options.index(opts)].append(sysOut_GWT)
    results_new[newsys_options.index(opts)].append(sysOut_Pin)
    results_new[newsys_options.index(opts)].append(sysOut_VH)
    
    
for i in range(len(results)): 
    print system_names[i]
    print "Objective        FPoS     Centroid"
    fpos1 = fuzzyOps.fuzzyPOS(results[i][0][0],results[i][0][1], goalVals['sys_phi'])
    cent = fuzz.defuzz(np.array(results[i][0][0]),np.array(results[i][0][1]), 'centroid')
    print "Phi:             %.3f     %.3f" % (fpos1, cent)
    fpos2 = fuzzyOps.fuzzyPOS(results[i][1][0],results[i][1][1], goalVals['sys_FoM'], plot=False)
    cent = fuzz.defuzz(np.array(results[i][1][0]),np.array(results[i][1][1]), 'centroid')
    print "FoM:             %.3f     %.3f" % (fpos2, cent)
    fpos3 = fuzzyOps.fuzzyPOS(results[i][2][0],results[i][2][1], goalVals['sys_LoD'])
    cent = fuzz.defuzz(np.array(results[i][2][0]),np.array(results[i][2][1]), 'centroid')
    print "L/D:             %.3f     %.3f" % (fpos3, cent)
    fpos4 = fuzzyOps.fuzzyPOS(results[i][3][0],results[i][3][1], goalVals['sys_etaP'])
    cent = fuzz.defuzz(np.array(results[i][3][0]),np.array(results[i][3][1]), 'centroid')
    print "etaP:            %.3f     %.3f" % (fpos4, cent)
    fpos5 = fuzzyOps.fuzzyPOS(results[i][5][0],results[i][5][1], goalVals['sys_Pin'], direction='min')
    cent = fuzz.defuzz(np.array(results[i][5][0]),np.array(results[i][5][1]), 'centroid')
    print "Pin:             %.3f     %.3f" % (fpos5, cent)
    print "TOTAL:           %.3f" % min(fpos1,fpos2,fpos3,fpos4,fpos5)
    
for i in range(len(results_new)): 
    print new_sys_names[i]
    print "Objective        FPoS     Centroid"
    fpos1 = fuzzyOps.fuzzyPOS(results_new[i][0][0],results_new[i][0][1], goalVals['sys_phi'])
    cent = fuzz.defuzz(np.array(results_new[i][0][0]),np.array(results_new[i][0][1]), 'centroid')
    print "Phi:             %.3f     %.3f" % (fpos1, cent)
    fpos2 = fuzzyOps.fuzzyPOS(results_new[i][1][0],results_new[i][1][1], goalVals['sys_FoM'], plot=False)
    cent = fuzz.defuzz(np.array(results_new[i][1][0]),np.array(results_new[i][1][1]), 'centroid')
    print "FoM:             %.3f     %.3f" % (fpos2, cent)
    fpos3 = fuzzyOps.fuzzyPOS(results_new[i][2][0],results_new[i][2][1], goalVals['sys_LoD'])
    cent = fuzz.defuzz(np.array(results_new[i][2][0]),np.array(results_new[i][2][1]), 'centroid')
    print "L/D:             %.3f     %.3f" % (fpos3, cent)
    fpos4 = fuzzyOps.fuzzyPOS(results_new[i][3][0],results_new[i][3][1], goalVals['sys_etaP'])
    cent = fuzz.defuzz(np.array(results_new[i][3][0]),np.array(results_new[i][3][1]), 'centroid')
    print "etaP:            %.3f     %.3f" % (fpos4, cent)
    fpos5 = fuzzyOps.fuzzyPOS(results_new[i][5][0],results_new[i][5][1], goalVals['sys_Pin'], direction='min')
    cent = fuzz.defuzz(np.array(results_new[i][5][0]),np.array(results_new[i][5][1]), 'centroid')
    print "Pin:             %.3f     %.3f" % (fpos5, cent)
    print "TOTAL:           %.3f" % min(fpos1,fpos2,fpos3,fpos4,fpos5)
   
   
colors_1 = ['darkred', 'r', 'orange', 'sienna', 'brown'] 
colors_2 = ['blue', 'green', 'cyan', 'purple', 'forest']
alpha = 0.55

## Empty Weight
fig = plt.figure(figsize=(10,2.5))
res_ind = 0
goal = 6.7
ax=fig.add_subplot(111)
hdls = []
for i in range(len(results)):
    x1 = ax.plot(results[i][res_ind][0],results[i][res_ind][1], '--', c=colors_1[i], lw=2.0, label= system_names[i], alpha=alpha)
    hdls.append(x1)
for i in range(len(results_new)):
    x1 = ax.plot(results_new[i][res_ind][0],results_new[i][res_ind][1], c=colors_2[i], lw=2.0, label= new_sys_names[i], alpha=alpha)
    hdls.append(x1)    
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles, labels, fontsize=9, loc=2)
plt.plot([goal, goal],[0.0,1.1], ':k',)
plt.text(goal, 0.3, 'GOAL', rotation=90, fontsize=12, weight='bold')
ax.set_ylim([0.0, 1.01])
ax.set_ylabel(r' $\mu(x)$')
ax.set_xlim([1.0, 9.0])
ax.set_xlabel(r'$\phi$ (1-Poor, 9-Excellent)') 
ax.set_xticks([1,2,3,4,5,6,7,8,9])
plt.subplots_adjust(bottom=0.22, left=0.07, right=0.97, top=0.96)


## FOM 
fig = plt.figure(figsize=(10,2.5))
res_ind = 1
goal = 0.775

ax=fig.add_subplot(111)
hdls = []
for i in range(len(results)):
    x1 = ax.plot(results[i][res_ind][0],results[i][res_ind][1], '--', c=colors_1[i], lw=2.0, label= system_names[i], alpha=alpha)
    hdls.append(x1)
for i in range(len(results_new)):
    x1 = ax.plot(results_new[i][res_ind][0],results_new[i][res_ind][1], c=colors_2[i], lw=2.0, label= new_sys_names[i], alpha=alpha)
    hdls.append(x1)    
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles, labels, fontsize=9, loc=2)
plt.plot([goal, goal],[0.0,1.1], ':k',)
plt.text(goal, 0.3, 'GOAL', rotation=90, fontsize=12, weight='bold')
ax.set_ylim([0.0, 1.01])
ax.set_ylabel(r' $\mu(x)$')
ax.set_xlim([0.5, 1.0])
ax.set_xlabel('Figure of Merit') 
#ax.set_xticks([1,2,3,4,5,6,7,8,9])
plt.subplots_adjust(bottom=0.22, left=0.07, right=0.97, top=0.96)

## L/D 
fig = plt.figure(figsize=(10,2.5))
res_ind = 2
goal = 12.5

ax=fig.add_subplot(111)
hdls = []
for i in range(len(results)):
    x1 = ax.plot(results[i][res_ind][0],results[i][res_ind][1], '--', c=colors_1[i], lw=2.0, label= system_names[i], alpha=alpha)
    hdls.append(x1)
for i in range(len(results_new)):
    x1 = ax.plot(results_new[i][res_ind][0],results_new[i][res_ind][1], c=colors_2[i], lw=2.0, label= new_sys_names[i], alpha=alpha)
    hdls.append(x1)    
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles, labels, fontsize=9, loc=2)
plt.plot([goal, goal],[0.0,1.1], ':k',)
plt.text(goal, 0.3, 'GOAL', rotation=90, fontsize=12, weight='bold')
ax.set_ylim([0.0, 1.01])
ax.set_ylabel(r' $\mu(x)$')
#ax.set_xlim([5,20])
ax.set_xlabel('Lift/Drag Ratio') 
#ax.set_xticks([1,2,3,4,5,6,7,8,9])
plt.subplots_adjust(bottom=0.22, left=0.07, right=0.97, top=0.96)

## etaP
fig = plt.figure(figsize=(10,2.5))
res_ind = 3
goal = 0.875
ax=fig.add_subplot(111)
hdls = []
for i in range(len(results)):
    x1 = ax.plot(results[i][res_ind][0],results[i][res_ind][1], '--', c=colors_1[i], lw=2.0, label= system_names[i], alpha=alpha)
    hdls.append(x1)
for i in range(len(results_new)):
    x1 = ax.plot(results_new[i][res_ind][0],results_new[i][res_ind][1], c=colors_2[i], lw=2.0, label= new_sys_names[i], alpha=alpha)
    hdls.append(x1)    
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles, labels, fontsize=9, loc=2)
plt.plot([goal, goal],[0.0,1.1], ':k',)
plt.text(goal, 0.3, 'GOAL', rotation=90, fontsize=12, weight='bold')
ax.set_ylim([0.0, 1.01])
ax.set_ylabel(r' $\mu(x)$')
#ax.set_xlim([5,20])
ax.set_xlabel('Propulsive Efficiency') 
#ax.set_xticks([1,2,3,4,5,6,7,8,9])
plt.subplots_adjust(bottom=0.22, left=0.07, right=0.97, top=0.96)


## Pinst
fig = plt.figure(figsize=(10,2.5))
res_ind = 5
goal = 4500
ax=fig.add_subplot(111)
hdls = []
for i in range(len(results)):
    x1 = ax.plot(results[i][res_ind][0],results[i][res_ind][1], '--', c=colors_1[i], lw=2.0, label= system_names[i], alpha=alpha)
    hdls.append(x1)
for i in range(len(results_new)):
    x1 = ax.plot(results_new[i][res_ind][0],results_new[i][res_ind][1], c=colors_2[i], lw=2.0, label= new_sys_names[i], alpha=alpha)
    hdls.append(x1)    
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles, labels, fontsize=9, loc=2)
plt.plot([goal, goal],[0.0,1.1], ':k',)
plt.text(goal, 0.3, 'GOAL', rotation=90, fontsize=12, weight='bold')
ax.set_ylim([0.0, 1.01])
ax.set_ylabel(r' $\mu(x)$')
ax.set_xlim([1000,6000])
ax.set_xlabel('Installed Power') 
#ax.set_xticks([1,2,3,4,5,6,7,8,9])
plt.subplots_adjust(bottom=0.22, left=0.07, right=0.97, top=0.96)

"""
### PLOT phi results
fig = plt.figure(figsize=(10,8))
res_ind = 0
alpha = 1.0
for i in range(len(results)):
    ax = plt.subplot(5+len(newsys_options)/2,2,i+1)
    ax.plot(results[i][res_ind][0],results[i][res_ind][1], lw=2.0)
    cent = fuzz.defuzz(np.array(results[i][res_ind][0]), np.array(results[i][res_ind][1]), 'centroid')
    plt.plot([cent, cent], [0.0,2.0], '-r', lw=2.5)
    
    ax.set_ylim([0.0, 1.01])
    ax.set_xlim([1.0, 9.0])
    ax.set_yticks([0.0, 0.5, 1.0])
    ax.set_ylabel(r' $\mu(x)$')
    if i > 7: ax.set_xlabel(r'$\phi$ (1-Poor, 9-Excellent)') 
    if i < 8: ax.set_xticks([1,2,3,4,5,6,7,8,9])
    #ax.text(np.average(system_evals[i][0]), 0.8, system_names[i], rotation=90, fontsize=10)
    ax.xaxis.grid(True)
    if i == 0: ax.legend(['DFES Output', 'Benchmark MF', 'Benchmark Range'],
                         bbox_to_anchor=(1.85, 1.38), ncol=3, fontsize=12)

for i in range(len(results_new)):
    ax = plt.subplot(5+len(newsys_options)/2,2,i+1+len(system_options))
    ax.plot(results_new[i][res_ind][0],results_new[i][res_ind][1], lw=2.0)
    cent = fuzz.defuzz(np.array(results_new[i][res_ind][0]), np.array(results_new[i][res_ind][1]), 'centroid')
    plt.plot([cent, cent], [0.0,2.0], '-r', lw=2.5)
    
    ax.set_ylim([0.0, 1.01])
    ax.set_xlim([1.0, 9.0])
    ax.set_yticks([0.0, 0.5, 1.0])
    ax.set_ylabel(r' $\mu(x)$')
    if i > 7: ax.set_xlabel(r'$\phi$ (1-Poor, 9-Excellent)') 
    if i < 8: ax.set_xticks([1,2,3,4,5,6,7,8,9])
    #ax.text(np.average(system_evals[i][0]), 0.8, system_names[i], rotation=90, fontsize=10)
    ax.xaxis.grid(True)
    if i == 0: ax.legend(['DFES Output', 'Benchmark MF', 'Benchmark Range'],
                         bbox_to_anchor=(1.85, 1.38), ncol=3, fontsize=12)
                         
plt.subplots_adjust(left=0.09, bottom=0.08, right=0.94, top=0.94, wspace=None, hspace=0.18)
   
   
### PLOT FoM results
fig = plt.figure(figsize=(10,8))
res_ind = 1
alpha = 1.0
for i in range(len(results)):
    ax = plt.subplot(5+len(newsys_options)/2,2,i+1)
    ax.plot(results[i][res_ind][0],results[i][res_ind][1], lw=2.0)
    cent = fuzz.defuzz(np.array(results[i][res_ind][0]), np.array(results[i][res_ind][1]), 'centroid')
    plt.plot([cent, cent], [0.0,2.0], '-r', lw=2.5)
    
    ax.set_ylim([0.0, 1.01])
    ax.set_yticks([0.0, 0.5, 1.0])
    ax.set_ylabel(r' $\mu(x)$')
    ax.set_xlim([0.4, 1.0])
    if i > 7: ax.set_xlabel('FoM')
    if i < 8: ax.set_xticklabels([])
    #ax.text(np.average(system_evals[i][2]), 0.8, system_names[i], rotation=90, fontsize=10)
    ax.xaxis.grid(True)
    if i == 0: ax.legend(['DFES Output', 'Benchmark MF', 'Benchmark Range'],
                         bbox_to_anchor=(1.85, 1.38), ncol=3, fontsize=12)

for i in range(len(results_new)):
    ax = plt.subplot(5+len(newsys_options)/2,2,i+1+len(system_options))
    ax.plot(results_new[i][res_ind][0],results_new[i][res_ind][1], lw=2.0)
    cent = fuzz.defuzz(np.array(results_new[i][res_ind][0]), np.array(results_new[i][res_ind][1]), 'centroid')
    plt.plot([cent, cent], [0.0,2.0], '-r', lw=2.5)
    
    ax.set_ylim([0.0, 1.01])
    ax.set_yticks([0.0, 0.5, 1.0])
    ax.set_ylabel(r' $\mu(x)$')
    ax.set_xlim([0.4, 1.0])
    if i > 7: ax.set_xlabel('FoM')
    if i < 8: ax.set_xticklabels([])
    #ax.text(np.average(system_evals[i][2]), 0.8, system_names[i], rotation=90, fontsize=10)
    ax.xaxis.grid(True)
    if i == 0: ax.legend(['DFES Output', 'Benchmark MF', 'Benchmark Range'],
                         bbox_to_anchor=(1.85, 1.38), ncol=3, fontsize=12)
                         
plt.subplots_adjust(left=0.09, bottom=0.08, right=0.94, top=0.94, wspace=None, hspace=0.18)


### PLOT L/D results
fig = plt.figure(figsize=(10,8))
res_ind = 2
alpha = 1.0
for i in range(len(results)):
    ax = plt.subplot(5+len(newsys_options)/2,2,i+1)
    ax.plot(results[i][res_ind][0],results[i][res_ind][1], lw=2.0)
    cent = fuzz.defuzz(np.array(results[i][res_ind][0]), np.array(results[i][res_ind][1]), 'centroid')
    plt.plot([cent, cent], [0.0,2.0], '-r', lw=2.5)
    
    ax.set_ylim([0.0, 1.01])
    ax.set_yticks([0.0, 0.5, 1.0])
    ax.set_ylabel(r' $\mu(x)$')
    ax.set_xlim([5., 25.])
    if i > 7: ax.set_xlabel('L/D')
    if i < 8: ax.set_xticklabels([])
    #ax.text(np.average(system_evals[i][3]), 0.8, system_names[i], rotation=90, fontsize=10)
    ax.xaxis.grid(True)
    if i == 0: ax.legend(['DFES Output', 'Benchmark MF', 'Benchmark Range'],
                         bbox_to_anchor=(1.85, 1.38), ncol=3, fontsize=12)

plt.subplots_adjust(left=0.09, bottom=0.08, right=0.94, top=0.94, wspace=None, hspace=0.18)
    
### PLOT etaP results
fig = plt.figure(figsize=(10,8))
res_ind = 3
alpha = 1.0
for i in range(len(results)):
    ax = plt.subplot(5+len(newsys_options)/2,2,i+1)
    ax.plot(results[i][res_ind][0],results[i][res_ind][1], lw=2.0)
    cent = fuzz.defuzz(np.array(results[i][res_ind][0]), np.array(results[i][res_ind][1]), 'centroid')
    plt.plot([cent, cent], [0.0,2.0], '-r', lw=2.5)
    
    ax.set_ylim([0.0, 1.01])
    ax.set_yticks([0.0, 0.5, 1.0])
    ax.set_ylabel(r' $\mu(x)$')
    ax.set_xlim([0.5, 1.0])
    if i > 7: ax.set_xlabel(r'System $\eta_P$')
    if i < 8: ax.set_xticklabels([])
    #ax.text(np.average(system_evals[i][4]), 0.8, system_names[i], rotation=90, fontsize=10)
    ax.xaxis.grid(True)
    if i == 0: ax.legend(['DFES Output', 'Benchmark MF', 'Benchmark Range'],
                         bbox_to_anchor=(1.85, 1.38), ncol=3, fontsize=12)

plt.subplots_adjust(left=0.09, bottom=0.08, right=0.94, top=0.94, wspace=None, hspace=0.18)
    
### PLOT GWT results
fig = plt.figure(figsize=(10,8))
res_ind = 4
alpha = 1.0
for i in range(len(results)):
    ax = plt.subplot(5+len(newsys_options)/2,2,i+1)
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
    ax = plt.subplot(5+len(newsys_options)/2,2,i+1)
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
    ax = plt.subplot(5+len(newsys_options)/2,2,i+1)
    ax.plot(results[i][res_ind][0],results[i][res_ind][1], lw=2.0)
    cent = fuzz.defuzz(np.array(results[i][res_ind][0]), np.array(results[i][res_ind][1]), 'centroid')
    plt.plot([cent, cent], [0.0,2.0], '-r', lw=2.5)
    
    ax.set_ylim([0.0, 1.01])
    ax.set_xlim([250, 500])
    if i > 7: ax.set_xlabel('Max Airspeed (kts)')
    if i < 8: ax.set_xticklabels([])
    ax.set_yticks([0.0, 0.5, 1.0])
    ax.set_ylabel(r' $\mu(x)$')
    #ax.text(np.average(system_evals[i][1]), 0.8, system_names[i], rotation=90, fontsize=10)
    ax.xaxis.grid(True)
plt.subplots_adjust(left=0.09, bottom=0.08, right=0.94, top=0.94, wspace=None, hspace=0.18)
"""

alts = [0]
res_all = results+results_new
x_ind = 0 # Empty Weight
y_ind = 3 # 
pts = 20

colors = ['grey', 'r', 'orange', 'coral', 'orange', 'lawngreen', 'c', 'b', 'purple', 'pink']
fig = plt.figure(figsize=(10,8))
ax = fig.add_subplot(111, projection='3d')
for i in alts:#range(len(results)):
    #ax = plt.subplot(4,2,i+1)
    a_ix = fuzzyOps.alpha_cut(0.05, (res_all[i][x_ind][0],res_all[i][x_ind][1]))
    a_iy = fuzzyOps.alpha_cut(0.05, (res_all[i][y_ind][0],res_all[i][y_ind][1]))
    if a_ix <> None and a_iy <> None:
        xs = np.arange(a_ix[0], a_ix[1], (a_ix[1]-a_ix[0])/pts)
        ys = np.arange(a_iy[0], a_iy[1], (a_iy[1]-a_iy[0])/pts)
        mxs = np.interp(xs, results[i][x_ind][0], results[i][x_ind][1])
        mys = np.interp(ys, results[i][y_ind][0], results[i][y_ind][1])
        ms = np.ones((len(mxs), len(mys)))
        for j in range(len(mxs)):
            for k in range(len(mys)):
                ms[j,k] = min(mxs[j],mys[k])
        
        xdat, ydat = np.meshgrid( mxs, mys )
        xdat.flatten()
        ydat.flatten()
        ms.flatten()

        dx = (a_ix[1]-a_ix[0])/pts
        dy = (a_iy[1]-a_iy[0])/pts
        
        ax.bar3d(xdat, ydat, np.zeros(len(ms)), dx, dy, ms )
        #plt.pcolormesh(xs,ys,ms, cmap='Blues', vmin=0.0, vmax=1.0)
        #cb = plt.colorbar()
        #tick_locator = ticker.MaxNLocator(nbins=5)
        #cb.locator = tick_locator
        #cb.update_ticks()
        #cb.set_label('Joint Membership Value', rotation=270, labelpad=15, fontsize=10)
        
    #ax.set_xlim([1.,9.])
    #ax.set_ylim([5,25])
    #ax.set_ylim([0.0,1.0])
    #if i%2 == 0: ax.set_ylabel('Propulsive Efficiency') #ax.set_ylabel('Lift/Drag Ratio')
    #if i > 7: ax.set_xlabel(r'$\phi$ (1-Poor, 9-Excellent)')
    #if i < 8: ax.set_xticklabels([])
plt.subplots_adjust(left=0.09, bottom=0.08, right=0.94, top=0.94, wspace=None, hspace=0.18)


plt.show()


