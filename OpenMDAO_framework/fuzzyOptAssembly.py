# -*- coding: utf-8 -*-
"""
Created on Thu Jan  8 10:54:15 2015

@author: frankpatterson
"""
import sys
import os
import time


from openmdao.main.api import Assembly
#from openmdao.lib.drivers.api import CaseIteratorDriver
#from pyopt_driver.pyopt_driver import pyOptDriver
from testdriver import GenDriver

from fuzzy_inputs import Build_Fuzzy_Input, Input_List, Quantify
from fuzzy_system import Fuzzy_System
from fuzzy_outputs import Postprocess_Fuzzy_Outputs
from morph_compatibility import checkCompatibility
from dfes import DFES

from openmdao.lib.casehandlers.api import JSONCaseRecorder, CSVCaseRecorder


class FuzzyAssemblyOpt(Assembly):
    """TEST ASSEMBLY FFAS"""

    def configure(self):

        # Create Optimizer instance
        #self.add('driver', CaseIteratorDriver())
        self.add('driver', GenDriver())
        
        # Create component instances
        self.add('input_list', Input_List())
        self.add('postprocess', Postprocess_Fuzzy_Outputs())
        self.add('compatibility', checkCompatibility())

        #PHI SYSTEM
        self.add('fuzz_combine_phi', Build_Fuzzy_Input())
        self.add('phi_sys', Fuzzy_System())

        #Lift/Drag SYSTEM
        self.add('fuzz_combine_LD', Build_Fuzzy_Input())
        self.add('LoD_sys', Fuzzy_System())

        #FoM SYSTEM
        self.add('fuzz_combine_FoM', Build_Fuzzy_Input())
        self.add('FoM_sys', DFES())

        #Propulsive Efficiency System
        self.add('fuzz_combine_etaP', Build_Fuzzy_Input())
        self.add('etaP_sys', Fuzzy_System())

        #QUANTIFICATION
        self.add('quantifyPHI', Quantify())
        self.add('quantifySFC', Quantify())

        #RF SYSTEMs
        self.add('fuzz_combine_GWT', Build_Fuzzy_Input())
        self.add('GWT_sys', DFES())
        self.add('fuzz_combine_P', Build_Fuzzy_Input())
        self.add('P_sys', DFES())
        self.add('fuzz_combine_VH', Build_Fuzzy_Input())
        self.add('VH_sys', DFES())
        

        # Iteration Hierarchy
        self.driver.workflow.add([  'compatibility', 'input_list',
                                    'fuzz_combine_phi', 'phi_sys',
                                    'fuzz_combine_LD',  'LoD_sys', 
                                    'fuzz_combine_FoM', 'FoM_sys', 
                                    'fuzz_combine_etaP', 'etaP_sys',
                                    'quantifyPHI', 'quantifySFC', 
                                    'GWT_sys', 'P_sys', 'VH_sys',
                                    'postprocess' ])

        self.connect('compatibility.option1_out', 'input_list.option1')
        self.connect('compatibility.option2_out', 'input_list.option2')
        self.connect('compatibility.option3_out', 'input_list.option3')
        self.connect('compatibility.option4_out', 'input_list.option4')
        self.connect('compatibility.option5_out', 'input_list.option5')
        self.connect('compatibility.option6_out', 'input_list.option6')
        self.connect('compatibility.option7_out', 'input_list.option7')
        self.connect('compatibility.option8_out', 'input_list.option8')
        self.connect('compatibility.option9_out', 'input_list.option9')
        self.connect('compatibility.compatibility', 'input_list.passthrough')
        self.connect('compatibility.compatibility', 'postprocess.passthrough')
        self.connect('compatibility.incompatCount', 'postprocess.incompatCount')

        self.connect('compatibility.compatibility', 'phi_sys.passthrough')
        self.connect('compatibility.compatibility', 'LoD_sys.passthrough')
        self.connect('compatibility.compatibility', 'FoM_sys.passthrough')
        self.connect('compatibility.compatibility', 'etaP_sys.passthrough')
        self.connect('compatibility.compatibility', 'GWT_sys.passthrough')
        self.connect('compatibility.compatibility', 'P_sys.passthrough')
        self.connect('compatibility.compatibility', 'VH_sys.passthrough')

        self.connect('compatibility.compatibility', 'quantifyPHI.passthrough')
        self.connect('compatibility.compatibility', 'quantifySFC.passthrough')


        ## CONNECT phi system
        #connect inputs for phi system
        self.connect("input_list.VL_SYS_TYPE_phi", "fuzz_combine_phi.in_1.input_value")
        self.connect("input_list.VL_SYS_TYPE_w", "fuzz_combine_phi.in_2.input_value")
        self.connect("input_list.VL_SYS_TYPE_f", "fuzz_combine_phi.in_3.input_value")
        
        self.connect("input_list.VL_SYS_PROP_w", "fuzz_combine_phi.in_4.input_value")
        self.connect("input_list.VL_SYS_PROP_phi", "fuzz_combine_phi.in_5.input_value")
        
        self.connect("input_list.VL_SYS_TECH_phi", "fuzz_combine_phi.in_6.input_value")
        self.connect("input_list.VL_SYS_TECH_w", "fuzz_combine_phi.in_7.input_value")
        self.connect("input_list.VL_SYS_TECH_f", "fuzz_combine_phi.in_8.input_value")
        self.connect("input_list.VL_SYS_TECH_LD", "fuzz_combine_phi.in_9.input_value")
        
        self.connect("input_list.FWD_SYS_PROP_eta_p", "fuzz_combine_phi.in_10.input_value")
        
        self.connect("input_list.FWD_SYS_DRV_eta_d", "fuzz_combine_phi.in_11.input_value")
        
        self.connect("input_list.FWD_SYS_TYPE_phi", "fuzz_combine_phi.in_12.input_value")
        self.connect("input_list.FWD_SYS_TYPE_TP", "fuzz_combine_phi.in_13.input_value")
        
        self.connect("input_list.WING_SYS_TYPE_LD", "fuzz_combine_phi.in_14.input_value")
        self.connect("input_list.WING_SYS_TYPE_f", "fuzz_combine_phi.in_15.input_value")


        self.connect("fuzz_combine_phi.system_inputs","phi_sys.input_list")
        self.connect("fuzz_combine_phi.runFlag_out", "phi_sys.runFlag_in")

        self.postprocess.fuzzSys_in_1.mf_key = 'sys_phi'
        self.connect('phi_sys.outputs_all', 'postprocess.fuzzSys_in_1.mf_dict')


        ## CONNECT LoD System
        self.connect("input_list.SYSTEM_f", "fuzz_combine_LD.in_1.input_value")
        self.connect("input_list.WING_LoD", "fuzz_combine_LD.in_2.input_value")
        
        self.connect("fuzz_combine_LD.system_inputs", "LoD_sys.input_list")
        self.connect("fuzz_combine_LD.runFlag_out", "LoD_sys.runFlag_in")

        self.postprocess.fuzzSys_in_2.mf_key = 'sys_LoD'
        self.connect('LoD_sys.outputs_all', 'postprocess.fuzzSys_in_2.mf_dict')


        ## CONNECT FoM System
        self.connect("input_list.VL_SYS_e_d", "fuzz_combine_FoM.in_1.input_value")
        self.connect("input_list.VL_SYS_PROP_sigma", "fuzz_combine_FoM.in_2.input_value")
        self.connect("input_list.VL_SYS_w", "fuzz_combine_FoM.in_3.input_value")
        self.connect("input_list.VL_SYS_DRV_eta_d", "fuzz_combine_FoM.in_4.input_value")
   
        self.connect("fuzz_combine_FoM.system_inputs","FoM_sys.input_list")
        self.connect("fuzz_combine_FoM.runFlag_out", "FoM_sys.runFlag_in")

        self.postprocess.fuzzSys_in_3.mf_key = 'sys_FoM'
        self.connect('FoM_sys.outputs_all', 'postprocess.fuzzSys_in_3.mf_dict')


        ## CONNECT etaP System
        self.connect("input_list.FWD_SYS_eta_p", "fuzz_combine_etaP.in_1.input_value")
        self.connect("input_list.FWD_DRV_eta_d", "fuzz_combine_etaP.in_2.input_value")
        
        self.connect("fuzz_combine_etaP.system_inputs", "etaP_sys.input_list")
        self.connect("fuzz_combine_etaP.runFlag_out", "etaP_sys.runFlag_in")

        self.postprocess.fuzzSys_in_4.mf_key = 'sys_etaP'
        self.connect('etaP_sys.outputs_all', 'postprocess.fuzzSys_in_4.mf_dict')

        self.postprocess.PLOTMODE = 0

        ## CONNECT RF Systems
        self.connect('phi_sys.outputs_all', 'quantifyPHI.inDict') #key defined below
        self.connect('input_list.ENG_SYS_TYPE_SFC', 'quantifySFC.qualVal')

        #GWT:
        self.connect("quantifyPHI.quantVal",        "fuzz_combine_GWT.in_1.input_value")
        self.connect("input_list.VL_SYS_w",         "fuzz_combine_GWT.in_2.input_value")
        self.connect("input_list.WING_SYS_TYPE_WS", "fuzz_combine_GWT.in_3.input_value")
        self.connect("etaP_sys.outputs_all",        "fuzz_combine_GWT.inDict_1.input_dict")
        self.connect("input_list.VL_SYS_DRV_eta_d", "fuzz_combine_GWT.in_4.input_value")
        self.connect("FoM_sys.outputs_all",         "fuzz_combine_GWT.inDict_2.input_dict")
        self.connect("input_list.VL_SYS_e_d",       "fuzz_combine_GWT.in_5.input_value")
        self.connect("quantifySFC.quantVal",        "fuzz_combine_GWT.in_6.input_value")
        self.connect("input_list.SYS_type",         "fuzz_combine_GWT.in_7.input_value")
        #self.connect("input_list.SYS_tech",         "fuzz_combine_GWT.in_8.input_value")

        self.connect("fuzz_combine_GWT.system_inputs","GWT_sys.input_list")
        self.connect("fuzz_combine_GWT.runFlag_out", "GWT_sys.runFlag_in")

        self.postprocess.fuzzSys_in_5.mf_key = 'sys_GWT'
        self.connect('GWT_sys.outputs_all', 'postprocess.fuzzSys_in_5.mf_dict')

        #Power Installed:
        self.connect("quantifyPHI.quantVal",        "fuzz_combine_P.in_1.input_value")
        self.connect("input_list.VL_SYS_w",         "fuzz_combine_P.in_2.input_value")
        self.connect("input_list.WING_SYS_TYPE_WS", "fuzz_combine_P.in_3.input_value")
        self.connect("etaP_sys.outputs_all",        "fuzz_combine_P.inDict_1.input_dict")
        self.connect("input_list.VL_SYS_DRV_eta_d", "fuzz_combine_P.in_4.input_value")
        self.connect("FoM_sys.outputs_all",         "fuzz_combine_P.inDict_2.input_dict")
        self.connect("input_list.VL_SYS_e_d",       "fuzz_combine_P.in_5.input_value")
        self.connect("quantifySFC.quantVal",        "fuzz_combine_P.in_6.input_value")
        self.connect("input_list.SYS_type",         "fuzz_combine_P.in_7.input_value")
        #self.connect("input_list.SYS_tech",         "fuzz_combine_P.in_8.input_value")

        self.connect("fuzz_combine_P.system_inputs","P_sys.input_list")
        self.connect("fuzz_combine_P.runFlag_out", "P_sys.runFlag_in")

        self.postprocess.fuzzSys_in_6.mf_key = 'sys_P'
        self.connect('P_sys.outputs_all', 'postprocess.fuzzSys_in_6.mf_dict') 

        #VH:
        self.connect("quantifyPHI.quantVal",        "fuzz_combine_VH.in_1.input_value")
        self.connect("input_list.VL_SYS_w",         "fuzz_combine_VH.in_2.input_value")
        self.connect("input_list.WING_SYS_TYPE_WS", "fuzz_combine_VH.in_3.input_value")
        self.connect("etaP_sys.outputs_all",        "fuzz_combine_VH.inDict_1.input_dict")
        self.connect("input_list.VL_SYS_DRV_eta_d", "fuzz_combine_VH.in_4.input_value")
        self.connect("FoM_sys.outputs_all",         "fuzz_combine_VH.inDict_2.input_dict")
        self.connect("input_list.VL_SYS_e_d",       "fuzz_combine_VH.in_5.input_value")
        self.connect("quantifySFC.quantVal",        "fuzz_combine_VH.in_6.input_value")
        self.connect("input_list.SYS_type",         "fuzz_combine_VH.in_7.input_value")
        #self.connect("input_list.SYS_tech",         "fuzz_combine_VH.in_8.input_value")

        self.connect("fuzz_combine_VH.system_inputs","VH_sys.input_list")
        self.connect("fuzz_combine_VH.runFlag_out", "VH_sys.runFlag_in")

        self.postprocess.fuzzSys_in_7.mf_key = 'sys_VH'
        self.connect('VH_sys.outputs_all', 'postprocess.fuzzSys_in_7.mf_dict')       


        ######
        # Design Variables
        self.input_list.input_file = 'data/morphInputs_13Jun15.csv'

        ## SET VARIABLES phi system
        self.phi_sys.TESTMODE = 0
        self.phi_sys.TESTPLOT = 0
        self.phi_sys.fcl_file = 'FCL_files/FRBS_phi/PHIsys_trained_5-2In9-2Out_gauss250tData_50vData_optInputsBEST_GA.fcl'    

        self.fuzz_combine_phi.in_1.input_key = 'VL_SYS_TYPE_phi'
        self.fuzz_combine_phi.in_2.input_key = 'VL_SYS_TYPE_w'
        self.fuzz_combine_phi.in_3.input_key = 'VL_SYS_TYPE_f'
        self.fuzz_combine_phi.in_4.input_key = 'VL_SYS_PROP_w'
        self.fuzz_combine_phi.in_5.input_key = 'VL_SYS_PROP_phi'
        self.fuzz_combine_phi.in_6.input_key = 'VL_SYS_TECH_phi'
        self.fuzz_combine_phi.in_7.input_key = 'VL_SYS_TECH_w'
        self.fuzz_combine_phi.in_8.input_key = 'VL_SYS_TECH_f'
        self.fuzz_combine_phi.in_9.input_key = 'VL_SYS_TECH_LD'
        self.fuzz_combine_phi.in_10.input_key = 'FWD_SYS_PROP_eta_p'
        self.fuzz_combine_phi.in_11.input_key = 'FWD_SYS_DRV_eta_d'
        self.fuzz_combine_phi.in_12.input_key = 'FWD_SYS_TYPE_phi'
        self.fuzz_combine_phi.in_13.input_key = 'FWD_SYS_TYPE_TP'
        self.fuzz_combine_phi.in_14.input_key = 'WING_SYS_TYPE_LD'
        self.fuzz_combine_phi.in_15.input_key = 'WING_SYS_TYPE_f'

        ## SET VARIABLES LoD system
        self.LoD_sys.TESTMODE = 0
        self.LoD_sys.TESTPLOT = 0
        self.LoD_sys.fcl_file = 'FCL_files/LoDsys_simple_13Jun15.fcl'

        self.fuzz_combine_LD.in_1.input_key = 'SYSTEM_f'
        self.fuzz_combine_LD.in_2.input_key = 'WING_LoD'


        ## SET VARIABLES FoM system
        self.FoM_sys.weight_file = 'FCL_files/DFES_FoM/DFES_FOMdata_data(500)_nodes(160_50_50).nwf'
        self.FoM_sys.inRanges = { 'VL_SYS_e_d':           [0.0,  0.3 ],
                                  'VL_SYS_PROP_sigma':    [0.05, 0.4 ],
                                  'VL_SYS_w':             [0.,   150.],
                                  'VL_SYS_DRV_eta':       [0.5,  1.0 ],}
        self.FoM_sys.inOrder = ['VL_SYS_e_d', 'VL_SYS_w', 'VL_SYS_DRV_eta', 'VL_SYS_PROP_sigma']
        self.FoM_sys.outRanges = {  'sys_FoM' :           [0.4,  1.0 ] }
        self.FoM_sys.actType = 'sigmoid'
        self.FoM_sys.hidNodes = 160
        self.FoM_sys.inGran   = 50
        self.FoM_sys.outGran  = 50

        self.FoM_sys.TESTPLOT = 0

        self.fuzz_combine_FoM.in_1.input_key ='VL_SYS_e_d'
        self.fuzz_combine_FoM.in_2.input_key ='VL_SYS_PROP_sigma'
        self.fuzz_combine_FoM.in_3.input_key ='VL_SYS_w'
        self.fuzz_combine_FoM.in_4.input_key ='VL_SYS_DRV_eta'


        ## SET VARIABLES etaP system
        self.etaP_sys.TESTMODE = 0
        self.etaP_sys.TESTPLOT = 0
        self.etaP_sys.fcl_file = 'FCL_files/etaPsys_simple_14Aug15.fcl'

        self.fuzz_combine_etaP.in_1.input_key = 'FWD_SYS_eta_p'
        self.fuzz_combine_etaP.in_2.input_key = 'FWD_DRV_eta_d'


        ## SET VARIABLES RF systems
        self.quantifyPHI.quantRange = [0.85, 0.50]
        self.quantifyPHI.inKey = 'sys_phi'
        self.quantifySFC.quantRange = [0.75, 0.35]

        ## SET VARIABLES RF:
        #GWT:
        self.GWT_sys.weight_file = 'FCL_files/DFES_RF/BEST/RFdata20Aug_DFES_GWT_30_250_50.nwf'
        self.GWT_sys.inRanges = { 'SYSTEM_QUANT_PHI':   [0.5,  0.85 ],
                                  'VL_SYS_w':           [1., 150.],
                                  'WING_SYS_TYPE_WS':   [15., 300.],
                                  'sys_etaP':           [0.6,  1.0 ],
                                  'VL_SYS_DRV_eta_d':   [0.7,  1.0 ],
                                  'sys_FoM':            [0.3,  1.0 ],
                                  'VL_SYS_e_d':         [0.0,  0.3 ],
                                  'ENG_SYS_TYPE_SFC':   [0.35, 0.75 ],
                                  'SYS_type':           [0.5,  3.5 ],}
                               
        self.GWT_sys.inOrder = ['VL_SYS_e_d', 'WING_SYS_TYPE_WS', 'SYSTEM_QUANT_PHI', 'SYS_type', 'VL_SYS_DRV_eta_d', 'sys_FoM', 'VL_SYS_w', 'sys_etaP', 'ENG_SYS_TYPE_SFC']
                                #['SYSTEM_QUANT_PHI', 'VL_SYS_w', 'WING_SYS_TYPE_WS', 'sys_etaP',
                                #    'VL_SYS_DRV_eta_d', 'sys_FoM','VL_SYS_e_d', 'ENG_SYS_TYPE_SFC', 'SYS_type']
        self.GWT_sys.outRanges = {'sys_GWT' :           [5000, 50000]}
        self.GWT_sys.actType = 'sigmoid'
        self.GWT_sys.hidNodes = 250
        self.GWT_sys.inGran   = 30
        self.GWT_sys.outGran  = 50
        
        self.fuzz_combine_GWT.in_1.input_key ='SYSTEM_QUANT_PHI'
        self.fuzz_combine_GWT.in_2.input_key ='VL_SYS_w'
        self.fuzz_combine_GWT.in_3.input_key ='WING_SYS_TYPE_WS'
        self.fuzz_combine_GWT.inDict_1.input_keys = ['sys_etaP']
        self.fuzz_combine_GWT.in_4.input_key ='VL_SYS_DRV_eta_d'
        self.fuzz_combine_GWT.inDict_2.input_keys = ['sys_FoM']
        self.fuzz_combine_GWT.in_5.input_key ='VL_SYS_e_d'
        self.fuzz_combine_GWT.in_6.input_key ='ENG_SYS_TYPE_SFC'
        self.fuzz_combine_GWT.in_7.input_key ='SYS_type'
        #self.fuzz_combine_GWT.in_8.input_key ='SYS_tech'

        #P installed:
        self.P_sys.weight_file = 'FCL_files/DFES_RF/BEST/RFdata20Aug_DFES_Pin_30_250_50.nwf'
        self.P_sys.TESTPLOT = 0
        self.P_sys.inRanges = {   'SYSTEM_QUANT_PHI':   [0.5,  0.85 ],
                                  'VL_SYS_w':           [1., 150.],
                                  'WING_SYS_TYPE_WS':   [15., 300.],
                                  'sys_etaP':           [0.6,  1.0 ],
                                  'VL_SYS_DRV_eta_d':   [0.7,  1.0 ],
                                  'sys_FoM':            [0.3,  1.0 ],
                                  'VL_SYS_e_d':         [0.0,  0.3 ],
                                  'ENG_SYS_TYPE_SFC':   [0.35, 0.75 ],
                                  'SYS_type':           [0.5,  3.5 ],}
        self.P_sys.inOrder = ['VL_SYS_e_d', 'WING_SYS_TYPE_WS', 'SYSTEM_QUANT_PHI', 'SYS_type', 'VL_SYS_DRV_eta_d', 'sys_FoM', 'VL_SYS_w', 'sys_etaP', 'ENG_SYS_TYPE_SFC']
                             #['SYSTEM_QUANT_PHI', 'VL_SYS_w', 'WING_SYS_TYPE_WS', 'sys_etaP',
                             # 'VL_SYS_DRV_eta_d', 'sys_FoM','VL_SYS_e_d', 'ENG_SYS_TYPE_SFC', 'SYS_type']
        self.P_sys.outRanges = {'sys_P' :           [1000, 15000]}
        self.P_sys.actType = 'sigmoid'
        self.P_sys.hidNodes = 250
        self.P_sys.inGran   = 30
        self.P_sys.outGran  = 50
        
        self.fuzz_combine_P.in_1.input_key ='SYSTEM_QUANT_PHI'
        self.fuzz_combine_P.in_2.input_key ='VL_SYS_w'
        self.fuzz_combine_P.in_3.input_key ='WING_SYS_TYPE_WS'
        self.fuzz_combine_P.inDict_1.input_keys = ['sys_etaP']
        self.fuzz_combine_P.in_4.input_key ='VL_SYS_DRV_eta_d'
        self.fuzz_combine_P.inDict_2.input_keys = ['sys_FoM']
        self.fuzz_combine_P.in_5.input_key ='VL_SYS_e_d'
        self.fuzz_combine_P.in_6.input_key ='ENG_SYS_TYPE_SFC'
        self.fuzz_combine_P.in_7.input_key ='SYS_type'
        #self.fuzz_combine_P.in_8.input_key ='SYS_tech'

        #VH:
        self.VH_sys.weight_file = 'FCL_files/DFES_RF/BEST/RFdata20Aug_DFES_VH_40_250_50.nwf'
        self.VH_sys.inRanges = { 'SYSTEM_QUANT_PHI':    [0.5,  0.85 ],
                                  'VL_SYS_w':           [1., 150.],
                                  'WING_SYS_TYPE_WS':   [15., 300.],
                                  'sys_etaP':           [0.6,  1.0 ],
                                  'VL_SYS_DRV_eta_d':   [0.7,  1.0 ],
                                  'sys_FoM':            [0.3,  1.0 ],
                                  'VL_SYS_e_d':         [0.0,  0.3 ],
                                  'ENG_SYS_TYPE_SFC':   [0.35, 0.75 ],
                                  'SYS_type':           [0.5,  3.5 ],}
        self.VH_sys.inOrder = ['VL_SYS_e_d', 'WING_SYS_TYPE_WS', 'SYSTEM_QUANT_PHI', 'SYS_type', 'VL_SYS_DRV_eta_d', 'sys_FoM', 'VL_SYS_w', 'sys_etaP', 'ENG_SYS_TYPE_SFC']
                                #['SYSTEM_QUANT_PHI', 'VL_SYS_w', 'WING_SYS_TYPE_WS', 'sys_etaP',
                                #'VL_SYS_DRV_eta_d', 'sys_FoM','VL_SYS_e_d', 'ENG_SYS_TYPE_SFC', 'SYS_type']
        self.VH_sys.outRanges = {'sys_VH' :           [200, 500]}
        self.VH_sys.actType = 'sigmoid'
        self.VH_sys.hidNodes = 250
        self.VH_sys.inGran   = 40
        self.VH_sys.outGran  = 50
        
        self.fuzz_combine_VH.in_1.input_key ='SYSTEM_QUANT_PHI'
        self.fuzz_combine_VH.in_2.input_key ='VL_SYS_w'
        self.fuzz_combine_VH.in_3.input_key ='WING_SYS_TYPE_WS'
        self.fuzz_combine_VH.inDict_1.input_keys = ['sys_etaP']
        self.fuzz_combine_VH.in_4.input_key ='VL_SYS_DRV_eta_d'
        self.fuzz_combine_VH.inDict_2.input_keys = ['sys_FoM']
        self.fuzz_combine_VH.in_5.input_key ='VL_SYS_e_d'
        self.fuzz_combine_VH.in_6.input_key ='ENG_SYS_TYPE_SFC'
        self.fuzz_combine_VH.in_7.input_key ='SYS_type'
        

        # CONFIGURE DRIVER
        self.driver.iprint = 0 # Driver Flags

        self.driver.add_parameter('compatibility.gen_num')
        self.driver.add_parameter('compatibility.option1') #list of options as input
        self.driver.add_parameter('compatibility.option2') #list of options as input
        self.driver.add_parameter('compatibility.option3') #list of options as input
        self.driver.add_parameter('compatibility.option4') #list of options as input
        self.driver.add_parameter('compatibility.option5') #list of options as input
        self.driver.add_parameter('compatibility.option6') #list of options as input
        self.driver.add_parameter('compatibility.option7') #list of options as input
        self.driver.add_parameter('compatibility.option8') #list of options as input
        self.driver.add_parameter('compatibility.option9') #list of options as input
        
        #self.driver.add_response('driver.gen_num')

        self.driver.add_objective('postprocess.response_1') #output from system (dict)
        self.driver.add_objective('postprocess.response_2') #output from system (dict)
        self.driver.add_objective('postprocess.response_3') #output from system (dict)
        self.driver.add_objective('postprocess.response_4') #output from system (dict)
        self.driver.add_objective('postprocess.response_6') #output from system (dict)


        #self.driver.add_response('postprocess.ranges_out') #output from system (dict)
        #self.driver.add_response('postprocess.response_1_r') #output from system (dict)
        #self.driver.add_response('postprocess.response_2_r') #output from system (dict)
        #self.driver.add_response('postprocess.response_3_r') #output from system (dict)
        t = time.strftime("%H-%M-%S")
        self.recorders = [CSVCaseRecorder(filename='optCases_'+t+'.csv')] #  [JSONCaseRecorder(out='opt_record.json')]  #
        self.recording_options.includes = ['compatibility.gen_num', 'compatibility.option1', 'compatibility.option2', 'compatibility.option3', 'compatibility.option4',
                                           'compatibility.option5', 'compatibility.option6', 'compatibility.option7', 'compatibility.option8', 'compatibility.option9', "compatibility.compatibility", 
                                           'postprocess.response_1', 'postprocess.response_2', 'postprocess.response_3', 'postprocess.response_4', 
                                           'postprocess.response_5', 'postprocess.response_6', 'postprocess.response_7',
                                           'postprocess.response_1_r', 'postprocess.response_2_r', 'postprocess.response_3_r', 'postprocess.response_4_r', 
                                           'postprocess.response_5_r', 'postprocess.response_6_r', 'postprocess.response_7_r',
                                           'postprocess.response_1_POS', 'postprocess.response_2_POS', 'postprocess.response_3_POS', 'postprocess.response_4_POS', 
                                           'postprocess.response_5_POS', 'postprocess.response_6_POS', 'postprocess.response_7_POS', 'postprocess.fuzzyPOS']

        self.driver.print_results = True

        
#if __name__ == "__main__":
#    None