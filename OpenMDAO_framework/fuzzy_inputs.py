#author - Frank Patterson
from openmdao.main.api import Component, VariableTree
from openmdao.lib.datatypes.api import Float, Str, Dict, Int, List, VarTree, Enum

import numpy as np
import skfuzzy as fuzz
import csv

import fuzzy_operations as fuzzyOps

import matplotlib.pylab as plt #for testing... 

#-----------------------------------------------------------------------------#   
#-----------------------------------------------------------------------------#
class Input_List(Component):
    
    SYS_list = ['VL_SYS', 'FWD_SYS', 'WING_SYS', 'ENG_SYS'] #a list of systems
    ASPECT_list = ['VL_SYS_TYPE', 'VL_SYS_PROP', 'VL_SYS_DRV', 'VL_SYS_TECH', \
                'FWD_SYS_PROP', 'FWD_SYS_DRV', 'FWD_SYS_TYPE',\
                'WING_SYS_TYPE', \
                'ENG_SYS_TYPE']    #list of system functional aspects 

    input_file = Str('', iotype='in', desc='input csv file with morph matrix data')
    #options = List([], iotype='in', desc='list of integers representing options for a given alternative')    


    option1 = Float(1, iotype='in', low=0.5, high=6.5, desc='option selection for functional aspect VL_SYS_TYPE (1-6)')
    option2 = Float(1, iotype='in', low=0.5, high=3.5, desc='option selection for functional aspect VL_SYS_PROP (1-3)')
    option3 = Float(1, iotype='in', low=0.5, high=3.5, desc='option selection for functional aspect VL_SYS_DRV (1-3)')
    option4 = Float(1, iotype='in', low=0.5, high=5.5, desc='option selection for functional aspect VL_SYS_TECH (1-5)')
    option5 = Float(1, iotype='in', low=0.5, high=4.5, desc='option selection for functional aspect FWD_SYS_PROP (1-4)')
    option6 = Float(1, iotype='in', low=0.5, high=3.5, desc='option selection for functional aspect FWD_SYS_DRV (1-3)')
    option7 = Float(1, iotype='in', low=0.5, high=4.5, desc='option selection for functional aspect FWD_SYS_TYPE (1-4)')
    option8 = Float(1, iotype='in', low=0.5, high=6.5, desc='option selection for functional aspect WING_SYS_TYPE (1-6)')
    option9 = Float(1, iotype='in', low=0.5, high=4.5, desc='option selection for functional aspect ENG_SYS_TYPE (1-4)')

    passthrough = Int(0, iotype='in', low=0, high=1, desc='passthrough flag for incompatible options')

    ### LIST INPUTS GENERATED (OUTPUTS) HERE ###
    
    #PHI SYSTEM INPUTS:
    #VL_SYS_TYPE_w      = List(['VL_SYS_TYPE_w'], iotype='out', desc='')
    #VL_SYS_TYPE_e_d    = List(['VL_SYS_TYPE_e_d'], iotype='out', desc='')
    #VL_SYS_PROP_w      = List(['VL_SYS_PROP_w'], iotype='out', desc='')

    #FWD_SYS_PROP_eta_p = List(['FWD_SYS_PROP_eta_p'], iotype='out', desc='')

    #WING_SYS_TYPE_phi  = List(['WING_SYS_TYPE_phi'], iotype='out', desc='')
    #WING_SYS_TYPE_LD   = List(['WING_SYS_TYPE_LD'], iotype='out', desc='')
    
    #ENG_SYS_TYPE_phi   = List(['ENG_SYS_TYPE_phi'], iotype='out', desc='')
    

    #UPDATED
    VL_SYS_TYPE_phi    = List(['VL_SYS_TYPE_phi'], iotype='out', desc='') 
    VL_SYS_TYPE_w      = List(['VL_SYS_TYPE_w'], iotype='out', desc='')
    VL_SYS_TYPE_f      = List(['VL_SYS_TYPE_f'], iotype='out', desc='')
    
    VL_SYS_PROP_phi    = List(['VL_SYS_PROP_phi' ], iotype='out', desc='')
    VL_SYS_PROP_w      = List(['VL_SYS_PROP_w' ], iotype='out', desc='')

    VL_SYS_TECH_phi    = List(['VL_SYS_TECH_phi'], iotype='out', desc='')
    VL_SYS_TECH_w      = List(['VL_SYS_TECH_w'], iotype='out', desc='')
    VL_SYS_TECH_f      = List(['VL_SYS_TECH_f'], iotype='out', desc='')
    VL_SYS_TECH_LD     = List(['VL_SYS_TECH_LD'], iotype='out', desc='')
    
    FWD_SYS_PROP_eta_p = List(['FWD_SYS_PROP_eta_p'], iotype='out', desc='')    
    FWD_SYS_DRV_eta_d  = List(['FWD_SYS_DRV_eta_d'], iotype='out', desc='')    

    FWD_SYS_TYPE_phi   = List(['FWD_SYS_TYPE_phi'], iotype='out', desc='')
    FWD_SYS_TYPE_TP    = List(['FWD_SYS_TYPE_TP'], iotype='out', desc='')

    WING_SYS_TYPE_LD   = List(['WING_SYS_TYPE_LD'], iotype='out', desc='')
    WING_SYS_TYPE_f   = List(['WING_SYS_TYPE_f'], iotype='out', desc='')


    #LoD SYSTEM INPUTS:
    SYSTEM_f    = List(['SYSTEM_f'], iotype='out', desc='average system flat plate drag (fuzzy gauss)')
    WING_LoD    = List(['WING_LoD'], iotype='out', desc='union of all LoD values (fuzzy gauss)')

    #FoM SYSTEM INPUTS:
    VL_SYS_w          = List(['VL_SYS_w'], iotype='out', desc='')
    VL_SYS_PROP_sigma = List(['VL_SYS_PROP_sigma'], iotype='out', desc='')
    VL_SYS_e_d        = List(['VL_SYS_e_d'], iotype='out', desc='')
    VL_SYS_DRV_eta_d  = List(['VL_SYS_DRV_eta_d'], iotype='out', desc='')

    #Propulsive Efficiency INPUTS:
    FWD_SYS_eta_p     = List(['FWD_SYS_eta_p'], iotype='out', desc='')
    FWD_DRV_eta_d     = List(['FWD_DRV_eta_d'], iotype='out', desc='')

    #RF System INPUTS:
    WING_SYS_TYPE_WS   = List(['WING_SYS_TYPE_WS'], iotype='out', desc='')  
    SYS_type           = List(['VL_type'], iotype='out', desc='')
    SYS_tech           = List(['VL_tech'], iotype='out', desc='')
    ENG_SYS_TYPE_SFC   = List(['ENG_SYS_TYPE_SFC'], iotype='out', desc='')

    #def __init__(self):
    """ Initialize component """
    #super(Input_List, self).__init__()  
    data = [] #list for input data 

    options = [] #place holder for selected options

    ###
    def read_fuzzy_data(self):
        """
        Read the (morph) input file and store as a list for pulling data from
        """
        #input_file = self.input_file

        print 'Reading', self.input_file, 'data file and translating inputs.'        
        
        with open(self.input_file, 'rU') as csvfile:
            input_reader = csv.reader(csvfile, delimiter=',')
            
            data = []
            for row in input_reader:        #read each row
                if row[2] <> '':                #if variable is there...
                    data_row = []
                    for x in row:               #for each item in the row
                        if '[' in x:            #check for '[min,max]' entry
                            y = x.strip(' []').split(',')                   #remove brackets from string and split at comma
                            if '.' in y: y[y.index('.')] = 0.0
                            data_row.append([ float(y[0]), float(y[1]) ])   #add data min/max as floats to list
                        else:
                            data_row.append(x)
                    data.append(data_row)
                    
        if len(data)>1: data.pop(0) #remove header line
        
        self.data = data        #save data as output var
        
        print len(self.data), 'lines of input data read... inputs translated.'  

    
    def execute(self):    

        #print "Getting:   ", [self.option1, self.option2, self.option3, self.option4, self.option5, self.option6, self.option7, self.option8, self.option9]

        #if no data read in, read it in.
        if len(self.data) == 0:
            self.read_fuzzy_data()

        #set options from individual functional attibutes
        self.options = [int(self.option1), int(self.option2), int(self.option3),
                        int(self.option4), int(self.option5), int(self.option6),
                        int(self.option7), int(self.option8), int(self.option9)]
        #print "Morph Opts:", self.options

        #print 'Building input list from', len(self.data), 'lines of data...'
        #get all inputs for selected options
        #[ system, aspect, varname, [min,max], quant_option, qual_option ] for each line
        var_list = []   
        for line in self.data:
            var_list.append([ line[0], line[1], line[3], line[4], \
                              line[self.options[self.ASPECT_list.index(line[1])]+4], \
                              line[self.options[self.ASPECT_list.index(line[1])]+10] ])
        
        ### CALCULATE INPUTS: (all use quant data: qual data => q = 10)
        q = 4 #for quant data, qual data in q=5

        """ PHI SYSTEM: """
        fuzzInType = 'gauss'
        #VL_SYS_TYPE_phi: 
        _x = next( var[q] for var in var_list if all((var[2] == 'phi', var[1] == 'VL_SYS_TYPE')) )
        self.VL_SYS_TYPE_phi = fuzzyOps.rangeToMF(_x, fuzzInType)
        #VL_SYS_TYPE_w : 
        _x = next( var[q] for var in var_list if all((var[2] == 'w', var[1] == 'VL_SYS_TYPE')) )
        self.VL_SYS_TYPE_w = fuzzyOps.rangeToMF(_x, fuzzInType)
        #VL_SYS_TYPE_f    
        _x = next( var[q] for var in var_list if all((var[2] == 'f', var[1] == 'VL_SYS_TYPE')) )
        self.VL_SYS_TYPE_f = fuzzyOps.rangeToMF(_x, fuzzInType)

        #'VL_SYS_PROP_phi'
        _x = next( var[q] for var in var_list if all((var[2] == 'phi', var[1] == 'VL_SYS_PROP')) )
        self.VL_SYS_PROP_phi = fuzzyOps.rangeToMF(_x, fuzzInType)
        #'VL_SYS_PROP_w'
        _x = next( var[q] for var in var_list if all((var[2] == 'w', var[1] == 'VL_SYS_PROP')) )
        self.VL_SYS_PROP_w = fuzzyOps.rangeToMF(_x, fuzzInType)

        #'VL_SYS_TECH_phi'
        _x = next( var[q] for var in var_list if all((var[2] == 'phi', var[1] == 'VL_SYS_TECH')) )
        self.VL_SYS_TECH_phi = fuzzyOps.rangeToMF(_x, fuzzInType)  
        #'VL_SYS_TECH_w'
        _x = next( var[q] for var in var_list if all((var[2] == 'w', var[1] == 'VL_SYS_TECH')) )
        self.VL_SYS_TECH_w = fuzzyOps.rangeToMF(_x, fuzzInType)  
        #'VL_SYS_TECH_f'
        _x = next( var[q] for var in var_list if all((var[2] == 'f', var[1] == 'VL_SYS_TECH')) )
        self.VL_SYS_TECH_f = fuzzyOps.rangeToMF(_x, fuzzInType)  
        #'VL_SYS_TECH_LD'
        _x = next( var[q] for var in var_list if all((var[2] == 'LD', var[1] == 'VL_SYS_TECH')) )
        self.VL_SYS_TECH_LD = fuzzyOps.rangeToMF(_x, fuzzInType)  

        #FWD_SYS_PROP_eta_p  : FWD system prop efficiency
        _x = next( var[q] for var in var_list if all((var[2] == 'eta_p', var[1] == 'FWD_SYS_PROP')) )
        self.FWD_SYS_PROP_eta_p = fuzzyOps.rangeToMF(_x, fuzzInType)

        #FWD_SYS_DRV_eta_d  : FWD drive efficiency
        _x = next( var[q] for var in var_list if all((var[2] == 'eta_d', var[1] == 'FWD_SYS_DRV')) )
        self.FWD_SYS_DRV_eta_d = fuzzyOps.rangeToMF(_x, fuzzInType)

        #FWD_SYS_TYPE_phi    
        _x = next( var[q] for var in var_list if all((var[2] == 'phi', var[1] == 'FWD_SYS_TYPE')) )
        self.FWD_SYS_TYPE_phi = fuzzyOps.rangeToMF(_x, fuzzInType)
        #FWD_SYS_TYPE_TP    
        _x = next( var[q] for var in var_list if all((var[2] == 'TP', var[1] == 'FWD_SYS_TYPE')) )
        self.FWD_SYS_TYPE_TP = fuzzyOps.rangeToMF(_x, fuzzInType)

        #WING_SYS_TYPE_LD   
        _x = next( var[q] for var in var_list if all((var[2] == 'LD', var[1] == 'WING_SYS_TYPE')) )
        self.WING_SYS_TYPE_LD = fuzzyOps.rangeToMF(_x, fuzzInType)
        #WING_SYS_TYPE_f  : WING system wing loading 
        _x = next( var[q] for var in var_list if all((var[2] == 'f', var[1] == 'WING_SYS_TYPE')) )
        self.WING_SYS_TYPE_f = fuzzyOps.rangeToMF(_x, fuzzInType)

        #ENG_SYS_TYPE_SFC   
        _x = next( var[q] for var in var_list if all((var[2] == 'SFC', var[1] == 'ENG_SYS_TYPE')) )
        self.ENG_SYS_TYPE_SFC = fuzzyOps.rangeToMF(_x, fuzzInType)
        

        """ LoD SYSTEM: """
        fuzzInType = 'trap'
        # SYSTEM_f : average system flat plate drag (fuzzy gauss)
        _f = [var[q] for var in var_list if (var[2] == 'f') ] 
        data_range = [np.average([v[0] for v in _f]), np.average([v[1] for v in _f])] #get union 
        self.SYSTEM_f = fuzzyOps.rangeToMF(data_range, fuzzInType)

        #WING_LoD : union of all LoD values (fuzzy gauss)
        _ld = [var[q] for var in var_list if var[2] == 'LD'] 
        data_range = [max([v[0] for v in _ld]), min([v[1] for v in _ld])] #get union 
        self.WING_LoD = fuzzyOps.rangeToMF(data_range, fuzzInType)
                

        """ FOM SYSTEM: """
        fuzzInType = 'gauss'
        #VL_SYS_w
        #_x = [var[q] for var in var_list if all((var[2] == 'w', var[0] == 'VL_SYS'))]
        #_x = [ np.average([x[0] for x in _x]), np.average([x[1] for x in _x]) ]
        _x1 = next( var[q] for var in var_list if all((var[2] == 'w', var[1] == 'VL_SYS_TYPE')) )
        _x2 = next( var[q] for var in var_list if all((var[2] == 'w', var[1] == 'VL_SYS_PROP')) )
        _x3 = next( var[q] for var in var_list if all((var[2] == 'w', var[1] == 'VL_SYS_TECH')) )
        _x = [ max(np.average([_x1[0],_x2[0]]), _x3[0]), min(np.average([_x1[1],_x2[1]]),_x3[1]) ]
        self.VL_SYS_w = fuzzyOps.rangeToMF(_x, fuzzInType)
        _w = _x

        #VL_SYS_PROP_sigma 
        _x = next( var[q] for var in var_list if all((var[2] == 'sigma', var[1] == 'VL_SYS_PROP')) )
        self.VL_SYS_PROP_sigma = fuzzyOps.rangeToMF(_x, fuzzInType)
        #x_2 = _x

        #VL_SYS_e_d        
        _x = next( var[q] for var in var_list if all((var[2] == 'e_d', var[1] == 'VL_SYS_TYPE')) )
        self.VL_SYS_e_d = fuzzyOps.rangeToMF(_x, fuzzInType)
        _ed = _x

        #VL_SYS_DRV_eta_d  
        _x = next( var[q] for var in var_list if all((var[2] == 'eta_d', var[1] == 'VL_SYS_DRV')) )
        self.VL_SYS_DRV_eta_d = fuzzyOps.rangeToMF(_x, fuzzInType)
        _eta = _x

        #print 'w: %s, sigma: %s, e_d: %s, eta_d: %s' % (x_1, x_2, x_3, x_4) 

        """ etaP SYSTEM: """
        fuzzInType = 'trap'
        # FWD_SYS_eta_p : intersection of forward system propulsive efficiencies
        _f = [var[q] for var in var_list if all((var[2] == 'eta_p', var[0] == 'FWD_SYS')) ] 
        data_range = [max([v[0] for v in _f]), min([v[1] for v in _f])] #get union 
        data_range = sorted(data_range)
        self.FWD_SYS_eta_p = fuzzyOps.rangeToMF(data_range, fuzzInType)

        #FWD_SYS_eta_d : foward system drive efficiency
        _etad = next( var[q] for var in var_list if all((var[2] == 'eta_d', var[1] == 'FWD_SYS_DRV')) )
        self.FWD_DRV_eta_d = fuzzyOps.rangeToMF(_etad, fuzzInType)
        

        """ RF SYSTEMs (GWT/Pinst/VH): """
        # SYSTEM_QUANT_PHI 
        #       (from phi system)
        #       NEEDS TO BE QUANTIFIED

        # VL_SYS_w
        #       self.VL_SYS_w (already calculatd with FoM system)

        # WING_SYS_TYPE_WS  : WING system wing loading 
        _x = next( var[q] for var in var_list if all((var[2] == 'WS', var[1] == 'WING_SYS_TYPE')) )
        self.WING_SYS_TYPE_WS = fuzzyOps.rangeToMF(_x, 'gauss')
        # sys_etaP: system propulsive efficiency
        #       (from etaP system)

        # VL_SYS_DRV_eta_d : VL system drive efficiency
        #       VL_SYS_DRV_eta_d (already calcualted with FoM system)

        # sys_FoM: system Figure of Merit
        #       (from FoM system)

        # VL_SYS_e_d
        #       self.VL_SYS_e_d (already calcualted with FoM system)   

        # ENG_SYS_TYPE_SFC 

        #       self.ENG_SYS_TYPE_SFC (already calcualted with phi system)
        #       NEEDS TO BE QUANTIFIED?

        # SYS_TYPE (1-tilt, 2-compound, 3-other)
        VL_SYS_TYPE = int(self.option1)
        FWD_SYS_TYPE = int(self.option7) 

        if FWD_SYS_TYPE == 2: 
            T = 1 #tiling VL
        else:
            if VL_SYS_TYPE < 4: 
                T = 2 #compound
            if VL_SYS_TYPE == 4 or VL_SYS_TYPE == 5: 
                T = 3 #other
            if VL_SYS_TYPE == 6:
                T = 1 #tilting tailsitter

        self.SYS_type = fuzzyOps.rangeToMF([T,T], 'gauss')

        # SYS_TECH (0 - None, 1 - varRPM, 2 - varDiameter, 3 - stop rotor, 4 - autogyro) (switch 2-3)
        VL_SYS_TECH = int(self.option4)
        if   VL_SYS_TECH == 2: T = 3
        elif VL_SYS_TECH == 3: T = 2
        else:                  T = VL_SYS_TECH

        self.SYS_tech = fuzzyOps.rangeToMF([T,T], 'gauss')






        if self.passthrough == 1: #catch for incompatible options
            return None 
#-----------------------------------------------------------------------------#
#-----------------------------------------------------------------------------#
class InputPair(VariableTree):
    """ input pair to add to dictionary for input to fuzzy system"""
    input_key = Str('', desc='Key for Input')
    input_value = List([], desc = 'Value(s) for Input ([crisp] or [fuzzy_x, fuzzy_y]')

#-----------------------------------------------------------------------------#
#-----------------------------------------------------------------------------#
class InputDict(VariableTree):
    """ input pair to add to dictionary for input to fuzzy system"""
    input_dict = Dict({}, desc='Input Dictionary (usually from another output)')
    input_keys = List([], desc = 'Keys for items in input_dict to pull as inputs')

#-----------------------------------------------------------------------------#
#-----------------------------------------------------------------------------#
class Build_Fuzzy_Input(Component):
    """
    creates a dictionary of up to 15 inputs with their corresponding keys 
    input values can be a single crisp value [c] or two iterable 
    objects representing a fuzzy number: [ [x values], [y values] ]
    """
    
    # set up interface to the framework
    #N_inputs = Int(1, iotype'in', desc='number of input pairs to accept')
    in_1 = VarTree(InputPair(), iotype='in')
    in_2 = VarTree(InputPair(), iotype='in')
    in_3 = VarTree(InputPair(), iotype='in')
    in_4 = VarTree(InputPair(), iotype='in')
    in_5 = VarTree(InputPair(), iotype='in')
    in_6 = VarTree(InputPair(), iotype='in')
    in_7 = VarTree(InputPair(), iotype='in')
    in_8 = VarTree(InputPair(), iotype='in')
    in_9 = VarTree(InputPair(), iotype='in')
    in_10 = VarTree(InputPair(), iotype='in')
    in_11 = VarTree(InputPair(), iotype='in')
    in_12 = VarTree(InputPair(), iotype='in')
    in_13 = VarTree(InputPair(), iotype='in')
    in_14 = VarTree(InputPair(), iotype='in')
    in_15 = VarTree(InputPair(), iotype='in')

    inDict_1 = VarTree(InputDict(), iotype='in', desc='Input Dictionary (usually from another output)')
    inDict_2 = VarTree(InputDict(), iotype='in', desc='Input Dictionary (usually from another output)')
    inDict_3 = VarTree(InputDict(), iotype='in', desc='Input Dictionary (usually from another output)')
    inDict_4 = VarTree(InputDict(), iotype='in', desc='Input Dictionary (usually from another output)')
    inDict_5 = VarTree(InputDict(), iotype='in', desc='Input Dictionary (usually from another output)')

    runFlag_in = Int(0, iotype='in', desc='test')
    
    system_inputs = Dict({}, iotype='out', desc='input dict for fuzzy sys')
    runFlag_out = Int(0, iotype='out', desc='test')

    def execute(self):
        """combine all input pairs into output dict"""     
        #try:   
        inputs = [self.in_1,  self.in_2,  self.in_3,  self.in_4,  self.in_5, 
                  self.in_6,  self.in_7,  self.in_8,  self.in_9,  self.in_10, 
                  self.in_11, self.in_12, self.in_13, self.in_14, self.in_15, ]
        inDicts = [self.inDict_1, self.inDict_2, self.inDict_3, self.inDict_4, self.inDict_5]

        system_inputs = {}          
        #for each input, assign it to dict if it's not empty         
        for ix in inputs:
            if ix.input_key <> '':
                if len(ix.input_value) == 1:
                    system_inputs[str(ix.input_key).strip("'")] = ix.input_value[0]
                else:
                    system_inputs[str(ix.input_key).strip("'")] = ix.input_value
        
        #for each input dict, add the selected keys to the system inputs
        for ix in inDicts:
            #print "INPUT KEYS:", ix.input_keys, len(ix.input_dict)
            if len(ix.input_keys) > 0:
                for k in ix.input_keys:
                    if len(ix.input_dict) == 0 or k not in ix.input_dict:
                        system_inputs[str(str(k).strip("'"))] = None
                    #elif len(ix.input_dict[k]) == 1: #catch for crisp value?
                    #    system_inputs[str(str(k).strip("'"))] = ix.input_dict[k][0]
                    else: 
                        system_inputs[str(str(k).strip("'"))] = ix.input_dict[k]

        self.system_inputs = system_inputs                    
        self.runFlag_out = self.runFlag_in
        #except Exception, e:
        #    print "EXCEPTION:", e
        #    self.system_inputs = {}              
        #    self.runFlag_out = self.runFlag_in

 #-----------------------------------------------------------------------------#
 #-----------------------------------------------------------------------------#       
class Quantify(Component):
    """ 
    Linearly interpolates to change a quantitative value to a qualitative one
    """

    # set up interface to the framework
    qualRange  = List([1.0,9.0], iotype='in', desc='The qualitative range to use.')
    quantRange = List([],        iotype='in', desc='The quantitative range to use.')
    inDict     = Dict({},        iotype='in', desc='Input dictionary to get qualvalue from')
    inKey      = Str('',         iotype='in', desc='Key to use in inDict')
    defuzz     = Str('centroid', iotype='in', desc='Defuzzification method to use') 
    qualVal    = List([],        iotype='in', desc='The qualitative value')

    passthrough = Int(0, iotype='in', low=0, high=1, desc='passthrough flag for incompatible options')

    quantVal   = List([],        iotype='out', desc='Resulting quantitative one')


    def execute(self):
        """Interpolate linearly
        """
        if self.passthrough == 1: return None

        x0 = self.qualRange[0]
        x1 = self.qualRange[1]
        y0 = self.quantRange[0]
        y1 = self.quantRange[1]
        
        if self.inDict <> {}:
            inVal = self.inDict[self.inKey]


            if len(inVal) > 1: #translate universe if fuzzy
                newXs = [y0 + (y1 - y0)*((v-x0)/(x1-x0)) for v in inVal[0]]
                self.quantVal = [newXs, inVal[1]]
            else: #interpolate if crisp
                self.quantVal = y0 + (y1 - y0)*((inVal[0]-x0)/(x1-x0))
            #print "Quantified: [%.2f,%.2f] => [%.2f,%.2f]" % (self.qualRange[0],self.qualRange[1],self.quantRange[0],self.quantRange[1])
            #plt.figure()
            #plt.subplot(2,1,1)
            #plt.plot(inVal[0],inVal[1])
            #plt.subplot(2,1,2)
            #plt.plot(self.quantVal[0],self.quantVal[1])
            #plt.show()
        #if no dict
        elif isinstance(self.qualVal, list):
            inVal = self.qualVal
            if len(inVal) > 1: #translate universe if fuzzy
                newXs = [y0 + (y1 - y0)*((v-x0)/(x1-x0)) for v in inVal[0]]
                self.quantVal = [newXs, inVal[1]]
            else: #interpolate if crisp
                self.quantVal = y0 + (y1 - y0)*((inVal[0]-x0)/(x1-x0))

        #elif isinstance(self.qualVal, float) or isinstance(self.qualVal, int):
        #    self.quantVal = y0 + (y1 - y0)*((self.qualVal-x0)/(x1-x0))

        if False:
            print "Plotting interpolation..."
            plt.figure()
            plt.subplot(2,1,1)
            plt.plot(inVal[0], inVal[1])
            plt.xlim(self.qualRange)
            plt.ylim([0.0, 1.0])
            plt.subplot(2,1,2)
            plt.plot(self.quantVal[0],self.quantVal[1])
            plt.xlim(self.quantRange)
            plt.ylim([0.0, 1.0])
            plt.show()
    