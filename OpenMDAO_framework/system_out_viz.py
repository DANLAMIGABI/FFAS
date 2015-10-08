"""
   system_out_viz.py
"""

from openmdao.main.api import Component
from openmdao.lib.datatypes.api import Float, Str, Dict, Int, List

#for viz:
import matplotlib.pyplot as plt

class fuzzy_out_viz(Component):
    """ visualize the fuzzy outputs of the system
    """
    viz_on = Int(0, iotype='in', desc='flag to turn on and off visualization (0=off, 1=on)')

    system_inputs = Dict({}, iotype='in', desc='input dict from fuzzy sys')
    system_outputs = Dict({}, iotype='in', desc='output dict from fuzzy sys')
    runFlag_in = Int(0, iotype='in')
    
    
    input_mfs = Dict({}, iotype='in', desc='dict of fuzzy system inputs')
    output_mfs = Dict({}, iotype='in', desc='dict of fuzzy system outputs')
    runFlag_out = Int(0, iotype='out')
    
    def execute(self):
        
        if self.viz_on == 1:
            
            print 'Plotting Fuzzy System Result'
            print 'Inputs:', len(self.input_mfs), '  Input Values:', len(self.system_inputs)
            print 'Outputs:', len(self.output_mfs), '  Output Values:', len(self.system_outputs)
            
            plt.figure()
            i = 1
            
            print 'Plotting Inputs'
            for k1 in self.input_mfs:
                
                #plot each input against MFs
                plt.subplot(len(self.input_mfs)+len(self.output_mfs), 1, i)
                for k2 in self.input_mfs[k1].MFs:
                    plt.plot(self.input_mfs[k1].MFs[k2][0], self.input_mfs[k1].MFs[k2][1])
                i = i + 1
                
                #plot input
                if isinstance(self.system_inputs[k1], list): 
                    plt.plot(self.system_inputs[k1][0],self.system_inputs[k1][1], lw=3.0, color='k')
                else:
                    plt.plot([self.system_inputs[k1],self.system_inputs[k1]],[0,1.0], lw=3.0, color='k')
                plt.ylabel(k1)
                plt.ylim([0,1.1])
                plt.xlim([1,9])
            
            print 'Plotting Outputs'
            #plot output against MFs     
            for k1 in self.output_mfs:
                plt.subplot(len(self.input_mfs)+len(self.output_mfs), 1, i)
                for k2 in self.output_mfs[k1].MFs:
                    plt.plot(self.output_mfs[k1].MFs[k2][0], self.output_mfs[k1].MFs[k2][1])
                i = i + 1
                plt.plot(self.system_outputs[k1][0],self,system_outputs[k1][1], lw=3.5, color='b')
                plt.ylabel(k1)
                plt.ylim([0,1.1])
                plt.xlim([1,9])
                
            print 'Plots Generated'
            plt.show()
            self.runFlag_out = self.runFlag_in