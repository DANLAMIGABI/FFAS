#Discrete Fuzzy Expert System:
# See Hayashi/Buckley papers
# Hayashi - Approximations between fuzzy expert systems and neural networks - 1994.pdf
# Fuller - Neural Fuzzy Systems (Book) - 1995 pg 245
# 
# THIS VERSION MODIFIED TO RUN AS OpenMDAO COMPONENT
#
# Author: Frank Patterson
import math
import copy
import random
import numpy as np

from timer import Timer

from collections import OrderedDict

import matplotlib.pyplot as plt

def functions(net, type):
    """
    "Activation" function, often a sigmoid or some other function
    ----- INPUTS -----
    net : float
        input to the function
    type: string
        type of function to use
    """
    if   type == 'sigmoid':
        try:
            return  1./(1. + math.exp(-net/1.0))
        except OverflowError:
            print "math range error: Overflow"
            return 0.0
    
    elif type == 'tanh':
        return math.tanh(net)
        
    elif type == 'identity':
        return net   
    
    
    else: raise StandardError('No function type specified!')
    
def dfunctions(y, type):
    """
    Derivatives of "activation" function, 
    ----- INPUTS -----
    net : float
        input to the function
    type: string
        type of function to use
    """
    if   type == 'sigmoid':
        try:
            return  math.exp(-y/1.0)/(1. + math.exp(-y/1.0))**2
        except OverflowError:
            print "math range error: Overflow"
            return 1.0
            
    elif type == 'tanh':
        return  1/(math.cosh(y))**2
    
    elif type == 'identity':
        return 1.0
    
    else: raise StandardError('No function type specified!')
    
        
# implement the fuzzy system (get output(s))
from openmdao.main.api import Component
from openmdao.lib.datatypes.api import Float
from openmdao.lib.datatypes.api import Int
from openmdao.lib.datatypes.api import Str
from openmdao.lib.datatypes.api import Dict   
from openmdao.lib.datatypes.api import List   

    
class DFES(Component):

    #component inputs and outputs
    weight_file = Str('',  iotype='in', desc='File name for FCL file')
    actType     = Str('',  iotype='in', desc='Type of Activation Function')
    hidNodes    = Int(1, iotype='in', desc='Number of Hidden Nodes')
    inGran      = Int(1,  iotype='in', desc='Number of Input Nodes per Input')
    outGran     = Int(1,  iotype='in', desc='Number of Output Nodes per Output')
    inRanges    = Dict({},  iotype='in', desc='Dict of Inputs (input:[min,max])')
    inOrder     = List([], iotype='in', desc='List of Inputs in correct order.')
    outRanges   = Dict({},  iotype='in', desc='Dict of Outputs (input:[min,max])')


    input_list = Dict({}, iotype='in', desc='Dict of Input Values')
    TESTPLOT = Int(0, iotype='in', desc='Flag for plottin')
    runFlag_in = Int(0, iotype='in', desc='test')

    passthrough = Int(0, iotype='in', low=0, high=1, desc='passthrough flag for incompatible options')

    #outputs 
    outputs_all = Dict({}, iotype='out', desc='Output Value Dict')
    runFlag_out = Int(0, iotype='out', desc='test')

    """
    Discrete Fuzzy Expert System ::: MODIFIED for OpenMDAO
    
    ----- INPUTS -----
    inRanges : dict
        dict of input names with ranges for inputs {'inputName':[x_min, x_max], ... }
    outRanges : dict
        dict of output names with ranges for inputs {'inputName':[x_min, x_max], ... }
    actType : str
        type of activation function to use
    hidNodes : int
        number of hidden nodes to use
    inGran : int
        number of discrete values to divide each input into (input nodes = inGran*#inputs)
    outGran : int
        number of discrete values to divide each output into (output nodes = outGran*#outputs)
    """
    def __init__(self): # inRanges, outRanges, actType, hidNodes, inGran=50, outGran=50, inWeights=None, outputWeights=None):
        """ Creates a new System object """
        super(DFES, self).__init__()        
        self.old_weight_file = self.weight_file #save current weight file

        #self.actType    = actType
        #self.inRanges   = inRanges
        #self.outRanges  = outRanges
        #self.inGran     = inGran
        #self.outGran    = outGran
        
        #FUZZY INPUT/OUTPUT MFs
        self.inputXs = OrderedDict((inp, np.arange( min(self.inRanges[inp]), max(self.inRanges[inp]),
                       (max(self.inRanges[inp]) - min(self.inRanges[inp]))/float(self.inGran) ) ) \
                       for inp in self.inOrder) #x values for input MFs       
        self.outputXs = {otp : np.arange( min(self.outRanges[otp]), max(self.outRanges[otp]),
                       (max(self.outRanges[otp]) - min(self.outRanges[otp]))/float(self.outGran) ) \
                       for otp in self.outRanges} #x values for output MFs   
                       
        #NEURAL NET 
        self.nIn = len(self.inRanges)*self.inGran + len(self.inRanges) #num of input nodes (+1 bias for each input)
        self.nHid = self.hidNodes                                      #num of hidden nodes
        self.nOut = len(self.outRanges)*self.outGran                   #number of output nodes
        
        self.actIn  = [1.0]*self.nIn    #input activations
        self.actHid = [1.0]*self.nHid   #hidden activations
        self.actOut = [1.0]*self.nOut   #output activations
        
        #create weight matrices (randomize)
        self.weightIn  = np.ones((self.nIn, self.nHid))
        self.weightOut = np.ones((self.nHid, self.nOut))
        
        #create momentum matrices (last change in weights)
        self.momIn  = np.zeros((self.nIn, self.nHid))
        self.momOut = np.zeros((self.nHid, self.nOut))
        
        #no randomization of weights... only trained systems
        #print 'New System Loaded...',  len(self.inputXs), 'inputs. ', len(self.outputXs), 'outputs. ', 
        #print self.nIn, 'input nodes. ', self.nHid, 'hidden nodes. ', self.nOut, 'output nodes. ' 
             

    def feedforward(self, inputs):
        """
        Calculates network through feedforward
        ----- INPUTS -----
        inputs : dict
            inputs to system in form of fuzzy MFs {'input name': [x,y] or x (for singleton, ...}
        ----- OUTPUTS -----
        outputs : dict
            outputs to system in form {'output name': [x,y], ... }
        """
        mu_min = 0.4 #if no input node is greater than this, it'll find one
        
        with Timer() as t:
            #translate input MFs to input membership nodes by interpolating
            inNodes = []
            for inp in self.inputXs:
                if isinstance(inputs[inp], list) or isinstance(inputs[inp], tuple): #for mf input
                    inpYs = np.interp(self.inputXs[inp], inputs[inp][0], inputs[inp][1])
                elif isinstance(inputs[inp], float) or isinstance(inputs[inp], int): #for singleton inputs
                    inpYs = np.interp(self.inputXs[inp], [inputs[inp]*0.9, inputs[inp], inputs[inp]*1.1], [0.0, 1.0, 0.0])
                else: 
                    raise StandardError("Inputs of unusable type! Input %s (value %s) is of type %s" % (inp, inputs[inp], type(inputs[inp]) ))
                    
                #check for miss-interpolated input
                if all([y < mu_min for y in inpYs]): 
                    #print "modding inputs",
                    #print inYs
                    max_mu = max(inputs[inp][1]) #get max input membership
                    max_x  = inputs[inp][0][list(inputs[inp][1]).index(max_mu)] #get x value at max mf
                    node_x = min(self.inputXs[inp], key=lambda x:abs(x-max_x)) #get node with closest value to max mf x value
                    inpYs[list(self.inputXs[inp]).index(node_x)] = max_mu #assign maximum mf at closest value to max mf
                
                inNodes = inNodes + list(inpYs) + [1.0] #combine inputs and a bias for each input
    
            self.actIn  = inNodes #replace input nodes with new ones
            
            #activations for hidden nodes:
            for i in range(len(self.actHid)):
                self.actHid[i] = sum([self.actIn[j]*self.weightIn[j][i] \
                                    for j in range(len(self.actIn))]) #sum of individual weights*input activations
                self.actHid[i] = functions(self.actHid[i], self.actType) #apply function
            
            
            #activations for output nodes
            for i in range(len(self.actOut)):
                self.actOut[i] = sum([self.actHid[j]*self.weightOut[j][i] \
                                    for j in range(len(self.actHid))]) #sum of individual weights*Hidden activations
                self.actOut[i] = functions(self.actOut[i], self.actType) #apply function
                
            
            #get output MFs
            outputMFs = {}
            i = 0
            for otp in self.outputXs:
                outputMFs[otp] = [self.outputXs[otp], self.actOut[i:i+self.outGran]] #get [x,y] of mf
                i = i+self.outGran
        
        #print '=> completed in', t.secs, 'sec'
               
        return outputMFs
        
    def backpropagate(self, targets, LR, M):
        """
        Backpropagate result through system to adjust weights. Uses a momentum factor
        to speed training
        
        ----- INPUTS -----
        targets : dict
            target outputs in form {output name: [x,y], ... }
        LR : float
            learning rate
        M : float 
            momentum multiplier 
        """
        if len(targets)*self.outGran != self.nOut:
            raise ValueError('wrong number of target values')
        
        #interpolate output data to output node X values and build target nodes
        nTarget = []
        for otp in self.outputXs:
            tarYs = np.interp(self.outputXs[otp], targets[otp][0], targets[otp][1])
            nTarget = nTarget + list(tarYs)
        
        #get deltas for output nodes
        outDels = [nTarget[i] - self.actOut[i] for i in range(len(nTarget))]
        outDels = [dfunctions(self.actOut[i], self.actType)*outDels[i] \
                    for i in range(len(outDels))]
                            
        #get deltas for hidden nodes
        hidDels = [0.0]*len(self.actHid)
        for i in range(len(self.actHid)):
            errors = [outDels[j]*self.weightOut[i][j] for j in range(len(outDels))]
            hidDels[i] = dfunctions(self.actHid[i], self.actType) * sum(errors)
            
        #update output weights
        for i in range(len(self.weightOut)):
            for j in range(len(self.weightOut[i])):
                del_w = outDels[j]*self.actHid[i]
                self.weightOut[i][j] = self.weightOut[i][j] + del_w*LR + self.momOut[i][j]*M
                self.momOut[i][j] = del_w
        
        #update hidden weights
        for i in range(len(self.weightIn)):
            for j in range(len(self.weightIn[i])):
                del_w = hidDels[j]*self.actIn[i]
                self.weightIn[i][j] = self.weightIn[i][j] + del_w*LR + self.momIn[i][j]*M
                self.momIn[i][j] = del_w
                
        RSME = sum([(nTarget[i] - self.actOut[i])**2 for i in range(len(nTarget))])/len(nTarget)
        return RSME**0.5

    def getError(self, targets):
        """
        Get Error for a given target
        
        ----- INPUTS -----
        targets : dict
            target outputs in form {output name: [x,y], ... }
        """
        if len(targets)*self.outGran != self.nOut:
            raise ValueError('wrong number of target values')
        
        #interpolate output data to output node X values and build target nodes
        nTarget = []
        for otp in self.outputXs:
            tarYs = np.interp(self.outputXs[otp], targets[otp][0], targets[otp][1])
            nTarget = nTarget + list(tarYs)
        
        #get deltas for output nodes
        #print "TARGETS:", nTarget
        #print "OUTPUTS:", self.actOut
        #plt.figure()
        #plt.plot([i for i in range(len(nTarget))], nTarget, '--b')
        #plt.plot([i for i in range(len(self.actOut))], self.actOut, '-r')
        #plt.show()
        
        outDels = [nTarget[i] - self.actOut[i] for i in range(len(nTarget))]
        RSME    = ((sum([oD**2 for oD in outDels]))/len(outDels))**0.5 #get RSME for training
                
        return RSME
            
    def train(self, data, holdback=0.2, LR=0.1, M=0.02, maxIterations=300, xConverge=0.0005, interactive=False):        
        """
        Train the system through back propagation. Stops at maxIterations or when
        convereged with running average standard dev. Checks standard deviation over
        a running average of last 10 iterations to see if it's smaller than xConverge.
        Uses stoichastic method, randomizing data order each epoch, but updating weights on each 
        pass through. 
        
        ----- INPUTS -----
        data : list
            data list for training network. data in form:
            [({in1: [x,y], in2[x,y], ... }, {out1: [x,y], out2: [x,y], ...} ), ...]
        holdback : float[0-1]
            percent of data to be used for validation
        LR : float
            learning rate
        M : float
            momentum multiplier (speeds training)
        maxIterations : int
            maximum number of iterations through the data
        xConverge : float
            stops training when last 10 training points have a std. dev. < xConverge
        ----- OUTPUTS -----
        system : instance of DFES
            instance of the trained system
        """
        
        #separate the data
        valData, trainData = [], []
        valIndecies = random.sample( range(len(data)), int(holdback*len(data)) )           
        for i in range(len(data)):
            if i in valIndecies: valData.append(data[i])
            else:                trainData.append(data[i])
            
        print "Using", len(trainData), "training points and", len(valData), "validation points.",
        print "Holdback =", round(float(len(valData))/(len(valData)+len(trainData)), 3)
        
        convergeFlag = False   # flag for convergence 
        iter = 0               # iteration counter
        totRSME_0 = 10.**10 # initialize RSME
        valRSME_min = 10.**10 # initialize RSME
        trackTrainERR = []
        trackValERR = []
        trackERR = []
        normStdDev_last10 = None
        
        if interactive:
            plt.figure()
            plt.xlabel('Training Iteration')
            plt.ylabel('Average System RSME')
            plt.ion()
            plt.show()
        
        with Timer() as t:
    
            while not convergeFlag and iter < maxIterations: #main training loop            
                iter = iter + 1
                
                #randomize data order
                iRef = range(len(trainData)) #get indecies for data
                random.shuffle(iRef)         #shuffle indecies
                trainData2 = copy.deepcopy(trainData) #copy data
                trainData = [trainData2[i] for i in iRef] #assign new order to data
                
                #pass data through backpropagate
                trainRSME = 0.0
                for item in trainData:
                    self.feedforward(item[0])
                    trainRSME = trainRSME + self.backpropagate(item[1], LR, M)
                                    
                #get validation data error
                valRSME = 0.0
                for item in valData:
                    self.feedforward(item[0])
                    valRSME = valRSME + self.getError(item[1])
                
                trackTrainERR.append(trainRSME) #track training Error    
                trackValERR.append(valRSME) #track validation error
                trackERR.append(trainRSME+valRSME) #track total error
                
                #save best systems
                if valRSME < valRSME_min:
                    self.write_weights('data/temp_opt_weights_file.nwf')
                    valRSME_min = valRSME

                #check for convergence
                if len(trackERR)>10: #only after 10 iterations
                    normStdDev_last10 = np.std(trackERR[-10:])/np.average(trackERR[-10:]) #get normalized standard deviation of last 10 total errors
                    if abs(normStdDev_last10) < xConverge: 
                        convergeFlag = True
                        print 'Training Converved, normalized stdDev =', normStdDev_last10
                
                #plot if interactive                
                if interactive:
                    plt.cla()
                    plt.plot([i for i in range(len(trackTrainERR))], [e/len(data) for e in trackTrainERR],)
                    plt.plot([i for i in range(len(trackValERR))], [e/len(data) for e in trackValERR],)
                    plt.plot([i for i in range(len(trackERR))], [e/len(data) for e in trackERR],)
                    plt.legend(["Training Error", "Validation Error", "Total Error"])
                    plt.draw()
            
                #display progress
                if iter % 5 == 0: 
                    print 'Iteration', iter, 'trainErr:', round(trainRSME,3), 
                    print 'valErr:', round(valRSME,3), 
                    print 'normStdDev(last 10)', normStdDev_last10,
                    print '=> run time', t.getTime(), 's'
    
    
        print "Best Validation RSME:", valRSME_min
        self.read_weights('data/temp_opt_weights_file.nwf')
        
        plt.cla()
        plt.plot([i for i in range(len(trackTrainERR))], [e/len(data) for e in trackTrainERR],)
        plt.plot([i for i in range(len(trackValERR))], [e/len(data) for e in trackValERR],)
        plt.plot([i for i in range(len(trackERR))], [e/len(data) for e in trackERR],)
        plt.legend(["Training Error", "Validation Error", "Total Error"])
        plt.draw()
                
        testRSME = self.test(valData, plotPoints=0)
    
        return testRSME
        

    def test(self, valData, plotPoints=3):
        """
        Tests the system, returning the sum of the RSMEs of the val data. Shows plots
        to illustrate accuracy of system
        """   
        totRSME = 0
        
        #get validation data error
        for item in valData:
            self.feedforward(item[0])
            totRSME = totRSME + self.getError(item[1])
            
        plotData = [] #random val data points to plot
        while len(plotData) < plotPoints:
            plotData.append(valData[random.randrange(0, len(valData)-1)])
        
        print "Test RSME:", totRSME, '-', totRSME/len(valData), '(normalized to data set size)'
        
        for dat in plotData:
            output = self.feedforward(dat[0])
            plt.figure()
            i=1
            for inp in self.inputXs: #plot input MFs
                ax = plt.subplot(len(self.inputXs), 2, i)
                i = i + 2
                ax.plot(dat[0][inp][1], dat[0][inp][0])
                ax.scatter(list(dat[0][inp][1]), list(dat[0][inp][0]), marker='o', c='r')
                ax.plot([0,1.1], [self.inRanges[inp][0],self.inRanges[inp][0]], '--k')
                ax.plot([0,1.1], [self.inRanges[inp][1],self.inRanges[inp][1]], '--k')
                ax.set_xlim([0.0, 1.1])
                ax.set_ylabel( str([self.inRanges[inp][0], self.inRanges[inp][1]]) ) 
                
            i = 1
            for otp in self.outputXs: #plot sys output MFs and data MFs
                ax = plt.subplot(len(self.outputXs), 2, i+1)
                i = i+2
                ax.plot(dat[1][otp][1], dat[1][otp][0])
                ax.plot(output[otp][1], output[otp][0])
                ax.set_xlim([0.0, 1.1])
                ax.set_ylim([self.outRanges[otp][0], self.outRanges[otp][1]])
            
                
            plt.show()
            
            
        return totRSME
     
    def write_weights(self, filename):
        """
        Write out the weights to a file to recreate the system
        
        """
        c = open(filename, 'w')
        c.truncate()
        
        with open(filename, 'w') as writer:   #with the file open
            writer.seek(0) #start at beginning
            writer.write('INPUT WEIGHTS'+'\n') #start fcl block     
            for row in self.weightIn:
                for w in row: 
                    writer.write(str(w) + ',')
                writer.write('\n')
            writer.write('')
            writer.write('OUTPUT WEIGHTS'+'\n') #start fcl block     
            for row in self.weightOut:
                for w in row: 
                    writer.write(str(w) + ',')
                writer.write('\n')
                            
        
    def read_weights(self, filename):
        """
        Reads in a weights file to recreate a trained system
        """
        
        f = open(filename, 'r')
        lines = f.readlines()
        
        #get input weights        
        inW = []
        flag = 0
        for line in lines:
            if 'INPUT WEIGHTS' in line: 
                flag = 1
                continue
            if 'OUTPUT WEIGHTS' in line: 
                flag = 0
                continue
            if flag == 1 and line.strip() <> "":
                row = line.rstrip().split(',')
                while '' in row: row.pop(row.index(''))
                row = [float(r) for r in row]
                inW.append(row)
                
        #check weight size before accepting as system weights
        if (not len(inW) == self.nIn) or (not len(inW[0]) == self.nHid):
            print "Input Matrix Size:", len(inW), len(inW[0])
            raise StandardError('Input weight matrix in file the wrong size!')    
        else:
            self.weightIn = np.array(inW)
            
        #get output weights        
        outW = []
        flag = 0
        for line in lines:
            if 'OUTPUT WEIGHTS' in line: 
                flag = 1
                continue
            
            if flag == 1 and line.strip() <> "":
                row = line.rstrip().split(',')
                while '' in row: row.pop(row.index(''))
                row = [float(r) for r in row]
                outW.append(row)
        
        #check weight size before accepting as system weights
        if (not len(outW) == self.nHid) or (not len(outW[0]) == self.nOut):
            print "Output Matrix Size:", len(outW), len(outW[0])
            raise StandardError('Output weight matrix in file the wrong size!')    
        else: 
            self.weightOut = np.array(outW)
        
        
    
    def execute(self):
        """
        Runs the system (feed forward) and returns the output dic
        
        ----- INPUTS -----
        inputs : dict
            inputs to system in form of fuzzy MFs {'input name': [x,y] or x (for singleton, ...}
        
        ----- OUTPUTS -----
        outputs : dict
            outputs to system in form {'output name': [x,y], ... }
        """
        
        
        #inputs = self.input_list

        if self.old_weight_file <> self.weight_file:    #check for changed weights file
            self.__init__()                             #reinitialize system
            self.read_weights(self.weight_file)
            self.old_weight_file = self.weight_file     #save old weight

            print "DFES File Loaded:", self.weight_file 

        #print "Running DFES for outputs:", [otp for otp in self.outRanges]
        
        #print len(self.input_list), "INPUTS!"
        if self.passthrough == 1: 
            self.outputs_all = {otp: None for otp in self.outputXs.keys()} #catch for incompatible option (does nothing if incompatible)
        else:
            self.outputs_all = self.feedforward(self.input_list)
        
        if self.TESTPLOT == 1: 
            data = [[self.input_list, self.outputs_all], [self.input_list, self.outputs_all]]
            e = self.test(data, plotPoints=1)

        self.runFlag_out = self.runFlag_in
