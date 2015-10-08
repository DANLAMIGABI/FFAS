#Discrete Fuzzy Expert System:
# See Hayashi/Buckley papers
# Hayashi - Approximations between fuzzy expert systems and neural networks - 1994.pdf
# Fuller - Neural Fuzzy Systems (Book) - 1995 pg 245
#
#
#
# Author: Frank Patterson
import math
import copy
import random
import numpy as np

from timer import Timer

#from systems import *
#from training import *
from fuzzy_error import fuzDistAC

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
    
        
    
    
class DFES():
    """
    Discrete Fuzzy Expert System
    
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
    def __init__(self, inRanges, outRanges, actType, hidNodes, inGran=50, outGran=50, inWeights=None, outputWeights=None, inputOrder=None):
        self.actType    = actType
        self.inRanges   = inRanges
        self.outRanges  = outRanges
        self.inGran     = inGran
        self.outGran    = outGran
        
         
        if inputOrder == None:
            print "NO INPUT ORDER!"
            self.inOrd = [inp for inp in inRanges]
        else:
            self.inOrd = inputOrder
            
        # ONLY 1 OUTPUT PLEASE!
        
        #FUZZY INPUT/OUTPUT MFs
        self.inputXs = OrderedDict((inp, np.arange( min(inRanges[inp]), max(inRanges[inp]),
                       (max(inRanges[inp]) - min(inRanges[inp]))/float(self.inGran) ) ) \
                       for inp in self.inOrd) #x values for input MFs   
        print "Initialized Input Order:", self.inputXs.keys()
        self.outputXs = {otp : np.arange( min(outRanges[otp]), max(outRanges[otp]),
                       (max(outRanges[otp]) - min(outRanges[otp]))/float(self.outGran) ) \
                       for otp in outRanges} #x values for output MFs   
                       
        #NEURAL NET 
        self.nIn = len(self.inRanges)*inGran + len(inRanges) #num of input nodes (+1 bias for each input)
        self.nHid = hidNodes                                 #num of hidden nodes
        self.nOut = len(self.outRanges)*outGran              #number of output nodes
        
        self.actIn  = [1.0]*self.nIn    #input activations
        self.actHid = [1.0]*self.nHid   #hidden activations
        self.actOut = [1.0]*self.nOut   #output activations
        
        #create weight matrices (randomize)
        self.weightIn  = np.ones((self.nIn, self.nHid))
        self.weightOut = np.ones((self.nHid, self.nOut))
        
        #create momentum matrices (last change in weights)
        self.momIn  = np.zeros((self.nIn, self.nHid))
        self.momOut = np.zeros((self.nHid, self.nOut))
        
        #randomize weights
        for i in range(len(self.weightIn)):
            for j in range(len(self.weightIn[i])):
                self.weightIn[i][j] = self.weightIn[i][j]*random.uniform(-0.2, 0.2)
        for i in range(len(self.weightOut)):
            for j in range(len(self.weightOut[i])):
                self.weightOut[i][j] = self.weightOut[i][j]*random.uniform(-0.2, 0.2)
        
        
        print "DFES Initialized: %d inputs, %d nodes/input, %d hidden nodes, %d output nodes" % (len(self.inputXs), self.inGran, self.nHid, len(self.outputXs))
        print "Inputs in order:",
        for inp in self.inputXs: print inp,
        print ""
        
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
                    inpYs = np.interp(self.inputXs[inp], [inputs[inp]*0.95, inputs[inp], inputs[inp]*1.05], [0.0, 1.0, 0.0])
                else: 
                    raise StandardError("Inputs of unusable type:", type(inputs[inp]), '!')
                    
                #check for miss-interpolated input
                if all([y < mu_min for y in inpYs]): 
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
            
    def train(self, data, holdback=0.2, LR=0.1, M=0.02, maxIterations=300, xConverge=0.0005, interactive=False, combError=False):        
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
        totRSME_0 = 10.**8 # initialize RSME
        valRSME_min = 10.**8 # initialize RSME
        combRSME_min = 10.**8 #again
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
                trainRSME = trainRSME#/len(trainData)
                
                
                #get validation data error
                valRSME = 0.0
                for item in valData:
                    self.feedforward(item[0])
                    valRSME = valRSME + self.getError(item[1])
                valRSME = valRSME#/len(valData)
                
                combRSME = trainRSME + valRSME 
                #combRSME = (trainRSME*len(trainData) + valRSME*len(valData)) / (len(trainData)+len(valData))
                trackTrainERR.append(trainRSME) #track training Error    
                trackValERR.append(valRSME) #track validation error
                trackERR.append(combRSME) #track total error
                
                #save best systems
                if not combError:
                    if valRSME < valRSME_min:
                        #print "New Best System Found!"
                        self.write_weights('data/temp_opt_weights_file.nwf')
                        valRSME_min = valRSME
                else: 
                    if combRSME < combRSME_min:
                        #print "New Best System Found!"
                        self.write_weights('data/temp_opt_weights_file.nwf')
                        combRSME_min = combRSME

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
                if iter % 2 == 0: 
                    print 'Iteration', iter, 'trainErr:', round(trainRSME,3), 
                    print 'valErr:', round(valRSME,3), 
                    print 'totErr:', round(combRSME,3),
                    print 'normSD(last 10)', 
                    if normStdDev_last10 <> None: print round(normStdDev_last10,5),
                    print '=> run time', t.getTime(), 's'
    
                #check results:
                if False:
                    if iter % 10 == 0: 
                        self.test(valData, plotPoints=3)
    
        if not combError: print "Best Validation RSME:", valRSME_min
        else: "Best Combined Error:", combRSME_min
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
        error = [] #check fuzzy error
        for item in valData:
            self.feedforward(item[0])
            totRSME = totRSME + self.getError(item[1])
            err = [item[1], self.outputXs[self.outputXs.keys()[0]]] #get actual and output
            err.append(fuzDistAC(err[0], err[1]))
            #print err
            error.append(err)
        plotData = random.sample(valData,plotPoints)
        #plotData = [] #random val data points to plot
        #while len(plotData) < plotPoints:
        #    plotData.append(valData[random.randrange(0, len(valData)-1)])
        
        print "Test RSME:", totRSME, '-', totRSME/len(valData), '(normalized to data set size)'    
        if len([e[2] for e in error if e[2] <> None]) > 0:    
            print "Fuzzy MSE:", sum([e[2] for e in error if e[2] <> None])/len([e[2] for e in error if e[2] <> None])
            print "Fuzzy RMSE:", (sum([e[2] for e in error if e[2] <> None])/len([e[2] for e in error if e[2] <> None]))**0.5
        
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

            #plt.draw()
            plt.figure()

            i = 1
            for inp in self.inputXs: #plot input MFs
                ax = plt.subplot(len(self.inputXs), 3, i)
                i = i + 1
                ax.plot(dat[0][inp][0], dat[0][inp][1])
                #inputYs = self.actIn[(i-1)*(self.inGran+1):i*(self.inGran+1)]
                #inputYs.pop(-1)
                #print "Xs:", len(self.inputXs[inp]), self.inputXs[inp]
                #print "Ys:", len(inputYs), (i-1)*(self.inGran+1), i*(self.inGran+1), inputYs
                ax.scatter(self.inputXs[inp], [0.01 for x in self.inputXs[inp]], marker='o', c='r')
                #ax.scatter(list(dat[0][inp][1]), list(dat[0][inp][0]), marker='o', c='r')
                ax.plot([self.inRanges[inp][0],self.inRanges[inp][0]], [0,1.1], '--k', lw=3.0)
                ax.plot([self.inRanges[inp][1],self.inRanges[inp][1]], [0,1.1], '--k', lw=3.0)
                ax.set_ylim([0.0, 1.1])
                ax.set_xlabel(inp) 

            #plt.draw()    
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
            raise StandardError('Input weight matrix in file the wrong size! Weight matrix requires %d input nodes and %d hidden nodes' % ( len(inW), len(inW[0]) ) ) 
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
            raise StandardError('Output weight matrix in file the wrong size! Weight matrix requires %d hidden nodes and %d output nodes' % ( len(outW), len(outW[0]) ))    
        else: 
            self.weightOut = np.array(outW)
        
        
    
    def run(self, inputs):
        """
        Runs the system (feed forward) and returns the output dic
        
        ----- INPUTS -----
        inputs : dict
            inputs to system in form of fuzzy MFs {'input name': [x,y] or x (for singleton, ...}
        
        ----- OUTPUTS -----
        outputs : dict
            outputs to system in form {'output name': [x,y], ... }
        """
        return self.feedforward(inputs)
        
        
if __name__ == '__main__':
    #TESTING
    import skfuzzy as fuzz
   
    plt.ioff()
    
    """
    inRanges = {'SYSTEM_f': [1,9], 
                'WING_LoD': [5,30]}
    outRanges = {'sys_LoD': [5,30]}
    
    sys = DFES(inRanges, outRanges, 'sigmoid', hidNodes=75, inGran=40, outGran=25)
    
    
    #test read in
    sys.read_weights('FCL_files/test_DFES_write')
    
    #test system feed forward singleton
    for i in range(20):
        inputs = {'SYSTEM_f': random.uniform(1.0, 9.0), 
                'WING_LoD': random.uniform(5.0, 30.0)}
        sys.run(inputs)

    #test system feed forward MF
    for i in range(20):
        f_range = np.arange(1.0, 9.0, 0.01)
        LoD_range = np.arange(5.0, 30.0, 0.01)
        inputs = {'SYSTEM_f': [f_range, 
                               fuzz.trimf(f_range, sorted([random.uniform(1.0, 9.0),
                                                          random.uniform(1.0, 9.0),
                                                          random.uniform(1.0, 9.0)]))],
                'WING_LoD': [LoD_range, 
                             fuzz.trimf(LoD_range, sorted([random.uniform(5.0, 30.0),
                                                          random.uniform(5.0, 30.0),
                                                          random.uniform(5.0, 30.0)]))]}
        outputs = sys.run(inputs)
    """

    ## BUILD FUZZY DATA TO TEST
    
    from training import *
    from systems import *
    import fuzzy_operations as fuzzyOps
    import matplotlib.pyplot as plt
    
    """
    inputs, outputs, rulebase, AND_operator, OR_operator, aggregator, implication, defuzz = \
        build_fuzz_system('FCL_files/LoDsys_simple_13Jun15.fcl')
    sys_data = Fuzzy_System(inputs, outputs, rulebase, AND_operator, OR_operator, aggregator, implication, defuzz)

    #generate data (for testing other system)
    import random
    data = []
    for n in range(100):
        f = sorted([random.randrange(1,9), random.randrange(1,9)])
        LoD = sorted([random.randrange(5,30), random.randrange(5,30)])
        SYSTEM_f_x = np.arange(0.9*f[0], 1.1*f[1], (1.1*f[1]-0.9*f[0])/100.0)
        SYSTEM_f = fuzz.trapmf(SYSTEM_f_x, [f[0],f[0],f[1],f[1]])sys.test
        WING_LoD_x = np.arange(0.9*LoD[0], 1.1*LoD[1], (1.1*LoD[1]-0.8*LoD[0])/100.0)
        WING_LoD = fuzz.trapmf(WING_LoD_x, [LoD[0],LoD[0],LoD[1],LoD[1]])
        inputs = {'SYSTEM_f': [SYSTEM_f_x,  SYSTEM_f], 'WING_LoD': [WING_LoD_x, WING_LoD]}
        outputs = sys_data.run(inputs)
        data.append((inputs, outputs))
        

    sys.train(data, holdback=0.2, LR=0.15, M=0.05, maxIterations=1000, xConverge=0.01)
    sys.write_weights('FCL_files/test_DFES_write')
    """
    
    ## TEST WITH FOM DATA
    ASPECT_list = ['VL_SYS_TYPE', 'VL_SYS_PROP', 'VL_SYS_DRV', 'VL_SYS_TECH', \
                'FWD_SYS_PROP', 'FWD_SYS_DRV', 'FWD_SYS_TYPE',\
                'WING_SYS_TYPE', \
                'ENG_SYS_TYPE']
                
                
    """
    inRanges = {    'DATA_e_d':     [0,0.3],
                    'DATA_sigma':   [0.05, 0.4],
                    'DATA_w':       [0.,150.],
                    'DATA_eta':     [0.5,1.0]}
    outRanges = {'sys_FoM' :        [0.4,1.0] }
    
    sys = DFES(inRanges, outRanges, 'sigmoid', hidNodes=100, inGran=30, outGran=40)
    
    #CREATE DATA
    
    combData = buildInputs(ASPECT_list, None, 'data/FoM_generatedData_15Jun15.csv', False,        #training data set
                        inputCols={'w':1, 'sigma':0, 'e_d':2, 'eta':3,},
                        outputCols={'sysFoM':4})
    
    q=0 #use first (quant inputs)
    # turn data into fuzzy MFs
    fuzzData = []
    for point in combData[0:350]:
        
        fuzIn = {} #create input MFs for each input
        for inp in point[q]:
            #create singleton input MFs
            mean = sum(point[q][inp])/len(point[q][inp]) #get mean of range (single value)
            fuzIn[inp[0]+'_'+inp[1]] = fuzzyOps.rangeToMF([mean], 'gauss')
        
        fuzOut = {} #create trapezoidal output MFs
        fuzOut['sys_FoM'] = fuzzyOps.rangeToMF(point[2], 'gauss')
        
        fuzzData.append([fuzIn, fuzOut])
        
        
    sys.train(fuzzData, holdback=0.3, LR=0.15, M=0.05, maxIterations=100, xConverge=0.001, interactive=False)
    sys.write_weights('FCL_files/FOMdata_DFES_test.nwf') #write network weight file
    """
    
    ## RF TEST ###
    inRanges = {    'DATA_phi':        [0.5, 0.85],
                    'DATA_w':          [1.0, 150.0],
                    'DATA_WS':         [15.0, 300],
                    'DATA_sys_etaP':   [0.6, 1.0],
                    'DATA_eta_d':      [0.7, 1.0],
                    'DATA_sys_FoM':    [0.3, 1.0],
                    'DATA_e_d':        [0.0, 0.3],
                    'DATA_SFC_quant':  [0.35,0.75],
                    'DATA_type':       [0.5, 3.5],}

    outRanges_GWT = { 'sys_GWT'  : [5000,50000]}
    outRanges_P = { 'sys_Pinst'  : [1000,15000] }
    outRanges_VH = { 'sys_VH' : [175,525] }
    
    sys = DFES(inRanges, outRanges_VH, 'sigmoid', hidNodes=250, inGran=40, outGran=50)
    
    combData = buildInputs(ASPECT_list, None, 'data/RF_GWT-P-VH_genData_20Aug15.csv', False,        #training data set
                            inputCols={'phi':0, 'w':1, 'WS':2, 'sys_etaP':3, 'eta_d':4, 'sys_FoM':5, 
                                        'e_d':6, 'SFC_quant':7, 'type':8},
                            outputCols={'sys_VH':26})
                                                    
    q=0 #use first (quant inputs)
    # turn data into fuzzy MFs
    fuzzData = []
    for point in combData:
        
        fuzIn = {} #create input MFs for each input
        for inp in point[q]:
            #create singleton input MFs
            mean = sum(point[q][inp])/len(point[q][inp]) #get mean of range (single value)
            fuzIn[inp[0]+'_'+inp[1]] = fuzzyOps.rangeToMF([mean], 'gauss')
        
        fuzOut = {} 
        fuzOut['sys_VH'] = fuzzyOps.rangeToMF(point[2], 'gauss')
        
        fuzzData.append([fuzIn, fuzOut])
    sys.train(fuzzData, holdback=0.05, LR=0.1, M=0.05, maxIterations=170, xConverge=0.005, interactive=False, combError=True)
    sys.write_weights('FCL_files/RFdata_DFES_VH_40_250_50.nwf') #write network weight file
    #sys.test(random.sample(fuzzData,100), plotPoints=10)
    
    """
    ## RF TEST ###
    inRanges = {    'DATA_phi':        [0.5, 0.95],
                    'DATA_w':          [1.0, 150.0],
                    'DATA_WS':         [15.0, 300],
                    'DATA_sys_etaP':   [0.6, 1.0],
                    'DATA_eta_d':      [0.4, 1.0],
                    'DATA_sys_FoM':    [0.6, 1.0],
                    'DATA_e_d':        [0.0, 0.3],
                    'DATA_dragX':      [0.6, 1.15],
                    'DATA_SFC_quant':  [0.45,1.05],
                    'DATA_type':       [0.5, 3.5],}

    outRanges_GWT = { 'sys_GWT'  : [5000.,85000.]}
    outRanges_P = { 'sys_Pinst'  : [1000.,15000.] }
    outRanges_VH = { 'sys_VH' : [200.,500.] }
    
    sys = DFES(inRanges, outRanges_GWT, 'sigmoid', hidNodes=250, inGran=35, outGran=60)
    
    combData = buildInputs(ASPECT_list, None, 'data/RF_GWT-P-VH-T-eWT_genData_2Sep15_14Sep15_LIMITEDFINAL.csv', False,        #training data set
                            inputCols={'phi':0, 'w':1, 'WS':2, 'sys_etaP':3, 'eta_d':4, 'sys_FoM':5, 
                                        'e_d':6, 'dragX':7, 'SFC_quant':8, 'type':9},
                            outputCols={'sys_GWT':12})

    q=0 #use first (quant inputs)
    # turn data into fuzzy MFs
    outlimits = [5000.,85000.]
    fuzzData = []
    for point in combData:
        fuzIn = {} #create input MFs for each input
        for inp in point[q]:
            #create singleton input MFs
            mean = sum(point[q][inp])/len(point[q][inp]) #get mean of range (single value)
            fuzIn[inp[0]+'_'+inp[1]] = fuzzyOps.rangeToMF([mean], 'gauss')
        
        fuzOut = {} 
        if point[2][0] < outlimits[0]: point[2][0] =  outlimits[0]
        if point[2][1] > outlimits[1]: point[2][1] =  outlimits[1]
        fuzOut['sys_GWT'] = fuzzyOps.rangeToMF(point[2], 'gauss')
        
        fuzzData.append([fuzIn, fuzOut])
    sys.train(fuzzData, holdback=0.1, LR=0.1, M=0.05, maxIterations=200, xConverge=0.005, interactive=False, combError=True)
    sys.write_weights('FCL_files/RFdata_DFES_VH_smallHoldback.nwf') #write network weight file
    #sys.test(random.sample(fuzzData,100), plotPoints=10)
    """