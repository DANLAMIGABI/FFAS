# -*- coding: utf-8 -*-
"""
Created on Mon Mar 30 08:33:12 2015

@author: frankpatterson
"""
__all__ = ['paramsToMF',
           'singleton_to_fuzzy',
           'alpha_cut',
           'firing_strength',
           'alpha_at_val',
          ]
        
import sys   
import numpy as np
import skfuzzy as fuzz

import matplotlib.pyplot as plt #testing only

##
def paramsToMF(params):
    """
    Translate the piecewise params list of x_values to an MF function. Assumes a 
    list of length: 
    1 - singleton MF
    2 - gaussian MF function (mean, standard deviation)
    3 - triangular MF function 
    4 - trapezoidal MF function
    """
    c = 100.0
    if len(params) == 1: #singleton
        c = 50 #special short MF for singleton
        xrange = [0.9*params[0], 1.1*params[0], 2*0.2*params[0]/c]
        x, y = singleton_to_fuzzy(params[0], xrange)
        
    if len(params) == 2: #gaussian MF
        #print "PARAMS:", params
        if params[1] == 0.0: v = 0.01*params[0]
        else:                v = params[1]
        x = np.arange( params[0] - 6*v, 
                       params[0] + 6*v,
                      (14.0*v/c) ) #use 6 sigmas
        y = fuzz.gaussmf(x, params[0], params[1])
        
    elif len(params) == 3: #triangular MF
        if max(params) == min(params): prange = max(params)
        else:                          prange = max(params) - min(params)
        x = np.arange( min(params), max(params),prange/c)
        y = fuzz.trimf(x, params)  
                   
    elif len(params) == 4: #trapezoidal MF
        if max(params) == min(params): prange = max(params)
        else:                          prange = max(params) - min(params)
        x = np.arange( min(params), max(params),prange/c)
        y = fuzz.trapmf(x, params)
        
    return [np.asarray(x), np.asarray(y)] #create MF 

#######################
def rangeToMF(range, type):
    """
    Translate the range into a list of x_values and an MF function. 
    
    range : list/tuple
        range to translate to MF
        
    type : string
        type of MF:
        'sing' - singleton MF
        'gauss' - gaussian MF function (mean, standard deviation)
        'tri' - triangular MF function 
        'trap' - trapezoidal MF function
    """
    c = 100 #number of x points

    minR = float(min(range))
    maxR = float(max(range))
    
    if type == 'sing': #singleton
        c = 10 #special short mf for singleton
        if maxR == minR: 
            #print "SAME R"
            ran = max(abs(0.15*minR),0.0001)
            xrange = [minR-ran, minR+ran, ran/c]
            Xs = [minR-2.*ran, minR-1.*ran, minR, minR+1.*ran, minR+2.*ran,]
            Ys = [0.0, 0.0, 1.0, 0.0, 0.0]
        else:            
            ran = abs(maxR-minR)
            xrange = [minR, maxR, ran/c]
            Xs, Ys = singleton_to_fuzzy(sum(range)/len(range), xrange)
        
    elif type == 'gauss': #gaussian MF
        std_range = (1./4.) #0.25
        if minR == maxR: ran = max(0.0001,abs(0.05*minR))#0.05
        else:            ran = abs(maxR - minR) #check for min=max and get range
        Xs = np.arange(minR - 0.5*ran, maxR + 0.5*ran, 2*ran/c) 
        Ys = fuzz.gaussmf(Xs, sum(range)/len(range), std_range*(ran)) #gaussian mf with 4sigma = range (to 97.7% "certainty")
    
    elif type == 'tri': #triangular MF
        if minR == maxR:
            ran = max(abs(0.2*minR),0.001)
            xrange = [0.9*minR, 1.1*maxR, ran/c,]
            Xs, Ys = singleton_to_fuzzy(sum(range)/len(range), xrange)
        else:
            Xs = np.arange(0.9*minR, 1.1*maxR, (1.1*maxR-0.9*minR)/c) #create fuzzy MF for output
            Ys = fuzz.trimf(Xs, [minR, sum(range)/len(range), maxR])

    elif type == 'trap': #trapezoidal MF
        if minR == maxR:
            ran = max(abs(0.2*minR),0.001)
            xrange = [0.9*minR, 1.1*maxR, ran/c,]
            Xs, Ys = singleton_to_fuzzy(sum(range)/len(range), xrange)
        else:
            Xs = np.arange(0.9*minR, 1.1*maxR, (1.1*maxR-0.9*minR)/c) #create fuzzy MF for output
            Ys = fuzz.trapmf(Xs, [minR, minR, maxR, maxR])
    else: 
        raise StandardError("Unknown type of membership function: %s" % type)
        
    return [np.asarray(Xs), np.asarray(Ys)] #create MF 
    
    
#######################
def singleton_to_fuzzy(s, x_range):
    """
    take in float singleton value and range and return fuzzy value
     s - single float value
     x_range - range to build MF on ([x1,x2,step])
    """
    
    x = [s]
    y = [1.0]
    step = abs(x_range[2])
    #print 'step:', step
    
    #catch for 0 step size
    if step == 0.0: step = 0.01
    if x_range[0] == x_range[1]: 
        x_range[0] = x_range[0] - 0.1
        x_range[1] = x_range[1] + 0.1
        
    z = s + step
    i = 0
    while z < x_range[1] and i < 500: #append to x and y till end of specified range
        x = x + [z]
        y = y + [0.0]
        z = z + step
        i = i+1
        
    z = s - step
    i = 0
    while z > x_range[0] and i < 500: #append to x and y till end of specified range
        x = [z] + x
        y = [0.0] + y
        z = z - step
        i = i+1
    
    x1 = np.asarray(x) #turn list to array (numpy)
    y1 = np.asarray(y)
    
    return x1,y1

###########################
def alpha_cut(alpha, fuz_num):
    """
    Take in fuz_num and alpha and return alpha cut (min,max).
    
     alpha - single float value
     fuz_num - [x,y] fuzzy membership function
    """
    try:
        ins = [i for i in range(len(fuz_num[1])) if fuz_num[1][i] >= alpha]
        mi = min([fuz_num[0][i] for i in ins])
        ma = max([fuz_num[0][i] for i in ins])
        return (mi,ma)
    except:
    #    alpha = max(fuz_num[1])
    #    ins = [i for i in range(len(fuz_num[1])) if fuz_num[1][i] >= alpha]
    #    mi = min([fuz_num[0][i] for i in ins])
    #    ma = max([fuz_num[0][i] for i in ins])
        #print 'No alpha cut exists here! Returning cut at maximum membership value.'
        return None



###########################
def firing_strength(self, input_name, input_, input_sys):
    """
    get firing stregth of an input/output
    input_name - linguistic input name
    input_ - list corresponding to input [x,y] or singleton
    input_sys - object corresponding to system input MFs
    """
    if not isinstance(input_, list): #if a singleton and not a list
        fs = self.fuzzy_single_AND(input_, [input_sys.MFs[input_name][0],input_sys.MFs[input_name][1]])        
        return fs        
    x_min,y_min = fuzz.fuzzy_and(input_sys.MFs[input_name][0],input_sys.MFs[input_name][1],input_[0],input_[1]) #use AND operator to get minimum of two functions
    return max(y_min)
    
    
###########################
def alpha_at_val(x,y,alpha=None):
    """
    Takes in a fuzzy membership function, determines, the maximum membership degree,
    and then returns the alpha-cut interval at that maximum
    
    ------- INPUTS ------
    x : list/array
        x values for membership function
    y : list/array
        y values for membership function
    alpha : float
        alpha value to find alpha-cut at (If None, uses max of MF)
    """
    
    try:
        if alpha == None: alpha = 0.999*max(y) #get maximum membership
        lc = fuzz.lambda_cut(y, alpha) #get lamda cut
        mm = [x[i] for i in range(len(lc)) if lc[i] == 1] #get x values for alpha cut
        if not isinstance(mm, list): mm = [mm]
    
        if len(mm) > 0: #get range of cut
            return [ min(mm), max(mm)]
        elif alpha == None: #catch for single peak value
            y = list(y)
            imax = y.index(max(y))
            print imax
            return [x[imax], x[imax]]
        else:           #if cut is empty
            return [0, 0] 
    except:
        return [None, None]
    
###########################
def fuzzyPOS(x,y,goal, direction='max', plot=False):
    """
    Takes in a fuzzy membership function (x,y), finds dominance of the function
    over the goal and returns the value. 
    
    ------- INPUTS ------
    x : list/array
        x values for membership function
    y : list/array
        y values for membership function
    goal : float
        value of goal as crisp number
    """
    try:
        if plot:
            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.plot(x,y)
            
        maxY = 0.999999*max(y) #maximum membership value
        #yMs = fuzz.lambda_cut(y,maxY) (REMOVED THIS CAUSE IT SUCKED)
        yMs = []
        for i in range(len(y)): 
            if y[i] >= maxY: yMs.append(1)
            else: yMs.append(0)
            
        #print 'yMS', yMs
    
        if direction is 'max':
            maxX = max([i for i in range(len(yMs)) if yMs[i] == 1])
            y2 = []
            for i in range(len(y)):
                if i <= maxX: y2.append(maxY) #extend function towards min (left)
                else:         y2.append(y[i])
        if direction is 'min':
            minX = min([i for i in range(len(yMs)) if yMs[i] == 1])
            y2 = []
            for i in range(len(y)):
                if i >= minX: y2.append(maxY) #extend function towards max (right)
                else:         y2.append(y[i])
                
        if plot:
            ax.plot(x,y2, '--c')
            ax.plot([goal, goal], [0.,1.], ':k')
            ax.set_ylim([0.0,1.01])
            plt.show()
    
        d = fuzz.interp_membership(x,y2,goal)
        return d
    except:
        e = sys.exc_info()[0]
        print "Error calculating fuzzy POS:", e
        return 0.0

################################################################################
if __name__=="__main__":
    Ax = np.arange(1,9,0.1)
    Ay = fuzz.trapmf(Ax, [4,5,6,7])
    POS = fuzzyPOS(Ax,Ay,4.5,direction='min', plot=True)
    print "POSmin:", POS
    POS = fuzzyPOS(Ax,Ay,6.5, plot=True)
    print "POSmax:", POS