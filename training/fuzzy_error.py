"""
author: Frank Patterson - 4Apr2015
- test scripts and stuff


"""
import numpy as np
import skfuzzy as fuzz
import fuzzy_operations as fuzzyOps

##
def fuzDistAC(dataF, outF, nCuts=11, disp=False):
    #=use McCulloch, Wagner, Aickelin - Measuring the Directional Distance Between Fuzzy Sets - 2013
    #A = data, B = output
    acuts = np.arange(0.05, 1.0, 1.0/(nCuts))

    dataCuts = []
    outCuts = []
    #get all the alpha cuts
    for i in range(len(acuts)):
        dataCuts.append(fuzzyOps.alpha_cut(acuts[i], dataF))
        outCuts.append(fuzzyOps.alpha_cut(acuts[i], outF))
    
    #check for no alpha cuts and return None
    if all([x == None for x in dataCuts]) or \
       all([x == None for x in outCuts]): return None

    #get the Hausdorff metrics to get numerator
    sumCuts = 0.0
    hVals = []
    for i in range(len(acuts)):
        if dataCuts[i] <> None and outCuts[i] <> None:
            #use ammended hausdorff measure for directional distance
            if abs(outCuts[i][0] - dataCuts[i][0]) >  abs(outCuts[i][1] - dataCuts[i][1]):
                h = outCuts[i][0] - dataCuts[i][0]
            else: 
                h = outCuts[i][1] - dataCuts[i][1]
            
            hVals.append(h)
        else: hVals.append(None)
        
    #add max instead of empty sets
    for i in range(len(hVals)): 
        if hVals[i] == None: hVals[i] = max(hVals)
    
    if disp:
        print "ERROR:"
        for i in range(len(acuts)): 
            print " ==> a =", acuts[i], "dCut =", dataCuts[i], "oCut =", outCuts[i], "hVal =", hVals[i]
    
    dAB = sum([hVals[i]*acuts[i] for i in range(len(acuts))]) / sum(acuts)
    
    return dAB

def fuzErrorAC(dataF, outF, acuts=11):
    #test fuzzy error function based on interval math on alpha cuts
    #get fuzzy sets for outF (x,y) and dataF (x,y) and get error using acuts # of cuts
    #dataF - 
    #outF  - 
    #acuts - number of alpha cuts to compare system at (default = 11) - 
    
    ac = np.array(range(acuts))/float(acuts-1) #set alpha cuts
    
    #get lambda cuts for data and output
    dataLC = [fuzz.lambda_cut(dataF[1], c) for c in ac]
    outLC  = [fuzz.lambda_cut(outF[1], c)  for c in ac]
   
    #get alpha cut intervals for each MF from lambda cuts
    dataACi = []
    outACi  = []
    for i in range(len(dataLC)):
        dataACi.append( [j for j, x in enumerate(dataLC[i]) if x == 1])
    for i in range(len(outLC)):
        outACi.append(  [j for j, x in enumerate(outLC[i])  if x == 1])
    dataAC = []
    outAC  = []                     
    for i in range(len(dataACi)):
        if len(dataACi[i]) > 0: 
            dataAC.append([ dataF[0][min(dataACi[i])], 
                            dataF[0][max(dataACi[i])] ])
        else: dataAC.append([])
    for i in range(len(outACi)):
        if len(outACi[i]) > 0: 
            outAC.append([ outF[0][min(outACi[i])], 
                           outF[0][max(outACi[i])] ])
        else: outAC.append([])                        
    
    #get differences in intervals
    error_ints = []
    for i in range(len(dataAC)):
        if len(dataAC[i]) > 0 and len(outAC[i]) > 0:
            error_ints.append([ min(dataAC[i]) - min(outAC[i]),
                                max(dataAC[i]) - max(outAC[i]) ])

    #get the position error (do the fuzzy sets line up)
    if len([sum(err)/len(err) for err in error_ints]) > 0.0:
        pos_error = sum([sum(err)/len(err) for err in error_ints])/ \
                    len([sum(err)/len(err) for err in error_ints])
    else: pos_error = 100.0
    
    #get the spread error (do the sets have the same MF shape)
    if len([(err[0])+(err[1]) for err in error_ints]) > 0.0:
        spread_error = sum([(err[0])+(err[1]) for err in error_ints])/ \
                       len([(err[0])+(err[1]) for err in error_ints])
    else: spread_error = 100.0
                    
    pos_sign = 1                     
    if pos_error < 0: pos_sign = -1 #get sign of error
    
    
    return pos_error, spread_error


##    
def fuzErrorInt(dataF, outF):
    """
    Gets fuzzy error between two MF functions based on Shtovba - 
    "FUZZY MODEL TUNING BASED ON A TRAINING SET WITH FUZZY MODEL OUTPUT VALUES". 
    
    RSME = sqrt( (integral (muA(y)-muB(y))^2 dy) / (y_min - ymax) )
    
    dataF : list
        input MF
    output : list
        output MF [x,y]
    acuts : int
        number of alpha cuts to compare system at (default = 11) -
    """ 
    dataF[0] = np.asarray(dataF[0]) #convert inputs to numpy arrays
    dataF[1] = np.asarray(dataF[1])
    outF[0] = np.asarray(outF[0])
    outF[1] = np.asarray(outF[1])
    
    #print type(dataF[0]), type(dataF[1]), type(outF[0]), type(outF[1])
    
    #Resamples fuzzy universes
    minstep = np.asarray( [np.diff(dataF[0]).min(), np.diff(outF[0]).min()] ).min()

    mi = min(dataF[0].min(), outF[0].min())
    ma = max(dataF[0].max(), outF[0].max())
    z = np.r_[mi:ma:minstep] #new universe (combined x range)

    xidx = np.argsort(dataF[0])
    mfx = dataF[1][xidx]
    x = dataF[0][xidx]
    mfx2 = np.interp(z, x, mfx) #new data MF y's

    #import pdb; pdb.set_trace()

    yidx = np.argsort(outF[0])
    mfy = outF[1][yidx] 
    y = outF[0][yidx]
    mfy2 = np.interp(z, y, mfy) #new out MF y's
    
    #integrate numerically
    intSum = 0
    for i in range(len(z)):
        intSum = intSum + minstep*(mfx2[i] - mfy2[i])**2
    
    RSME = (intSum/(max(z) - min(z)))**0.5
    #RSME = (intSum)**0.5
    return RSME
    
##
def getError(truthData, system, inMF='sing', outMF='sing', sysOutType='crisp', errType='dist'):
    """
    Get a system's error against a set of truth data.
    ------INPUTS------
    truthData - list
        truth data to compare system output to in form:
                [Quant Input, Qual Input, Output]
                with inputs as {('input'): [values], ('input'): [values], ... }
                and outputs as [output value(s)]
    system : module
        instance of system to get error on
    inMF : string
        type of MF to use for input data ('sing', 'tri', 'trap', 'gauss')
    outMF : string
        type of MF to use for output data ('sing', 'tri', 'trap', 'gauss')
    sysOutType : string
        type of output to use from system ('crisp' or 'fuzzy')
        converts fuzzy outputs to crisp via 'centroid' defuzzification
    errType : string
        type of fuzzy error measurement to use ('int' : integration type, 'ac' : alpha-cut type)
    ------OUTPUTS------
    error : list
        list of [truth, output, errors]
    """
    q = 0 #use quantitative data
    
    #Turn Data to MFs
    for point in truthData:
        for inp in point[q]:         #create input MFs for each input
            #if type(point[q][inp]) <> list: point[q][inp] = [point[q][inp]]
            if len(point[q][inp]) == 1 and inMF == 'sing': #for crisp inputs don't get singleton MF yet
                point[q][inp] = point[q][inp][0]
            else:
                if not hasattr(point[q][inp][0], '__iter__'): #if not already fuzzy
                    point[q][inp] = fuzzyOps.rangeToMF(point[q][inp], inMF)
        if not hasattr(point[2][0],'__iter__'): #if not already fuzzy
            point[2] = fuzzyOps.rangeToMF(point[2], outMF)       #create output MFs
    
    #Get system responses to data
    i = 0
    for point in truthData:
        i = i + 1
        if not isinstance(point[0].keys()[0], str): #if data inputs are ('system', 'input')
            inputs = {key[0]+"_"+key[1]:point[0][key] for key in point[0]}
        else:  #if data inputs are ('system_input')
            inputs = point[0]
                    
        output = system.run(inputs)
        if isinstance(output, dict): output = output.itervalues().next()
        point.append(output)
            
    error = [] #track error for output
    for point in truthData: 
        #get error
        if not outMF=='sing' and not sysOutType == 'crisp':  #if both output and data are fuzzy
            if errType == 'dist':
                err = fuzDistAC(point[2], point[3]) #inflate error if can't solve for it
            elif errType == 'int':    
                err = fuzErrorInt(point[2], point[3])
            elif errType == 'ac':   
                p_err, s_err = fuzErrorAC(point[2], point[3]) #get position and spread error
                err = p_err + 0.5*s_err
        
        elif outMF=='sing' and sysOutType=='crisp':  #if both output and data are crisp
            err = point[2] - point[3]
            
        elif not outDataMFs=='sing' and isinstance(output, float):  #if both output is fuzzy and data is crisp
            raise ValueError('You have not created a case for this yet')
            
        elif outDataMFs=='sing' and isinstance(output, list):  #if both output is crisp and data is fuzzy
            raise ValueError('You have not created a case for this yet')
        
        error.append([point[2], point[3], err])
        
    return error

##
def getRangeError(truthData, system, inMF='sing', outMF='sing'):
    """
    Get a system's error against a set of truth data using Range outputs. (MUST 
    BE SYSTEM WITH FUZZY OUTPUT, NOT CRISP). For crisp (sing) outMF, error is 
    distance outside alpha-cut at maximum of system's output fuzzy membership fucntion.
    For fuzzy outMF (gauss, tri, trap), gets output alpha-cut range at alpha=1.0, 
    and error is error function by fuzErrorAC.
    ------INPUTS------
    truthData - list
        truth data to compare system output to in form:
                [Quant Input, Qual Input, Output]
                with inputs as {('input'): [values], ('input'): [values], ... }
                and outputs as [output value(s)]
    system : module
        instance of system to get error on !MUST BE FUZZY OUTPUT!
    inMF : string
        type of MF to use for input data ('sing', 'tri', 'trap', 'gauss')
    outMF : string
        type of MF to use for output data ('sing', 'tri', 'trap', 'gauss')
    sysOutType : string
        type of output to use from system ('crisp' or 'fuzzy')
        converts fuzzy outputs to crisp via 'centroid' defuzzification
    ------OUTPUTS------
    error : list
        list of [truth[min,max], truth[min,max], error]
    """
    q = 0 #use quantitative data
    
    error = [] #track error for output

    #Turn Data to MFs
    for point in truthData:
        #create input MFs for each input
        for inp in point[q]:
            if inMF == 'sing':   #create singleton MF
                point[q][inp] = sum(point[q][inp])/len(point[q][inp]) #get mean of range (single value)
            elif inMF == 'tri':   #create triangluar MF (min, avg, max)
                x_range = np.arange(point[q][inp][0]*0.9, point[q][inp][1]*1.1, 
                                    (point[q][inp][1]*1.1 - point[q][inp][0]*0.9)/150)
                y_vals = fuzz.trimf(x_range, [point[q][inp][0], sum(point[q][inp])/len(point[q][inp]), point[q][inp][1]])
                point[q][inp] = [x_range, y_vals]
            elif inMF == 'trap':  #create traoeziodal MF (min, min, max, max)
                x_range = np.arange(point[q][inp][0]*0.9, point[q][inp][1]*1.1, 
                                    (point[q][inp][1]*1.1 - point[q][inp][0]*0.9)/150)
                y_vals = fuzz.trimf(x_range, [dataIt[q][inp][0], point[q][inp][0],
                                    point[q][inp][0], point[q][inp][1]])
                point[q][inp] = [x_range, y_vals]    
        #create output MFs
        if outMF == 'sing':   #create singleton MF
            point[2] = sum(point[2])/len(point[2]) #get average for singleton value
        elif outMF == 'tri':   #create singleton MF
            x_range = np.arange(point[2][0]*0.9, point[2][1]*1.1, 
                                (point[2][1]*1.1 - point[2][0]*0.9)/150)
            y_vals = fuzz.trimf(x_range, [point[2][0], sum(point[2])/len(point[2]), point[2][1]])
            point[2] = [x_range, y_vals]        
        elif outMF == 'trap':   #create singleton MF
            x_range = np.arange(point[2][0]*0.9, point[2][1]*1.1, (point[2][1]*1.1 - point[2][0]*0.9)/150)
            y_vals = fuzz.trimf(x_range, [point[2][0], point[2][0], point[2][1], point[2][1]])
            point[2] = [x_range, y_vals]
    
    #Get system responses to data
    for point in truthData:
        inputs = {key[0]+"_"+key[1]:point[0][key] for key in point[0]}
        output = system.run(inputs)
        if isinstance(output, dict): output = output.itervalues().next()
        point.append(output)
    
    for point in truthData:             
        #get error
        if not outMF=='sing':  #if data is fuzzy
            err = fuzzy_error.fuzErrorAC(point[2], point[3])
            truthRange = fuzzyOps.alpha_at_val(point[2][0], point[2][1], alpha=1.0)
            outRange = fuzzyOps.alpha_at_val(point[3], point[3])
        else:
            outRange = fuzzyOps.alpha_at_val(point[3][0], point[3][1])
            
            if None in outRange: err = None
            elif point[2] >= min(outRange) and point[2] <= max(outRange):
                err = 0
            elif point[2] < min(outRange):
                err = point[2] - min(outRange)
            elif point[2] > max(outRange):
                err = point[2] - max(outRange)
            truthRange = point[2]
            
        error.append([truthRange, outRange, err])
        
    return error   
    
if __name__ == '__main__':
    
    #testerror functions
    import matplotlib.pyplot as plt
    
    A = fuzzyOps.paramsToMF([6.,7.,8.])
    B = fuzzyOps.paramsToMF([5.5,7.4,8.7])
    C = fuzzyOps.paramsToMF([1.5,2.8,4.1])
    D = fuzzyOps.paramsToMF([12.,14.,16.])
    
    A_A_int = fuzErrorInt(A, A)
    A_B_int = fuzErrorInt(A, B)
    A_C_int = fuzErrorInt(A, C)
    A_D_int = fuzErrorInt(A, D)
    print "INT ERRORS:", A_A_int, A_B_int, A_C_int, A_D_int
    
    #A = fuzzyOps.paramsToMF([3.,4.,5.])
    #B = fuzzyOps.paramsToMF([3.5,4.2,5.7])
    #C = fuzzyOps.paramsToMF([1.5,2.8,4.1])
    #D = fuzzyOps.paramsToMF([16.,17.,19.])
    
    A_A_ac = fuzDistAC(A, A)
    A_B_ac = fuzDistAC(A, B)
    A_C_ac = fuzDistAC(A, C)
    A_D_ac = fuzDistAC(A, D)
    print "AC ERRORS:", A_A_ac, A_B_ac, A_C_ac, A_D_ac

    #plt.rc('text', usetex=True)
    #plt.rc('font', family='serif')
    
    plt.figure(figsize=(9.,4.))
    plt.plot(A[0],A[1], '--k', lw=1.0)
    plt.text(np.median(A[0])*0.92, 0.7, 'A', fontsize=18)

    plt.plot(B[0],B[1], 'r', lw=1.0)
    plt.text(np.median(B[0])*1.07, 0.9, 'B', fontsize=18)
    t = r"$RSME(B,A)="+str(round(A_B_int,2))+'$'
    plt.text(0.8*np.median(B[0]), 0.15, t, fontweight='bold', fontsize=14)
    t = r"$err_{AC}(B,A)="+str(round(A_B_ac,2))+'$'
    plt.text(0.8*np.median(B[0]), 0.05, t, fontweight='bold', fontsize=14)
    
    plt.plot(C[0],C[1], 'b', lw=1.0)
    plt.text(np.median(C[0])*0.88, 0.7, 'C', fontsize=18)
    t = r"$RSME(C,A)="+str(round(A_C_int,2))+'$'
    plt.text(0.7*np.median(C[0]), 0.15, t, fontweight='bold', fontsize=14)
    t = r"$err_{AC}(C,A)="+str(round(A_C_ac,2))+'$'
    plt.text(0.7*np.median(C[0]), 0.05, t, fontweight='bold', fontsize=14)
    
    plt.plot(D[0],D[1], 'g', lw=1.0)
    plt.text(np.median(D[0])*0.94, 0.7, 'D', fontsize=18)
    t = r"$RSME(D,A)="+str(round(A_D_int,2))+'$'
    plt.text(0.9*np.median(D[0]), 0.15, t, fontweight='bold', fontsize=14)
    t = r"$err_{AC}(D,A)="+str(round(A_D_ac,2))+'$'
    plt.text(0.9*np.median(D[0]), 0.05, t, fontweight='bold', fontsize=14)
    
        
        
    plt.ylabel(r'$\mu(x)$')
    plt.xlabel(r'$x$')
    
    
    plt.show()