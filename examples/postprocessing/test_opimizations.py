"""
author: Frank Patterson - 4Apr2015
Testing training modules
"""
import random

import numpy as np
import skfuzzy as fuzz
import string

from training import *
from systems import *
import fuzzy_operations as fuzzyOps
from timer import Timer

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Rectangle

from matplotlib.backends.backend_pdf import PdfPages

plt.ioff()

##
ASPECT_list = ['VL_SYS_TYPE', 'VL_SYS_PROP', 'VL_SYS_DRV', 'VL_SYS_TECH', \
            'FWD_SYS_PROP', 'FWD_SYS_DRV', 'FWD_SYS_TYPE',\
            'WING_SYS_TYPE', \
            'ENG_SYS_TYPE']    #list of system functional aspects 

inputRanges = {     'LD'   : [5, 25],
                    'w'    : [0, 150],
                    'e_d'  : [0.0, 0.3],
                    'phi'  : [1, 9],
                    'FM'   : [0.3, 1.0],
                    'f'    : [1,9],
                    'V_max': [150, 550],
                    'eta_p': [0.6, 1.0],
                    'sigma': [0.05 ,0.4],
                    'TP'   : [0.0, 20.0],
                    'PW'   : [0.01, 5.0],
                    'eta_d': [0.5,1.0],
                    'eta'  : [0.5,1.0],
                    'WS'   : [15,300],
                    'SFC'  : [1,9],    
                    }
outputRanges = {    'sys_FoM'   : [0.4, 1.0],
                    'sys_phi'   : [1.0, 9.0],
                    'sys_GWT'   : [5000.,50000.],
                    'sys_Pinst' : [1000, 15000],
                    'sys_Tinst' : [1000, 25000],
                    'sys_VH'    : [100, 500],
                    'sys_eWT'   : [200, 8000] }

preSel_names = ['Alt BaseTR', 'Alt BaseTW', 'Alt FIW_TJ','Alt Tilt_FIW', 'Alt StopRot',
                'Alt AutoGyro', 'Alt TwinTS', 'Alt FixedFIW', 'Alt HeliPL',  'Alt FIB-TD']  
preSel_options = [ [2,2,1,2,2,1,2,1,1], #Alt BaseTR
                   [2,1,1,2,1,1,1,5,1], #Alt BaseTW
                   [4,3,3,1,4,3,1,2,2], #Alt FIW_TJ
                   [4,3,1,2,3,1,2,4,1], #Alt Tilt_FIW
                   [1,2,2,3,4,3,1,6,3], #Alt StopRot
                   [1,2,1,5,1,1,1,1,1], #Alt AutoGyro 
                   [6,2,1,2,2,1,1,1,1], #Alt TwinTS
                   [4,3,1,1,1,1,1,4,1], #Alt FixedFIW
                   [1,2,2,5,4,3,1,1,3], #Alt HeliPL'
                   [5,3,1,1,3,1,4,1,4] ] #Alt FIB-TD



##----------------------------------------------------------------------------##
#                        TEST FRBS: PHI System 
##----------------------------------------------------------------------------##
def testPhiFRBSfit(testFCLFile, inDataForm, outDataForm, inForm, outForm):
    """
    Test EMPTY WEIGHT from Expert Opinion!
    """
    testDataFile = 'data/phiData_300pts.csv'
    
    dataIn = readFuzzyInputData('data/morphInputs_13Jun15.csv')
    combinedData = buildInputs(ASPECT_list, dataIn, testDataFile, True)
        
    # BUILD TEST SYSTEM
    inputs, outputs, rulebase, AND_operator, OR_operator, aggregator, implication, defuzz = build_fuzz_system(testFCLFile)
    sys = Fuzzy_System(inputs, outputs, rulebase, AND_operator, OR_operator, aggregator, implication, defuzz)
    
    ### Plot System
    #if True:
    plt.figure(figsize=(7,10))
    i = 1
    for k in inputs:
        #plot each input against MFs
        ax = plt.subplot(len(inputs)+len(outputs), 1, i)
        for k2 in inputs[k].MFs:
            ax.plot(inputs[k].MFs[k2][0], inputs[k].MFs[k2][1])
        i = i + 1
        ax.set_ylabel('IN: '+k, rotation='horizontal', ha='right', fontsize=9)
        ax.set_yticks([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        var = string.join(k.split('_')[3:],'_')
        ax.set_xlim(inputRanges[var])
        #ax.set_xlim(inputs[k].data_range)
        ax.tick_params(axis='x', labelsize=9)
        ax.tick_params(axis='y', labelsize=9)
        ax.get_xaxis().set_tick_params(pad=1)
        
    #plot output against MFs     
    for k in outputs:
        ax = plt.subplot(len(inputs)+len(outputs), 1, i)
        for k2 in outputs[k].MFs:
            ax.plot(outputs[k].MFs[k2][0], outputs[k].MFs[k2][1])
        ax.set_ylabel('OUT: '+k, rotation='horizontal', ha='right', fontsize=9)
        ax.set_yticks([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xticks([1,2,3,4,5,6,7,8,9])
        ax.set_xlim([1,9])
        ax.tick_params(axis='x', labelsize=9)
        ax.tick_params(axis='y', labelsize=9)
        ax.get_xaxis().set_tick_params(pad=1)

    plt.subplots_adjust(left=0.27, bottom=0.03, right=0.98, top=0.97, wspace=None, hspace=0.48)


    ###
    if outDataForm == 'singleton': sysOutType = 'crisp'
    else: sysOutType = 'fuzzy'
    
    combinedData = combinedData[:]
    
    with Timer() as t:
        error = getError(combinedData, sys, inMF=inDataForm, outMF=outDataForm, sysOutType=sysOutType, errType='dist')
    
    print '=> ', t.secs, 'secs to check error'
    print 'Fuzzy Error Statistics: (alpha-cut distance error)'
    print 'Total System Error:', sum([err[2] for err in error if err[2] <> None])
    print 'Mean Square System Error:', (1.0/len([e for e in error if e[2] <> None]))*sum([e[2]**2 for e in error if e[2] <> None])
    print 'Root Mean Square System Error:', ( (1.0/len([e for e in error if e[2] <> None]))*sum([e[2]**2 for e in error if e[2] <> None]) )**0.5
    

    check = 50
    t_tot = 0.0
    for j in range(check):
        i = random.randrange(0, len(combinedData))
        inputs = {key[0]+"_"+key[1]:combinedData[i][0][key] for key in combinedData[i][0]}
        with Timer() as t:
            sysOut = sys.run(inputs)
        t_tot = t_tot + float(t.secs)
    print 'Average system time of %d points => %.5f s' % (check, t_tot/check)
    print '                                 => %.5f s per rule' % ((t_tot/check)/len(sys.rulebase))
    
    #actual vs. predicted plot (alpha-cuts)
    alpha = 0.8
    AC_actual_ = []
    AC_pred_ = []
    plt.figure()
    tit = 'Actual vs. Predicted (alpha =', alpha, ')'
    plt.title(tit)
    for err in error:
        if outDataForm == 'gauss': AC_actual = fuzzyOps.alpha_cut(alpha, [err[0][0],err[0][1]])
        else: AC_actual = fuzzyOps.alpha_at_val(err[0][0],err[0][1])
        if outForm == 'gauss': AC_pred = fuzzyOps.alpha_cut(alpha, (err[1][0],err[1][1]))
        else: AC_pred = fuzzyOps.alpha_at_val(err[1][0],err[1][1])
        
        if AC_actual <> None and AC_pred <> None: 
            AC_actual_.append(AC_actual)
            AC_pred_.append(AC_pred)
            plt.scatter(AC_actual[0], AC_pred[0], marker='o', c='r')
            plt.scatter(AC_actual[1], AC_pred[1], marker='x', c='b')
    plt.plot([1,9],[1,9], '--k')     
    plt.xlim([1,9])
    plt.ylim([1,9])
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.legend(['Min', 'Max'])
    
    #EXPANDED actual vs. predicted plot (alpha-cuts)
    alphas = [0.55, 0.7, 0.85]
    aplots = [0.2, 0.6, 1.0]
    cplots = plt.cm.Blues([0.2, 0.6, 1.0]) 
    sizes = [5.0, 15.0, 30.0]
    plt.figure()
    tit = 'Actual vs. Predicted (%d total points)' % len(combinedData)
    plt.title(tit)
    hands1 = []
    hands2 = []
    for i in range(len(alphas)):
        #print "Using cut:", i, alphas[i]
        for err in error:
            AC_actual = fuzzyOps.alpha_cut(alphas[i], [err[0][0],err[0][1]])
            AC_pred = fuzzyOps.alpha_cut(alphas[i], (err[1][0],err[1][1]))
            
            if AC_actual <> None and AC_pred <> None: 
                #print "found one!"
                l1 = r'Min at $ \alpha = %s $' % alphas[i]
                l2 = r'Max at $ \alpha = %s $' % alphas[i]
                min_scat = plt.scatter(AC_actual[0], AC_pred[0], s=sizes[i],  marker='o', c=cplots[i], label=l1, alpha=aplots[i])
                max_scat = plt.scatter(AC_actual[1], AC_pred[1], s=sizes[i], marker='d', c=cplots[i], label=l2, alpha=aplots[i])
                if len(hands1) < i + 1: 
                    hands1.append(min_scat)
                    #print i, hands1
                if len(hands2) < i + 1: 
                    hands2.append(max_scat)
                    #print i, hands2
    plt.plot([0,900],[0,900], '--k')     
    plt.xlim([1.0,9.0])
    plt.ylim([1.0,9.0])
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.legend(handles=hands1+hands2, fontsize=10, loc='upper left')
    
    #actual vs. predicted plot (centroid)
    plt.figure()
    plt.title('Actual vs. Predicted (Centroid)')
    for err in error:
        actCent = fuzz.defuzz(err[0][0], err[0][1], 'centroid')
        predCent = fuzz.defuzz(err[1][0], err[1][1], 'centroid')
        if actCent <> None and predCent <> None: 
            plt.scatter(actCent, predCent, marker='+', c='m')
    plt.plot([1,9],[1,9], '--k')     
    plt.xlim([1,9])
    plt.ylim([1,9])
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    
    #check random data points
    plt.figure()
    plt.title('Random Tests')
    for j in range(9):
        i = random.randrange(0, len(combinedData))
        inputs = {key[0]+"_"+key[1]:combinedData[i][0][key] for key in combinedData[i][0]}
        sysOut = sys.run(inputs)
        sysOut = sysOut[sysOut.keys()[0]]
        plt.subplot(3,3,j+1)
        plt.plot(sysOut[0], sysOut[1], '-r')
        plt.plot(combinedData[i][2][0], combinedData[i][2][1], '--k')
        plt.ylim([0,1.1])
        plt.xlim([1,9])
    
    #AC act vs. error plots 
    min_errPct = [(AC_pred_[i][0] - AC_actual_[i][0])/AC_actual_[i][0] for i in range(len(AC_actual_))]
    max_errPct = [(AC_pred_[i][1] - AC_actual_[i][1])/AC_actual_[i][1] for i in range(len(AC_actual_))]
    avg_ACact  = [np.average(x) for x in AC_actual_]
    plt.figure()
    plt.title('Actual (Alpha-Cut) vs. %Error')
    plt.scatter([c[0] for c in AC_actual_], min_errPct, marker='o', c='r')
    plt.scatter([c[1] for c in AC_actual_], max_errPct, marker='x', c='b')
    plt.plot([0,10],[0,0], '--k')     
    plt.xlim([1,9])
    #plt.ylim([1,9])
    plt.xlabel('Actual (Alpha-Cut)')
    plt.ylabel('%Error')
    plt.legend(['Min', 'Max'])

    #actual centroid vs. error plot
    plt.figure()
    plt.title('Actual (Centroid) vs. Error')
    cents = [fuzz.defuzz(err[0][0], err[0][1], 'centroid') for err in error if err[2] <> None]
    #pctErr = [err[2]/fuzz.defuzz(err[0][0], err[0][1], 'centroid') for err in error if err[2] <> None]
    plt.scatter(cents, [err[2] for err in error if err[2] <> None])
    plt.xlabel('Actual (Centroid)')
    plt.ylabel('Fuzzy Error')
    
    #actual centroid vs. percent error plot
    plt.figure()
    plt.title('Actual (Centroid) vs. %Error')
    cents = [fuzz.defuzz(err[0][0], err[0][1], 'centroid') for err in error if err[2] <> None]
    pctErr = [(fuzz.defuzz(err[1][0], err[1][1], 'centroid')-fuzz.defuzz(err[0][0], err[0][1], 'centroid'))/fuzz.defuzz(err[0][0], err[0][1], 'centroid') \
              for err in error if err[2] <> None]
    plt.scatter(cents, pctErr)
    plt.xlabel('Actual (Centroid)')
    plt.ylabel('%Error (Centroids)')
    
    #actual vs. error plot
    plt.figure()
    plt.title('Error/Actual vs Actual (Centroid)')
    cents_act  = [fuzz.defuzz(err[0][0], err[0][1], 'centroid') for err in error if err[2] <> None]
    cents_pre = [fuzz.defuzz(err[1][0], err[1][1], 'centroid') for err in error if err[2] <> None]
    pctErr = [err[2]/fuzz.defuzz(err[0][0], err[0][1], 'centroid') for err in error if err[2] <> None]
    plt.scatter(cents_act, pctErr)
    plt.xlabel('Actual (Centroid)')
    plt.ylabel('%Error (Centroids)')
    
    #actual vs. fuzzy error plot
    plt.figure()
    plt.title('Actual (MoM) vs. %Error')
    cents = [fuzz.defuzz(err[0][0], err[0][1], 'mom') for err in error if err[2] <> None]
    pctErr = [err[2]/fuzz.defuzz(err[0][0], err[0][1], 'mom') for err in error if err[2] <> None]
    plt.scatter(cents, pctErr)
    plt.xlabel('Actual (Mean of Max)')
    plt.ylabel('%Error (MoM)')
    
    txt =        'Fuzzy Error Statistics: (alpha-cut distance error)' + '\n'
    txt =  txt + 'Total System Error:  ' + str(sum([err[2] for err in error if err[2] <> None])) + '\n'
    txt =  txt + 'Mean Square System Error:  ' + str( (1.0/len([e for e in error if e[2] <> None]))*sum([e[2]**2 for e in error if e[2] <> None])) + '\n'
    txt =  txt + 'Root Mean Square System Error:  ' + str(( (1.0/len([e for e in error if e[2] <> None]))*sum([e[2]**2 for e in error if e[2] <> None]) )**0.5) + '\n'
    txt =  txt + 'Average system time of => ' + str(t_tot/check) + '\n'
    txt =  txt + '              per rule => ' + str((t_tot/check)/len(sys.rulebase))

    f = plt.figure(figsize=(6,8))
    f.text(0.1, 0.5, txt)
    
    #visuzlize system with random data point
    #i = random.randrange(0, len(combinedData))
    #inputs = {key[0]+"_"+key[1]:combinedData[i][0][key] for key in combinedData[i][0]}
    #sys.run(inputs, TESTMODE=True)
    
    pp = PdfPages(testFCLFile[:-4] + '_VALIDATION.pdf')
    figs = [plt.figure(n) for n in plt.get_fignums()]
    for fig in figs:
        fig.savefig(pp, format='pdf')
    pp.close()
    plt.close('all')
    plt.clf()   

    """
    #TEST PRESELECTED OPTIONS
    for i in range(len(preSel_options)):
        
    sys.run(inputs, TESTMODE=True) #vis system
    """
##----------------------------------------------------------------------------##
#                        TEST FRBS: FoM System 
##----------------------------------------------------------------------------##
def testFoMFRBSfit(testFCLFile, inDataForm, outDataForm, inForm, outForm):
    """
    Test FoM system from Expert Opinion in FRBS!
    """
    
    combData = buildInputs(ASPECT_list, None, 'data/FoM_generatedData_15Jun15.csv', False,        #training data set
                        inputCols={'w':1, 'sigma':0, 'e_d':2, 'eta':3,},
                        outputCols={'sysFoM':4}) 
    
    #get random data
    dataMax = 300
    combinedData = random.sample(combData, dataMax) #get random data points for testing
    
    print "data read... "    
    # BUILD TEST SYSTEM
    inputs, outputs, rulebase, AND_operator, OR_operator, aggregator, implication, defuzz = build_fuzz_system(testFCLFile)
    sys = Fuzzy_System(inputs, outputs, rulebase, AND_operator, OR_operator, aggregator, implication, defuzz)
    
    print "system created... "
    
    ### Plot System
    #if True:
    plt.figure(figsize=(8,5))
    i = 1
    for k in inputs:
        #plot each input against MFs
        ax = plt.subplot(len(inputs)+len(outputs), 1, i)
        for k2 in inputs[k].MFs:
            ax.plot(inputs[k].MFs[k2][0], inputs[k].MFs[k2][1])
        i = i + 1
        ax.set_ylabel('IN: '+k, rotation='horizontal', ha='right', fontsize=9)
        ax.set_yticks([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        var = string.join(k.split('_')[1:],'_')
        #print k, var
        ax.set_xlim(inputRanges[var])
        #ax.set_xlim(inputs[k].data_range)
        ax.tick_params(axis='x', labelsize=9)
        ax.tick_params(axis='y', labelsize=9)
        ax.get_xaxis().set_tick_params(pad=1)
        
    #plot output against MFs     
    for k in outputs:
        ax = plt.subplot(len(inputs)+len(outputs), 1, i)
        for k2 in outputs[k].MFs:
            ax.plot(outputs[k].MFs[k2][0], outputs[k].MFs[k2][1])
        ax.set_ylabel('OUT: '+k, rotation='horizontal', ha='right', fontsize=9)
        ax.set_yticks([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        #ax.set_xticks([1,2,3,4,5,6,7,8,9])
        print k
        ax.set_xlim(outputRanges[k])
        ax.tick_params(axis='x', labelsize=9)
        ax.tick_params(axis='y', labelsize=9)
        ax.get_xaxis().set_tick_params(pad=1)

    plt.subplots_adjust(left=0.15, bottom=0.03, right=0.98, top=0.97, wspace=None, hspace=0.48)
    #plt.show()

    if outDataForm == 'singleton': sysOutType = 'crisp'
    else: sysOutType = 'fuzzy'
    
    error = getError(combinedData, sys, inMF=inDataForm, outMF=outDataForm, sysOutType=sysOutType, errType='dist')
    
    #actual vs. predicted plot (alpha-cuts)
    alpha = 0.65
    plt.figure()
    tit = 'Actual vs. Predicted (alpha = %.2f, %d total points)' % (alpha, dataMax)#, testFCLFile
    plt.title(tit, y=1.05)
    for err in error:
        if outDataForm == 'gauss': AC_actual = fuzzyOps.alpha_cut(alpha, [err[0][0],err[0][1]])
        else: AC_actual = fuzzyOps.alpha_at_val(err[0][0],err[0][1])
        if outForm == 'gauss': AC_pred = fuzzyOps.alpha_cut(alpha, (err[1][0],err[1][1]))
        else: AC_pred = fuzzyOps.alpha_at_val(err[1][0],err[1][1])
        
        if AC_actual <> None and AC_pred <> None: 
            plt.scatter(AC_actual[0], AC_pred[0], marker='o', c='r')
            plt.scatter(AC_actual[1], AC_pred[1], marker='x', c='b')
    plt.plot([0,900],[0,900], '--k')     
    plt.xlim([0.4,0.9])
    plt.ylim([0.4,0.9])
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    
    #EXPANDED actual vs. predicted plot (alpha-cuts)
    alphas = [0.4, 0.6, 0.8]
    aplots = [0.4, 0.7, 1.0]
    cplots = plt.cm.Blues([0.3, 0.6, 1.0]) 
    sizes = [10.0, 20.0, 30.0]
    plt.figure()
    tit = 'Actual vs. Predicted (%d total points)' % dataMax
    plt.title(tit, y=1.05)
    hands1 = []
    hands2 = []
    for err in error:
        for i in range(len(alphas)):
            if outDataForm == 'gauss': AC_actual = fuzzyOps.alpha_cut(alphas[i], [err[0][0],err[0][1]])
            else: AC_actual = fuzzyOps.alpha_at_val(err[0][0],err[0][1])
            if outForm == 'gauss': AC_pred = fuzzyOps.alpha_cut(alphas[i], (err[1][0],err[1][1]))
            else: AC_pred = fuzzyOps.alpha_at_val(err[1][0],err[1][1])
            
            if AC_actual <> None and AC_pred <> None: 
                l1 = r'Min at $ \alpha = %s $' % alphas[i]
                l2 = r'Max at $ \alpha = %s $' % alphas[i]
                min_scat = plt.scatter(AC_actual[0], AC_pred[0], s=sizes[i],  marker='o', c=cplots[i], label=l1, alpha=aplots[i])
                max_scat = plt.scatter(AC_actual[1], AC_pred[1], s=sizes[i], marker='d', c=cplots[i], label=l2, alpha=aplots[i])
                if len(hands1) < i + 1: hands1.append(min_scat)
                if len(hands2) < i + 1: hands2.append(max_scat)
    plt.plot([0,900],[0,900], '--k')     
    plt.xlim([0.4,0.9])
    plt.ylim([0.4,0.9])
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.legend(handles=hands1+hands2, fontsize=10, loc='upper left')
    
    #actual vs. predicted plot (centroid)
    plt.figure()
    plt.title('Actual vs. Predicted (Centroid)') #+ testFCLFile)
    for err in error:
        try:
            actCent = fuzz.defuzz(err[0][0], err[0][1], 'centroid')
            predCent = fuzz.defuzz(err[1][0], err[1][1], 'centroid')
        except:
            actCent, predCent = None, None
        if actCent <> None and predCent <> None: 
            plt.scatter(actCent, predCent, marker='+', c='m')
    plt.plot([0,900],[0,900], '--k')     
    plt.xlim([0.4,0.9])
    plt.ylim([0.4,0.9])
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    
    #actual vs. predicted plot (mom)
    #plt.figure()
    #plt.title('Actual vs. Predicted (Mean of Max)')# + testFCLFile)
    #for err in error:
    #    actCent = fuzz.defuzz(err[0][0], err[0][1], 'mom')
    #    predCent = fuzz.defuzz(err[1][0], err[1][1], 'mom')
    #    if actCent <> None and predCent <> None: 
    #        plt.scatter(actCent, predCent, marker='+', c='m')
    #plt.plot([0,900],[0,900], '--k')     
    #plt.xlim([0.4,1.0])
    #plt.ylim([0.4,1.0])
    #plt.xlabel('Actual')
    #plt.ylabel('Predicted')
        
        
    #check random data points (9)
    plt.figure()
    #plt.title('Test Random Data Points:')#+testFCLFile)
    for j in range(9):
        i = random.randrange(0, len(error))
        err = fuzDistAC(error[i][0], error[i][1])
        plt.subplot(3,3,j+1)
        plt.plot(error[i][0][0], error[i][0][1], '-r') #plot actual
        plt.plot(error[i][1][0], error[i][1][1], '--b') #plot predicted
        if err <> None:
            plt.text(np.median(error[i][0][0]), 0.7, 'err = %.3f' % err, color='k')
        plt.ylim([0,1.1])
        plt.xlim([0.4, 0.9])  
        
    #visuzlize system with random data point
    #i = random.randrange(0, len(combinedData))
    #inputs = {key[0]+"_"+key[1]:combinedData[i][0][key] for key in combinedData[i][0]}
    #sys.run(inputs, TESTMODE=True)
    
    #actual centroid vs. error plot
    plt.figure()
    plt.title('Actual (Centroid) vs. Error')
    cents = [fuzz.defuzz(err[0][0], err[0][1], 'mom') for err in error]
    plt.scatter(cents, [err[2] for err in error])
    plt.xlabel('Actual (Centroid)')
    plt.ylabel('Fuzzy Error')
    
    #actual centroid vs. percent error plot
    plt.figure()
    plt.title('Actual (Centroid) vs. %Error')
    cents = [fuzz.defuzz(err[0][0], err[0][1], 'centroid') for err in error if err[2] <> None]
    pctErr = [(fuzz.defuzz(err[1][0], err[1][1], 'centroid')-fuzz.defuzz(err[0][0], err[0][1], 'centroid'))/fuzz.defuzz(err[0][0], err[0][1], 'centroid') \
              for err in error if err[2] <> None]
    plt.scatter(cents, pctErr)
    plt.xlabel('Actual (Centroid)')
    plt.ylabel('%Error (Centroids)')
    
    txt =        'Fuzzy Error Statistics: (alpha-cut distance error)' + '\n'
    txt =  txt + 'Total System Error:  ' + str(sum([err[2] for err in error if err[2] <> None])) + '\n'
    txt =  txt + 'Mean Square System Error:  ' + str( (1.0/len([e for e in error if e[2] <> None]))*sum([e[2]**2 for e in error if e[2] <> None])) + '\n'
    txt =  txt + 'Root Mean Square System Error:  ' + str(( (1.0/len([e for e in error if e[2] <> None]))*sum([e[2]**2 for e in error if e[2] <> None]) )**0.5) + '\n'
    #txt =  txt + 'Average system time of => ' + str(t_tot/check) + '\n'
    #txt =  txt + '              per rule => ' + str((t_tot/check)/len(sys.rulebase))

    f = plt.figure(figsize=(6,8))
    f.text(0.1, 0.5, txt)
    
    pp = PdfPages(testFCLFile[:-4] + '_VALIDATION.pdf')
    figs = [plt.figure(n) for n in plt.get_fignums()]
    for fig in figs:
        fig.savefig(pp, format='pdf')
    pp.close()
    plt.close('all')
    plt.clf()   

    """
    #TEST PRESELECTED OPTIONS
    for i in range(len(preSel_options)):
        
    sys.run(inputs, TESTMODE=True) #vis system
    """
    
##----------------------------------------------------------------------------##
#                        TEST DFES: FoM System 
##----------------------------------------------------------------------------## 
def testFoMDFESfit(wtFileName, hidNodes, inGran, outGran, inDataForm='gauss', outDataForm='gauss'):
    """
    Test the fit of DFES systems to Figure of Merit Data
    """
    print "*****************************************"
    print " => Testing", wtFileName

    combData = buildInputs(ASPECT_list, None, 'data/FoM_generatedData_15Jun15.csv', False,        #training data set
                        inputCols={'w':1, 'sigma':0, 'e_d':2, 'eta':3,},
                        outputCols={'sysFoM':4}) 

    q=0 #use first (quant inputs)
    
    inType  = inDataForm
    outType = outDataForm
    
    #Turn data into fuzzy MFs
    fuzzData = []
    for point in combData[:500]:
    
        fuzIn = {} #create input MFs for each input
        for inp in point[q]:
            #create singleton input MFs
            mean = sum(point[q][inp])/len(point[q][inp]) #get mean of range (single value)
            fuzIn[inp[0]+'_'+inp[1]] = fuzzyOps.rangeToMF([mean], inType)
        
        fuzOut = {} #create trapezoidal output MFs
        fuzOut['sys_FoM'] = fuzzyOps.rangeToMF(point[2], outType)
        
        fuzzData.append([fuzIn, fuzOut])
    
    inRanges = {    'DATA_e_d':     inputRanges['e_d'],
                    'DATA_sigma':   inputRanges['sigma'],
                    'DATA_w':       inputRanges['w'],
                    'DATA_eta':     inputRanges['eta_d']}
                    
    outRanges = {'sys_FoM' : outputRanges['sys_FoM'] }


    
    #build system                        
    sys = DFES(inRanges, outRanges, 'sigmoid', hidNodes=hidNodes, inGran=inGran, outGran=outGran)
    sys.read_weights(wtFileName) #read network weight foil
    
    error = []
    for point in fuzzData[:]:
        output = sys.run(point[0])
        err = [point[1]['sys_FoM'], np.asarray(output['sys_FoM'])] #get actual and output
        err.append(fuzDistAC(err[0], err[1]))
        error.append(err)
        
    check = 50
    t_tot = 0.0
    for j in range(check):
        i = random.randrange(0, len(fuzzData))
        with Timer() as t:
            sysOut = sys.run(fuzzData[i][0])
        t_tot = t_tot + float(t.secs)
    #print 'Average system time of %d points => %.5f s' % (check, t_tot/check)
    #print '                                 => %.5f s per rule' % ((t_tot/check)/len(sys.rulebase))
    

    #actual vs. predicted plot
    alpha = 0.8
    fig_avp = plt.figure()
    dataMax = len(fuzzData)
    plt.title('Actual vs. Predicted at Max Alpha Cut', y=1.05, fontsize=11)
    lflag = 1
    for err in error:
        AC_actual = fuzzyOps.alpha_cut(alpha, [err[0][0],err[0][1]])
        AC_pred = fuzzyOps.alpha_cut(alpha, (err[1][0],err[1][1]))
        
        if AC_actual <> None and AC_pred <> None:
            plt.scatter(AC_actual[0], AC_pred[0], marker='o', c='r')
            plt.scatter(AC_actual[1], AC_pred[1], marker='x', c='b')
        if lflag == 1:
            plt.legend([r'Min at $ \alpha=%.2f $' % alpha ,r'Max at $ \alpha=%.2f $' % alpha ], loc=2, fontsize=11)  
            lflag = 0
    
    plt.plot([0.0,100.0],[0.0,100.0], '--k')   
    plt.xlim([0.4,0.9])
    plt.ylim([0.4,0.9])
    plt.xlabel('Actual')
    plt.ylabel('Predicted')


    #EXPANDED actual vs. predicted plot (alpha-cuts)
    alphas = [0.4, 0.6, 0.8]
    aplots = [0.4, 0.7, 1.0]
    cplots = plt.cm.Blues([0.3, 0.6, 1.0]) 
    sizes = [10.0, 20.0, 30.0]
    plt.figure()
    tit = 'Actual vs. Predicted (%d total points)' % dataMax
    plt.title(tit, y=1.05)
    hands1 = []
    hands2 = []
    for err in error:
        for i in range(len(alphas)):
            AC_actual = fuzzyOps.alpha_cut(alphas[i], [err[0][0],err[0][1]])
            AC_pred   = fuzzyOps.alpha_cut(alphas[i], (err[1][0],err[1][1]))
            
            if AC_actual <> None and AC_pred <> None: 
                l1 = r'Min at $ \alpha = %s $' % alphas[i]
                l2 = r'Max at $ \alpha = %s $' % alphas[i]
                min_scat = plt.scatter(AC_actual[0], AC_pred[0], s=sizes[i],  marker='o', c=cplots[i], label=l1, alpha=aplots[i])
                max_scat = plt.scatter(AC_actual[1], AC_pred[1], s=sizes[i], marker='d', c=cplots[i], label=l2, alpha=aplots[i])
                if len(hands1) < i + 1: hands1.append(min_scat)
                if len(hands2) < i + 1: hands2.append(max_scat)
    plt.plot([0,900],[0,900], '--k')     
    plt.xlim([0.4,0.9])
    plt.ylim([0.4,0.9])
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.legend(handles=hands1+hands2, fontsize=10, loc='upper left')
    
    #actual vs. predicted plot (centroid)
    plt.figure()
    plt.title('Actual vs. Predicted (Centroid)')
    for err in error:
        try:
            actCent = fuzz.defuzz(err[0][0], err[0][1], 'centroid')
            predCent = fuzz.defuzz(err[1][0], err[1][1], 'centroid')
        except:
            actCent, predCent = None, None
        if actCent <> None and predCent <> None: 
            plt.scatter(actCent, predCent, marker='+', c='m')
    plt.plot([0,5],[0,5], '--k')     
    plt.xlim([0.4,0.9])
    plt.ylim([0.4,0.9])
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    
    
    #check random data points (9)
    fig_rand = plt.figure()
    plt.title('Random Tests:'+wtFileName, y=1.05, fontsize=11)
    for j in range(9):
        i = random.randrange(0, len(error))
        plt.subplot(3,3,j+1)
        plt.plot(error[i][0][0], error[i][0][1], '-r') #plot actual
        plt.plot(error[i][1][0], error[i][1][1], '--b') #plot predicted
        plt.ylim([0,1.1])
        plt.xlim([0.4, 1.0])
        if j > 5: plt.xlabel('FoM')
        if j in [0,3,6]: plt.ylabel('$\mu(x)$')   
    
    plt.legend(['Act.', 'Pred.'], fontsize=8)
    
    #model RSME
    RSME = sys.test(fuzzData, plotPoints=0)
    
    #actual vs. error plot
    fig_ave = plt.figure()
    plt.title('Actual (Centroid) vs. Error'+wtFileName, y=1.05, fontsize=11)
    cents = [fuzz.defuzz(err[0][0], err[0][1], 'centroid') for err in error]
    plt.scatter(cents, [err[2] for err in error])
    plt.xlabel('Actual (Centroid)')
    plt.ylabel('Fuzzy Error') 
    plt.xlim([0.4,0.9])
    
    txt =        'Fuzzy Error Statistics: (alpha-cut distance error)' + '\n'
    txt =  txt + 'Discrete RSME (system test):' + str(RSME) + '\n'
    txt =  txt + 'Total System Error:  ' + str(sum([err[2] for err in error if err[2] <> None])) + '\n'
    txt =  txt + 'Mean Square System Error:  ' + str( (1.0/len([e for e in error if e[2] <> None]))*sum([e[2]**2 for e in error if e[2] <> None])) + '\n'
    txt =  txt + 'Root Mean Square System Error:  ' + str(( (1.0/len([e for e in error if e[2] <> None]))*sum([e[2]**2 for e in error if e[2] <> None]) )**0.5) + '\n'
    txt =  txt + 'Average system time of => ' + str(t_tot/check) + '\n'
    txt =  txt + '         per node => ' + str((t_tot/check)/sum([sys.nIn, sys.nOut, sys.nHid]))

    f = plt.figure(figsize=(6,8))
    f.text(0.1, 0.5, txt)
    
    pp = PdfPages(wtFileName[:-4] + '_VALIDATION.pdf')
    figs = [plt.figure(n) for n in plt.get_fignums()]
    for fig in figs:
        fig.savefig(pp, format='pdf')
    pp.close()
    plt.close('all')
    plt.clf()  
    
##----------------------------------------------------------------------------##
#                        TEST DFES: PHI System 
##----------------------------------------------------------------------------##
def testphiDFESfit(wtFileName, hidNodes, inGran, outGran):
    """
    Test the fit of DFES systems to Figure of Merit Data
    """
    print "*****************************************"
    print " => Testing", wtFileName
    
    testFCLFile = wtFileName
    
    inType='gauss'
    outType = 'gauss'
    
    #Read in Input Data for Morph
    dataIn = readFuzzyInputData('data/morphInputs_13Jun15.csv')
    combinedData = buildInputs(ASPECT_list, dataIn, 'data/phiData_300pts.csv', True)
    combinedData = combinedData[:]
    q=0 #use first (quant inputs) 
    
    inputList = [('VL_SYS_TECH', 'phi'),
                ('FWD_SYS_DRV', 'eta_d'),
                ('FWD_SYS_TYPE', 'phi'),
                ('VL_SYS_TECH', 'f'),
                ('FWD_SYS_PROP', 'eta_p'),
                ('VL_SYS_TECH', 'w'),
                ('VL_SYS_TYPE', 'f'),
                ('VL_SYS_TECH', 'LD'),
                ('WING_SYS_TYPE', 'LD'),
                ('FWD_SYS_TYPE', 'TP'),
                ('VL_SYS_TYPE', 'w'),
                ('WING_SYS_TYPE', 'f'),
                ('VL_SYS_PROP', 'w'),
                ('VL_SYS_TYPE', 'phi'),
                ('VL_SYS_PROP', 'phi'),   ]
    
    inRanges = {inp[0]+"_"+inp[1] : inputRanges[inp[1]] for inp in inputList} #set exact inputs
    outRanges = {'sys_phi' : outputRanges['sys_phi'] } 
    
    #Turn data into fuzzy MFs
    fuzzData = []
    for point in combinedData:
    
        fuzIn = {} #create input MFs for each input
        for inp in point[q]:
            #create singleton input MFs
            mean = sum(point[q][inp])/len(point[q][inp]) #get mean of range (single value)
            fuzIn[inp[0]+'_'+inp[1]] = fuzzyOps.rangeToMF([mean], inType)
        
        fuzOut = {} #create trapezoidal output MFs
        fuzOut['sys_phi'] = fuzzyOps.rangeToMF(point[2], outType)
        
        fuzzData.append([fuzIn, fuzOut])
        
    if False: #random fuzzy data check
        for n in range(3): #number of random points to check
            dat = random.sample(fuzzData, 1)[0] #random fuzzy data point
            fig = plt.figure()
            i = 1
            for inp in inRanges: #for each input
                ax = fig.add_subplot(len(inRanges), 2, i)
                ax.plot(dat[0][inp][0],dat[0][inp][1])
                ax.set_xlim(inRanges[inp])
                ax.set_ylabel(inp, rotation=90)
                ax.set_yticklabels([])
                ax.set_xticklabels([])
                i = i+2
            i = 2
            for otp in outRanges: #for each output
                ax = fig.add_subplot(len(outRanges), 2, i)
                ax.plot(dat[1][otp][0], dat[1][otp][1])
                ax.set_xlim(outRanges[otp])
        plt.show()
        

    
    #build system                        
    sys = DFES(inRanges, outRanges, 'sigmoid', hidNodes=hidNodes, inGran=inGran, outGran=outGran)
    sys.read_weights(wtFileName) #read network weight foil
    
    error = []
    for point in fuzzData[:]:
        output = sys.run(point[0])
        err = [point[1]['sys_phi'], np.asarray(output['sys_phi'])] #get actual and output
        pointErr = fuzDistAC(err[0], err[1])
        if pointErr <> None: err.append(pointErr)
        error.append(err)

    #actual vs. predicted plot
    alpha = 0.6
    fig_avp = plt.figure()
    plt.title('Actual vs. Predicted $(\alpha=%.f, %d data points)$' % (alpha, len(error)))
    lflag = 1
    for err in error:
        AC_actual = fuzzyOps.alpha_cut(alpha, [err[0][0],err[0][1]])
        AC_pred = fuzzyOps.alpha_cut(alpha, (err[1][0],err[1][1]))
        
        if AC_actual <> None and AC_pred <> None:
            plt.scatter(AC_actual[0], AC_pred[0], marker='o', c='r')
            plt.scatter(AC_actual[1], AC_pred[1], marker='x', c='b')
        if lflag == 1:
            plt.legend([r'$Min(^{'+str(alpha)+'}X)$',r'$Max(^{'+str(alpha)+'}X)$'], loc=2)  
            lflag = 0
    
    plt.plot([0.0,100.0],[0.0,100.0], '--k')   
    plt.xlim([1.0, 9.0])
    plt.ylim([1.0, 9.0])
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    
    #EXPANDED actual vs. predicted plot (alpha-cuts)
    alphas = [0.4, 0.6, 0.8]
    aplots = [0.4, 0.7, 1.0]
    cplots = plt.cm.Blues([0.3, 0.6, 1.0]) 
    sizes = [10.0, 20.0, 30.0]
    plt.figure()
    tit = 'Actual vs. Predicted (%d total points)' % len(error)
    plt.title(tit)
    hands1 = []
    hands2 = []
    for err in error:
        for i in range(len(alphas)):
            AC_actual = fuzzyOps.alpha_cut(alphas[i], [err[0][0],err[0][1]])
            AC_pred = fuzzyOps.alpha_cut(alphas[i], (err[1][0],err[1][1]))
                
            if AC_actual <> None and AC_pred <> None: 
                l1 = r'Min at $ \alpha = %s $' % alphas[i]
                l2 = r'Max at $ \alpha = %s $' % alphas[i]
                min_scat = plt.scatter(AC_actual[0], AC_pred[0], s=sizes[i],  marker='o', c=cplots[i], label=l1, alpha=aplots[i])
                max_scat = plt.scatter(AC_actual[1], AC_pred[1], s=sizes[i], marker='d', c=cplots[i], label=l2, alpha=aplots[i])
                if len(hands1) < i + 1: hands1.append(min_scat)
                if len(hands2) < i + 1: hands2.append(max_scat)
    plt.plot([0,900],[0,900], '--k')     
    plt.xlim([1,9])
    plt.ylim([1,9])
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.legend(handles=hands1+hands2, fontsize=10, loc='upper left')
    
    #actual vs. predicted plot (centroid)
    plt.figure()
    plt.title('Actual vs. Predicted (Centroid): ')
    for err in error:
        #try:
        actCent = fuzz.defuzz(err[0][0], err[0][1], 'centroid')
        predCent = fuzz.defuzz(err[1][0], err[1][1], 'centroid')
        #except:
        #    actCent, predCent = None, None
        if actCent <> None and predCent <> None: 
            plt.scatter(actCent, predCent, marker='+', c='m')
    plt.plot([0,900],[0,900], '--k')     
    plt.xlim([1.0,9.0])
    plt.ylim([1.0,9.0])
    plt.xlabel('Actual (Centroid)')
    plt.ylabel('Predicted (Centroid)')
    
    #actual vs. predicted plot (mom)
    plt.figure()
    plt.title('Actual vs. Predicted (Mean of Max): ')
    for err in error:
        actCent = fuzz.defuzz(err[0][0], err[0][1], 'mom')
        predCent = fuzz.defuzz(err[1][0], err[1][1], 'mom')
        if actCent <> None and predCent <> None: 
            plt.scatter(actCent, predCent, marker='+', c='m')
    plt.plot([0,900],[0,900], '--k')     
    plt.xlim([1.0,9.0])
    plt.ylim([1.0,9.0])
    plt.xlabel('Actual (MoM)')
    plt.ylabel('Predicted (MoM)')
    
    #check random data points (9)
    fig_rand = plt.figure()
    plt.title('Random Tests:'+wtFileName, y=1.05, fontsize=11)
    for j in range(9):
        i = random.randrange(0, len(error))
        plt.subplot(3,3,j+1)
        plt.plot(error[i][0][0], error[i][0][1], '-r') #plot actual
        plt.plot(error[i][1][0], error[i][1][1], '--b') #plot predicted
        plt.ylim([0,1.1])
        plt.xlim([1.0, 9.0])   
    
    plt.legend(['Act.', 'Pred.'], fontsize=8)


    #actual centroid vs. error plot
    plt.figure()
    plt.title('Actual (Centroid) vs. Error')
    cents = [fuzz.defuzz(err[0][0], err[0][1], 'centroid') for err in error if err[2] <> None]
    #pctErr = [err[2]/fuzz.defuzz(err[0][0], err[0][1], 'centroid') for err in error if err[2] <> None]
    plt.scatter(cents, [err[2] for err in error if err[2] <> None])
    plt.xlabel('Actual (Centroid)')
    plt.ylabel('Fuzzy Error')
    
    #actual centroid vs. percent error plot
    plt.figure()
    plt.title('Actual (Centroid) vs. %Error')
    cents = [fuzz.defuzz(err[0][0], err[0][1], 'centroid') for err in error if err[2] <> None]
    pctErr = [err[2]/fuzz.defuzz(err[0][0], err[0][1], 'centroid') for err in error if err[2] <> None]
    #pctErr = [(fuzz.defuzz(err[1][0], err[1][1], 'centroid')-fuzz.defuzz(err[0][0], err[0][1], 'centroid'))/fuzz.defuzz(err[0][0], err[0][1], 'centroid') \
    #          for err in error if err[2] <> None]
    plt.scatter(cents, pctErr)
    plt.xlabel('Actual (Centroid)')
    plt.ylabel('%Error (Centroids)')
    
    
    #model RSME
    RSME = sys.test(fuzzData, plotPoints=0)
    
    #actual vs. error plot
    fig_ave = plt.figure()
    plt.title('Actual (Centroid) vs. Error', y=1.05, fontsize=11)
    cents = [fuzz.defuzz(err[0][0], err[0][1], 'centroid') for err in error]
    plt.scatter(cents, [err[2] for err in error])
    plt.xlabel('Actual (Centroid)')
    plt.ylabel('Fuzzy Error') 
    
    txt =        "***RESULTS FOR SYSTEM:" + wtFileName + "***" + '\n'
    txt =  txt + 'Discrete RSME (system test):' + str(RSME) + '\n'
    txt =  txt + 'Total System Error (Fuzzy - AC):  ' + str(sum([err[2] for err in error if err[2] <> None])) + '\n'
    txt =  txt + 'Mean Square System Error (Fuzzy - AC):  ' + str( (1.0/len([e for e in error if e[2] <> None]))*sum([e[2]**2 for e in error if e[2] <> None])) + '\n'
    txt =  txt + 'Root Mean Square System Error( Fuzzy - AC):  ' + str(( (1.0/len([e for e in error if e[2] <> None]))*sum([e[2]**2 for e in error if e[2] <> None]) )**0.5) + '\n'

    f = plt.figure(figsize=(6,8))
    f.text(0.1, 0.5, txt)
    
    pp = PdfPages(wtFileName[:-4] + '_VALIDATION.pdf')
    figs = [plt.figure(n) for n in plt.get_fignums()]
    for fig in figs:
        fig.savefig(pp, format='pdf')
    pp.close()
    plt.close('all')
    plt.clf()   

##----------------------------------------------------------------------------##
#                        TEST DFES: RF System 
##----------------------------------------------------------------------------##
def testRFDFESfit(wtFileName_GWT, hidNodes_GWT, inGran_GWT, outGran_GWT, 
                  wtFileName_P, hidNodes_P, inGran_P, outGran_P, 
                  wtFileName_T, hidNodes_T, inGran_T, outGran_T,
                  wtFileName_VH, hidNodes_VH, inGran_VH, outGran_VH,
                  wtFileName_eWT, hidNodes_eWT, inGran_eWT, outGran_eWT):
    """
    Test the fit of DFES systems to RF Data
    """
    print "*****************************************"
    print " => Testing", wtFileName_GWT, wtFileName_P, wtFileName_VH
    
    inType='gauss'
    outType = 'gauss'
    
        
    def limitData(dataList, outputKey):
        """Limit input data prior to training"""
        print 'Limiting', outputKey, 'to', outputRanges[outputKey],  
        limitedData = []
        for i in range(len(dataList)):
            ldi = [dataList[i][0], dataList[i][1]]
            flag = 0
            if dataList[i][2][0] < outputRanges[outputKey][0]:
                flag = 1
                dataList[i][2][0] = outputRanges[outputKey][0]
            if dataList[i][2][1] > outputRanges[outputKey][1]:
                flag = 1
                dataList[i][2][1] = outputRanges[outputKey][1]
            
            if flag == 0:
                ldi.append(dataList[i][2])
                limitedData.append(ldi)
            
        return limitedData #change to datalist to just limit all raw data
    
    def fuzzifyData(dataFile, outputKey, inMFtype='gauss', outMFtype='gauss'):
        fuzzData = []
        
        for point in dataFile:
            fuzIn = {} #create input MFs for each input
            for inp in point[q]:
                #create singleton input MFs
                mean = sum(point[q][inp])/len(point[q][inp]) #get mean of range (single value)
                fuzIn[inp[0]+'_'+inp[1]] = fuzzyOps.rangeToMF([mean], inMFtype)
            fuzOut = {} 
            fuzOut[outputKey] = fuzzyOps.rangeToMF(point[2], outMFtype)
            fuzzData.append([fuzIn, fuzOut])
            
        return fuzzData    
        
    #Read in Input Data for Morph
    ### FIGURE OF MERIT SYSETMS (FUZZY DATA) ###
    inRanges = {'DATA_phi':        [0.5, 0.95],
                'DATA_w':          [1.0, 150.0],
                'DATA_WS':         [15.0, 300],
                'DATA_sys_etaP':   [0.6, 1.0],
                'DATA_eta_d':      [0.4, 1.0],
                'DATA_sys_FoM':    [0.6, 1.0],
                'DATA_e_d':        [0.0, 0.3],
                'DATA_SFC_quant':  [0.6, 1.05],
                'DATA_dragX':      [0.6, 1.15],
                'DATA_type':       [0.5, 3.5],
                #'DATA_tech':       [-0.5, 4.5],
                #'DATA_jet':        [0.5, 2.5],
            }
    outRanges_GWT = {'sys_GWT'   : [5000.,85000.] }
    outRanges_Pin = {'sys_Pinst' : outputRanges['sys_Pinst'] }
    outRanges_Tin = {'sys_Tinst' : outputRanges['sys_Tinst'] }
    outRanges_VH  = {'sys_VH'    : outputRanges['sys_VH'] }
    outRanges_eWT = {'sys_eWT'   : outputRanges['sys_eWT'] }
    
    print "READING DATA"
    combData_GWT = buildInputs(ASPECT_list, None, 'data/RF_GWT-P-VH-T-eWT_genData_2Sep15_14Sep15_LIMITEDFINAL.csv', False,        #training data set
                            inputCols={'phi':0, 'w':1, 'WS':2, 'sys_etaP':3, 'eta_d':4, 'sys_FoM':5,
                                        'e_d':6, 'dragX':7, 'SFC_quant':8, 'type':9,},# 'tech':10, 'jet':11},
                            outputCols={'sys_GWT':12})
    combData_Pin = buildInputs(ASPECT_list, None, 'data/RF_GWT-P-VH-T-eWT_genData_2Sep15_14Sep15_LIMITEDFINAL.csv', False,        #training data set
                            inputCols={'phi':0, 'w':1, 'WS':2, 'sys_etaP':3, 'eta_d':4, 'sys_FoM':5,
                                        'e_d':6, 'dragX':7, 'SFC_quant':8, 'type':9,},# 'tech':10, 'jet':11},
                            outputCols={'sys_Pinst':13})
    combData_Tin = buildInputs(ASPECT_list, None, 'data/RF_GWT-P-VH-T-eWT_genData_2Sep15_14Sep15_LIMITEDFINAL.csv', False,        #training data set
                            inputCols={'phi':0, 'w':1, 'WS':2, 'sys_etaP':3, 'eta_d':4, 'sys_FoM':5,
                                        'e_d':6, 'dragX':7, 'SFC_quant':8, 'type':9,},# 'tech':10, 'jet':11},
                            outputCols={'sys_Tinst':14})
    combData_VH  = buildInputs(ASPECT_list, None, 'data/RF_GWT-P-VH-T-eWT_genData_2Sep15_14Sep15_LIMITEDFINAL.csv', False,        #training data set
                            inputCols={'phi':0, 'w':1, 'WS':2, 'sys_etaP':3, 'eta_d':4, 'sys_FoM':5,
                                        'e_d':6, 'dragX':7, 'SFC_quant':8, 'type':9,},# 'tech':10, 'jet':11},
                            outputCols={'sys_VH':15})
    combData_eWT = buildInputs(ASPECT_list, None, 'data/RF_GWT-P-VH-T-eWT_genData_2Sep15_14Sep15_LIMITEDFINAL.csv', False,        #training data set
                            inputCols={'phi':0, 'w':1, 'WS':2, 'sys_etaP':3, 'eta_d':4, 'sys_FoM':5,
                                        'e_d':6, 'dragX':7, 'SFC_quant':8, 'type':9,},# 'tech':10, 'jet':11},
                            outputCols={'sys_eWT':16})
    q=0 #use first (quant inputs)
    
    nTest = 300
    print "SAMPLING DATA"
    combData_GWT = random.sample(combData_GWT,nTest)
    combData_Pin = random.sample(combData_Pin,nTest)
    combData_Tin = random.sample(combData_Tin,nTest)
    combData_VH  = random.sample(combData_VH, nTest)
    combData_eWT = random.sample(combData_eWT,nTest)
    
    # limit data
    print "LIMITING DATA"
    combData_GWT = limitData(combData_GWT, 'sys_GWT')
    combData_Pin = limitData(combData_Pin, 'sys_Pinst')
    combData_Tin = limitData(combData_Tin, 'sys_Tinst')
    combData_VH  = limitData(combData_VH,  'sys_VH')
    combData_eWT = limitData(combData_eWT, 'sys_eWT')
    
    #Turn data into fuzzy MFs
    print "FUZZIFYING DATA"
    fuzzData_GWT = fuzzifyData(combData_GWT, 'sys_GWT')
    fuzzData_Pin = fuzzifyData(combData_Pin, 'sys_Pinst')
    fuzzData_Tin = fuzzifyData(combData_Tin, 'sys_Tinst')
    fuzzData_VH  = fuzzifyData(combData_VH, 'sys_VH')
    fuzzData_eWT = fuzzifyData(combData_eWT, 'sys_eWT')
    

    #build system                        
    sys_GWT = DFES(inRanges, outRanges_GWT, 'sigmoid', hidNodes=hidNodes_GWT, inGran=inGran_GWT, outGran=outGran_GWT)
    sys_GWT.read_weights(wtFileName_GWT) #read network weights
    sys_P = DFES(inRanges, outRanges_Pin, 'sigmoid', hidNodes=hidNodes_P, inGran=inGran_P, outGran=outGran_P)
    sys_P.read_weights(wtFileName_P) #read network weights
    sys_T = DFES(inRanges, outRanges_Tin, 'sigmoid', hidNodes=hidNodes_T, inGran=inGran_T, outGran=outGran_T)
    sys_T.read_weights(wtFileName_T) #read network weights
    sys_VH = DFES(inRanges, outRanges_VH, 'sigmoid', hidNodes=hidNodes_VH, inGran=inGran_VH, outGran=outGran_VH)
    sys_VH.read_weights(wtFileName_VH) #read network weights
    sys_eWT = DFES(inRanges, outRanges_eWT, 'sigmoid', hidNodes=hidNodes_eWT, inGran=inGran_eWT, outGran=outGran_eWT)
    sys_eWT.read_weights(wtFileName_eWT) #read network weights
    
    error_GWT = []
    error_P   = []
    error_T   = []
    error_VH  = []
    error_eWT = []
    
    for point in fuzzData_GWT[:]:
        output = sys_GWT.run(point[0])
        err = [point[1]['sys_GWT'], np.asarray(output['sys_GWT'])] #get actual and output
        pointErr = fuzDistAC(err[0], err[1])
        err.append(pointErr)
        error_GWT.append(err)
    for point in fuzzData_Pin[:]:
        output = sys_P.run(point[0])
        err = [point[1]['sys_Pinst'], np.asarray(output['sys_Pinst'])] #get actual and output
        pointErr = fuzDistAC(err[0], err[1])
        err.append(pointErr)
        error_P.append(err)
    for point in fuzzData_VH[:]:
        output = sys_VH.run(point[0])
        err = [point[1]['sys_VH'], np.asarray(output['sys_VH'])] #get actual and output
        pointErr = fuzDistAC(err[0], err[1])
        err.append(pointErr)
        error_VH.append(err)
    for point in fuzzData_Tin[:]:
        output = sys_T.run(point[0])
        err = [point[1]['sys_Tinst'], np.asarray(output['sys_Tinst'])] #get actual and output
        pointErr = fuzDistAC(err[0], err[1])
        err.append(pointErr)
        error_T.append(err)
    for point in fuzzData_eWT[:]:
        output = sys_eWT.run(point[0])
        err = [point[1]['sys_eWT'], np.asarray(output['sys_eWT'])] #get actual and output
        pointErr = fuzDistAC(err[0], err[1])
        err.append(pointErr)
        error_eWT.append(err)
        
    #print "GWT System", sys_GWT.test(fuzzData_GWT, plotPoints=0) 
    #print "Pinst System -", sys_P.test(fuzzData_Pin, plotPoints=0) 
    #print "Tinst System -", sys_T.test(fuzzData_Tin, plotPoints=0) 
    #print "VH System -", sys_VH.test(fuzzData_VH, plotPoints=0)     
    #print "eWT System -", sys_eWT.test(fuzzData_eWT, plotPoints=0) 
    
    files = [wtFileName_GWT, wtFileName_P, wtFileName_T, wtFileName_VH, wtFileName_eWT]
    fData = [fuzzData_GWT, fuzzData_Pin, fuzzData_Tin, fuzzData_VH, fuzzData_eWT]
    limits = [[1000,50000], [1000,15000], [1000,25000], [100,500], [300,8000]]
    systems = [sys_GWT, sys_P, sys_T, sys_VH, sys_eWT]
    k = 0
    for error in [error_GWT]:#, error_P, error_P, error_VH, error_T, error_eWT]:
        
        wtFileName = files[k]
        fuzzData = fData[k]
        xlimits = limits[k]
        system = systems[k]
        k = k+1
        
        #actual vs. predicted plot
        alpha = 0.5
        fig_avp = plt.figure()
        plt.title('Actual vs. Predicted at Max Alpha Cut'+wtFileName, y=1.05, fontsize=11)
        lflag = 1
        for err in error:
            AC_actual = fuzzyOps.alpha_cut(alpha, [err[0][0],err[0][1]])
            AC_pred = fuzzyOps.alpha_cut(alpha, (err[1][0],err[1][1]))
            
            if AC_actual <> None and AC_pred <> None:
                plt.scatter(AC_actual[0], AC_pred[0], marker='o', c='r')
                plt.scatter(AC_actual[1], AC_pred[1], marker='x', c='b')
            if lflag == 1:
                plt.legend([r'$Min(^{'+str(alpha)+'}X)$',r'$Max(^{'+str(alpha)+'}X)$'], loc=2)  
                lflag = 0
        
        plt.plot([0.0,50000.0],[0.0,50000.0], '--k')   
        plt.xlim(xlimits)
        plt.ylim(xlimits)
        plt.xlabel('Actual')
        plt.ylabel('Predicted')
        
        #check random data points (9)
        fig_rand = plt.figure()
        plt.title('Random Tests:'+wtFileName, y=1.05, fontsize=11)
        for j in range(9):
            i = random.randrange(0, len(error))
            plt.subplot(3,3,j+1)
            plt.plot(error[i][0][0], error[i][0][1], '-r') #plot actual
            plt.plot(error[i][1][0], error[i][1][1], '--b') #plot predicted
            plt.ylim([0,1.1])
            plt.xlim(xlimits)   
        
        plt.legend(['Act.', 'Pred.'], fontsize=8)
        
        #actual vs. error plot
        
        fig_ave = plt.figure()
        plt.title('Actual (Centroid) vs. Rel. Fuzzy Error'+wtFileName, y=1.05, fontsize=11)
        cents = [fuzz.defuzz(err[0][0], err[0][1], 'centroid') for err in error if err[2] <> None]
        pctErr = [error[i][2]/fuzz.defuzz(err[0][0], err[0][1], 'centroid') for i in range(len(error)) if error[i][2] <> None]
        plt.scatter(cents, pctErr)
        plt.xlabel('Actual (Centroid)')
        plt.ylabel('Relative Fuzzy Error') 
        
        #model RSME
        RSME = system.test(fuzzData, plotPoints=5)/len(fuzzData)
        
        print "***RESULTS FOR SYSTEM:", wtFileName, "***"
        print "System Fuzzy MSE (AC):", sum([err[2]**2 for err in error if err[2] <> None])/len([err for err in error if err[2] <> None])
        print "System Fuzzy RMSE (AC):", (sum([err[2]**2 for err in error if err[2] <> None])/len([err for err in error if err[2] <> None]))**0.5

        txt =        "***RESULTS FOR SYSTEM:" + wtFileName + "***" + '\n'
        txt =  txt + 'Discrete RSME (system test):' + str(RSME) + '\n'
        txt =  txt + 'Total System Error (Fuzzy - AC):  ' + str(sum([err[2] for err in error if err[2] <> None])) + '\n'
        txt =  txt + 'Mean Square System Error (Fuzzy - AC):  ' + str( (1.0/len([e for e in error if e[2] <> None]))*sum([e[2]**2 for e in error if e[2] <> None])) + '\n'
        txt =  txt + 'Root Mean Square System Error( Fuzzy - AC):  ' + str(( (1.0/len([e for e in error if e[2] <> None]))*sum([e[2]**2 for e in error if e[2] <> None]) )**0.5) + '\n'
    
        f = plt.figure(figsize=(6,8))
        f.text(0.1, 0.5, txt)
        
        
        #visuzlize system with random data point
        #i = random.randrange(0, len(combinedData))
        #inputs = {key[0]+"_"+key[1]:combinedData[i][0][key] for key in combinedData[i][0]}
        #sys.run(inputs, TESTMODE=True)
        
        pp = PdfPages(wtFileName[:-4] + '_VALIDATION.pdf')
        figs = [plt.figure(n) for n in plt.get_fignums()]
        for fig in figs:
            fig.savefig(pp, format='pdf')
        pp.close()
        plt.close('all')
        plt.clf()   

    return None



##----------------------------------------------------------------------------##
#                        TEST DFES: RF System 
##----------------------------------------------------------------------------##
def testRFDFESfitOLD(wtFileName_GWT, hidNodes_GWT, inGran_GWT, outGran_GWT, 
                     wtFileName_P, hidNodes_P, inGran_P, outGran_P, 
                     wtFileName_VH, hidNodes_VH, inGran_VH, outGran_VH):
    """
    Test the fit of DFES systems to RF Data
    """
    print "*****************************************"
    print " => Testing", wtFileName_GWT, wtFileName_P, wtFileName_VH
    
    inType='gauss'
    outType = 'gauss'
    
    
    def limitData(dataFile, outputKey):
        """Limit input data prior to training"""
        print 'limiting w/ outputKey', outputKey
        for i in range(len(dataFile)):
            if dataFile[i][2][0] < outputRanges[outputKey][0]:
                print "Limited", dataFile[i][2][0], "to", outputRanges[outputKey][0]
                dataFile[i][2][0] = outputRanges[outputKey][0]
                
            if dataFile[i][2][1] > outputRanges[outputKey][1]:
                print "Limited", dataFile[i][2][1], "to", outputRanges[outputKey][1]
                dataFile[i][2][1] = outputRanges[outputKey][1]
                
        return dataFile
    
    def fuzzifyData(dataFile, outputKey, inMFtype='gauss', outMFtype='gauss'):
        fuzzData = []
        q=0
        print "Fuzzifying", outputKey
        for point in dataFile:
            fuzIn = {} #create input MFs for each input
            for inp in point[q]:
                #create singleton input MFs
                mean = sum(point[q][inp])/len(point[q][inp]) #get mean of range (single value)
                fuzIn[inp[0]+'_'+inp[1]] = fuzzyOps.rangeToMF([mean], inMFtype)
            fuzOut = {} 
            fuzOut[outputKey] = fuzzyOps.rangeToMF(point[2], outMFtype)
            fuzzData.append([fuzIn, fuzOut])
            
        return fuzzData    
        
    #Read in Input Data for Morph
    ### FIGURE OF MERIT SYSETMS (FUZZY DATA) ###
    inRanges = {  'DATA_phi':        [0.5, 0.85],
                    'DATA_w':          [1.0, 150.0],
                    'DATA_WS':         [15.0, 300],
                    'DATA_sys_etaP':   [0.6, 1.0],
                    'DATA_eta_d':      [0.7, 1.0],
                    'DATA_sys_FoM':    [0.3, 1.0],
                    'DATA_e_d':        [0.0, 0.3],
                    'DATA_SFC_quant':  [0.35,0.75],
                    'DATA_type':       [0.5, 3.5],}  

    outRanges_GWT = {'sys_GWT'   : outputRanges['sys_GWT'] }
    outRanges_Pin = {'sys_Pinst' : outputRanges['sys_Pinst'] }
    outRanges_VH  = {'sys_VH'    : [200,500] }
    
    print "READING DATA"
    combData_GWT = buildInputs(ASPECT_list, None, 'data/RF_GWT-P-VH_genData_20Aug15.csv', False,        #training data set
                            inputCols={'phi':0, 'w':1, 'WS':2, 'sys_etaP':3, 'eta_d':4, 'sys_FoM':5, 
                                       'e_d':6, 'SFC_quant':7, 'type':8, },#'tech':9 },
                            outputCols={'sys_GWT':14})
    combData_Pin = buildInputs(ASPECT_list, None, 'data/RF_GWT-P-VH_genData_20Aug15.csv', False,        #training data set
                            inputCols={'phi':0, 'w':1, 'WS':2, 'sys_etaP':3, 'eta_d':4, 'sys_FoM':5, 
                                        'e_d':6, 'SFC_quant':7, 'type':8},
                            outputCols={'sys_Pinst':20})
    combData_VH  = buildInputs(ASPECT_list, None, 'data/RF_GWT-P-VH_genData_20Aug15.csv', False,        #training data set
                            inputCols={'phi':0, 'w':1, 'WS':2, 'sys_etaP':3, 'eta_d':4, 'sys_FoM':5, 
                                       'e_d':6, 'SFC_quant':7, 'type':8, },#'tech':9 },
                            outputCols={'sys_VH':26})
    q=0 #use first (quant inputs)
    
    nTest = 300
    
    print "SAMPLING DATA"
    combData_GWT = random.sample(combData_GWT,nTest)
    combData_Pin = random.sample(combData_Pin,nTest)
    combData_VH  = random.sample(combData_VH, nTest)
    
    #Turn data into fuzzy MFs
    print "FUZZIFYING DATA"
    fuzzData_GWT = fuzzifyData(combData_GWT, 'sys_GWT')
    fuzzData_Pin = fuzzifyData(combData_Pin, 'sys_Pinst')
    fuzzData_VH  = fuzzifyData(combData_VH, 'sys_VH')
    

    #build system                        
    sys_GWT = DFES(inRanges, outRanges_GWT, 'sigmoid', hidNodes=hidNodes_GWT, inGran=inGran_GWT, outGran=outGran_GWT)
    sys_GWT.read_weights(wtFileName_GWT) #read network weights
    sys_P = DFES(inRanges, outRanges_Pin, 'sigmoid', hidNodes=hidNodes_P, inGran=inGran_P, outGran=outGran_P)
    sys_P.read_weights(wtFileName_P) #read network weights
    sys_VH = DFES(inRanges, outRanges_VH, 'sigmoid', hidNodes=hidNodes_VH, inGran=inGran_VH, outGran=outGran_VH)
    sys_VH.read_weights(wtFileName_VH) #read network weights
   
    error_GWT = []
    error_P   = []
    error_VH  = []
    
    for point in fuzzData_GWT[:]:
        output = sys_GWT.run(point[0])
        err = [point[1]['sys_GWT'], np.asarray(output['sys_GWT'])] #get actual and output
        pointErr = fuzDistAC(err[0], err[1])
        err.append(pointErr)
        error_GWT.append(err)
    for point in fuzzData_Pin[:]:
        output = sys_P.run(point[0])
        err = [point[1]['sys_Pinst'], np.asarray(output['sys_Pinst'])] #get actual and output
        pointErr = fuzDistAC(err[0], err[1])
        err.append(pointErr)
        error_P.append(err)
    for point in fuzzData_VH[:]:
        output = sys_VH.run(point[0])
        err = [point[1]['sys_VH'], np.asarray(output['sys_VH'])] #get actual and output
        pointErr = fuzDistAC(err[0], err[1])
        err.append(pointErr)
        error_VH.append(err)
    
    files = [wtFileName_GWT, wtFileName_P, wtFileName_VH]
    fData = [fuzzData_GWT, fuzzData_Pin, fuzzData_VH]
    limits = [[5000,45000], [2000,14000], [250,500]]
    systems = [sys_GWT, sys_P, sys_VH, ]
    k = 0
    for error in [error_GWT, error_P, error_VH]:#, error_P, error_VH, error_T, error_eWT]:
        
        wtFileName = files[k]
        fuzzData = fData[k]
        xlimits = limits[k]
        system = systems[k]
        k = k+1
        
        #actual vs. predicted plot
        alpha = 0.85
        fig_avp = plt.figure()
        plt.title('Actual vs. Predicted at Max Alpha Cut'+wtFileName, y=1.05, fontsize=11)
        lflag = 1
        for err in error:
            AC_actual = fuzzyOps.alpha_cut(alpha, [err[0][0],err[0][1]])
            AC_pred = fuzzyOps.alpha_cut(alpha, (err[1][0],err[1][1]))
            
            if AC_actual <> None and AC_pred <> None:
                plt.scatter(AC_actual[0], AC_pred[0], marker='o', c='r')
                plt.scatter(AC_actual[1], AC_pred[1], marker='x', c='b')
            if lflag == 1:
                plt.legend([r'$Min(^{'+str(alpha)+'}X)$',r'$Max(^{'+str(alpha)+'}X)$'], loc=2)  
                lflag = 0
        
        plt.plot([0.0,10**10],[0.0,10**10], '--k')   
        plt.xlim(xlimits)
        plt.ylim(xlimits)
        plt.xlabel('Actual')
        plt.ylabel('Predicted')
        
        
        #EXPANDED actual vs. predicted plot (alpha-cuts)
        alphas = [0.4, 0.6, 0.8]
        aplots = [0.4, 0.7, 1.0]
        cplots = plt.cm.Blues([0.3, 0.6, 1.0]) 
        sizes = [10.0, 20.0, 30.0]
        plt.figure()
        tit = 'Actual vs. Predicted (%d total points)' % len(error)
        plt.title(tit, y=1.05)
        hands1 = []
        hands2 = []
        for err in error:
            for i in range(len(alphas)):
                AC_actual = fuzzyOps.alpha_cut(alphas[i], [err[0][0],err[0][1]])
                AC_pred = fuzzyOps.alpha_cut(alphas[i], (err[1][0],err[1][1]))
                    
                if AC_actual <> None and AC_pred <> None: 
                    l1 = r'Min at $ \alpha = %s $' % alphas[i]
                    l2 = r'Max at $ \alpha = %s $' % alphas[i]
                    min_scat = plt.scatter(AC_actual[0], AC_pred[0], s=sizes[i],  marker='o', c=cplots[i], label=l1, alpha=aplots[i])
                    max_scat = plt.scatter(AC_actual[1], AC_pred[1], s=sizes[i], marker='d', c=cplots[i], label=l2, alpha=aplots[i])
                    if len(hands1) < i + 1: hands1.append(min_scat)
                    if len(hands2) < i + 1: hands2.append(max_scat)
        plt.plot([0,10**10],[0,10**10], '--k')     
        plt.xlim(xlimits)
        plt.ylim(xlimits)
        plt.xlabel('Actual')
        plt.ylabel('Predicted')
        plt.legend(handles=hands1+hands2, fontsize=10, loc='upper left')
        
        #actual vs. predicted plot (centroid)
        plt.figure()
        plt.title('Actual vs. Predicted (Centroid)', y=1.05)
        for err in error:
            #try:
            actCent = fuzz.defuzz(err[0][0], err[0][1], 'centroid')
            predCent = fuzz.defuzz(err[1][0], err[1][1], 'centroid')
            #except:
            #    actCent, predCent = None, None
            if actCent <> None and predCent <> None: 
                plt.scatter(actCent, predCent, marker='+', c='m')
        plt.plot([0,10**10],[0,10**10], '--k')     
        plt.xlim(xlimits)
        plt.ylim(xlimits)
        plt.xlabel('Actual (Centroid)')
        plt.ylabel('Predicted (Centroid)')        
        
        
        
        #check random data points (9)
        fig_rand = plt.figure()
        plt.title('Random Tests:'+wtFileName, y=1.05, fontsize=11)
        for j in range(9):
            i = random.randrange(0, len(error))
            plt.subplot(3,3,j+1)
            plt.plot(error[i][0][0], error[i][0][1], '-r') #plot actual
            plt.plot(error[i][1][0], error[i][1][1], '--b') #plot predicted
            plt.ylim([0,1.1])
            plt.xlim(xlimits)   
        
        plt.legend(['Act.', 'Pred.'], fontsize=8)
        
        #actual vs. error plot
        
        fig_ave = plt.figure()
        plt.title('Actual (Centroid) vs. Rel. Fuzzy Error'+wtFileName, y=1.05, fontsize=11)
        cents = [fuzz.defuzz(err[0][0], err[0][1], 'centroid') for err in error if err[2] <> None]
        pctErr = [error[i][2]/fuzz.defuzz(err[0][0], err[0][1], 'centroid') for i in range(len(error)) if error[i][2] <> None]
        plt.scatter(cents, pctErr)
        plt.xlim(xlimits)
        plt.xlabel('Actual (Centroid)')
        plt.ylabel('Relative Fuzzy Error') 
        
        #model RSME
        RSME = system.test(fuzzData, plotPoints=0)/len(fuzzData)
        
        print "***RESULTS FOR SYSTEM:", wtFileName, "***"
        print "System Fuzzy MSE (AC):", sum([err[2]**2 for err in error if err[2] <> None])/len([err for err in error if err[2] <> None])
        print "System Fuzzy RMSE (AC):", (sum([err[2]**2 for err in error if err[2] <> None])/len([err for err in error if err[2] <> None]))**0.5

        txt =        "***RESULTS FOR SYSTEM:" + wtFileName + "***" + '\n'
        txt =  txt + 'Discrete RSME (system test):' + str(RSME) + '\n'
        txt =  txt + 'Total System Error (Fuzzy - AC):  ' + str(sum([err[2] for err in error if err[2] <> None])) + '\n'
        txt =  txt + 'Mean Square System Error (Fuzzy - AC):  ' + str( (1.0/len([e for e in error if e[2] <> None]))*sum([e[2]**2 for e in error if e[2] <> None])) + '\n'
        txt =  txt + 'Root Mean Square System Error( Fuzzy - AC):  ' + str(( (1.0/len([e for e in error if e[2] <> None]))*sum([e[2]**2 for e in error if e[2] <> None]) )**0.5) + '\n'
    
        f = plt.figure(figsize=(6,8))
        f.text(0.1, 0.5, txt)
        
        
        #visuzlize system with random data point
        #i = random.randrange(0, len(combinedData))
        #inputs = {key[0]+"_"+key[1]:combinedData[i][0][key] for key in combinedData[i][0]}
        #sys.run(inputs, TESTMODE=True)
        
        pp = PdfPages(wtFileName[:-4] + '_VALIDATION.pdf')
        figs = [plt.figure(n) for n in plt.get_fignums()]
        for fig in figs:
            fig.savefig(pp, format='pdf')
        pp.close()
        plt.close('all')
        plt.clf()   

    return None    
################################################################################
if __name__=="__main__":
    
    #TEST PHI FRBS
    
    #testPhiFRBSfit('FCL_files/FRBS_phi/PHIsys_trained_5-2In9-2Out_gauss250tData_tri50vDataBEST.fcl', 'tri', 'tri', 'gauss', 'gauss')
    #testPhiFRBSfit('FCL_files/FRBS_phi/PHIsys_trained_5-3In9-3Out_tri250tData_50vData_diffEvBEST.fcl', 'tri', 'tri', 'tri', 'tri')
    #testPhiFRBSfit('FCL_files/FRBS_phi/PHIsys_trained_5-2In9-2Out_gauss250-50Data_diffEvBEST.fcl', 'gauss', 'gauss', 'gauss', 'gauss')
    #testPhiFRBSfit('FCL_files/FRBS_phi/PHIsys_trained_5-2In9-2Out_gauss250-50Data_GABEST.fcl', 'gauss', 'gauss', 'gauss', 'gauss')
    #testPhiFRBSfit('FCL_files/FRBS_phi/PHIsys_trained_5-3In9-3Out_tri250-50Data_GABEST.fcl', 'tri', 'tri', 'tri', 'tri')
    #testPhiFRBSfit('FCL_files/FRBS_phi/phi_FRBS_5-2_9-2_OptIns_unOpt.fcl', 'gauss', 'gauss', 'gauss', 'gauss')
    #testPhiFRBSfit('FCL_files/FRBS_phi/PHIsys_trained_5-2In9-2Out_gauss250tData_50vData_optInputsBEST.fcl', 'gauss', 'gauss', 'gauss', 'gauss')
    #testPhiFRBSfit('FCL_files/FRBS_phi/PHIsys_trained_5-2In9-2Out_gauss250tData_50vData_optInputsBEST_GA.fcl', 'gauss', 'gauss', 'gauss', 'gauss')
    #testPhiFRBSfit('FCL_files/FRBS_phi/PHIsys_trained_5-2In9-2Out_gauss250tData_50vData_optInputsBEST_GA_fullRange.fcl', 'gauss', 'gauss', 'gauss', 'gauss')

    #testFCLFile1 = 'FCL_files/FRBS_FoM/FoMsys_trained 5-2In_9-2Out_gauss400tData_100vData_unOPT.fcl'
    #testFCLFile2 = 'FCL_files/FRBS_FoM/FoMsys_trained_5-2In9-2Out_sing-gauss400tData_100vData_tempBEST_0.00380270602952.fcl'
    #testFCLFile3 = 'FCL_files/FoMsys_trained_5-2In9-2Out_sing-gauss400tData_100vData_tempBEST_0.00252829727145.fcl'
    #testFCLFile4 = 'FCL_files/FRBS_FoM/FoMsys_trained_5-2In9-2Out_sing-gauss400tData_100vData_tempBEST_0.00396757693885.fcl'
    #testFoMFRBSfit(testFCLFile3, 'sing', 'gauss', 'gauss', 'gauss')
    #testFoMFRBSfit(testFCLFile2, 'sing', 'gauss', 'gauss', 'gauss')
    #testFoMFRBSfit(testFCLFile1, 'sing', 'gauss', 'gauss', 'gauss')
    #testFoMFRBSfit(testFCLFile4, 'sing', 'gauss', 'gauss', 'gauss')

    
  
    #TEST DFES FoM
    #testFoMDFESfit('FCL_files/DFES_Fom/DFES_FOMdata_data(500)_nodes(100_30_30).nwf', 100, 30, 30)
    #testFoMDFESfit('FCL_files/DFES_Fom/DFES_FOMdata_data(500)_nodes(100_30_40).nwf', 100, 30, 40)
    #testFoMDFESfit('FCL_files/DFES_Fom/DFES_FOMdata_data(500)_nodes(120_30_50).nwf', 120, 30, 50)
    #testFoMDFESfit('FCL_files/DFES_Fom/DFES_FOMdata_data(500)_nodes(130_40_40).nwf', 130, 40, 40)
    #testFoMDFESfit('FCL_files/DFES_Fom/DFES_FOMdata_data(500)_nodes(160_50_30).nwf', 160, 50, 30)
    #testFoMDFESfit('FCL_files/DFES_Fom/DFES_FOMdata_data(500)_nodes(160_50_50).nwf', 160, 50, 50)
    #testFoMDFESfit('FCL_files/DFES_FOMdata_gauss(500)_nodes(100_30_30).nwf', 100, 30, 30,)
    #testFoMDFESfit('FCL_files/DFES_FOMdata_trap(500)_nodes(100_30_30).nwf', 100, 30, 30, inDataForm='trap', outDataForm='trap')
    #testFoMDFESfit('FCL_files/DFES_FOMdata_tri(500)_nodes(100_30_30).nwf', 100, 30, 30, inDataForm='tri', outDataForm='tri')

    
    #TEST DFES_phi
    #testphiDFESfit('FCL_files/DFES_phi/DFES_PHIwExperts_data(300)_nodes(200_30_40)_6Sep15.nwf', 200,  30,  40)


    
    #TEST DFES RF
    """
    testRFDFESfit('FCL_files/temp_opt_weights_file.nwf', 250, 35, 60, 
                  'FCL_files/temp_opt_weights_file.nwf', 250, 35, 60,
                  'FCL_files/temp_opt_weights_file.nwf', 250, 35, 60,
                  'FCL_files/temp_opt_weights_file.nwf', 250, 35, 60,
                  'FCL_files/temp_opt_weights_file.nwf', 250, 35, 60,)
    """
    testRFDFESfitOLD('FCL_files/DFES_RF/BEST/RFdata20Aug_DFES_GWT_30_250_50.nwf', 250, 30, 50, #'FCL_files/DFES_RF/OLD/DFES_RFdata_GWT_data(600)_nodes(250_40_50)_combError.nwf', 250, 40, 50,
                     'FCL_files/DFES_RF/BEST/RFdata20Aug_DFES_Pin_30_250_50.nwf', 250, 30, 50, #'FCL_files/DFES_RF/OLD/DFES_RFdata_Pinst_data(600)_nodes(250_40_50)_combError.nwf', 250, 40, 50,
                     'FCL_files/DFES_RF/BEST/RFdata20Aug_DFES_VH_40_250_50.nwf', 250, 40, 50)#'FCL_files/DFES_RF/OLD/DFES_RFdata_VH_data(600)_nodes(250_40_50)_combError.nwf', 250, 40, 50,)
    