# -*- coding: utf-8 -*-
"""
Created on Thu Feb  5 12:28:43 2015

@author: frankpatterson
"""
import matplotlib.pyplot as plt



def plotFuzzy(values, fuzzy_type=None, labels=None):
    """
    takes in set of fuzzy numbers as values and plots them
    fuzzy_type:
         None: custom fuzzy fuction in the form [[x values], [membership values]]
        'FT': fuzzy triangular in the form mu([a,b,c]) = [0,1,0]
        'FZ': fuzzy triangular in the form mu([a,b,c,d]) = [0,1,1,0]
    """
    ax = plt.subplot()

    for v in values:    
        if fuzzy_type == 'FT':
            ax.plot(v,[0.,1.,0.])
    
    
    
    
    
    
    return ax
    
    
if __name__=="__main__": 
    
    t1 = [1,3,5]
    t2 = [3,4,7]
    t3 = [3,5,6]
    f = plt.figure()
    p1 = plotFuzzy([t1,t2], fuzzy_type='FT')
    p2 = plotFuzzy([t3], fuzzy_type='FT')
    p1.set_xlim((1.,9.))
    #f.add_axes(p1,p2)
    plt.show()
    