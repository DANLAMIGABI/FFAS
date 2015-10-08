#author - Frank Patterson
import math
import copy
import random
import numpy as np

from timer import Timer


from training import *
from systems import *
import fuzzy_operations as fuzzyOps
import matplotlib.pyplot as plt
import matplotlib.patches as patches
plt.ioff()

import scipy.stats as stats

ASPECT_list = ['VL_SYS_TYPE', 'VL_SYS_PROP', 'VL_SYS_DRV', 'VL_SYS_TECH', \
            'FWD_SYS_PROP', 'FWD_SYS_DRV', 'FWD_SYS_TYPE',\
            'WING_SYS_TYPE', \
            'ENG_SYS_TYPE']
            
combData_GWT = buildInputs(ASPECT_list, None, 'data/RF_GWT-P-VH_genData_20Aug15.csv', False,        #training data set
                        inputCols={'phi':0, 'w':1, 'WS':2, 'sys_etaP':3, 'eta_d':4, 'sys_FoM':5, 
                                    'e_d':6, 'SFC_quant':7, 'type':8},
                        outputCols={'sys_VH':14})            
combData_Pin = buildInputs(ASPECT_list, None, 'data/RF_GWT-P-VH_genData_20Aug15.csv', False,        #training data set
                        inputCols={'phi':0, 'w':1, 'WS':2, 'sys_etaP':3, 'eta_d':4, 'sys_FoM':5, 
                                    'e_d':6, 'SFC_quant':7, 'type':8},
                        outputCols={'sys_Pin':20})            
combData_VH = buildInputs(ASPECT_list, None, 'data/RF_GWT-P-VH_genData_20Aug15.csv', False,        #training data set
                        inputCols={'phi':0, 'w':1, 'WS':2, 'sys_etaP':3, 'eta_d':4, 'sys_FoM':5, 
                                    'e_d':6, 'SFC_quant':7, 'type':8},
                        outputCols={'sys_VH':26})

### Pinst
plt.figure()      
for dat in combData_Pin[:]:
    if dat[2][0] == dat[2][1]: r = 0.01*dat[2][0]
    else: r = dat[2][1] - dat[2][0]
    xs = np.arange(dat[2][0]-r, dat[2][1]+r, r/20)
    mean = sum(dat[2])/2.0
    std = r/3.0
    #h = list(np.random.normal(mean, std, size=200))
    #alldata = alldata + h
    ys = stats.norm.pdf(xs,mean,std)
    ys = [y/max(ys) for y in ys]
    #plt.plot(xs, ys, c='b', alpha=0.2)
    fx,fy = fuzzyOps.rangeToMF(dat[2], 'gauss')
    plt.plot(fx,fy, c='b', alpha=0.25)



plt.xlabel('Power Installed (shp)')
plt.xlim([1000,14000])
plt.ylabel('Membership')



"""
plt.figure()
### GWT
alldata = []  
for dat in combData_GWT[:]:
    if dat[2][0] == dat[2][1]: r = 0.1*dat[2][0]
    else:                   r = dat[2][1] - dat[2][0]
    #print dat[2][0]-r, dat[2][1]+r, r
    #if dat[0] == dat[2][1]: xs = np.arange(dat[2][0]-r, dat[2][1]+r, r/5.0)
    xs = np.arange(dat[2][0]-r, dat[2][1]+r, r/20.0)
    mean = sum(dat[2])/2.0
    std = r/3.0
    h = list(np.random.normal(mean, std, size=300))
    alldata = alldata + h
    #plt.plot(stats.norm.pdf(xs, mean, std))
    #plt.hist(h, bins=10, alpha=0.2)
    
alldata = sorted(alldata)
nfit1 = stats.norm.pdf(alldata, np.average(alldata), np.std(alldata))
nfit2 = stats.norm.cdf(alldata, np.average(alldata), np.std(alldata))
a, b, loc, scale = stats.beta.fit(alldata)
bfit1 = stats.beta.pdf(alldata, a,b, loc=loc, scale=scale)
bfit2 = stats.beta.cdf(alldata, a,b, loc=loc, scale=scale)


#plt.figure()
plt.subplot(3,2,1)
#plt.plot(alldata,nfit1,'-b')
#plt.plot(alldata,bfit1)
plt.hist(alldata, bins=30, normed=True)
plt.xlabel('Gross Weight (lbs)')
plt.xlim([500,50000])
plt.ylabel('Frequency (Normalized)')

#plt.figure()
plt.subplot(3,2,2)
#plt.plot(alldata,nfit2)
#plt.plot(alldata,bfit2)
plt.hist(alldata, bins=40, cumulative=True, normed=True)
plt.xlabel('Gross Weight (lbs)')
plt.xlim([500,50000])
plt.ylabel('Cum. Frequency (Normalized)')

### Pinst
alldata = []       
for dat in combData_Pin[:]:
    if dat[2][0] == dat[2][1]: r = 0.1*dat[2][0]
    else:                   r = dat[2][1] - dat[2][0]
    #print dat[2][0]-r, dat[2][1]+r, r
    #if dat[0] == dat[2][1]: xs = np.arange(dat[2][0]-r, dat[2][1]+r, r/5.0)
    xs = np.arange(dat[2][0]-r, dat[2][1]+r, r/20.0)
    mean = sum(dat[2])/2.0
    std = r/3.0
    h = list(np.random.normal(mean, std, size=300))
    alldata = alldata + h
    
alldata = sorted(alldata)
nfit1 = stats.norm.pdf(alldata, np.average(alldata), np.std(alldata))
nfit2 = stats.norm.cdf(alldata, np.average(alldata), np.std(alldata))
a, b, loc, scale = stats.beta.fit(alldata)
bfit1 = stats.beta.pdf(alldata, a,b, loc=loc, scale=scale)
bfit2 = stats.beta.cdf(alldata, a,b, loc=loc, scale=scale)

#plt.figure()
plt.subplot(3,2,3)
#plt.plot(alldata,nfit1,'-b')
#plt.plot(alldata,bfit1)
plt.hist(alldata, bins=30, normed=True)
plt.xlabel('Power Installed (shp)')
plt.xlim([1000,15000])
plt.ylabel('Frequency (Normalized)')

#plt.figure()
plt.subplot(3,2,4)
#plt.plot(alldata,nfit2)
#plt.plot(alldata,bfit2)
plt.hist(alldata, bins=40, cumulative=True, normed=True)
plt.xlabel('Power Installed (shp)')
plt.xlim([1000,15000])
plt.ylabel('Cum. Frequency (Normalized)')

### VH
alldata = []       
for dat in combData_VH[:]:
    if dat[2][0] == dat[2][1]: r = 0.1*dat[2][0]
    else:                   r = dat[2][1] - dat[2][0]
    #print dat[2][0]-r, dat[2][1]+r, r
    #if dat[0] == dat[2][1]: xs = np.arange(dat[2][0]-r, dat[2][1]+r, r/5.0)
    xs = np.arange(dat[2][0]-r, dat[2][1]+r, r/20.0)
    mean = sum(dat[2])/2.0
    std = abs(r/3.0)
    h = list(np.random.normal(mean, std, size=300))
    alldata = alldata + h
    
alldata = sorted(alldata)
nfit1 = stats.norm.pdf(alldata, np.average(alldata), np.std(alldata))
nfit2 = stats.norm.cdf(alldata, np.average(alldata), np.std(alldata))
a, b, loc, scale = stats.beta.fit(alldata)
bfit1 = stats.beta.pdf(alldata, a,b, loc=loc, scale=scale)
bfit2 = stats.beta.cdf(alldata, a,b, loc=loc, scale=scale)

#plt.figure()
plt.subplot(3,2,5)
#plt.plot(alldata,nfit1,'-b')
#plt.plot(alldata,bfit1)
plt.hist(alldata, bins=40, normed=True)
plt.xlabel('Maximum Airspeed (kts)')
plt.xlim([300,500])
plt.ylabel('Frequency (Normalized)')

#plt.figure()
plt.subplot(3,2,6)
#plt.plot(alldata,nfit2)
#plt.plot(alldata,bfit2)
plt.hist(alldata, bins=50, cumulative=True, normed=True)
plt.xlabel('Maximum Airspeed (kts)')
plt.xlim([300,500])
plt.ylabel('Cum. Frequency (Normalized)')
"""

### COMBINED
"""
fig = plt.figure(figsize=(10,10))
ax1 = fig.add_subplot(111, aspect='equal')
for i in range(len(combData_GWT)):
    print i, ":",
    print combData_GWT[i][2][1]-combData_GWT[i][2][0], 
    print (combData_Pin[i][2][1]-combData_Pin[i][2][0])
    ax1.add_patch(
        patches.Rectangle(
            (combData_GWT[i][2][0], combData_Pin[i][2][0]),         # (x,y)
            (combData_GWT[i][2][1]-combData_GWT[i][2][0]),          # width
            (combData_Pin[i][2][1]-combData_Pin[i][2][0]),          # height
             alpha=0.03, lw=0.0
        )
    )
for i in range(len(combData_GWT)):
    print i, ":",
    print combData_GWT[i][2][1]-combData_GWT[i][2][0], 
    print (combData_Pin[i][2][1]-combData_Pin[i][2][0])
    ax1.add_patch(
        patches.Rectangle(
            (combData_GWT[i][2][0], combData_Pin[i][2][0]),         # (x,y)
            (combData_GWT[i][2][1]-combData_GWT[i][2][0]),          # width
            (combData_Pin[i][2][1]-combData_Pin[i][2][0]),          # height
             fill=False, lw=0.2
        )
    )

ax1.set_xlim([5000,50000])
ax1.set_ylim([1000, 15000])
"""


plt.show()
    