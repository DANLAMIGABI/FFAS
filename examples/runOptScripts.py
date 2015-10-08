#runs all optimization scripts

import runOptFramework
import runOptFramework_POS
import time
print "==========================================================================================="
print "OPTIMIZING FPoS OUTPUTS w/ HAMMING DISTANCE"
t = time.strftime("%H:%M:%S")
print "starting optimization at:", t
runOptFramework_POS.execute(True)

print "==========================================================================================="
print "OPTIMIZING FPoS OUTPUTS w/ CROWDING DISTANCE"
t = time.strftime("%H:%M:%S")
print "starting optimization at:", t
runOptFramework_POS.execute(False)

print "==========================================================================================="
print "OPTIMIZING BASELINE OUTPUTS w/ HAMMING DISTANCE"
t = time.strftime("%H:%M:%S")
print "starting optimization at:", t
runOptFramework.execute(True)

print "==========================================================================================="
print "OPTIMIZING BASELINE OUTPUTS w/ CROWDING DISTANCE"
t = time.strftime("%H:%M:%S")
print "starting optimization at:", t
runOptFramework.execute(False)





