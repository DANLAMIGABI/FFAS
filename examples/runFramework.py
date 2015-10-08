
import sys
import os
import time
import numpy as np
import skfuzzy as fuzz

from OpenMDAO_framework.fuzzyAssembly import FuzzyAssembly
from openmdao.lib.casehandlers.api import CaseDataset

import matplotlib.pyplot as plt
plt.ioff()

print "\n\n"

a = FuzzyAssembly()
a.configure()


tt = time.time()  


print "\n"
a.run()
#results = a.driver.case_outputs.postprocess.ranges_out
#print results
#print "\n"
#for i in range(len(results)): 
#    print "CASE:", i
#    print results[i]

#print a.responses

print "\n"
#print len(a.fuzz_combine.system_inputs)
print "Elapsed time: ", time.time()-tt, "seconds"


#----------------------------------------------------
# Print out history of our objective for inspection
#----------------------------------------------------
#case_dataset = CaseDataset('test.json', 'json')
#data = case_dataset.data.by_case().fetch()


#for case in data:
#    print case['postprocess.ranges_out']

if False:
	print ""
	i = 1
	for case in data: 
		print "ALTERNATIVE", i, " - ",
		print "PHI: %.1f, (%.1f)" % (case['postprocess.response_1'], case['postprocess.response_1_r']), 
		print "  FoM: %.3f, (%.3f)" % (case['postprocess.response_2'], case['postprocess.response_2_r']), 
		print "  L/D: %.1f, (%.1f)" % (case['postprocess.response_3'], case['postprocess.response_3_r']),
		print "  etaP: %.3f, (%.3f)" % (case['postprocess.response_4'], case['postprocess.response_4_r']),
		print "  GWT: %.0f, (%.0f)" % (case['postprocess.response_5'], case['postprocess.response_5_r'])

		i = i+1