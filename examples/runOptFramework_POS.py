#!/usr/local/bin/python

import sys
import os
import time
import numpy as np
import skfuzzy as fuzz
import csv

from OpenMDAO_framework.fuzzyOptAssembly_POS import FuzzyAssemblyOpt
from openmdao.lib.casehandlers.api import CaseDataset
from openmdao.main.api import set_as_top

def execute(hD):
	print "\n\n"
	print "Running", sys.version, "in", sys.executable

	print "----------------------------------------------------------------------"
	a = FuzzyAssemblyOpt()
	set_as_top(a)

	a.compatibility.count = 0 #flag for counting incompatibilities #if count = 1: return the total number of incompatibilities
	a.postprocess.printResults = 0 #print results flag

	a.driver.opt_type        = 'max'
	a.driver.generations     = 50
	a.driver.popMult         = 15.0
	a.driver.crossover_Rate  = 0.65
	a.driver.mutation_rate   = 0.07
	a.driver.crossN          = 3
	a.driver.bitsPerGene 	 = 4
	a.driver.tourneySize	 = 3
	a.driver.hammingDist 	 = hD
	a.driver.livePlotGens    = 0


	tt = time.time()  
	a.run()

	print "\n"
	#print len(a.fuzz_combine.system_inputs)
	print "---------- Optimization Complete ----------"
	print "Elapsed time: ", time.time()-tt, "seconds"
	print "----------------------------------------------------------------------"