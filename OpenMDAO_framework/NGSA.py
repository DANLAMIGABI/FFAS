# -*- coding: utf-8 -*-
"""
Created on Sun Apr  1 10:23:51 2012

@author: - Frank Patterson
"""
import struct
import sys
import math
import random
import copy
import itertools
import time

import numpy as np

#from timer import Timer

import matplotlib.pylab as plt
import matplotlib.ticker as ticker
from mpl_toolkits.mplot3d.art3d import Line3DCollection
#from timer import Timer


def gray2bin(gray):
    """Convert gray string to binary string.
    E.g. gray2bin('1111') ==> '1010' and gray2bin('') ==> '0."""

    binary = gray[0]
    i=0
    while( len(gray) > i + 1 ):
        binary += `int(binary[i]) ^ int(gray[i+1])`
        i += 1
    return binary

def hamdist(str1, str2):
    """
    Count the # of differences between equal length strings str1 and str2
    """    
    diffs = 0
    for ch1, ch2 in zip(str1, str2):
            if ch1 != ch2:
                    diffs += 1
    return diffs
##
class GenAlg():
    """
    Generic Genetic Algorithm Class
    
    ----- INPUTS -----
    x : iterable
        initial list of variables
    bounds : list
        (min, max) for each variable in x
    types : list (optional)
        if types <> None, a list of variables types. supports: int, float
    crossover : float [0,1] (0.6) 
        crossover rate
    mutation : float [0,1] (0.03)
        mutation rate
    popMult : float (0,] (3.0)
        multiplier for population size. n = popMult*len(x)
    bestRepeat=10 : 
        number of generations without improvement for convergence
    bitsPerGene=12 : 
        number of bits in each gene. gives 2^bits possible answers for each variable
    direction : string
        direction to optimize ('min' or 'max')
    """
    
    def __init__(self, fitness, x, bounds, types=None, crossover=0.6, mutation=0.03, 
                 popMult=3, bitsPerGene=12, direction='max', crossN=2, 
                 maxGens=None, tourneySize=2, livePlotGens=3, hammingDist=True, removeDups=True):
        """
        Init algorithm
        """
        self.fitness = fitness
        
        if types <> None:
            if len(types) <> len(x):
                raise StandardError('Length of variable types must match length of phenotype!')
            else: self.types = types
        
        #============ALGORITHM GLOBALS=============
        self.crossover_rate = crossover             #crossover rate
        self.mutation_rate = mutation               #mutation rate
        self.population_size = int(popMult*len(x))  #size of population
        self.direction = direction
        self.crossN = crossN
        self.maxGens = maxGens  
        self.tourneySize = tourneySize              #number of individuals in each tourny selection
        self.livePlotGens = livePlotGens
        self.hammingDist = hammingDist
        self.removeDups = removeDups
        self.prevGens = [] #store previous generations

        #============GENOME DEFINITION=============
        #define the genome by number of genes, the bits for each, and the min and max value  
        #discretize the genes throughout the range(min,max)
        self.genes = len(x)                         #number of genes in the chromosome
        self.bitsPerGene = [bitsPerGene]*len(x)     #bits in each gene as list [gene1, gene2, ... geneN]
        self.gene_min = [b[0] for b in bounds]      #minimum value of a gene as list [gene1, gene2, ... geneN]
        self.gene_max = [b[1] for b in bounds]      #maximum value of a gene as list [gene1, gene2, ... geneN]
        
        #setup
        self.population = [ [None,None,{}] for i in range (self.population_size)] #defines population 
        self.child_pop  = []

        self.genCount = 0 #count generation
            #[genome, actual, char_dict]
            #char_dict = {... rank: n, distance: x, }
            
    
    def randGenomes(self):
        """
        Creates random population of genomes to start algorithm  
        """
        for i in range(self.population_size):        #for each population member
            
            #create random genes for encoding
            genome = ''
            for j in range(self.genes):
                
                gene = ''
                #create random bits for each gene
                for k in range(self.bitsPerGene[j]):
                    
                    if random.random() > 0.5: gene = gene + '0'
                    else: gene = gene + '1'
                    
                genome = genome + gene
            
            #add to population
            actual = self.decodeGenome(genome)
            self.population[i][0] = genome
            self.population[i][1] = actual        
    
    def decodeGenome(self, genome):
        """ Take in genome and decodes decodedVals from it, then creates list with fixed values
        """
        decodedVals = []    
        l = 0
        
        for i in range(self.genes):
            
            #grab the right genes from the genome based on the pack/unpack strings
            gene = genome[l:l + self.bitsPerGene[i]]
            gene = gray2bin(gene) #get binary version of gray code

            l = l + self.bitsPerGene[i] #update counter

            #convert the gene to an integer
            x1 = int(gene,2)
            #print x1/(2.0**bitsPerGene[i]-1.0)
            #find the value of that integer
            x2 = self.gene_min[i] + (self.gene_max[i]-self.gene_min[i])*(x1/(2.0**self.bitsPerGene[i]))
            #print x2
    
            decodedVals.append(x2)
                
        return decodedVals 

    def domination(self, p, q):
        """ 
        Check if p dominates q. P and Q are lists of objective values.
        """
        #if self.direction == "min":
        #if each element of p is less than (more desirable) it's 
        #counterpart in q, then p dominates q, else it does not
        #print p, q, all([p[i] <= q[i] for i in range(len(p))]) 
        return all([p[i] <= q[i] for i in range(len(p))]) 
        #if self.direction == "max":
            #flip the direction for max search
        #    return all([p[i] >= q[i] for i in range(len(p))])
        
    #sorts a population
    def sortPopulation(self, populationX):
        """
        Sort the population by dominance (fast non-dominated sort, see Deb,Pratab-2002)
        """
        #print "sorting population 1 here..."
        F = [[]] #first front
        for p in populationX: p[2]['rank'] = None #initalize all ranks
        
        for p in populationX:
            eta_p = 0
            S_p = []
            for q in populationX:
                if self.domination(p[2]['objs'], q[2]['objs']): #if p dominates q
                    S_p.append(q)                          #add q to the set dominated by p
                elif self.domination(q[2]['objs'], p[2]['objs']): #elif q dominates p: 
                    eta_p = eta_p + 1

            p[2]['Sp'] = S_p #capture S_p
            p[2]['etap'] = eta_p #capture eta_p
            #print populationX.index(p), ":", len(S_p), eta_p

            if p[2]['etap'] == 0: #check if p is on the first front
                p[2]['rank'] = 0
                F[0].append(p)
        #print "sorting population 2 here..."
        i = 0
        while i < len(F):
        #while len([p for p in populationX if p[2]['rank'] == None]) > 0:
            Q = []          #next front
            for p in F[i]:
                for q in p[2]['Sp']: 
                    q[2]['etap'] = q[2]['etap'] - 1
                    if q[2]['etap'] == 0:

                        #print "q:", len(q), type(q), type(q[0]), 
                        #print "Q:", len(Q), type(Q), 
                        #if len(Q)>0: print type(Q[0])
                        #else: print "empty"

                        #if not q[0] in [q_i[0] for q_i in Q]: 
                        q[2]['rank'] = i+1
                        Q.append(q) #add to next front

            if len(Q) > 0: F.append(Q)
            i = i+1

        #print "pXlen:", len(populationX)
        return populationX

    def crowdingDistanceAssignment(self, populationX):
        """
        Calculate crowding distances, averaging the distances of points on 
        either side of each points
        """
        nd_sets = list(set([p[2]['rank'] for p in populationX])) #list of ranks
        objectives = len(populationX[0][2]['objs']) #num of objectives

        for p in populationX: p[2]['distance'] = 0.0 #initialize distance

        for i in nd_sets:#range(nd_sets+1): #for each rank
            I = [p for p in populationX if p[2]['rank'] == i] #get all solutions on given frontier
            if len(I) > 0:
                for j in range(objectives): #for each objective
                    I.sort(key=lambda p: p[2]['objs'][j]) #sort by the objective
                    I[0][2]['distance'] = 10**10 #first set to "infinity"
                    I[-1][2]['distance'] = 10**10 #last set to "infinity"
                    for k in range(1,len(I)-1): #add distance measure for each member
                        if (I[-1][2]['objs'][j] - I[0][2]['objs'][j]) <> 0.0:
                            d = (I[k+1][2]['objs'][j] - I[k-1][2]['objs'][j]) / \
                                (I[-1][2]['objs'][j] - I[0][2]['objs'][j])
                        else: 
                            d = (I[k+1][2]['objs'][j] - I[k-1][2]['objs'][j])/ 10.0**-8
                        I[k][2]['distance'] = I[k][2]['distance'] + d
            
        return populationX    

    def hammingDistanceAssignment(self, populationX):
        """
        Calculate population hamming distances, averaging hamming distance from that 
        point all other in it's frontier (rank)
        """
        nd_sets = list(set([p[2]['rank'] for p in populationX])) #list of ranks
        for p in populationX: p[2]['hammdistance'] = 0.0 #initialize distance

        for i in nd_sets:#range(nd_sets+1): #for each rank
            I = [p for p in populationX if p[2]['rank'] == i] #get all solutions on given frontier
            if len(I) > 0:
                for j in range(len(I)): #for each member
                    hd = 0 #init hamming distance
                    for k in range(len(I)): #calculate average hamming distance from other members on that frontier
                        if j <> k:
                            hd = hd + float(hamdist(I[j][0], I[k][0]))/float(len(I))
                    I[j][2]['hammdistance'] = hd #assign hamming distance as distance

        return populationX    

    #create new generaton
    def generation(self): 
        """ Performs each geneation as follows:
            1) Sort child population based on non-domination
            2) Assign child fitness based on non-domination level
            3) Initial population is evolved with tournament selection, 
               On Second Generation: 
                3a) Parent and Child populations are combined
                3b) A fast-nondominated sort is performed on Entire (2N) population 
                    (ensures elitism)
                3c) To build new population, add all members from most dominate 
                    from (F_i) until len(F_i) < remaining space in N
                3d) Sort the last front (F_l) using the crowded comparison operator
                    in decending order and choose the best solutions to fill out N 
                    population members
            4) This new population becomes the parent population and selection,
               crossover  and mutation are performed to create a new child population

        """
                
        def crossover(parent1, parent2, eflag=False):
            """ Perform crossover
            """
            #perform crossover
            if random.random() <= self.crossover_rate or eflag:           #if we perform a crossover
                if self.crossN == 2: #two way crossover
                    pos = int(random.random()*len(parent1[0]))
                    p1a = parent1[0][:pos]
                    p1b = parent1[0][pos:]
                    p2a = parent2[0][:pos]
                    p2b = parent2[0][pos:]
                
                    child1_g = p1a + p2b  #create the new children's genomes
                    child2_g = p2a + p1b    
                
                elif self.crossN == 3: #three way crossover
                    pos1 = int(random.random()*len(parent1[0]))
                    pos2 = int(random.random()*len(parent1[0]))
                    [pos1, pos2] = sorted([pos1, pos2])
                    
                    p1a = parent1[0][:pos1]
                    p1b = parent1[0][pos1:pos2]
                    p1c = parent1[0][pos2:]
                    
                    p2a = parent2[0][:pos1]
                    p2b = parent2[0][pos1:pos2]
                    p2c = parent2[0][pos2:]
                                    
                    #create the new children's genomes
                    child1_g = p1a + p2b + p1c
                    child2_g = p2a + p1b + p2c
                    
            #if not crossing over, the children's genomes are clones of the parents
            else:
                child1_g = parent1[0]
                child2_g = parent2[0]        
            
            child1 = [child1_g, None, {}]
            child2 = [child2_g, None, {}]
    
            return child1, child2        
                        
        def mutation(genome):  
            """ Perform mutation sequence on a genome.
            """  
            Lgenome = list(genome)                      #break string into list
            for i in range(len(Lgenome)):               #parse list and mutate as probable
                if random.uniform(0.0, 1.0) <= self.mutation_rate:
                    if Lgenome[i] == '1': Lgenome[i] = '0'
                    if Lgenome[i] == '0': Lgenome[i] = '1'
    
            g = "".join(Lgenome)                        #join list genome back into string
            return g
    
        def tourneySelect():
            """
            Tournament selection: from random pool of self.tournySize, returns 
            participant with largest distance from among those with best rank (lowest)
            """
            tourny_p = random.sample(self.population, self.tourneySize)
            winners = [p for p in tourny_p if p[2]['rank'] == min(tourny_p, key=lambda q:q[2]['rank'])[2]['rank']] #get all participants with lowest rank
            if not self.hammingDist:  return max(winners, key=lambda q:q[2]['distance']) #return winner with largest distance
            else:                     return max(winners, key=lambda q:q[2]['hammdistance']) #return winner with largest hamming distance

        #first generation rank/sort
        if len(self.child_pop) < self.population_size:
            print "Rank Sorting First Population..."
            self.population = self.sortPopulation(self.population) #assign ranks
            self.population.sort(key=lambda p:p[2]['rank'])

            print "Assigning Crowding Distance to First Population..."
            self.population = self.crowdingDistanceAssignment(self.population) #get crowding distances
            self.population = self.hammingDistanceAssignment(self.population)

        #Loop to fill children on first generation only
        while len(self.child_pop) < self.population_size: #populate children
            parent1 = tourneySelect()
            parent2 = tourneySelect()
            
            child1, child2 = crossover(parent1,parent2) #crossover
            child1[0] = mutation(child1[0])    #mutation
            child2[0] = mutation(child2[0])    #mutation

            child1[1] = self.decodeGenome(child1[0]) #decode genomes
            child2[1] = self.decodeGenome(child2[0]) #decode genomes


            child1[2]['objs'] = self.fitness(child1[1]) #get fitness objectives
            child2[2]['objs'] = self.fitness(child2[1]) #get fitness objectives
            if self.direction == "max":
                child1[2]['objs'] = [-1.0*x for x in child1[2]['objs']]
                child2[2]['objs'] = [-1.0*x for x in child2[2]['objs']]

            if len(self.child_pop) < self.population_size: self.child_pop.append(child1) #update new generation
            if len(self.child_pop) < self.population_size: self.child_pop.append(child2)

        ##---## START GENERATION ##---##
        print "Combining Population..."
        combined_pop = self.population + self.child_pop  #combined populations
        for p in combined_pop: p[2]['rank'] = None
        combined_pop = self.sortPopulation(combined_pop) #sort population to get ranks

        #create a new population from the best of the combined population
        self.population = [] 
        i = 0
        print "Selecting New Population..."
        while len(self.population) < self.population_size:
            F_i = [p for p in combined_pop if p[2]['rank'] == i] #get frontier in comb population with best remaining rank
            #print "Combined Frontier %d length: %d" % (i,len(F_i))
            if len(F_i) <= (self.population_size - len(self.population)): #if enough room
                if self.removeDups: #don't add duplicates
                    for candidate in F_i: 
                        if not candidate[0] in [p[0] for p in self.population]: self.population.append(candidate)
                else:
                    self.population = self.population + F_i #add whole frontier
            else:
                self.crowdingDistanceAssignment(F_i) #get crowding distance
                self.hammingDistanceAssignment(F_i)

                while len(self.population) < self.population_size and len(F_i) > 0:  #add p with largest distances until full
                    if not self.hammingDist: F_i_max = F_i.pop(F_i.index(max(F_i, key=lambda p:p[2]['distance'])))
                    else:                    F_i_max = F_i.pop(F_i.index(max(F_i, key=lambda p:p[2]['hammdistance'])))
                    if self.removeDups:
                        if F_i_max[0] not in [p[0] for p in self.population]: self.population.append(F_i_max)
                    else:                                                     self.population.append(F_i_max)
            i = i + 1

        #self.population = self.sortPopulation(self.population) #sort new population
        self.population = self.crowdingDistanceAssignment(self.population) #get crowding/hamming distances for remaining members of population
        self.population = self.hammingDistanceAssignment(self.population)

        if False: #print population (for TESTING)
            i = 0
            print ""
            print "PRINTING COMBINED POPULATION..."
            for p in combined_pop:
                print i, p[0], p[1], ": f =", p[2]['objs'], "  rank =", p[2]['rank'] 
                i = i + 1

        #use selection/crossover/mutation to create new child population
        self.child_pop = []
        #Loop to fill children
        print "Generating New Children..."
        while len(self.child_pop) < self.population_size: #populate children
            parent1 = tourneySelect()
            parent2 = tourneySelect()
            
            child1, child2 = crossover(parent1,parent2) #crossover
            child1[0] = mutation(child1[0])    #mutation
            child2[0] = mutation(child2[0])    #mutation

            child1[1] = self.decodeGenome(child1[0]) #decode genomes
            child2[1] = self.decodeGenome(child2[0]) #decode genomes
           
            if len(self.child_pop) < self.population_size: self.child_pop.append(child1) #update new generation
            if len(self.child_pop) < self.population_size: self.child_pop.append(child2)


    def run(self):
        """
        Run the algorithm
        """
        #create random population
        self.randGenomes()
        gen = 0 #generation number
        
        #place holders for best member and time he was repeated
        best_member = None
        best_fitness = 0.0
        best_objectives = None
        best_count = 0
        
        genList = []
        best = []
        avg = []

        hDistP1 = []
        cDistP1 = []
        pctPopP1 = []
        


        print "----------------------------------------------------------"
        print "Starting Optimization:"
        if self.direction == "max": print "Maximizing", 
        else: print "Minimizing",
        print "population: %d, Chromosome Size: %d," % (len(self.population), len(self.population[0][0])) 
        print "%d parameters, %d bits per parameter" % (self.genes, self.bitsPerGene[0])
        for i in range(self.genes): print "parameter x%d:  min=%.3f, max=%.3f" % (i, self.gene_min[i], self.gene_max[i])

        if self.livePlotGens > 0: 
            sampleobjs = self.fitness(self.population[0][1])
            obj_combs = [x for x in itertools.combinations(range(len(sampleobjs)), 2)]
            fig, axes = plt.subplots(nrows=int(math.ceil(len(obj_combs)**0.5)), ncols=int(len(obj_combs)**0.5), figsize=(16,9))
            if not isinstance(axes, np.ndarray): axes = np.array([axes])
            plt.ion()
            fig.tight_layout()
            #mng = plt.get_current_fig_manager() #get current fig
            #mng.frame.Maximize(True) #maximize it

            fig_in, axes_in = plt.subplots(1, self.genes-1, sharey=False) #add figure/axes for input parallel

            plt.show()

        #repeat until ?
        while gen < self.maxGens:
          
            gen = gen + 1
            self.genCount = gen

            print '========== Generation:', gen, '=========='
            print "Evaluating Population of %d :" % self.population_size

            # get fitness for each population member 
            for p in self.population: #check parent population (first generation)
                if not 'objs' in p[2]:
                    
                    #with Timer() as t:
                    p[2]['objs'] = self.fitness(p[1], gen=0) #get objectives
                    #print "RECEIVED:", p[2]['objs'] 
                    #print "==> member fitness calculated in %.1f s" % t.secs
                    #p[2]['actuals'] = copy.copy(p[2]['objs']) #copy in case of manipulation

                    if self.direction == "max":
                        p[2]['objs'] = [-1.0*x for x in p[2]['objs']]

            for p in self.child_pop: #check child population
                 if not 'objs' in p[2]:

                    #with Timer() as t:
                    p[2]['objs'] = self.fitness(p[1], gen=0) #get objectives
                    #print "==> child fitness calculated in %.1f s" % t.secs
                    #p[2]['actuals'] = copy.copy(p[2]['objs']) #copy in case of manipulation

                    if self.direction == "max":  #flip objectives for max
                        p[2]['objs'] = [-1.0*x for x in p[2]['objs']]
            print "" #move print to new line
            
            self.generation() #create new generation

            #### GET FITNESSES FOR RESULTING POPULATION FOR RECORDING 
            print "Recording Fitnesses..."
            #*** SLOWS SHIT WAY DOWN! ****
            for p in self.population: fitness_toss = self.fitness(p[1], gen=gen)
            #***                      ****

            hDistP1.append(np.average([p[2]['hammdistance'] for p in self.population if p[2]['rank'] == 0]))
            cDistP1.append(np.average([p[2]['distance'] for p in self.population if p[2]['rank'] == 0]))
            pctPopP1.append(float(len([p for p in self.population if p[2]['rank'] == 0]))/len(self.population))

            #####
            print "Generation Complete", 
            print len(self.population), "parents,", len(self.child_pop), "children."
            print "Average Crowding Distance (parents): %.4e" % np.average([p[2]['distance'] for p in self.population])
            print "Average Hamming Distance (parents): %.4e" % np.average([p[2]['hammdistance'] for p in self.population])
            print "Frontiers: 1: %d, 2: %d, 3: %d" % (len([p for p in self.population if p[2]['rank'] == 0]),
                                                      len([p for p in self.population if p[2]['rank'] == 1]),
                                                      len([p for p in self.population if p[2]['rank'] == 2]))
            print "Average Crowding Distance on Frontiers: 1: %.3e, 2: %.3e, 3: %.3e" % (np.average([p[2]['distance'] for p in self.population if p[2]['rank'] == 0]),
                                                                                         np.average([p[2]['distance'] for p in self.population if p[2]['rank'] == 1]),
                                                                                         np.average([p[2]['distance'] for p in self.population if p[2]['rank'] == 2]),)
            print "Average Hamming Distance on Frontiers: 1: %.3f, 2: %.3f, 3: %.3f" % (np.average([p[2]['hammdistance'] for p in self.population if p[2]['rank'] == 0]),
                                                                                        np.average([p[2]['hammdistance'] for p in self.population if p[2]['rank'] == 1]),
                                                                                        np.average([p[2]['hammdistance'] for p in self.population if p[2]['rank'] == 2]),)
            print "Objectives:"
            pf = [p for p in self.population if p[2]['rank'] == 0]
            for i in range(len(self.population[0][2]['objs'])): 
                if self.direction == "min": 
                    stats = (i, min(self.population, key=lambda p:p[2]['objs'][i])[2]['objs'][i],
                             max(self.population, key=lambda p:p[2]['objs'][i])[2]['objs'][i],
                             min(pf, key=lambda p:p[2]['objs'][i])[2]['objs'][i],
                             max(pf, key=lambda p:p[2]['objs'][i])[2]['objs'][i])
                else: #flip it around if maximizing
                    stats = (i, -1.0*min(self.population, key=lambda p:p[2]['objs'][i])[2]['objs'][i],
                             -1.0*max(self.population, key=lambda p:p[2]['objs'][i])[2]['objs'][i],
                             -1.0*min(pf, key=lambda p:p[2]['objs'][i])[2]['objs'][i],
                             -1.0*max(pf, key=lambda p:p[2]['objs'][i])[2]['objs'][i])
                print "f%d:   min=%.2f, max=%.2f, min(pareto)=%.2f, max(pareto)=%.2f" % stats

            if self.livePlotGens > 0:
                self.prevGens = [self.population[:]] + self.prevGens #add to previous generations
                while len(self.prevGens) > self.livePlotGens: self.prevGens.pop(-1) #remove old generations

            #####

            ##### LIVE PLOT
            if self.livePlotGens > 0:
                for a in axes: #clear axes
                    if isinstance(a, np.ndarray):
                        for b in a: b.cla()
                    else: a.cla()
                for a in axes_in: #clear axes
                    if isinstance(a, np.ndarray):
                        for b in a: b.cla()
                    else: a.cla()

            """
            #get pareto min/max for better plotting 
            mins = [10.0^10 for i in range(self.genes)]
            maxs = [-10.0^10 for i in range(self.genes)]
            for i in range(len(self.prevGens)):
                for j in range(len(self.genes)):
                    pf = [p for p in self.prevGens[i] if p['rank'] == 0]
                    miij = max(pf, key=lambda x:x[2]['objs'][j])[2]['objs'][j]
                    maij = max(pf, key=lambda x:x[2]['objs'][j])[2]['objs'][j]
                    if miij < mins[j]: mins[j] = miij
                    if maij < maxs[j]: maxs[j] = maij
            """

            if self.livePlotGens > 0:
                alphas = [1 - x/float(self.livePlotGens) for x in range(self.livePlotGens)] #alpha levels
                colors = [plt.cm.Blues(x) for x in alphas]
                for i in reversed(range(len(self.prevGens))): #for each generation saved (< full at first)
                    k,l = 0,0
                    for j in range(len(obj_combs)): #for each combinations
                        xs = [m[2]['objs'][obj_combs[j][0]] for m in self.prevGens[i]]
                        ys = [m[2]['objs'][obj_combs[j][1]] for m in self.prevGens[i]]
                        if self.direction == "max":
                            xs = [-1.0*x1 for x1 in xs]
                            ys = [-1.0*y1 for y1 in ys]


                        if len(axes.shape) > 1: ax = axes[k,l]
                        else:                   ax = axes[k]
                        k = k + 1
                        if k == len(axes):
                            k = 0
                            l = l+1

                        ax.scatter(xs, ys, color=colors[i], alpha=alphas[i])
                        ax.set_xlabel('Objective %d' % obj_combs[j][0], fontsize=10)
                        ax.set_ylabel('Objective %d' % obj_combs[j][1], fontsize=10)

                        ax.grid(True)                        
                ax.legend(["Gen n-%d" % x for x in reversed(range(self.livePlotGens))], loc=4)
                
                input_data = []
                par_colors = []
                lws = []

                for i in reversed(range(len(self.prevGens))):
                    for p in self.prevGens[i]:
                        input_data.append([int(x) for x in p[1]])
                        lws.append(2.0/(i+1))
                        par_colors.append(colors[i])
                inplot = parallel_coordinates(fig_in, axes_in, input_data, colors=par_colors, lws=lws)

                plt.figure(fig.number)
                plt.draw()
                plt.figure(fig_in.number)
                plt.draw()
                time.sleep(0.25)


        #OPTIMIZATION COMPLETE!        
        print "----------------------------------------------------------"
        print "Optimization Complete in %d generations." % gen
        print "Population: %d, Chromosome Size: %d," % (len(self.population), len(self.population[0][0])) 
        print "Final: %d frontiers" % (max(self.population, key=lambda p:p[2]['rank'])[2]['rank'] + 1)
        print ""    
        
        pf = [p for p in self.population if p[2]['rank'] == 0]
        print "Pareto Frontier: (%d points)" % len(pf)
         
        for i in range(len(pf[0][1])): 
            print "x%d".ljust(9) % i,
            for p in pf: 
                print str(round(p[1][i], 3)).ljust(10),
            print ""
        print ""
        for i in range(len(pf[0][2]['objs'])):
            print "f%d".ljust(9) % i,
            for p in pf: 
                if self.direction == "min":
                    print str(round(p[2]['objs'][i], 3)).ljust(10),
                else: 
                    print str(round(-1.0*p[2]['objs'][i], 3)).ljust(10),
            print "" 
        print "----------------------------------------------------------"

        if False: 
            plt.figure()
            plt.subplot(3,1,1)
            plt.plot(range(1,len(hDistP1)+1), hDistP1)
            plt.title('Average Hamming Distance on Pareto Fronteir')
            plt.ylabel('H-Dist') 
            plt.xlabel('Generation')
            plt.subplot(3,1,2)
            plt.plot(range(1,len(cDistP1)+1), cDistP1)
            plt.title('Average Crowding Distance on Pareto Fronteir')
            plt.ylabel('C-Dist') 
            plt.xlabel('Generation')
            plt.subplot(3,1,3)
            plt.plot(range(1,len(pctPopP1)+1), pctPopP1)
            plt.title('Pareto Frontier as Percentage of Population')
            plt.ylabel('%')
            plt.xlabel('Generation')
            plt.draw()
            plt.show()

        if False: #plot results
            plt.figure()
            i = 0
            for p in self.population:
                plt.scatter(p[2]['objs'][0],p[2]['objs'][1])
                plt.text(p[2]['objs'][0],p[2]['objs'][1], str(i))
                i = i+1
            for r in range(max([p[2]['rank'] for p in self.population])+1):
                I = [p for p in self.population if p[2]['rank'] == r]
                I.sort(key=lambda p:p[2]['objs'][0])
                plt.plot([p[2]['objs'][0] for p in I], [p[2]['objs'][1] for p in I], '--')
            plt.show()

        return [[p[1], p[2]['objs']] for p in self.population] #return [[[parameters], [objectives]], ... [] ] for final population
        


def parallel_coordinates(fig, axes, data_sets, colors=None, lws=None, style=None):
    """
    #Function to plot parallel_coordinates
    http://stackoverflow.com/questions/8230638/parallel-coordinates-plot-in-matplotlib
    """
    dims = len(data_sets[0])
    x    = range(dims)

    if style is None:
        style = ['r-']*len(data_sets)
    if colors is None:
        colors = ['r']*len(data_sets)
    if lws is None:
        lws = [1.0]*len(data_sets)
    # Calculate the limits on the data
    min_max_range = list()
    for m in zip(*data_sets):
        mn = min(m)
        mx = max(m)
        if mn == mx:
            mn -= 0.5
            mx = mn + 1.
        r  = float(mx - mn)
        min_max_range.append((mn, mx, r))

    # Normalize the data sets
    norm_data_sets = list()
    for ds in data_sets:
        nds = [(value - min_max_range[dimension][0]) / 
                min_max_range[dimension][2] 
                for dimension,value in enumerate(ds)]
        #nds = [value for dimension,value in enumerate(ds)]
        norm_data_sets.append(nds)
    data_sets = norm_data_sets

    # Plot the datasets on all the subplots
    for i, ax in enumerate(axes):
        for dsi, d in enumerate(data_sets):
            ax.plot(x, d, style[dsi], c=colors[dsi], lw=lws[dsi])
        ax.set_xlim([x[i], x[i+1]])

    # Set the x axis ticks 
    for dimension, (axx,xx) in enumerate(zip(axes, x[:-1])):
        axx.xaxis.set_major_locator(ticker.FixedLocator([xx]))
        ticks = len(axx.get_yticklabels())
        labels = list()
        step = min_max_range[dimension][2] / (ticks - 1)
        mn   = min_max_range[dimension][0]
        for i in xrange(ticks):
            v = mn + i*step
            labels.append('%.d' % v)
        axx.set_yticklabels(labels)


    # Move the final axis' ticks to the right-hand side
    axx = plt.twinx(axes[-1])
    dimension += 1
    axx.xaxis.set_major_locator(ticker.FixedLocator([x[-2], x[-1]]))
    ticks = len(axx.get_yticklabels())
    step = min_max_range[dimension][2] / (ticks - 1)
    mn   = min_max_range[dimension][0]
    labels = ['%4.2f' % (mn + i*step) for i in xrange(ticks)]
    axx.set_yticklabels(labels)

    # Stack the subplots 
    plt.subplots_adjust(wspace=0)

    return plt            
        
#===================Main Sequence===================
if __name__ == '__main__':  
    
    """
    def fitcalc(x):
        #print x
        #f =  sum([1.0 for y in x if all( [(y<2.0), (y>1.0)] ) ] )
        f = sum(x)**2 / max(1.0,len([x1 for x1 in x if x1 > 3.0]))
        
        return f
        
        
    x = [1]*15
    xbounds = [(-3.0, 5.0) for y in x]
    
    GA = GenAlg(fitcalc, x, xbounds, popMult=10, bitsPerGene=12, bestRepeat=20)
    results = GA.run()
    print "*** DONE ***"
    print results
    """
    import math

    def fitcalc1(x, gen=0):
        
        #s1 = sum([(x_i - (1./3.)**(0.5))**2 for x_i in x])
        #s2 = sum([(x_i + (1./3.)**(0.5))**2 for x_i in x])
        #f1 = 1 - math.exp(-s1)
        #f2 = 1 - math.exp(-s2)
        f1 = sum([-10.0*math.exp(-0.2*(x[i]**2.0 + x[i+1]**2)**0.5) for i in range(len(x)-1)])
        f2 = sum([(abs(x_i)**0.8 + 5*math.sin(x_i**3.0) ) for x_i in x])

        return [f1, f2]

    def fitcalc2(x, gen=0):
        s1 = sum([(x_i - (1./3.)**(0.5))**2 for x_i in x])
        s2 = sum([(x_i + (1./3.)**(0.5))**2 for x_i in x])
        s3 = sum([(x_i + (1./3.)**(0.5))**2 for x_i in x])
        f1 = 1 - math.exp(-s1)
        f2 = 1 - math.exp(-s2)
        f3 = abs(s1-s2)
        f4 = 1 - math.atan(s2/(1.0+s1))

        return [f1, f2, f3, f4]

    def test1():
        x = [0.5]*3
        xbounds = [(-5, 5) for y in x]


        GA = GenAlg(fitcalc1, x, xbounds, popMult=100, bitsPerGene=9, mutation=(1./9.), crossover=0.65, crossN=2, direction='min', maxGens=60, hammingDist=False)
        results = GA.run()
        print "*** DONE ***"
        #print results
        plt.ioff()
        #generate pareto frontier numerically
        x1_ = np.arange(-5., 0., 0.05)
        x2_ = np.arange(-5., 0., 0.05)
        x3_ = np.arange(-5., 0., 0.05)

        pfn = []
        for x1 in x1_:
            for x2 in x2_:
                for x3 in x3_:
                    pfn.append(fitcalc1([x1,x2,x3]))

        pfn.sort(key=lambda x:x[0])
        
        plt.figure()
        i = 0
        for x in results:
            plt.scatter(x[1][0], x[1][1], 20, c='r')

        plt.scatter([x[0] for x in pfn], [x[1] for x in pfn], 1.0, c='b', alpha=0.1)
        plt.xlim([-20,-1])
        plt.ylim([-12, 2])
        plt.draw()




    def test2():
        x = [0.5]*3
        xbounds = [(-5, 5) for y in x]


        GA = GenAlg(fitcalc2, x, xbounds, popMult=100, bitsPerGene=9, mutation=(1./9.), crossover=0.65, crossN=2, direction='min', maxGens=60)
        results = GA.run()
        print "*** DONE ***"
        #print results

        #generate pareto frontier numerically
        x1_ = np.arange(-5., 0., 0.05)
        x2_ = np.arange(-5., 0., 0.05)
        x3_ = np.arange(-5., 0., 0.05)

        pfn = []
        for x1 in x1_:
            for x2 in x2_:
                for x3 in x3_:
                    pfn.append(fitcalc1([x1,x2,x3]))

        #pfn.sort(key=lambda x:x[0])
        #plt.figure()
        #i = 0
        #for x in results:
        #    plt.scatter(x[1][0], x[1][1], 20, c='r')

        #plt.ioff
        #plt.scatter([x[0] for x in pfn], [x[1] for x in pfn], 1.0, c='b', alpha=0.1)
        #plt.xlim([-20,-1])
        #plt.ylim([-12, 2])
        #plt.show()
        #plt.draw()


    test1()
    #test2()

    