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

from decimal import *

import matplotlib.pylab as plt

from timer import Timer


##
class GenAlg():
    """
    Generic Genetic Algorithm Class
    
    ----- INPUTS -----
    x : iterable
        initial list of variables
    bounds : list
        (min, max) for each variable in x
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
    
    def __init__(self, fitness, x, bounds, crossover=0.6, mutation=0.03, 
                 popMult=3, bestRepeat=10, bitsPerGene=12, elite_flag=True,
                 direction='max', crossN=2, useBase=True):
        """
        Init algorithm
        """
        self.fitness = fitness
        self.base = x   #record baseline
        
        
        #============ALGORITHM GLOBALS=============
        self.crossover_rate = crossover             #crossover rate
        self.mutation_rate = mutation               #mutation rate
        self.population_size = int(popMult*len(x))  #size of population
        self.bestRepeat = bestRepeat                #number of generations where best is unchanged to converge
        self.elite_flag = elite_flag
        self.direction = direction
        self.crossN = crossN
        self.useBase = useBase
        
        #============GENOME DEFINITION=============
        #define the genome by number of genes, the bits for each, and the min and max value  
        #discretize the genes throughout the range(min,max)
        self.genes = len(x)                         #number of genes in the chromosome
        self.bitsPerGene = [bitsPerGene]*len(x)     #bits in each gene as list [gene1, gene2, ... geneN]
        self.gene_min = [b[0] for b in bounds]      #minimum value of a gene as list [gene1, gene2, ... geneN]
        self.gene_max = [b[1] for b in bounds]      #maximum value of a gene as list [gene1, gene2, ... geneN]
        
        #setup
        self.population = [ [None,None,None] for i in range (self.population_size)] #defines population [genome, actual, Fitness]
            
    #creates random population of genomes to start algorithm    
    def randGenomes(self):
        
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
        
    
    
    # Take in genome and decodes decodedVals from it, then creates list with fixed values
    def decodeGenome(self, genome):
        #genome - binary string representing aircraft values to translate 
        decodedVals = []    
        l = 0
        
        for i in range(self.genes):
            
            #grab the right genes from the genome based on the pack/unpack strings
            gene = genome[l:l + self.bitsPerGene[i]]
            l = l + self.bitsPerGene[i]
    
            #convert the gene to an integer
            x1 = int(gene,2)
            #print x1/(2.0**bitsPerGene[i]-1.0)
            #find the value of that integer
            x2 = self.gene_min[i] + (self.gene_max[i]-self.gene_min[i])*(x1/(2.0**self.bitsPerGene[i]))
            #print x2
    
            decodedVals.append(x2)
                
        return decodedVals   
      
    #take a variable list and turn it into a bianary genome
    def encodeGenome(self, vals):
        genome = ''
        
        
        for i in range(len(vals)):
            inList = range(2**self.bitsPerGene[i])
            decList = [self.gene_min[i] + float(self.gene_max[i]-self.gene_min[i])*(x/(2.0**self.bitsPerGene[i])) 
                       for x in inList]
            (j, y) = min(enumerate(decList), key=lambda x: abs(x[1]-vals[i]))
            gene = bin(inList[j])[2:].zfill(self.bitsPerGene[i])
            genome = genome + gene
        
        return genome
        
    #create new generaton
    def generation(self):    
                
        #perform crossover
        def crossover(parent1, parent2, eflag=False):
            
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
            
            child1 = [child1_g, None, None]
            child2 = [child2_g, None, None]
    
            return child1, child2        
            
            
        #perform mutation sequence on a genome.
        def mutation(genome):    
            Lgenome = list(genome)                      #break string into list
            for i in range(len(Lgenome)):               #parse list and mutate as probable
                if random.uniform(0.0, 1.0) <= self.mutation_rate:
                    if Lgenome[i] == '1': Lgenome[i] = '0'
                    if Lgenome[i] == '0': Lgenome[i] = '1'
    
            g = "".join(Lgenome)                        #join list genome back into string
            return g
    
    
        #selects a parent based on the proportional representation and returns that parent
        def propSelect(f_cum):
            
            r = random.random() #get random number between 0 and 1
    
            #find where r lies
            for i in range(1,len(f_cum)-1):
                if r >= f_cum[i-1] and r < f_cum[i]: break  
            
            #return appropriate population member
            return self.population[i]
            
    
        temp_pop = [] #temporary population
        
        #create cumulative fitness list for proportional seletion
        tot = 0
        for i in range(self.population_size):    #sum fitnesses
            tot = tot + self.population[i][2]  
        if tot <= 0: tot = 0.00001
        print "Averge Fitness:", tot/self.population_size
        f_norm = []
        for i in range(self.population_size):        #create normalized fitness list
            f_norm.append(self.population[i][2]/tot)
        f_cum = [0, ]                   
        for i in range(len(f_norm)):                #create cumulative fitness list 
            f_cum.append(f_cum[i-1] + f_norm[i])
            
            
        #perform elitist seletion of top fitness value
        #elite = population[0]
        elite_flag = copy.copy(self.elite_flag)
        if elite_flag:  #if elite flag, always carry forward best member
            elite_pure = [max(self.population, key=lambda x:x[2])[0], None, None]
            elite_mut  = [mutation(max(self.population, key=lambda x:x[2])[0]), None, None]
            temp_pop.append(elite_pure)
            temp_pop.append(elite_mut)
    
        #iterate until new population is full
        while len(temp_pop) < self.population_size:
            
            #select 2 parents
            if elite_flag:  #ensure elitest always makes it to selction
                parent1 = max(self.population, key=lambda x:x[2])
            else: 
                parent1 = propSelect(f_cum)
                
            parent2 = propSelect(f_cum)
            
            #crossover
            child1, child2 = crossover(parent1,parent2, elite_flag)
            elite_flag = False
            
            #mutation
            child1[0] = mutation(child1[0])    
            child2[0] = mutation(child2[0])
            
            #update new generation
            if len(temp_pop) < self.population_size: temp_pop.append(child1)
            if len(temp_pop) < self.population_size: temp_pop.append(child2)
    
        self.population = temp_pop       #replace population with new one
        
    def run(self):
        """
        Run the algorithm
        """
        #create random population
        self.randGenomes()
        if self.useBase:
            gnm = self.encodeGenome(self.base)
            self.population[0] = [gnm, self.base, None]
            #print "USING BASE:", self.population[0]
        
        gen = 0 #generation number
        
        #place holders for best member and time he was repeated
        best_member = None
        best_fitness = 0.0
        best_count = 0
        
        genList = []
        best = []
        avg = []
        
        #repeat until the right best_count
        while best_count < self.bestRepeat:
            
          
            gen = gen + 1
            print '========== Generation:', gen, '=========='
            
            with Timer() as t:
                # get fitness for each population member  
                print "Evaluating Population of %d :" % self.population_size,  
                
                for i in range(self.population_size):
                    #if i % 10 == 0: print round(100*(float(i)/self.population_size),0), '%,', 
                    f = self.fitness(self.population[i][1])
                    if self.direction == 'min': f = 1.0/f #switch fitness for minimization, default is maximize
                    self.population[i][2] = f
                
                print "" #move print to new line
                    
                avgFit = sum([x[2] for x in self.population])/self.population_size
                            
                #sort population members
                self.population.sort(key=lambda member: -1*member[2])
                tempBest = max(self.population, key=lambda x1: x1[2])
                
                #check if best has changed (compare genome)
                print 'Best Member Fitness:', tempBest[2]
                print 'Worst Member Fitness:', self.population[-1][2]
                #print 'Average Member Fitness:', avgFit
                if tempBest[0] == best_member:
                    best_count = best_count+1
                elif tempBest[2] > best_fitness: 
                    best_member = tempBest[0]
                    best_fitness = tempBest[2]
                    best_count = 0 
                else: 
                    best_count = 0
                    
                genList.append(gen)
                best.append(max(self.population, key=lambda x1: x1[2])[2])
                avg.append(avgFit)
                    
                #create new generation
                self.generation()
                        
                #decode genomes
                for i in range(self.population_size):
                    self.population[i][1] = self.decodeGenome(self.population[i][0])
            print '=> generation complete in %.2f min' % (t.secs/60.0)
            
        Labels = ['Best Fitness', 'Average Fitness']
        plt.figure
        plt.plot(genList,best,genList,avg)
        plt.xlabel('Generation')
        plt.ylabel('Fitness Value')
        plt.legend(Labels, 'best', ncol=2, borderaxespad=0.)
        
        #plt.show()    
        
        return [best_member, self.decodeGenome(best_member), best_fitness]
            
        
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
    def fitcalc(x):
        s = 0
        for i in x:
            if i > 0: s = s + 1
        return s
        
        
    x = [1]*100
    xbounds = [(0.0, 1.0) for y in x]
    
    GA = GenAlg(fitcalc, x, xbounds, popMult=5, bitsPerGene=5, bestRepeat=15, mutation=0.1, crossover=0.7, crossN=3)
    results = GA.run()
    print "*** DONE ***"
    print results
    
    GA = GenAlg(fitcalc, x, xbounds, popMult=5, bitsPerGene=5, bestRepeat=15, mutation=0.13, crossover=0.7, crossN=2)
    results = GA.run()
    print "*** DONE ***"
    print results
    plt.show()
    