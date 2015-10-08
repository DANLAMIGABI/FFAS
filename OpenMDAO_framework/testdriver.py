__all__ = ['GenDriver']

from openmdao.main.api import Driver
from openmdao.main.hasparameters import HasParameters
from openmdao.main.hasresponses import HasResponses
from openmdao.util.decorators import add_delegate
from openmdao.lib.datatypes.api import Float, Str, Dict, Int, List, VarTree, Enum, Bool

from openmdao.main.hasobjective import HasObjectives
from openmdao.main.hasevents import HasEvents
from openmdao.main.interfaces import implements, IHasParameters, IHasResponses, IHasObjective, IHasConstraints, IOptimizer
from openmdao.util.decorators import add_delegate
from openmdao.util.typegroups import real_types, int_types, iterable_types

import numpy as np

from NGSA import GenAlg
import random

import matplotlib.pylab as plt



@add_delegate(HasParameters, HasObjectives, HasResponses)#, HasEvents)
class GenDriver(Driver):
    """
    Custom Genetic Driver
    """
    implements(IHasParameters, IHasObjective, IHasResponses, IOptimizer)

    # pylint: disable-msg=E1101
    opt_type        = Enum("max", values=["min", "max"], iotype="in", desc='Sets the optimization to either minimize or maximize the objective function.')
    generations     = Int(10, iotype="in", desc="The maximum number of generations the algorithm will evolve to before stopping.")
    popMult         = Float(4.0, iotype="in", desc="Multiplier to size population (population = len(x)*popMult)")
    crossover_rate  = Float(0.8, iotype="in", low=0.0, high=1.0, desc="The crossover rate used when two parent genomes reproduce to form a child genome.")
    mutation_rate   = Float(0.03, iotype="in", low=0.0, high=1.0, desc="The mutation rate applied to population members.")
    crossN          = Int(2, iotype='in', low=2, high=3, desc="Number of crossover segments")
    bitsPerGene     = Int(9, iotype='in', desc="number of bits per gene(parameter)")
    tourneySize     = Int(2, iotype='in', desc="number of members in each tournament selection") 
    livePlotGens    = Int(3, iotype='in', desc="number of generations to plot live (0 for no live plotting)")
    hammingDist     = Bool(False, iotype='in', desc='use hamming distance as separation distance?')

    gen_num         = Int(0, iotype="out", desc="This counter is driven by the fitness function to indicate generation/iteration number")

    results = List([], iotype="out", desc="Final solution(s)")
    #selection_method = Enum("roulette_wheel", ("roulette_wheel","rank","uniform"), iotype="in", desc="The selection method used to pick population members who will survive for breeding into the next generation.")
    #elitism = Bool(False, iotype="in", desc="Controls the use of elitism in the creation of new generations.")
    #best_individual = Slot(klass=GenomeBase.GenomeBase, desc="The genome with the best score from the optimization.")

    def _build_problem(self):
        """ builds the problem """
        x = []
        x_bounds = []



        #check each param and get relevant parameter data
        for param in self.get_parameters().values()[1:]:
            low = param.low
            high = param.high

            metadata = param.get_metadata()[1]

            #get bounds
            min_i = [metadata['low'], low]
            max_i = [metadata['high'], high]

            try: #get most constrictive range
                mm = [ max([m for m in min_i if m <> None]), 
                       min([m for m in max_i if m <> None])    ]
            except:
                raise StandardError('Each parameter must have a min and max')  
            
            if "value" in metadata:
                x_i = metadata['value']
            else: x_i = random.uniform(mm[0],mm[1])
                
            x.append(x_i)
            x_bounds.append(mm)    

        return x, x_bounds

    def fitCalc(self):
        """
        Calculate Fitness
        """
        results = self.eval_objective()
        pass



    def execute(self):
        """Perform the optimization"""
        x, bounds = self._build_problem()

        outputs = self.get_responses().keys()

        GA = GenAlg(self._run_model, x, bounds, popMult=self.popMult, 
                    bitsPerGene=self.bitsPerGene, mutation=self.mutation_rate, 
                    direction=self.opt_type, crossover=self.crossover_rate, 
                    crossN=self.crossN, maxGens=self.generations, 
                    tourneySize=self.tourneySize, livePlotGens=self.livePlotGens,
                    hammingDist=self.hammingDist)
        
        results = GA.run()
        print "*** DONE ***"
        self.results = results

        
    def _run_model(self, chromosome, gen=0):
        #self.gen_num = gen
        self.set_parameters([val for val in [gen]+chromosome])
        self.run_iteration()
        f = np.array(self.eval_objectives())
        #print "OUTPUTS:", f
        return f



