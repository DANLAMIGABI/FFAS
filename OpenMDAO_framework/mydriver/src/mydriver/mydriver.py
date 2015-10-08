__all__ = ['GenDriver']

from openmdao.main.api import Driver
from openmdao.main.hasparameters import HasParameters
from openmdao.util.decorators import add_delegate
from openmdao.lib.datatypes.api import Float, Str, Dict, Int, List, VarTree, Enum
# Make sure that your class has some kind of docstring. Otherwise
# the descriptions for your variables won't show up in the
# source documentation.

#@add_delegate(HasParameters)  # uncomment this to add parameter handling
class GenDriver(Driver):
    """
    Custom Genetic Driver
    """
    implements(IHasParameters)#, IHasObjective, IOptimizer)

    # pylint: disable-msg=E1101
    opt_type = Enum("minimize", values=["minimize", "maximize"],
                    iotype="in", desc='Sets the optimization to either minimize or maximize the objective function.')
    generations = Int(10, iotype="in", desc="The maximum number of generations the algorithm will evolve to before stopping.")
    population_size = Float(50, iotype="in", desc="The size of the population in each generation.")
    crossover_rate = Float(0.8, iotype="in", low=0.0, high=1.0, desc="The crossover rate used when two parent genomes reproduce to form a child genome.")
    mutation_rate = Float(0.03, iotype="in", low=0.0, high=1.0, desc="The mutation rate applied to population members.")

    #selection_method = Enum("roulette_wheel", ("roulette_wheel","rank","uniform"), iotype="in", desc="The selection method used to pick population members who will survive for breeding into the next generation.")
    #elitism = Bool(False, iotype="in", desc="Controls the use of elitism in the creation of new generations.")
    #best_individual = Slot(klass=GenomeBase.GenomeBase, desc="The genome with the best score from the optimization.")


    def _build_problem(self):
        """ builds the problem """

        alleles = GAllele.GAlleles()
        count = 0
        for param in self.get_parameters().values():
            allele = None
            count += 1
            val = param.evaluate()[0] #now grab the value
            low = param.low
            high = param.high

            metadata = param.get_metadata()[1]
            #then it's a float or an int, or a member of an array
            if ('low' in metadata or 'high' in metadata) or \
                array_test.search(param.targets[0]):
                #some kind of int
                if isinstance(val, int_types):
                    allele = GAllele.GAlleleRange(begin=low, end=high, real=False)
                elif isinstance(val, real_types):
                    #some kind of float
                    allele = GAllele.GAlleleRange(begin=low, end=high, real=True)

            elif "values" in metadata and \
                 isinstance(metadata['values'], iterable_types):
                allele = GAllele.GAlleleList(metadata['values'])

            if allele:
                alleles.add(allele)
            else:
                self.raise_exception("%s is not a float, int, or enumerated "
                                     "datatype. Only these 3 types are allowed"
                                     % param.targets[0], ValueError)
        self.count = count
        return alleles

    def execute(self):
        """Perform the optimization"""
        self.set_events()

        alleles = self._make_alleles()

        genome = G1DList.G1DList(len(alleles))
        genome.setParams(allele=alleles)
        genome.evaluator.set(self._run_model)

        genome.mutator.set(Mutators.G1DListMutatorAllele)
        genome.initializator.set(Initializators.G1DListInitializatorAllele)
        #TODO: fix tournament size settings
        #genome.setParams(tournamentPool=self.tournament_size)

        # Genetic Algorithm Instance
        #print self.seed

        #configuring the options
        ga = GSimpleGA.GSimpleGA(genome, interactiveMode=False,
                                 seed=self.seed)
        pop = ga.getPopulation()
        pop = pop.scaleMethod.set(Scaling.SigmaTruncScaling)
        ga.setMinimax(Consts.minimaxType[self.opt_type])
        ga.setGenerations(self.generations)
        ga.setMutationRate(self.mutation_rate)
        if self.count > 1:
            ga.setCrossoverRate(self.crossover_rate)
        else:
            ga.setCrossoverRate(0)
        ga.setPopulationSize(self.population_size)
        ga.setElitism(self.elitism)

        #setting the selector for the algorithm
        ga.selector.set(self._selection_mapping[self.selection_method])

        #GO
        ga.evolve(freq_stats=0)

        self.best_individual = ga.bestIndividual()

        #run it once to get the model into the optimal state
        self._run_model(self.best_individual)

    def _run_model(self, chromosome):
        self.set_parameters([val for val in chromosome])
        self.run_iteration()
        return self.eval_objective()












    
    # uncomment this to add parameter handling
    #implements(IHasParameters)

    # declare inputs and outputs here, for example:
    #x = Float(0.0, iotype='in', desc='description for x')
    #y = Float(0.0, iotype='out', desc='description for y')
    
    def start_iteration(self):
        super(MyDriver, self).start_iteration()



    def continue_iteration(self):
        return super(MyDriver, self).continue_iteration()
    


    def pre_iteration(self):
        super(MyDriver, self).pre_iteration()
        




    def run_iteration(self):
        super(MyDriver, self).run_iteration()



    def post_iteration(self):
        super(MyDriver, self).post_iteration()
