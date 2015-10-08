
from openmdao.util.testutil import assert_rel_error
from openmdao.main.api import Assembly, set_as_top, Component, Driver
from openmdao.main.datatypes.api import Array, Float, Int
from openmdao.main.interfaces import IHasParameters, implements
from openmdao.main.hasparameters import HasParameters
from openmdao.util.decorators import add_delegate
from openmdao.examples.simple.paraboloid import Paraboloid
from openmdao.examples.simple.paraboloid_derivative import ParaboloidDerivative
from pyopt_driver.pyopt_driver import pyOptDriver

from testdriver import GenDriver

import math

class MultiFunction(Component):
    #Finds the minimum f(1) = x[1]
    #              and f(2) = (1+x[2])/x[1]

    # set up interface to the framework
    # pylint: disable=E1101
    x1 = Float(1.0, iotype='in', desc='The variable x1')
    x2 = Float(1.0, iotype='in', desc='The variable x2')
    x3 = Float(1.0, iotype='in', desc='The variable x2')

    f1_x = Float(iotype='out', desc='f1(x1,x2)')
    f2_x = Float(iotype='out', desc='f2(x1,x2)')

    #g1_x = Float(iotype='out', desc='g1(x1,x2)')
    #g2_x = Float(iotype='out', desc='g2(x1,x2)')

    def execute(self):

        x1 = self.x1
        x2 = self.x2
        x3 = self.x3

        x = [self.x1, self.x2, self.x3]
        #self.f1_x = x1
        #self.f2_x = (1+x2)/x1
        self.f1_x = sum([-10.0*math.exp(-0.2*(x[i]**2.0 + x[i+1]**2)**0.5) for i in range(len(x)-1)])
        self.f2_x = sum([(abs(x_i)**0.8 + 5.0*math.sin(x_i**3.0) ) for x_i in x])
        
        #self.g1_x =  x2+9.0*x1
        #self.g2_x = -x2+9.0*x1

        #print [self.f1_x, self.f2_x]

class MultiObjectiveOptimization_Test(Assembly):
    """Multi Objective optimization of the  with NSGA2."""

    def configure(self):
        """ Creates a new Assembly containing a MultiFunction and an optimizer"""

        # pylint: disable=E1101

        # Create MultiFunction component instances
        self.add('multifunction', MultiFunction())

        # Create NSGA2 Optimizer instance
        self.add('driver', GenDriver())

        # Driver process definition
        self.driver.workflow.add('multifunction')

        #self.driver.print_results = True

        # NSGA2 Objective
        self.driver.add_objective('multifunction.f1_x')
        self.driver.add_objective('multifunction.f2_x')

        # NSGA2 Design Variable
        self.driver.add_parameter('multifunction.x1', low=-5, high=5)
        self.driver.add_parameter('multifunction.x2', low=-5, high=5)
        self.driver.add_parameter('multifunction.x3', low=-5, high=5)



def test_GA_multi_obj_custom():
# Note, just verifying that things work functionally, rather than run
# this for many generations.

    top = MultiObjectiveOptimization_Test()
    set_as_top(top)

    # Optimization Variables 
    top.driver.opt_type        = 'min'
    top.driver.generations     = 20
    top.driver.popMult         = 40.0
    top.driver.crossover_Rate  = 0.9
    top.driver.mutation_rate   = 0.05
    top.driver.crossN          = 3

    top.run()
   
if __name__ == "__main__":
    #test_GA_multi_obj_multi_con()
    test_GA_multi_obj_custom()