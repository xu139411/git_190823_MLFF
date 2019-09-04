#   Workflow of the genetic algorithm that finds the parameters of Tersoff force
#   field for Transition Metal Dichalcogenides (TMDCs).

# Standard library imports
import random
import pickle

# Third party imports: Numpy, DEAP and SCOOP
import numpy as np
from deap import base, creator, tools
from scoop import futures
toolbox.register("map", futures.map) # Stay with scoop

# Local library imports


# DEAP setup
#           Minimize the fitness value
#           Each individual is a list.
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)

toolbox = base.Toolbox()

#           Attribute generator
#           The attributes (Tersoff force field parameters) are in the order of:
#           m (=1), gamma, lambda3, c, d, costheta0,
#           n, beta, lambda2, B, R, D, lambda1 and A
#           Each attribute is randomly chosen according to the range specified
#           in the file GA_control.txt.
#           Moreover, there are parameters that have constraints:
#           2.82 < R< 3.8; 0 < D < R-2.81821; lambda2 < lambda1 < 2 * lambda2
#           A > B.
#           The following codes register a population of individuals.

toolbox.register("attr_m", lambda: 1)
toolbox.register("attr_gamma", random.uniform, gamma_min, gamma_max)
toolbox.register("attr_lambda3", random.uniform, lambda3_min, lambda3_max)
toolbox.register("attr_c", random.uniform, c_min, c_max)
toolbox.register("attr_d", random.uniform, d_min, d_max)
toolbox.register("attr_costheta0", random.uniform, costheta0_min,
                 costheta0_max)
toolbox.register("attr_n", random.uniform, n_min, n_max)
toolbox.register("attr_beta", random.uniform, beta_min, beta_max)
toolbox.register("attr_lambda2", random.uniform, lambda2_min, lambda2_max)
toolbox.register("attr_B", random.uniform, B_min, B_max)
toolbox.register("attr_R", random.uniform, R_min, R_max)
toolbox.register("attr_D", random.uniform, D_min, D_max)
toolbox.register("attr_lambda1", random.uniform, lambda1_min, lambda1_max)
toolbox.register("attr_A", random.uniform, A_min, A_max)
toolbox.register("individual", tools.initCycle, creator.Individual,
                     (toolbox.attr_m, toolbox.attr_gamma, toolbox.attr_lambda3,
                      toolbox.attr_c, toolbox.attr_d, toolbox.attr_costheta0,
                      toolbox.attr_n, toolbox.attr_beta, toolbox.attr_lambda2,
                      toolbox.attr_B, toolbox.attr_R, toolbox.attr_D,
                      toolbox.attr_lambda1, toolbox.attr_A), n=1)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.population(n=2)

def main(checkpoint=None):
    if checkpoint:
        # A file name has been given, then load the data from the file
        with open(checkpoint, "r") as cp_file:
            cp = pickle.load(cp_file)
        population = cp["population"]
        start_gen = cp["generation"]
        halloffame = cp["halloffame"]
        logbook = cp["logbook"]
        random.setstate(cp["rndstate"])
    else:
        # Start a new evolution
