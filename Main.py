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
#           A > B
#           The following codes register a population of individuals that
#           satisfy those requirements


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
