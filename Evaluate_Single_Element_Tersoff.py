#   Evaluation part of the genetic algorithm that finds the parameters of
#   Tersoff force field for Transition Metal Dichalcogenides (TMDCs).
#   This file evaluates the single-element calculations

# Standard library imports
import os
import copy

# Third party imports: lammps
import numpy as np
from lammps import lammps

# Local library imports

#   Create a Tersoff force field file
def create_fffile(individual):
    #   With the input list individual that contains all the force field
    #   parameters, create a file with all parameters that is read by LAMMPS
    #   Individual has all the parameters in the following order:
    #   m (=1), gamma, lambda3, c, d, costheta0,
    #   n, beta, lambda2, B, R, D, lambda1 and A
    pass

#   Calculate the RMSD and cohesive energy of Se2 dimer
def rmsd_cohesive_se2(*criteria):
    #   Calculate the room mean square displacement and cohesive energy of Se2
    




# The evaluate function
def evaluate_single_element_Tersoff(individual, criteria_all):
    #   Deep copy to avoid changing the individual list
    ind = copy.deepcopy(individual)
    #   ind_all is the complete parameter set
    ind_all = [1] + ind
    #   Create a force field file
    create_fffile(ind_all)

    #   Parameters for evaluation
    eval_seq = [rmsd_cohesive_se2, dissociation_se2, rmsd_cohesive_se3,
                rmsd_cohesive_se6, rmsd_cohesive_se8ring,
                rmsd_cohesive_se8helix, stability_se2, stability_se6,
                stability_se8]
    fitness_step = 1000
    fitness_max = len(eval_seq) * fitness_step
    fitness_current = fitness_max

    #   Start of evaluation
    for i, eval in enumerate(eval_seq):
        #   Error Sum of Squares and whether the current evaluation succeeds
        #   For each evaluation function, if the convergence criteria is met,
        #   the current fitness is subtracted by fitness_step plus the current
        #   error sum of squares. If the convergence criteria is not met, the
        #   evaluation stops immediately and the current fitness is returned
        #   If all evaluation functions converge to the criteria, the final
        #   fitness value is returned
        sse, proceed = eval(*criteria_all[i])
        if proceed:
            fitness_current = fitness_current - fitness_step + sse
            if i == len(eval_seq) - 1:
                return fitness_current
            continue
        else:
            fitness_current = fitness_current + sse
            return fitness_current
