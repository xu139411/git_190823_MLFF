#   Workflow of the genetic algorithm that finds the parameters of Tersoff force
#   field for Transition Metal Dichalcogenides (TMDCs).

# Standard library imports
import random
import pickle

# Third party imports: Numpy, DEAP and SCOOP
import numpy as np
from deap import base, creator, tools
from scoop import futures
#toolbox.register("map", futures.map) # Stay with scoop

# Local library imports
from Read_Control import read_control

# Retrieve parameters
gamma_min, gamma_max, lambda3_min, lambda3_max, c_min, c_max, d_min, d_max,\
costheta0_min, costheta0_max, n_min, n_max, beta_min, beta_max, lambda2_min,\
lambda2_max, B_min, B_max, R_min, R_max, D_min, D_max, lambda1_min, lambda1_max,\
A_min, A_max, ELEMENT, RANDOM_SEED, POP_SIZE, MAX_GENERATION, CXPB,\
MUTPB = read_control()

# DEAP setup
#   Minimize the fitness value
#   Each individual is a list.
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)

toolbox = base.Toolbox()
#   Attribute generator
#   The attributes (Tersoff force field parameters) are in the order of:
#   m (=1), gamma, lambda3, c, d, costheta0,
#   n, beta, lambda2, B, R, D, lambda1 and A
#   Each attribute is randomly chosen according to the range specified
#   in the file GA_control.txt.
#   Moreover, there are parameters that have constraints:
#   2.82 < R< 3.8; 0 < D < R-2.81821; lambda2 < lambda1 < 2 * lambda2
#   A > B.
#   The following codes register a population of individuals with all
#   the force field parameters except m. Since the value of m is always
#   1, it is not included in the individuals, but is added during the
#   evaluation.
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
                     (toolbox.attr_gamma, toolbox.attr_lambda3,
                      toolbox.attr_c, toolbox.attr_d, toolbox.attr_costheta0,
                      toolbox.attr_n, toolbox.attr_beta, toolbox.attr_lambda2,
                      toolbox.attr_B, toolbox.attr_R, toolbox.attr_D,
                      toolbox.attr_lambda1, toolbox.attr_A), n=1)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

#   Operator registration
#   Dummy evaluate function
def evaluate_single_element_Tersoff(individual):
    return sum(individual),
#   register the goal / fitness function
if ELEMENT == "single":
    toolbox.register("evaluate", evaluate_single_element_Tersoff)
elif ELEMENT == "two":
    pass
    #toolbox.register("evaluate", evaluate_two_elements_Tersoff)
indiv_bounds_low = [gamma_min, lambda3_min, c_min, d_min, costheta0_min, n_min,
                    beta_min, lambda2_min, B_min, R_min, D_min, lambda1_min,
                    A_min]
indiv_bounds_up = [gamma_max, lambda3_max, c_max, d_max, costheta0_max, n_max,
                    beta_max, lambda2_max, B_max, R_max, D_max, lambda1_max,
                    A_max]
#   register the crossover operator. Simulated Binary Bounded algorithm
#   is used with an eta value of 20 and bounds for parameters.
toolbox.register("mate", tools.cxSimulatedBinaryBounded, eta=20,
                 low=indiv_bounds_low, up=indiv_bounds_up)
#   register the mutation operator. Polynomial algorithm is used with
#   an eta value of 20 and bounds for parameters. The probability for
#   each attribute to be mutated is equal to 1/num_attributes
toolbox.register("mutate", tools.mutPolynomialBounded, eta=20,
                 low=indiv_bounds_low, up=indiv_bounds_up,
                 indpb=1/len(indiv_bounds_low))
#   register the selection operator. Tournament algorithm is used. For
#   each selection, 3 individuals are chosen, and the best among them is
#   selected.
#   ***Note that DEAP use tournament with replacement, while in the
#      paper tournament without replacement is used. This could cause
#      deviation of results.
toolbox.register("select", tools.selTournament, tournsize=3)

# Main function
def main(checkpoint=None):
    if checkpoint:
        #   A file name has been given, then load the data from the file
        with open(checkpoint, "r") as cp_file:
            cp = pickle.load(cp_file)
        population = cp["population"]
        start_gen = cp["generation"]
        halloffame = cp["halloffame"]
        logbook = cp["logbook"]
        random.setstate(cp["rndstate"])
    else:
        #   Start a new evolution with a specified random_seed
        random.seed(RANDOM_SEED)
        #   Create an initial population of POP_SIZE individuals (where)
        #   each individual is a list of 13 force field parameters
        pop = toolbox.population(n=POP_SIZE)

        print("START OF EVALUATION")
        #   Evaluate the entire population
        fitnesses = list(map(toolbox.evaluate, pop))
        for ind, fit in zip(pop, fitnesses):
            ind.fitness.values = fit
        print(" EVALUATED {0} INDIVIDUALS".format(len(pop)))
        #   Extracting the fitnesses of all individuals in the current population
        fits = [ind.fitness.values[0] for ind in pop]

        #   Variable keeping track of the number of generations
        g = 0

        #   Begin the evolution
        while g < MAX_GENERATION:
            #   A new generation
            g = g + 1
            print("-- GENERATION {0}".format(g))
            #   Select individuals for the next generation
            offspring = toolbox.select(pop, len(pop))
            #   Clone the selected individuals
            offspring = list(map(toolbox.clone, offspring))

            #   Apply crossover and mutation on the offspring
            for child1, child2 in zip(offspring[::2], offspring[1::2]):
                #   Cross two individuals with probability CXPB
                if random.random() < CXPB:
                    toolbox.mate(child1, child2)
                    #   Fitness values of the children must be recalculated
                    del child1.fitness.values
                    del child2.fitness.values
            for mutant in offspring:
                #   Mutate an individual with probability MUTPB
                if random.random() < MUTPB:
                    toolbox.mutate(mutant)
                    del mutant.fitness.values
            #   Evaluate the individuals with invalid fitnesses
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            fitnesses = map(toolbox.evaluate, invalid_ind)
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit
            #   The population is entirely replaced by the offspring
            pop[:] = offspring
            #   Gather all the fitnesses in one list and print the stats
            fits = [ind.fitness.values[0] for ind in pop]
            length = len(pop)
            mean = sum(fits) / length
            sum2 = sum(x*x for x in fits)
            std = abs(sum2 / length - mean**2)**0.5
            print(" Min {0}".format(min(fits)))
            print(" Max {0}".format(max(fits)))
            print(" Avg {0}".format(mean))
            print(" Std {0}".format(std))

        print("-- End of evolution --")

        best_ind = tools.selBest(pop, 1)[0]
        print(" The best individual is {0}, the best fitness is {1}".format(
                best_ind, best_ind.fitness.values)
        )

if __name__ == "__main__":
    main()
