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
from Read_Control_Config import read_control_config
from Evaluate_Single_Element_Tersoff import evaluate_single_element_Tersoff

# Retrieve parameters
indiv_low, indiv_up, ELEMENT_NUM, ELEMENT_NAME,\
RANDOM_SEED, POP_SIZE, MAX_GEN, CXPB, MUTPB = read_control_config()

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
toolbox.register("attr_gamma", random.uniform, indiv_low[0], indiv_up[0])
toolbox.register("attr_lambda3", random.uniform, indiv_low[1], indiv_up[1])
toolbox.register("attr_c", random.uniform, indiv_low[2], indiv_up[2])
toolbox.register("attr_d", random.uniform, indiv_low[3], indiv_up[3])
toolbox.register("attr_costheta0", random.uniform, indiv_low[4], indiv_up[4])
toolbox.register("attr_n", random.uniform, indiv_low[5], indiv_up[5])
toolbox.register("attr_beta", random.uniform, indiv_low[6], indiv_up[6])
toolbox.register("attr_lambda2", random.uniform, indiv_low[7], indiv_up[7])
toolbox.register("attr_B", random.uniform, indiv_low[8], indiv_up[8])
toolbox.register("attr_R", random.uniform, indiv_low[9], indiv_up[9])
toolbox.register("attr_D", random.uniform, indiv_low[10], indiv_up[10])
toolbox.register("attr_lambda1", random.uniform, indiv_low[11], indiv_up[11])
toolbox.register("attr_A", random.uniform, indiv_low[12], indiv_up[12])
toolbox.register("individual", tools.initCycle, creator.Individual,
                     (toolbox.attr_gamma, toolbox.attr_lambda3,
                      toolbox.attr_c, toolbox.attr_d, toolbox.attr_costheta0,
                      toolbox.attr_n, toolbox.attr_beta, toolbox.attr_lambda2,
                      toolbox.attr_B, toolbox.attr_R, toolbox.attr_D,
                      toolbox.attr_lambda1, toolbox.attr_A), n=1)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

#   Operator registration
#   register the goal / fitness function
if ELEMENT_NUM == "single":
    toolbox.register("evaluate", evaluate_single_element_Tersoff)
elif ELEMENT_NUM == "two":
    pass
    #toolbox.register("evaluate", evaluate_two_elements_Tersoff)

#   register the crossover operator. Simulated Binary Bounded algorithm
#   is used with an eta value of 20 and bounds for parameters.
toolbox.register("mate", tools.cxSimulatedBinaryBounded, eta=20,
                 low=indiv_low, up=indiv_up)
#   register the mutation operator. Polynomial algorithm is used with
#   an eta value of 20 and bounds for parameters. The probability for
#   each attribute to be mutated is equal to 1/num_attributes
toolbox.register("mutate", tools.mutPolynomialBounded, eta=20,
                 low=indiv_low, up=indiv_up,
                 indpb=1/len(indiv_low))
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
