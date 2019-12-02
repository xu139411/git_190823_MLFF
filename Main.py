#   Workflow of the genetic algorithm that finds the parameters of Tersoff force
#   field for Transition Metal Dichalcogenides (TMDCs).

# Standard library imports
import random
import pickle
import os
PATH_ROOT = os.path.abspath('.')
import logging
logging_file = os.path.join(PATH_ROOT, 'logging.txt')
logging.basicConfig(filename=logging_file, level=logging.INFO)
# Third party imports: Numpy, DEAP and SCOOP
import numpy as np
from deap import base, creator, tools
from scoop import futures
# Local library imports
from Read_Functions import read_control_config
from Read_Functions import read_training_data
from Evaluate_Single_Element_Tersoff import evaluate_single_element_Tersoff
from Evaluate_Two_Elements_Tersoff import evaluate_two_elements_Tersoff

# Retrieve parameters: dictionary, dictionary, dictionary
ELEMENT_NAME, PARAMETERS_FF_RANGE, PARAMETERS_GA, CRITERIA,\
OPTIMIZED_PARAMETERS = read_control_config()
# Retrieve the DFT training data into a dictionary
training_data = read_training_data(ELEMENT_NAME)
if list(CRITERIA.keys()) != list(training_data.keys()):
    raise ValueError('Criteria does not match with the training data\n')

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
#   2.82 < R< 3.8; 0 < D < R-2.81821; lambda2 < lambda1 < 3 * lambda2
#   A > B.
#   The following codes register a population of individuals with all
#   the force field parameters except m. Since the value of m is always
#   1, it is not included in the individuals, but is added during the
#   evaluation.
#   Parameters other than m can also be spefied as a fixed value. When
#   registering the individual, the code will register parameters that are
#   not fixed. Parameters that are fixed will be stored in a dictionary called
#   fixed_para{}, whose keys are the indices of those parameters according to
#   the Tersoff parameter sequence.
indiv_low, indiv_up = [], []
fixed_para = {}
for index, parameter in enumerate(list(PARAMETERS_FF_RANGE.keys())):
    attr = "attr_" + parameter
    toolbox.register(attr, random.uniform, PARAMETERS_FF_RANGE[parameter][0],
        PARAMETERS_FF_RANGE[parameter][1])
    if PARAMETERS_FF_RANGE[parameter][0] < PARAMETERS_FF_RANGE[parameter][1]:
        indiv_low.append(PARAMETERS_FF_RANGE[parameter][0])
        indiv_up.append(PARAMETERS_FF_RANGE[parameter][1])
    else:
        fixed_para[index] = PARAMETERS_FF_RANGE[parameter][0]
attr = [toolbox.attr_gamma, toolbox.attr_lambda3, toolbox.attr_c, toolbox.attr_d,
        toolbox.attr_costheta0, toolbox.attr_n, toolbox.attr_beta, toolbox.attr_lambda2,
        toolbox.attr_B, toolbox.attr_R, toolbox.attr_D, toolbox.attr_lambda1,
        toolbox.attr_A]
for _ in list(fixed_para.keys())[::-1]:
    del attr[_]
attr = tuple(attr)
toolbox.register("individual", tools.initCycle, creator.Individual, attr, n=1)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
#   Register for guess individuals
def readIndividual(icls, content):
    return icls(content)
def readPopulation(pcls, ind_init, filename):
    with open(filename, "r") as pop_file:
        contents = np.loadtxt(filename)
        contents = contents.tolist()
    return pcls(ind_init(c) for c in contents)
toolbox.register("individual_guess", readIndividual, creator.Individual)
toolbox.register("population_guess", readPopulation, list, toolbox.individual_guess)

#   Operator registration
#   Register the goal / fitness function
if len(ELEMENT_NAME) == 1:
    toolbox.register("evaluate", evaluate_single_element_Tersoff,
                     element_name=ELEMENT_NAME,
                     training_data=training_data,
                     criteria=CRITERIA,
                     fixed_value=fixed_para)
else:
    toolbox.register("evaluate", evaluate_two_elements_Tersoff,
                     element_name=ELEMENT_NAME,
                     training_data=training_data,
                     criteria=CRITERIA,
                     fixed_value=fixed_para,
                     optimized_parameters=OPTIMIZED_PARAMETERS)

#   Register the crossover operator. Simulated Binary Bounded algorithm
#   is used with an eta value of 20 and bounds for parameters.
toolbox.register("mate", tools.cxSimulatedBinaryBounded, eta=20,
                 low=indiv_low, up=indiv_up)
#   Register the mutation operator. Polynomial algorithm is used with
#   an eta value of 20 and bounds for parameters. The probability for
#   each attribute to be mutated is equal to 1/num_attributes
toolbox.register("mutate", tools.mutPolynomialBounded, eta=20,
                 low=indiv_low, up=indiv_up, indpb=1/len(indiv_low))
#   Register the selection operator. Tournament algorithm is used. For
#   each selection, 3 individuals are chosen, and the best among them is
#   selected.
#   ***Note that DEAP use tournament with replacement, while in the
#      paper tournament without replacement is used. This could cause
#      deviation of results.
toolbox.register("select", tools.selTournament, tournsize=3)
#   register the statistics module that calculates and stores the statistics
stats = tools.Statistics(key=lambda ind: ind.fitness.values)
stats.register("avg", np.mean)
stats.register("std", np.std)
stats.register("min", np.min)
stats.register("max", np.max)
#   Register the
toolbox.register("map", futures.map) # Stay with scoop

# Main function
def main(checkpoint=None, guess=None):
    if checkpoint:
        #   A file name has been given, then load the data from the file
        with open(checkpoint, "rb") as cp_file:
            cp = pickle.load(cp_file)
        random.setstate(cp["rndstate"])
        pop = cp["population"]
        start_gen = cp["generation"]
        hof = cp["halloffame"]
        logbook = cp["logbook"]
    elif guess:
        #   A file name has been given, then load the data from the file
        pop_guess = toolbox.population_guess(guess)
        #   Generate a part of the population with a specified random_seed
        random.seed(PARAMETERS_GA['RANDOM_SEED'])
        random_size = PARAMETERS_GA['POP_SIZE'] - len(pop_guess)
        pop_random = toolbox.population(n=random_size)
        pop = pop_guess + pop_random
        start_gen = 0
        hof = tools.HallOfFame(1)
        logbook = tools.Logbook()
    else:
        #   Start a new evolution with a specified random_seed
        random.seed(PARAMETERS_GA['RANDOM_SEED'])
        #   Create an initial population of POP_SIZE individuals (where)
        #   Each individual is a list of 13 force field parameters
        pop = toolbox.population(n=PARAMETERS_GA['POP_SIZE'])
        #   Starting generation
        start_gen = 0
        #   Hall of fame object that stores the best three individuals thus far
        hof = tools.HallOfFame(3)
        #   Create a logbook
        logbook = tools.Logbook()

    #   Evaluate the entire population
    fitnesses = list(toolbox.map(toolbox.evaluate, pop))
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit
    #   Find the Hall of fame individuals
    hof.update(pop)
    #   Begin the evolution
    for g in range(start_gen+1, PARAMETERS_GA['MAX_GEN']+1):
        logging.info('-- GENERATION %s', str(g))
        #   Select individuals for the next generation, hof is always included.
        #   hof[:] provides a normal list
        offspring = toolbox.select(pop, len(pop)-len(hof)) + hof[:]
        #   Clone the selected individuals
        offspring = list(toolbox.map(toolbox.clone, offspring))

        #   Apply crossover and mutation on the offspring
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            #   Cross two individuals with probability CXPB
            if random.random() < PARAMETERS_GA['CXPB']:
                toolbox.mate(child1, child2)
                #   Fitness values of the children must be recalculated
                del child1.fitness.values
                del child2.fitness.values
        for mutant in offspring:
            #   Mutate an individual with probability MUTPB
            if random.random() < PARAMETERS_GA['MUTPB']:
                toolbox.mutate(mutant)
                del mutant.fitness.values
        #   Evaluate the individuals with invalid fitnesses
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit
        #   The population is entirely replaced by the offspring
        pop[:] = offspring
        #   Update the hof
        hof.update(pop)
        #   Compute the statistics and record them in the logbook
        record = stats.compile(pop)
        logbook.record(gen=g, evals=len(invalid_ind), **record)
        #   Save the optimization every FREQ generation
        if g % PARAMETERS_GA['FREQ'] == 0:
            #   Fill the dictionary using the dict(key=value[, ...]) constructor
            cp = dict(rndstate=random.getstate(),population=pop, generation=g,
                      halloffame=hof, logbook=logbook)
            #   Create the folder checkpoint, and dump the data
            path_log = os.path.join(PATH_ROOT, 'checkpoint', '')
            cp_file_name = os.path.join(path_log, 'cp_'+str(g)+'.pkl')
            log_file_name = os.path.join(path_log, 'log_'+str(g)+'.txt')
            if 'checkpoint' in os.listdir('.'):
                pass
            else:
                os.mkdir(path_log)
            with open(cp_file_name, 'wb') as cp_file:
                pickle.dump(cp, cp_file)
            #   Compute and record the statistics into the logbook
            log_gen, log_avg, log_std, log_min, log_max = \
            logbook.select('gen', 'avg', 'std', 'min', 'max')
            with open(log_file_name, 'w') as log_file:
                log_file.write('gen' + 12*' ' + 'avg' + 12*' '  + 'std' + 12*' '
                               + 'min' + 12*' ' + 'max\n')
                for _ in range(len(log_gen)):
                    log_file.write('{0:<15d}{1:<15.4f}{2:<15.4f}{3:<15.4f}{4:<15.4f}\n'.format(log_gen[_], log_avg[_], log_std[_], log_min[_], log_max[_]))
                log_file.write('HallofFame individuals thus far:\n')
                for best in hof:
                    log_file.writelines('{0} '.format(_) for _ in best)
                    log_file.write('\n')
                    log_file.writelines('The fitness value is {0}'.format(best.fitness.values))
                    log_file.write('\n')
                log_file.write('The best individuals in the current generation:\n')
                best_ind = tools.selBest(pop, 3)
                for best in best_ind:
                    log_file.writelines('{0} '.format(_) for _ in best)
                    log_file.write('\n')
                    log_file.writelines('The fitness value is {0}'.format(best.fitness.values))
                    log_file.write('\n')

    print("-- End of evolution --")

if __name__ == "__main__":
    #pass
    main(guess='./best/result_hof.txt')
    #main('./cp_600.pkl')
