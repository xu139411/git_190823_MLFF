#   Evaluation part of the genetic algorithm that finds the parameters of
#   Tersoff force field for Transition Metal Dichalcogenides (TMDCs).
#   This file evaluates the single-element calculations

# Standard library imports
import os
import shutil
import copy
# Third party imports: lammps
import numpy as np
# Local library imports

#   Create a Tersoff force field file
def create_fffile(path_tmp, element_name, individual):
    #   With the input list individual that contains all the force field
    #   parameters, create a file to be read by LAMMPS
    #   Individual value has all the parameters in the following order:
    #   [m (=1), gamma, lambda3, c, d, costheta0,
    #   n, beta, lambda2, B, R, D, lambda1, A]
    '''
    input:
        - element_name - str, name of the element
        - individual - list, Tersoff parameters of the mentioned element
    output:
        - a text file containing parameters
    '''
    file_name = os.path.join(path_tmp, element_name + '.tersoff')
    with open(file_name, 'w') as output_file:
        output_file.write('# Tersoff parameters for various elements and mixtures\n' +
                '# multiple entries can be added to this file, LAMMPS reads the\n' +
                '# ones it needs these entries are in LAMMPS metal units:\n' +
                '#   A,B = eV; lambda1,lambda2,lambda3 = 1/Angstroms; R,D = Angstroms\n' +
                '#   other quantities are unitless\n\n' +
                '# format of a single entry (one or more lines):\n' +
                '#   element 1, element 2, element 3,\n' +
                '#   m, gamma, lambda3, c, d, costheta0, n, beta, lambda2, B, R, D, lambda1, A\n\n' )
        output_file.write((element_name + ' ')*3)
        for parameter in individual:
            output_file.write(str(parameter) + ' ')

#   Read DFT training data
def read_training_data(path_tmp):
    training_data = {}
    with open(path_tmp, 'r') as input_file:
        all_lines = input_file.readlines()
        for line in all_lines:
            if '#' in line:
                continue
            else:
                line_split = line.split()
                training_data[line_split[0]] = list(map(float, line_split[1:]))
    return training_data

#   Calculate the Error sum of squares and decide whether or not to proceed the
#   evaluation
def calculate_sse_proceed(path_tmp, eval_label, training_data, criteria):
    #   Convert lists in the dictionary to numpy arrays
    training_data = {_: np.array(training_data[_]) for _ in training_data.keys()}
    criteria = {_: np.array(criteria[_]) for _ in criteria.keys()}
    sse_max = 999
    predictions = False
    #   Fetch data from log file
    with open(os.path.join(path_tmp, eval_label + '.log'), 'r') as output:
        all_lines = output.readlines()
        for line in all_lines:
            if 'Predictions' in line and 'print' not in line:
                predictions = np.array(list(map(float, line.split()[1:])))
    #   Return Error sum of squares and whether to proceed
    if predictions is False:
        return sse_max, False
    else:
        abs_error = np.absolute(predictions - training_data[eval_label])
        sse = np.sum(np.square(abs_error))
        sse = min(sse_max, sse)
        mae = np.mean(abs_error)
        if 'RMSD_COHESIVE' in eval_label:
            if np.all(abs_error < criteria[eval_label]):
                return sse, True
            else:
                return sse, False
        else:
            if mae < criteria[eval_label][0]:
                return sse, True
            else:
                return sse, False

# The evaluate function
def evaluate_single_element_Tersoff(individual, element_name=None, criteria=None):
    #   Set up working directory
    element_name = element_name[0]
    path_eval = os.path.join(os.path.abspath('.'),
                             'results_single_element_'+element_name, '')
    if 'results_single_element_' + element_name in os.listdir('.'):
        pass
    else:
        os.mkdir(path_eval)

    #   Create a force field file
    #   Deep copy to avoid changing the individual list
    ind = copy.deepcopy(individual)
    #   ind_all is the complete parameter set
    ind_all = [1] + ind
    create_fffile(path_eval, element_name, ind_all)

    #   Fetch the training data
    path_training = os.path.join(os.path.abspath('.'), 'training_data',
                                 element_name+'.txt')
    training_data = read_training_data(path_training)
    #   Set up evaluation
    #   eval_seq records the sequence of evaluation
    eval_seq = list(criteria.keys())
    fitness_step = 1000
    fitness_max = len(eval_seq) * fitness_step
    fitness_current = fitness_max

    #   Start the evaluation
    for i, eval_label in enumerate(eval_seq):
        #   Error Sum of Squares and whether the current evaluation succeeds
        #   For each evaluation function, if the convergence criteria is met,
        #   the current fitness is subtracted by fitness_step plus the current
        #   error sum of squares. If the convergence criteria is not met, the
        #   evaluation stops immediately and the current fitness is returned
        #   If all evaluation functions converge to the criteria, the final
        #   fitness value is returned

        #   Copy LAMMPS data to the result directory
        if eval_label is not 'RMSD_COHESIVE_SE2':
            continue
        lammps_file_path = os.path.join('.', 'lammps_input', eval_label, '')
        for _ in os.listdir(lammps_file_path):
            lammps_file_name = os.path.join(lammps_file_path, _)
            if os.path.isfile(lammps_file_name):
                shutil.copy(lammps_file_name, path_eval)

        os.chdir(path_eval)
        os.system('mpirun -np 1 lmp_mpi -log ' + eval_label + '.log -screen none -in ' + eval_label + '.in')
        os.chdir('..')

        sse, proceed = calculate_sse_proceed(path_eval, eval_label, training_data, criteria)

        if proceed:
            fitness_current = fitness_current - fitness_step + sse
            if i == len(eval_seq) - 1:
                return fitness_current,
            continue
        else:
            fitness_current = fitness_current + sse
            return fitness_current,

#           For testing purpose
#ind_test = [9.63864476911458, 4.649431802224302, 34.52946334483883,
            #41.497503218167566, -4.360361672979792, 17.059460886599563,
            #7.706376286903511, 3.7486736441243638, 549.8068421028667,
            #2.8742436240522284, 0.1941839547042561, 9.34690108401631,
            #1175.070667905]
#ELEMENT_NAME = ['Se']
#CRITERIA = {'RMSD_COHESIVE_SE2': [0.02, 0.1],
            #'DISSOCIATION_SE2': [0.2],
            #'RMSD_COHESIVE_SE3': [0.05, 0.2],
            #'RMSD_COHESIVE_SE6': [0.1, 0.05],
            #'RMSD_COHESIVE_SE8_RING': [0.1, 0.05],
            #'RMSD_COHESIVE_SE8_HELIX': [0.05, 0.05],
            #'RMSD_COHESIVE_SE8_LADDER': [0.2, 0.2]}

#result = evaluate_single_element_Tersoff(ind_test, element_name=ELEMENT_NAME,criteria=CRITERIA)
#print(result)
