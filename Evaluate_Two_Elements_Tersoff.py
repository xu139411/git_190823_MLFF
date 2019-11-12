#   Evaluation part of the genetic algorithm that finds the parameters of
#   Tersoff force field for Transition Metal Dichalcogenides (TMDCs).
#   This file evaluates the single-element calculations

# Standard library imports
import os
from itertools import product
PATH_ROOT = os.path.abspath('.')
import shutil
import copy
import time
import subprocess
import logging
warning_file = os.path.join(PATH_ROOT, 'warning.txt')
logging.basicConfig(filename=warning_file, level=logging.WARNING)
# Third party imports: lammps
import numpy as np
import scoop
# Local library imports

#   Create a Tersoff force field file
def create_fffile(path_tmp, element_name, individual, optimized_parameters):
    #   With the input list individual that contains all the force field
    #   parameters, create a file to be read by LAMMPS
    #   Individual value has all the parameters in the following order:
    #   [m (=1), gamma, lambda3, c, d, costheta0,
    #   n, beta, lambda2, B, R, D, lambda1, A]
    '''
    input:
        - element_name - str, name of the element
        - individual - list, Tersoff parameters of the mentioned element
        - optimized_parameters_first - list, Tersoff parameters of the first
          element
        - optimized_parameters_second - list, Tersoff parameters of the second
          element
    output:
        - a text file containing parameters
    '''
    file_name = os.path.join(path_tmp, element_name[0]+element_name[1]+'.tersoff')
    with open(file_name, 'w') as output_file:
        output_file.write('# Tersoff parameters for various elements and mixtures\n' +
                '# multiple entries can be added to this file, LAMMPS reads the\n' +
                '# ones it needs these entries are in LAMMPS metal units:\n' +
                '#   A,B = eV; lambda1,lambda2,lambda3 = 1/Angstroms; R,D = Angstroms\n' +
                '#   other quantities are unitless\n\n' +
                '# format of a single entry (one or more lines):\n' +
                '#   element 1, element 2, element 3,\n' +
                '#   m, gamma, lambda3, c, d, costheta0, n, beta, lambda2, B, R, D, lambda1, A\n\n' )
        # product returns the Cartesian product in the form of tuple
        for _ in product(element_name, repeat=3):
            element_seq = list(_)
            element_uniq = list(set(element_seq))
            output_file.write(' '.join(element_seq) + ' ')
            if len(element_uniq) == 1:
                if element_uniq[0] == element_name[0]:
                    output_file.write(' '.join(list(map(str, optimized_parameters[0]))))
                    output_file.write('\n')
                else:
                    output_file.write(' '.join(list(map(str, optimized_parameters[1]))))
                    output_file.write('\n')
            else:
                if element_seq[1] == element_seq[2]:
                    output_file.write(' '.join(list(map(str, individual))))
                    output_file.write('\n')
                else:
                    if element_seq[0] == element_seq[2] and element_seq[0] == element_name[0]:
                        para = optimized_parameters[0][0:6] + [0, 0, 0, 0] + optimized_parameters[0][10:12] + [0, 0]
                    elif element_seq[0] == element_seq[2] and element_seq[0] == element_name[1]:
                        para = optimized_parameters[1][0:6] + [0, 0, 0, 0] + optimized_parameters[1][10:12] + [0, 0]
                    else:
                        para = individual[0:6] + [0, 0, 0, 0] + individual[10:12] + [0, 0]
                    output_file.write(' '.join(list(map(str, para))))
                    output_file.write('\n')

#   Calculate the Error sum of squares and decide whether or not to proceed the
#   evaluation
def calculate_sse_proceed(path_tmp, job_id, eval_label, training_data, criteria):
    #   Convert lists in the dictionary to numpy arrays
    training_data = {_: np.array(training_data[_]) for _ in training_data.keys()}
    criteria = {_: np.array(criteria[_]) for _ in criteria.keys()}
    sse_max = 999
    predictions = []
    #   Fetch data from log file
    log_file_path = os.path.join(path_tmp, eval_label + '.log')
    while not os.path.exists(log_file_path):
        logging.warning('%s cannot read LAMMPS log file', str(job_id))
        time.sleep(0.1)
    if os.path.isfile(log_file_path):
        with open(log_file_path, 'r') as output:
            all_lines = output.readlines()
            for line in all_lines:
                if 'Predictions' in line and 'print' not in line:
                    if 'RMSD_COHESIVE' in eval_label:
                        predictions = list(map(float, line.split()[1:]))
                    elif 'EOS' in eval_label:
                        predictions.append(float(line.split()[1]))
                    elif 'MD' in eval_label:
                        predictions = list(map(float, line.split()[1:]))
                    else:
                        raise ValueError(str(job_id), 'cannot retrieve the target parameters for optimization\n')
    else:
        raise OSError(str(job_id), 'cannot open LAMMPS log file\n')
    #   Return Error sum of squares and whether to proceed
    if len(predictions) == 0:
        return sse_max, False
    elif len(predictions) != np.shape(training_data[eval_label])[0] and 'EOS' in eval_label:
        return sse_max, False
    else:
        if 'EOS' in eval_label:
            predictions = np.array(predictions)
            predictions = predictions - np.amin(predictions)
            abs_error = np.absolute(predictions - training_data[eval_label] + np.amin(training_data[eval_label]))
        else:
            predictions = np.array(predictions)
            abs_error = np.absolute(predictions - training_data[eval_label])
        sse = np.sum(np.square(abs_error))
        sse = min(sse_max, sse)
        mae = np.mean(abs_error)
        if 'RMSD_COHESIVE' in eval_label:
            if np.all(abs_error < criteria[eval_label]):
                return sse, True
            else:
                return sse, False
        elif 'EOS' in eval_label:
            if mae < criteria[eval_label][0]:
                return sse, True
            else:
                return sse, False
        elif 'MD' in eval_label and 'LOWT' in eval_label:
            # RMSD at low T is smaller than the criteria
            if predictions[0] <= criteria[eval_label][0]:
                return sse, True
            else:
                return sse, False
        elif 'MD' in eval_label and 'HIGHT' in eval_label:
            # RMSD at high T is larger than the criteria
            if predictions[0] >= criteria[eval_label][0]:
                return sse, True
            else:
                return sse, False
        else:
            raise ValueError(str(job_id), 'cannot retrieve the target parameters for optimization\n')

# The evaluate function
def evaluate_two_elements_Tersoff(individual, element_name=None,
                                    training_data=None, criteria=None,
                                    fixed_value=None,
                                    optimized_parameters=None):
    #   Set up working directory
    job_id = id(scoop.worker)
    path_eval = os.path.join(PATH_ROOT, 'results_'+element_name[0]+element_name[1], str(job_id), '')
    try:
        if os.path.isdir(path_eval):
            shutil.rmtree(path_eval)
            os.makedirs(path_eval)
        else:
            os.makedirs(path_eval)
    except:
        raise OSError(str(job_id), ' cannot make the directory\n')
    #   Create a force field file
    #   Deep copy to avoid changing the individual list
    ind = copy.deepcopy(individual)
    #   ind_all is the parameter set including the fixed values
    for i in list(fixed_value.keys()):
        ind.insert(i, fixed_value[i])
    ind_all = [1] + ind
    create_fffile(path_eval, element_name, ind_all, optimized_parameters)

    #   Set up evaluation
    #   eval_seq records the sequence of evaluation
    eval_seq = list(criteria.keys())
    fitness_step = 1000
    fitness_max = len(eval_seq) * fitness_step
    fitness_current = fitness_max
    proceed = False

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
        lammps_file_path = os.path.join(PATH_ROOT, 'lammps_input', eval_label, '')
        for _ in os.listdir(lammps_file_path):
            lammps_file_name = os.path.join(lammps_file_path, _)
            if os.path.isfile(lammps_file_name):
                shutil.copy(lammps_file_name, path_eval)

        os.chdir(path_eval)
        #subprocess.call(['lmp_mpi', '-log', eval_label+'.log', '-screen', 'none', '-in', eval_label+'.in'])
        os.system('lmp_mpi -log ' + eval_label + '.log -screen none -in ' + eval_label + '.in')
        time.sleep(0.1)
        #os.system('lmp_serial -log ' + eval_label + '.log -screen none -in ' + eval_label + '.in')
        os.chdir(PATH_ROOT)

        sse, proceed = calculate_sse_proceed(path_eval, job_id, eval_label, training_data, criteria)

        if proceed:
            fitness_current = fitness_current - fitness_step + sse
            if i == len(eval_seq) - 1:
                return fitness_current,
            continue
        else:
            fitness_current = fitness_current + sse
            return fitness_current,

    #For testing purpose
#path_tmp = PATH_ROOT
#element_name = ['W', 'Se']
#ndividual = [1, 1.83073271913, -0.0021970964964, 1.36806361463, 0.629172073364, 0.522704731142, 1.00558350622, 0.0792382145959, 1.34915995464, 175.933838187, 3.26742070742, 0.763969153161, 3.21479473302, 4350.90479176]
#optimized_parameters = [[1, 0.00188227, 0.45876, 2.14969, 0.17126, 0.2778, 1, 1, 1.411246, 306.49968, 3.5, 0.3, 2.719584, 3401.474424],
                        #[1, 0.349062129091, 0, 1.19864625442, 1.06060163186, -0.0396671926082, 1, 1, 1.95864203232, 880.038350037, 3.41105925109, 0.376880167844, 2.93792248396, 4929.6960118]]
#create_fffile(path_tmp, element_name, individual, optimized_parameters)
#criteria = {'RMSD_COHESIVE_WSE2': [0.07, 1.0],
            #'EOS_WSE2': [0.007],
            #'MD_WSE2_LOWT': [0.5]}
#training_data = {'RMSD_COHESIVE_WSE2': [0.0, -5.24438748],
                 #'EOS_WSE2': [-15.697756, -15.710685, -15.720611, -15.727517, -15.731777, -15.73316243, -15.731834, -15.727839, -15.721332, -15.712275, -15.700761],
                 #'MD_WSE2_LOWT': [0.0]}
#result =  evaluate_two_elements_Tersoff(individual, element_name=element_name, training_data=training_data, criteria=criteria, optimized_parameters=optimized_parameters)
#print(result)
