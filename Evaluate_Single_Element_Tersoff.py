#   Evaluation part of the genetic algorithm that finds the parameters of
#   Tersoff force field for Transition Metal Dichalcogenides (TMDCs).
#   This file evaluates the single-element calculations

# Standard library imports
import os
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
                    elif 'DISSOCIATION' in eval_label:
                        predictions.append(float(line.split()[1]))
                    elif 'TRANSITION_PATH' in eval_label:
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
    elif len(predictions) != np.shape(training_data[eval_label])[0] and 'DISSOCIATION' in eval_label:
        return sse_max, False
    elif len(predictions) != np.shape(training_data[eval_label])[0] and 'TRANSITION_PATH' in eval_label:
        return sse_max, False
    else:
        if 'TRANSITION_PATH' in eval_label:
            predictions = np.array(predictions)
            predictions = predictions[1:] - predictions[0]
            abs_error = np.absolute(predictions - training_data[eval_label][1:] + training_data[eval_label][0])
            max_abs_error = np.max(abs_error)
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
        elif 'DISSOCIATION' in eval_label:
            if mae < criteria[eval_label][0]:
                return sse, True
            else:
                return sse, False
        elif 'TRANSITION_PATH' in eval_label:
            if max_abs_error < criteria[eval_label][0]:
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
def evaluate_single_element_Tersoff(individual, element_name=None,
                                    training_data=None, criteria=None,
                                    fixed_value=None):
    #   Set up working directory
    job_id = id(scoop.worker)
    element_name = element_name[0]
    path_eval = os.path.join(PATH_ROOT, 'results_'+element_name, str(job_id), '')
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
    create_fffile(path_eval, element_name, ind_all)

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

#   For testing purpose
if __name__ == '__main__':
    ind_henry = [0.349062129091, 0, 1.19864625442, 1.06060163186, -0.0396671926082, 1, 1, 1.95864203232, 880.038350037, 3.41105925109, 0.376880167844, 2.93792248396, 4929.6960118]
    ind_trial = [3.9349214091113076, 0.4222861426544582, 7.8484525631732325, 8.114512113497252, -3.7615544430289107, 0.6693918364032599, 0.47181144590398555, 1.7062809592892105, 272.488527874936, 3.5221542542852435, 0.30344731963196897, 3.2087440466792434, 1600.0756347740457]
    ELEMENT_NAME = ['Se']
    CRITERIA = {'TRANSITION_PATH_SE3': [0.4]}
                #'RMSD_COHESIVE_SE2': [0.02, 0.1],
                #'DISSOCIATION_SE2': [0.2],
                #'RMSD_COHESIVE_SE3': [0.05, 0.25],
                #'RMSD_COHESIVE_SE6': [0.1, 0.05],
                #'RMSD_COHESIVE_SE8_RING': [0.1, 0.05],
                #'RMSD_COHESIVE_SE8_HELIX': [0.05, 0.06],
                #'RMSD_COHESIVE_SE8_LADDER': [0.6, 0.2],
                #'MD_SE6_LOWT': [0.3],
                #'MD_SE6_HIGHT': [0.8],
                #'MD_SE8_RING_LOWT': [0.6],
                #'MD_SE8_RING_HIGHT': [0.8]}
    TRAINING_DATA = {'TRANSITION_PATH_SE3': [-10.400714, -9.602875, -8.601496, -7.214531, -6.092061, -5.233269, -5.225650, -5.733959, -6.175055, -6.926634, -7.572948]}
                 #'RMSD_COHESIVE_SE2': [0.0, -2.029814],
                 #'DISSOCIATION_SE2': [8.138150, 1.948148, -0.696524, -1.804186, -2.029814, -1.894431, -1.608573, -1.278067, -0.954593, -0.661746, -0.389450],
                 #'RMSD_COHESIVE_SE3': [0.0, -2.170],
                 #'RMSD_COHESIVE_SE6': [0.0, -2.521578],
                 #'RMSD_COHESIVE_SE8_RING': [0.0, -2.585449],
                 #'RMSD_COHESIVE_SE8_HELIX': [0.0, -2.380094],
                 #'RMSD_COHESIVE_SE8_LADDER': [0.0, -2.346071],
                 #'MD_SE6_LOWT': [0.0],
                 #'MD_SE6_HIGHT': [4.0],
                 #'MD_SE8_RING_LOWT': [0.0],
                 #'MD_SE8_RING_HIGHT': [4.0]}
    fixed_para = {}
    result = evaluate_single_element_Tersoff(ind_trial, element_name=ELEMENT_NAME,training_data=TRAINING_DATA, criteria=CRITERIA, fixed_value=fixed_para)
    print(result)
