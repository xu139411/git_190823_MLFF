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

#   Calculate the RMSD and cohesive energy of Se2 dimer
def rmsd_cohesive_se2(path_tmp, *criteria):
    #   Calculate the room mean square displacement and cohesive energy of Se2
    #   From them, calculate the Error sum of squares (SSE).
    #   Copy files to the result directory
    lammps_file_path = os.path.join('.', 'lammps_input', 'rmsd_cohesive_se2')
    for _ in os.listdir(lammps_file_path):
        lammps_file_name = os.path.join(lammps_file_path, _)
        if os.path.isfile(lammps_file_name):
            shutil.copy(lammps_file_name, path_tmp)
    #   Run LAMMPS
    os.chdir(path_tmp)
    os.system("mpirun -np 1 lmp_mpi -log rmsd_cohesive_se2.log -screen none -in rmsd_cohesive_se2.in")
    os.chdir('..')
    #   Initialize the values to be False to handle LAMMPS errors
    rmsd, cohesive_eng = False, False
    sse_max = 999
    cohesive_eng_dft = -2.03
    criteria = [0.02, 0.1]
    #   Fetch data from log file
    with open(os.path.join(path_tmp, 'rmsd_cohesive_se2.log'), "r") as output:
        all_lines = output.readlines()
        for line in all_lines:
            if 'cohesive energy' in line and 'print' not in line:
                rmsd = float(line.split()[1])
                cohesive_eng = float(line.split()[4])
    #   Return Error sum of squares and whether to proceed
    if rmsd is False and cohesive_eng is False:
        return sse_max, False
    else:
        sse = min(sse_max, rmsd**2 + (cohesive_eng - cohesive_eng_dft)**2)
        if rmsd <= criteria[0] and abs(cohesive_eng -cohesive_eng_dft) <= criteria[1]:
            return sse, True
        else:
            return sse, False

# The evaluate function
def evaluate_single_element_Tersoff(individual, criteria_all):
    #   Set up working directory
    path_eval = os.path.join(os.path.abspath('.'), 'result_single_element', '')
    if 'results_single_element' in os.listdir('.'):
        pass
    else:
        os.mkdir(path_eval)

    #   Create a force field file
    #   Deep copy to avoid changing the individual list
    ind = copy.deepcopy(individual)
    #   ind_all is the complete parameter set
    ind_all = [1] + ind
    create_fffile(path_eval, element_name, ind_all)

    #   Set up evaluation. eval_seq stores the names of all evaluation
    #   functions
    eval_seq = [rmsd_cohesive_se2, dissociation_se2, rmsd_cohesive_se3,
                rmsd_cohesive_se6, rmsd_cohesive_se8ring,
                rmsd_cohesive_se8helix, stability_se2, stability_se6,
                stability_se8]
    fitness_step = 1000
    fitness_max = len(eval_seq) * fitness_step
    fitness_current = fitness_max

    #   Start the evaluation
    for i, eval in enumerate(eval_seq):
        #   Error Sum of Squares and whether the current evaluation succeeds
        #   For each evaluation function, if the convergence criteria is met,
        #   the current fitness is subtracted by fitness_step plus the current
        #   error sum of squares. If the convergence criteria is not met, the
        #   evaluation stops immediately and the current fitness is returned
        #   If all evaluation functions converge to the criteria, the final
        #   fitness value is returned
        sse, proceed = eval(path_eval, *criteria_all[i])
        if proceed:
            fitness_current = fitness_current - fitness_step + sse
            if i == len(eval_seq) - 1:
                return fitness_current
            continue
        else:
            fitness_current = fitness_current + sse
            return fitness_current
