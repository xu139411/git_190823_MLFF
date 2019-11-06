#   Retrieve parameters for the genetic algorithm from the GA_control.txt file
# Standard library imports
import configparser
import os

#   Read FF parameter ranges, hyperparameters of GA and convergence criteria
#   into dictionaries
def read_control_config():
    parameters_ff_range = {}
    parameters_GA = {}
    criteria = {}
    switch = 0
    path_tmp = os.path.join(os.path.abspath('.'), 'GA_control.txt')
    with open(path_tmp, 'r') as input_file:
        all_lines = input_file.readlines()
        for line in all_lines:
            if '#' in line or line == '\n':
                continue
            else:
                if '[FF Parameters]' in line:
                    switch = 1
                    continue
                elif '[Parameters for GA Work Flow]' in line:
                    switch = 2
                    continue
                elif '[Convergence criteria Se]' in line:
                    switch = 3
                    continue

                line_split = line.split()
                if switch == 1:
                    parameters_ff_range[line_split[0]] = list(map(float, line_split[1:]))
                if switch == 2:
                    if line_split[0] == 'ELEMENT_NAME':
                        parameters_GA[line_split[0]] = line_split[1:] # [1:] will create a list instead of a variable
                    elif line_split[0] == 'CXPB' or line_split[0] == 'MUTPB':
                        parameters_GA[line_split[0]] = float(line_split[1])
                    else:
                        parameters_GA[line_split[0]] = int(line_split[1])
                if switch == 3:
                    criteria[line_split[0]] = list(map(float, line_split[1:]))
    return parameters_ff_range, parameters_GA, criteria

#   Read DFT training data into a dictionary
def read_training_data(element_name):
    training_data = {}
    #   element_name can be ['Se'] or ['W', 'Se']
    if len(element_name) == 1:
        element_name = element_name[0]
    else:
        element_name = element_name[0] + element_name[1]
    path_tmp = os.path.join(os.path.abspath('.'), 'training_data',
                            element_name + '.txt')
    with open(path_tmp, 'r') as input_file:
        all_lines = input_file.readlines()
        for line in all_lines:
            if '#' in line:
                continue
            else:
                line_split = line.split()
                training_data[line_split[0]] = list(map(float, line_split[1:]))
    return training_data
