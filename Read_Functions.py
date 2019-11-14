#   Retrieve parameters for the genetic algorithm from the GA_control.txt file
# Standard library imports
import configparser
import os

#   Read FF parameter ranges, hyperparameters of GA and convergence criteria
#   into dictionaries
def read_control_config():
    element_name = []
    parameters_ff_range = {}
    parameters_GA = {}
    criteria = {}
    optimized_parameter_W = False
    optimized_parameter_Se = False
    switch = 0
    path_tmp = os.path.join(os.path.abspath('.'), 'GA_control.txt')
    with open(path_tmp, 'r') as input_file:
        all_lines = input_file.readlines()
        #   Get the element names and the corresponding section names
        for line in all_lines:
            if '[ Element' in line:
                if len(line.split()) == 4:
                    element_name = [line.split()[2]]
                    kw_switch1 = '[ FF Parameters Range '+element_name[0]+' ]\n'
                    kw_switch2 = '[ Parameters for GA Work Flow '+element_name[0]+' ]\n'
                    kw_switch3 = '[ Convergence Criteria '+element_name[0]+' ]\n'
                    kw_switch4 = False
                    kw_switch5 = False
                elif len(line.split()) == 5:
                    element_name = [line.split()[2], line.split()[3]]
                    kw_switch1 = '[ FF Parameters Range '+element_name[0]+' '+element_name[1]+' ]\n'
                    kw_switch2 = '[ Parameters for GA Work Flow '+element_name[0]+' '+element_name[1]+' ]\n'
                    kw_switch3 = '[ Convergence Criteria '+element_name[0]+' '+element_name[1]+' ]\n'
                    kw_switch4 = '[ Optimized Parameters '+element_name[0]+' '+element_name[0]+' ]\n'
                    kw_switch5 = '[ Optimized Parameters '+element_name[1]+' '+element_name[1]+' ]\n'
        #   Get the parameters
        for line in all_lines:
            if '#' in line or line == '\n':
                switch = 0
                continue
            elif line == '[ END ]\n':
                switch = 0
                continue
            elif line == kw_switch1:
                switch = 1
                continue
            elif line == kw_switch2:
                switch = 2
                continue
            elif line == kw_switch3:
                switch = 3
                continue
            elif kw_switch4 is not False and line == kw_switch4:
                switch = 4
                continue
            elif kw_switch5 is not False and line == kw_switch5:
                switch = 5
                continue
            line_split = line.split()
            if switch == 1:
                parameters_ff_range[line_split[0]] = list(map(float, line_split[1:]))
            if switch == 2:
                if line_split[0] == 'CXPB' or line_split[0] == 'MUTPB':
                    parameters_GA[line_split[0]] = float(line_split[1])
                else:
                    parameters_GA[line_split[0]] = int(line_split[1])
            if switch == 3:
                criteria[line_split[0]] = list(map(float, line_split[1:]))
            if switch == 4:
                optimized_parameter_W = list(map(float, line_split[:]))
            if switch == 5:
                optimized_parameter_Se = list(map(float, line_split[:]))
    return element_name, parameters_ff_range, parameters_GA, criteria,\
           [optimized_parameter_W, optimized_parameter_Se]

#   Read DFT training data into a dictionary
def read_training_data(element_name):
    training_data = {}
    # element_name can be ['Se'] or ['W', 'Se']
    if len(element_name) == 1:
        element_name = element_name[0]
    else:
        element_name = element_name[0] + element_name[1]
    path_tmp = os.path.join(os.path.abspath('.'), 'training_data',
                            element_name + '.txt')
    with open(path_tmp, 'r') as input_file:
        all_lines = input_file.readlines()
        for line in all_lines:
            if '#' in line or line == '\n':
                continue
            else:
                line_split = line.split()
                training_data[line_split[0]] = list(map(float, line_split[1:]))
    return training_data
