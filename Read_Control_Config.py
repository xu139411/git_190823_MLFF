#   Retrieve parameters for the genetic algorithm from the GA_control.txt file
# Standard library imports
import configparser

def read_control_config():
    config = configparser.ConfigParser()
    config.read('GA_control.ini')

    #   Force field parameters
    gamma_min = float(config.get('FF Parameters', 'gamma').split()[0])
    gamma_max = float(config.get('FF Parameters', 'gamma').split()[1])
    lambda3_min = float(config.get('FF Parameters', 'lambda3').split()[0])
    lambda3_max = float(config.get('FF Parameters', 'lambda3').split()[1])
    c_min = float(config.get('FF Parameters', 'c').split()[0])
    c_max = float(config.get('FF Parameters', 'c').split()[1])
    d_min = float(config.get('FF Parameters', 'd').split()[0])
    d_max = float(config.get('FF Parameters', 'd').split()[1])
    costheta0_min = float(config.get('FF Parameters', 'costheta0').split()[0])
    costheta0_max = float(config.get('FF Parameters', 'costheta0').split()[1])
    n_min = float(config.get('FF Parameters', 'n').split()[0])
    n_max = float(config.get('FF Parameters', 'n').split()[1])
    beta_min = float(config.get('FF Parameters', 'beta').split()[0])
    beta_max = float(config.get('FF Parameters', 'beta').split()[1])
    lambda2_min = float(config.get('FF Parameters', 'lambda2').split()[0])
    lambda2_max = float(config.get('FF Parameters', 'lambda2').split()[1])
    B_min = float(config.get('FF Parameters', 'B').split()[0])
    B_max = float(config.get('FF Parameters', 'B').split()[1])
    R_min = float(config.get('FF Parameters', 'R').split()[0])
    R_max = float(config.get('FF Parameters', 'R').split()[1])
    D_min = float(config.get('FF Parameters', 'D_D').split()[0])
    D_max = float(config.get('FF Parameters', 'D_D').split()[1])
    lambda1_min = float(config.get('FF Parameters', 'lambda1').split()[0])
    lambda1_max = float(config.get('FF Parameters', 'lambda1').split()[1])
    A_min = float(config.get('FF Parameters', 'A').split()[0])
    A_max = float(config.get('FF Parameters', 'A').split()[1])

    #   Parameters for GA work flow
    ELEMENT_NAME = config.get('Parameters for GA Work Flow', 'ELEMENT_NAME').split()
    RANDOM_SEED = int(config.get('Parameters for GA Work Flow', 'RANDOM_SEED'))
    POP_SIZE = int(config.get('Parameters for GA Work Flow', 'POP_SIZE'))
    MAX_GEN = int(config.get('Parameters for GA Work Flow', 'MAX_GENERATION'))
    CXPB = float(config.get('Parameters for GA Work Flow', 'CXPB'))
    MUTPB = float(config.get('Parameters for GA Work Flow', 'MUTPB'))

    #   Convergence criteria for evaluation
    RMSD_COHESIVE_SE2 = list(map(float, config.get('Convergence criteria Se', 'RMSD_COHESIVE_SE2').split()))
    DISSOCIATION_SE2 = [float(config.get('Convergence criteria Se', 'DISSOCIATION_SE2'))]
    RMSD_COHESIVE_SE3 = list(map(float, config.get('Convergence criteria Se', 'RMSD_COHESIVE_SE3').split()))
    RMSD_COHESIVE_SE6 = list(map(float, config.get('Convergence criteria Se', 'RMSD_COHESIVE_SE6').split()))
    RMSD_COHESIVE_SE8_RING = list(map(float, config.get('Convergence criteria Se', 'RMSD_COHESIVE_SE8_RING').split()))
    RMSD_COHESIVE_SE8_HELIX = list(map(float, config.get('Convergence criteria Se', 'RMSD_COHESIVE_SE8_HELIX').split()))
    RMSD_COHESIVE_SE8_LADDER = list(map(float, config.get('Convergence criteria Se', 'RMSD_COHESIVE_SE8_LADDER').split()))

    indiv_low = [gamma_min, lambda3_min, c_min, d_min, costheta0_min,
                 n_min, beta_min, lambda2_min, B_min, R_min, D_min,
                 lambda1_min, A_min]
    indiv_up = [gamma_max, lambda3_max, c_max, d_max, costheta0_max,
                n_max, beta_max, lambda2_max, B_max, R_max, D_max,
                lambda1_max, A_max]
    parameters_GA = {'ELEMENT_NAME': ELEMENT_NAME,
                     'RANDOM_SEED': RANDOM_SEED,
                     'POP_SIZE': POP_SIZE,
                     'MAX_GEN': MAX_GEN,
                     'CXPB': CXPB,
                     'MUTPB': MUTPB}
    CRITERIA = {'RMSD_COHESIVE_SE2': RMSD_COHESIVE_SE2,
                'DISSOCIATION_SE2': DISSOCIATION_SE2,
                'RMSD_COHESIVE_SE3': RMSD_COHESIVE_SE3,
                'RMSD_COHESIVE_SE6': RMSD_COHESIVE_SE6,
                'RMSD_COHESIVE_SE8_RING': RMSD_COHESIVE_SE8_RING,
                'RMSD_COHESIVE_SE8_HELIX': RMSD_COHESIVE_SE8_HELIX,
                'RMSD_COHESIVE_SE8_LADDER': RMSD_COHESIVE_SE8_LADDER}

    return indiv_low, indiv_up, parameters_GA, CRITERIA
