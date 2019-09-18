def create_fffile(individual):
    #   With the input list individual that contains all the force field
    #   parameters, create a file with all parameters that is read by LAMMPS
    #   Individual has all the parameters in the following order:
    #   m (=1), gamma, lambda3, c, d, costheta0,
    #   n, beta, lambda2, B, R, D, lambda1 and A
