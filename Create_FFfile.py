def create_fffile(element_name,individual):
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
    file_name = element_name + '.tersoff'
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
        
individual = [1, 0.349062129091, 0, 1.19864625442, 1.06060163186, -0.0396671926082,
              1, 1, 1.95864203232, 880.038350037, 3.41105925109, 0.376880167844, 
              2.93792248396, 4929.6960118]
    
create_fffile('Se',individual)      