#   Retrieve parameters for the genetic algorithm from the GA_control.txt file

def read_control():
    with open("GA_control.txt", 'r') as myfile:
        all_lines = myfile.readlines()
        for line in all_lines:
            line_split = line.split()
            if "gamma" in line_split:
                gamma_min = float(line_split[1])
                gamma_max = float(line_split[2])
                continue
            if "lambda3" in line_split:
                lambda3_min = float(line_split[1])
                lambda3_max = float(line_split[2])
                continue
            if "c" in line_split:
                c_min = float(line_split[1])
                c_max = float(line_split[2])
                continue
            if "d" in line_split:
                d_min = float(line_split[1])
                d_max = float(line_split[2])
                continue
            if "costheta0" in line_split:
                costheta0_min = float(line_split[1])
                costheta0_max = float(line_split[2])
                continue
            if "n" in line_split:
                n_min = float(line_split[1])
                n_max = float(line_split[2])
                continue
            if "beta" in line_split:
                beta_min = float(line_split[1])
                beta_max = float(line_split[2])
                continue
            if "lambda2" in line_split:
                lambda2_min = float(line_split[1])
                lambda2_max = float(line_split[2])
                continue
            if "B" in line_split:
                B_min = float(line_split[1])
                B_max = float(line_split[2])
                continue
            if "R" in line_split:
                R_min = float(line_split[1])
                R_max = float(line_split[2])
                continue
            if "D" in line_split:
                D_min = float(line_split[1])
                D_max = float(line_split[2])
                continue
            if "lambda1" in line_split:
                lambda1_min = float(line_split[1])
                lambda1_max = float(line_split[2])
                continue
            if "A" in line_split and "#" not in line_split:
                A_min = float(line_split[1])
                A_max = float(line_split[2])
                continue

            if "ELEMENT" in line_split and "#" not in line_split:
                ELEMENT = line_split[1]
                continue
            if "RANDOM_SEED" in line_split and "#" not in line_split:
                RANDOM_SEED = int(line_split[1])
                continue
            if "POP_SIZE" in line_split and "#" not in line_split:
                POP_SIZE = int(line_split[1])
                continue
            if "MAX_GENERATION" in line_split and "#" not in line_split:
                MAX_GENERATION = int(line_split[1])
                continue
            if "CXPB" in line_split and "#" not in line_split:
                CXPB = float(line_split[1])
                continue
            if "MUTPB" in line_split and "#" not in line_split:
                MUTPB = float(line_split[1])
                continue
    return gamma_min, gamma_max, lambda3_min, lambda3_max, c_min, c_max, d_min,\
           d_max, costheta0_min, costheta0_max, n_min, n_max, beta_min,\
           beta_max, lambda2_min, lambda2_max, B_min, B_max, R_min, R_max,\
           D_min, D_max, lambda1_min, lambda1_max, A_min, A_max, ELEMENT,\
           RANDOM_SEED, POP_SIZE, MAX_GENERATION, CXPB, MUTPB
