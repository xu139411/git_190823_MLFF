This package adopts Python genetic algorithm (GA) module DEAP to train Tersoff molecular dynamic force fields (FF) for Transition Metal Dichalcogenides (TMDC).

-------------------------------------------------------------------------------------

This package includes the following files and directories:

README.md                                        This file\n
Main.py                                          The main workflow for GA'\n'
GA_control.txt                                   Various parameters for GA\n
Create_FFfile.py                                 Contains the function create_fffile() that create Tersoff force field file to be read by LAMMPS\n
Evaluate_Single_Element_Tersoff.py               Contains the function evaluate_single_element() that conduct a hierarchical evaluation on the input individual (FF parameters), and return a fitness value.\n
Evaluate_Two_Elements_Tersoff.py                 Contains the function evaluate_two_elements() that conduct a hierarchical evaluation on the input individual (FF parameters), and return a fitness value.\n
training_set/                                    Folder that contains all the training data for the target TMDC.\n

-------------------------------------------------------------------------------------

Contact:
Xu Zhang (xuzhang2017@u.northwestern.edu)\n
Hoang T. Nguyen (hoangnguyen2015@u.northwestern.edu)
