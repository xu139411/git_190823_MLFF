This package adopts Python genetic algorithm (GA) module DEAP to train Tersoff molecular dynamic force fields (FF) for Transition Metal Dichalcogenides (TMDC).

-------------------------------------------------------------------------------------

This package includes the following files and directories:

README.md                                        This file
Main.py                                          The main workflow for GA
GA_control.txt                                   Various parameters for GA
Create_FFfile.py                                 Contains the function create_fffile() that create Tersoff force field file to be read by LAMMPS
Evaluate_Single_Element_Tersoff.py               Contains the function evaluate_single_element() that conduct a hierarchical evaluation on the input individual (FF parameters), and return a fitness value.
Evaluate_Two_Elements_Tersoff.py                 Contains the function evaluate_two_elements() that conduct a hierarchical evaluation on the input individual (FF parameters), and return a fitness value.
training_set/                                    Folder that contains all the training data for the target TMDC.

-------------------------------------------------------------------------------------

Contact:
Xu Zhang (xuzhang2017@u.northwestern.edu)
Hoang T. Nguyen (hoangnguyen2015@u.northwestern.edu)
