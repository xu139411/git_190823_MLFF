#! ~/anaconda2/bin/python

# Uses LAMMPS python library to predict
# properties for a given set of potential
# parameters. The weighted sum of errors in
# prediction are then calculated

#In module mode, passes cost function from phonon dispersion back to objective.py.
#In stand alone mode, generates phonon dispersion with arguments as frequency file, lattice file which is subsequently relaxed and band path

# In the previous version, type assignment can go wrong if the LAMMPS
# data ordering differs from the original atomid order during the
# course of minimization. The order should be ensured when the
# positions are extracted after minimization. This is done in the
# version - KsK.

import sys
import numpy as np
from lammps import lammps
from phonopy import Phonopy
from phonopy.structure.atoms import Atoms as PhonopyAtoms

def PHON(qfile,latfile,NRepx,NRepy,NRepz,elem,mass,
         pathfile=False, savedat=False,
         headerfile=False,
         fffile=False,
         savedos=False,saveCv=False):

    #lmp = lammps(cmdargs=['-sc','none','-log','lammps.log'])
    lmp = lammps(cmdargs=['-log', 'lammps.log', '-screen', 'none'])
    lmp.file(headerfile)
    lmp.file(latfile)
    lmp.file(fffile)

    lmp.command("compute frac all property/atom xs ys zs")
    lmp.command("compute typ all property/atom type")
    lmp.command("compute atids all property/atom id")
    lmp.command("fix 1 all box/relax x 0.0 y 0.0 couple none vmax 0.001")
    lmp.command("minimize 1e-6 1e-8 200 2000")
    lmp.command("unfix 1")

    Natoms = lmp.get_natoms()

    BoxBounds = np.array([lmp.extract_global("boxxlo",1),
                          lmp.extract_global("boxxhi",1),
                          lmp.extract_global("boxylo",1),
                          lmp.extract_global("boxyhi",1),
                          lmp.extract_global("boxzlo",1),
                          lmp.extract_global("boxzhi",1),
                          lmp.extract_global("xy",1),
                          lmp.extract_global("xz",1),
                          lmp.extract_global("yz",1)])

    Lattice = np.array([BoxBounds[1]-BoxBounds[0],
                        BoxBounds[3]-BoxBounds[2],
                        BoxBounds[5]-BoxBounds[4],
                        BoxBounds[6], BoxBounds[7], BoxBounds[8]])

    fcoords = lmp.extract_compute("frac",1,2)
    atidtmp = lmp.extract_compute("atids",1,1)
    fpos = np.zeros((Natoms,3))
    for i in range(Natoms):
        for j in range(3):
            fpos[int(atidtmp[i]-1)][j] += fcoords[i][j]

    typetmp = lmp.extract_compute("typ",1,1)
    typeid = np.zeros(Natoms,dtype='int32')
    for i in range(Natoms):
        typeid[int(atidtmp[i]-1)] += typetmp[i]

    typeid_0 = np.copy(typeid) # fix: multiple atomtypes for same element

    lmp.close()

    sym=[elem[typeid[i]-1] for i in range(Natoms)]

    atmass = np.zeros(Natoms)
    for i in range(Natoms):
        atmass[i] = mass[typeid[i]-1]

    PhononSingleCell = PhonopyAtoms(
                        symbols=sym, scaled_positions=fpos, masses=atmass,
                        cell=np.array([[Lattice[0],0.0,0.0],
                                      [Lattice[3],Lattice[1],0.0],
                                      [Lattice[4],Lattice[5],Lattice[2]]])
                       )
    phonon = Phonopy(PhononSingleCell,[[NRepx,0,0],[0,NRepy,0],[0,0,NRepz]])
    phonon.generate_displacements(distance=0.01)
    PhononSupercells = phonon.get_supercells_with_displacements()
    NSupercells = len(PhononSupercells)


    # Run LAMMPS calculation to get forces for each supercell
    sets_of_forces = []
    for PhonStruct in range(NSupercells):
        lmp = lammps(cmdargs=['-sc','none','-log','none'])
        lmp.file(headerfile)
        xhi = PhononSupercells[PhonStruct].get_cell()[0][0]
        yhi = PhononSupercells[PhonStruct].get_cell()[1][1]
        zhi = PhononSupercells[PhonStruct].get_cell()[2][2]
        xy  = PhononSupercells[PhonStruct].get_cell()[1][0]
        xz  = PhononSupercells[PhonStruct].get_cell()[2][0]
        yz  = PhononSupercells[PhonStruct].get_cell()[2][1]

        tmp = []
        tmp.append('lattice  custom 1.000000 spacing '+
                   '%12.6f %12.6f %12.6f a1 %12.6f %12.6f %12.6f '
                    %(xhi,yhi,zhi,xhi,0.0,0.0))
        tmp.append('\t\t a2 %12.6f %12.6f %12.6f '%(xy,yhi,0.0))
        tmp.append('\t\t a3 %12.6f %12.6f %12.6f '%(xz,yz,zhi))
        Nsites = len(PhononSupercells[PhonStruct].get_positions())
        sym = PhononSupercells[PhonStruct].get_chemical_symbols()
        try:
            typeid = np.zeros(Nsites,dtype='int32')
            for i in range(len(sym)):
                typeid[i] = np.where(np.asarray(elem) == sym[i])[0] + 1
        except:
            # fix: multiple atomtypes for same element
            Nreplicate = int(len(sym)/len(typeid_0))
            typeid = np.repeat(typeid_0,Nreplicate)
        for sites in range(Nsites):
            sxx = PhononSupercells[PhonStruct].get_scaled_positions()[sites][0]
            syy = PhononSupercells[PhonStruct].get_scaled_positions()[sites][1]
            szz = PhononSupercells[PhonStruct].get_scaled_positions()[sites][2]
            if round(sxx,6) < 0.0:
                sxx = sxx + 1.0
            if round(syy,6) < 0.0:
                syy = syy + 1.0
            if round(szz,6) < 0.0:
                szz = szz + 1.0
            if round(sxx,6) >= 1.0:
                sxx = sxx - 1.0
            if round(syy,6) >= 1.0:
                syy = syy - 1.0
            if round(szz,6) >= 1.0:
                szz = szz - 1.0
            tmp.append(" basis %12.6f %12.6f %12.6f" %(sxx,syy,szz))
        lmp.command(''.join(tmp))
        lmp.command('region box prism 0 1 0 1 0 1    '+
                    '%12.6f     %12.6f     %12.6f units lattice'
                    %(xy/xhi,xz/xhi,yz/yhi))
        lmp.command("create_box  %d box" %(len(elem)))

        tmp = []
        tmp.append("create_atoms %d box "%(len(elem)))
        for sites in range(Nsites-1):
            tmp.append(" basis %d %d " %(sites+1, typeid[sites]))
        tmp.append(" basis %d %d " %(Nsites, typeid[Nsites-1]))
        lmp.command(''.join(tmp))

        for atm in range(len(mass)):
            lmp.command("mass %d   %.3f" %(atm+1, mass[atm]))

        lmp.file(fffile)
        lmp.command("compute forces all property/atom fx fy fz")
        lmp.command("run 0")

        force_lammps = lmp.extract_compute("forces",1,2)
        forces = np.zeros((Nsites,3))

        for i in range(Nsites):
            for j in range(3):
                forces[i][j] += force_lammps[i][j]
        sets_of_forces.append(forces)

        lmp.close()

    #phonon.set_forces(sets_of_forces)
    phonon.produce_force_constants(forces=sets_of_forces)
    #phonon.get_force_constants()
    #phonon.set_post_process()


    # Read q points, and target frequencies
    qlist = np.loadtxt(qfile,comments='#')
    band = list(qlist[:,:3])

    # Band Structure
    bands = []
    bands.append(band)
    phonon.set_band_structure(bands)
    qpoints, distances, frequencies, eigvecs = phonon.get_band_structure()

    if savedat:
        f = open(savedat,'w')
        f.write('# qpoints (col 1-3), distances (col 4), frequencies\n')
        f.close()
        f = open(savedat,'ab')
        np.savetxt(f,np.c_[qpoints[0],distances[0],frequencies[0]],fmt='%.8f')
        f.close()

    # plot Band Structure
    if pathfile:
        bands = []
        NBANDPOINTS = qlist.shape[0] # nbandpts = points in target
        ifile=open(pathfile, 'r')
        for line in ifile:
            tmp=line.split()
            q_start = np.array([float(tmp[0]),float(tmp[1]),float(tmp[2])])
            line=ifile.next()
            tmp=line.split()
            q_end = np.array([float(tmp[0]),float(tmp[1]),float(tmp[2])])
            band = []
            for i in range(NBANDPOINTS):
                band.append(q_start + (q_end - q_start) * i /(NBANDPOINTS-1))
            bands.append(band)
        ifile.close()

        phonon.set_band_structure(bands)
        qpoints, distances, frequencies, eigvecs = phonon.get_band_structure()
        #print 'CONVERSION FACTOR %f' %phonon.get_unit_conversion_factor()
        phonon.write_yaml_band_structure()
        phonon.plot_band_structure().show()

    # save dos
    if savedos:
        mesh = [20, 20, 20]
        phonon.set_mesh(mesh)
        qpoints, weights, frequencies, eigvecs = phonon.get_mesh()
#        phonon.set_mesh([20, 20, 20], is_eigenvectors=True)
        phonon.set_total_DOS()

        phonon.plot_total_DOS().show()
        phonon.write_total_DOS()

    if saveCv:
        phonon.set_thermal_properties(t_step=10,
                              t_max=1000,
                              t_min=0)
        with open(saveCv,'w') as f:
            f.write('# T_[K] free_energy_[kJ/mol] entropy_[J/K/mol] Cv_[J/K/mol]\n')
        with open(saveCv,'ab') as f:
            np.savetxt(f,np.array(phonon.get_thermal_properties()).T,fmt="%12.3f "+"%15.7f"*3)
#        for t, free_energy, entropy, cv in np.array(phonon.get_thermal_properties()).T:
#            print ("%12.3f " + "%15.7f" * 3) % ( t, free_energy, entropy, cv )

        phonon.plot_thermal_properties().show()

# calculate objective
#    obj = 0.0
#    for i in range(len(frequencies[0])):
#        for j in range(len(frequencies[0][i])):
#            tmp=(frequencies[0][i][j] - qlist[i][j+3]) * (frequencies[0][i][j] - qlist[i][j+3])
#            obj += phon_wts[j]*tmp**0.5
#    freq = np.ravel(frequencies[0])
#    obj += freq[freq<0.].size*10.
#    print "PHONON COST", obj

    return (qlist[:,3:],frequencies[0])


if __name__ == '__main__':

#    qfile=sys.argv[1] #Target frequencies
#    latfile=sys.argv[2] #Structure file
#    Nx,Ny,Nz=6,6,1
#    elem = ['W', 'Se']
#    mass = np.array([183.84,78.9600])
#    try:
#        pathfile=sys.argv[3] # for plotting 'phons/band.conf'
#    except:
#        pathfile=False
#    savedat='phons/predicted.txt'
#    PHON(qfile,latfile,Nx,Ny,Nz,elem,mass,pathfile,savedat)


    qfile = 'phons/wse2.phonon'
    datafile = 'phons/datafile.mod'
    elem = ['W', 'Se']
    mass = [183.84,78.9600]
    savedat = 'output.txt'
    savedos = 'total_dos.dat' # cannot change
    saveCv = 'output_Cv.txt'

#    from phonon import PHON
    freq_target,freq_predict = PHON(qfile,datafile,
                                    6,6,1,
                                    elem,mass,savedat=savedat,
                                    savedos=savedos,saveCv=saveCv)
