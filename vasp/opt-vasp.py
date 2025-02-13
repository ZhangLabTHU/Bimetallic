#!/usr/bin/env python
from ase import Atoms, Atom
from ase.calculators.vasp import Vasp
from ase.io import read,write

            
p=read('POSCAR')
calc = Vasp(prec='normal',
            encut=400.0,
            xc='pbe',
            lreal='Auto',
            kpts=[3,3,1],
            gamma = True,
            nsw = 300,
            ibrion = 2,
	    ispin = 1,
            sigma = 0.050000,
            ediff = 5.00e-05,
            ediffg = -3.00e-02,
            algo = 'fast',
            ismear = 0,
            isif = 2,
            nelm = 60,
            npar = 2,
	    lplane = True,
            lvtot = False,
            lwave = False,
            lcharg = False,
)

calc.calculation_required = lambda x, y: True
p.set_calculator(calc)
pe=p.get_potential_energy()

write('fin.traj',p)

print ("Energy = "+str(pe))

