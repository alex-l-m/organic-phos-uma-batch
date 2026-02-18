'''Short script to try and figure out whether AtomicData.from_ase keeps charge and multiplicity'''

from fairchem.core import pretrained_mlip
from fairchem.core.calculate import FAIRChemCalculator
from fairchem.core.datasets.atomic_data import AtomicData, atomicdata_list_to_batch
from ase.build import molecule

predictor = pretrained_mlip.get_predict_unit('uma-s-1p1', device='cpu')

# Benzene in ground state
singlet_benzene_ase = molecule("C6H6")
singlet_benzene_ase.info['charge'] = 0
singlet_benzene_ase.info['spin'] = 1
singlet_benzene_ase.calc = FAIRChemCalculator(predictor, task_name="omol")

# Benzene in triplet state
triplet_benzene_ase = molecule("C6H6")
triplet_benzene_ase.info['charge'] = 0
triplet_benzene_ase.info['spin'] = 3
triplet_benzene_ase.calc = FAIRChemCalculator(predictor, task_name="omol")

singlet_energy_ase = singlet_benzene_ase.get_potential_energy()
triplet_energy_ase = triplet_benzene_ase.get_potential_energy()
gap_ase = triplet_energy_ase - singlet_energy_ase
print(f'Singlet-triplet gap computed using ASE calculator: {gap_ase:.4f} eV')

singlet_benzene_fair = AtomicData.from_ase(singlet_benzene_ase, task_name="omol")
triplet_benzene_fair = AtomicData.from_ase(triplet_benzene_ase, task_name="omol")

print(f'Triplet ASE Atoms spin: {triplet_benzene_ase.info["spin"]}')
print(f'Triplet AtomicData spin: {triplet_benzene_fair.spin}')
fair_list = [singlet_benzene_fair, triplet_benzene_fair]
batch = atomicdata_list_to_batch(fair_list)
singlet_energy_fair, triplet_energy_fair = \
        predictor.predict(batch)['energy'].tolist()
gap_fair = triplet_energy_fair - singlet_energy_fair
print(f'Singlet-triplet gap computed using AtomicData: {gap_fair:.4f} eV')
