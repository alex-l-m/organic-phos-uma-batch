from functools import partial
import pandas as pd
from rdkit import Chem
from rdkit.Chem import rdDistGeom
from ase import Atoms
from ase.optimize import BFGS
from fairchem.core import pretrained_mlip
from fairchem.core.calculate import FAIRChemCalculator, InferenceBatcher
from fairchem.core.datasets.atomic_data import AtomicData, atomicdata_list_to_batch

MODEL = 'uma-s-1p1'  # Models: uma-s-1p1, uma-m-1p1
DEVICE = 'cpu'  # or 'cpu'

# Load the pretrained model
predictor = pretrained_mlip.get_predict_unit(MODEL, device=DEVICE) # Models: uma-s-1p1, uma-m-1p1

batcher = InferenceBatcher(predictor, concurrency_backend_options={'max_workers': 8})

def uma_setup(multiplicity: int, predictor, atoms: Atoms) -> None:
    '''Setup function for UMA for doing ASE calculations'''
    atoms.info['charge'] = 0
    atoms.info['spin'] = multiplicity
    atoms.calc = FAIRChemCalculator(predictor, task_name='omol')

def uma_atomicdata_setup(multiplicity: int, atoms: Atoms):
    '''Setup function for UMA, for using AtomicData for batching'''
    atoms.info['charge'] = 0
    atoms.info['spin'] = multiplicity
    return AtomicData.from_ase(atoms, task_name='omol', r_data_keys=['charge', 'spin'])

def initial_geometry(mol_id: str, smile: str) -> Atoms:
    mol_nohs = Chem.MolFromSmiles(smile)
    mol = Chem.AddHs(mol_nohs)
    rdDistGeom.EmbedMolecule(mol)
    pos = mol.GetConformer().GetPositions()
    elem = [atom.GetSymbol() for atom in mol.GetAtoms()]
    atoms = Atoms(elem, positions=pos)
    atoms.info['name'] = mol_id
    return atoms

def triplet_geometry(predictor, atoms: Atoms) -> Atoms:
    '''Optimize geometry using BFGS'''
    uma_setup(multiplicity=3, predictor=predictor, atoms=atoms)
    opt = BFGS(atoms)
    opt.run(fmax=0.02)
    return atoms

# Read input
intbl = pd.read_csv('smiles_energy.csv')
mol_ids = intbl['mol_id'].tolist()
smiles_strings = intbl['smiles'].tolist()

initial_geometries = [initial_geometry(mol_id, smile) \
        for mol_id, smile in zip(mol_ids, smiles_strings)]

# Optimize geometries
# Use InferenceBatcher
# https://fair-chem.github.io/inference-batcher/
# I think this is the way to do anything iterative in ASE like geometry
# optimization or molecular dynamics
optimized_geometries = list(batcher.executor.map(partial(triplet_geometry, batcher.batch_predict_unit), initial_geometries))

# Calculate singlet-triplet gaps
# Use atomic_data_list_to_batch
# https://fair-chem.github.io/batch-inference/
# Could use InferenceBatcher instead, but this works for energy or force
# calculations
singlet_atomic_data = [uma_atomicdata_setup(1, atoms) for atoms in optimized_geometries]
singlet_batch = atomicdata_list_to_batch(singlet_atomic_data)
singlet_energies = predictor.predict(singlet_batch)['energy'].tolist()

triplet_atomic_data = [uma_atomicdata_setup(3, atoms) for atoms in optimized_geometries]
triplet_batch = atomicdata_list_to_batch(triplet_atomic_data)
triplet_energies = predictor.predict(triplet_batch)['energy'].tolist()

intbl['uma_st'] = [triplet - singlet for triplet, singlet in zip(triplet_energies, singlet_energies)]

intbl.to_csv('smiles_energy_uma.csv', index=False)
