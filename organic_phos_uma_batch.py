import pandas as pd
from rdkit import Chem
from functools import partial
from rdkit.Chem import rdDistGeom
from ase import Atoms
from ase.optimize import BFGS
from fairchem.core import pretrained_mlip
from fairchem.core.calculate import FAIRChemCalculator, InferenceBatcher


MODEL = 'uma-s-1p1'  # Models: uma-s-1p1, uma-m-1p1
DEVICE = 'cpu'  # or 'cpu'

# Load the pretrained model
predictor = pretrained_mlip.get_predict_unit(MODEL, device=DEVICE) # Models: uma-s-1p1, uma-m-1p1

batcher = InferenceBatcher(predictor, concurrency_backend_options={'max_workers': 32})

def uma_setup(multiplicity: int, predictor, atoms: Atoms) -> None:
    atoms.info['charge'] = 0
    atoms.info['spin'] = 1
    atoms.calc = FAIRChemCalculator(predictor, task_name='omol')

def initial_geometry(mol_id: str, smile: str) -> Atoms:
    mol_nohs = Chem.MolFromSmiles(smile)
    mol = Chem.AddHs(mol_nohs)
    rdDistGeom.EmbedMolecule(mol)
    pos = mol.GetConformer().GetPositions()
    elem = [atom.GetSymbol() for atom in mol.GetAtoms()]
    atoms = Atoms(elem, positions=pos)
    atoms.info['name'] = mol_id
    return atoms

def singlet_energy_step(predictor, atoms: Atoms) -> float:
    uma_setup(multiplicity=1, predictor=predictor, atoms=atoms)
    return atoms.get_potential_energy()

def triplet_energy_step(predictor, atoms: Atoms) -> float:
    uma_setup(multiplicity=3, predictor=predictor, atoms=atoms)
    return atoms.get_potential_energy()

# READ INPUT
intbl = pd.read_csv('smiles_energy.csv')
mol_ids = intbl['mol_id'].tolist()
smiles_strings = intbl['smiles'].tolist()

initial_geometries = [initial_geometry(mol_id, smile) \
        for mol_id, smile in zip(mol_ids, smiles_strings)]

singlet_energies = list(batcher.executor.map(partial(singlet_energy_step, predictor), initial_geometries))

triplet_energies = list(batcher.executor.map(partial(triplet_energy_step, predictor), initial_geometries))

for mol_id, singlet_energy, triplet_energy in zip(mol_ids, singlet_energies, triplet_energies):
    st_gap = singlet_energy - triplet_energy
    print(f'{mol_id}: S-T gap = {st_gap:.4f} eV')
