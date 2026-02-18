import pandas as pd
from rdkit import Chem
from rdkit.Chem import rdDistGeom
from ase import Atoms
from fairchem.core import pretrained_mlip
from fairchem.core.datasets.atomic_data import AtomicData, atomicdata_list_to_batch


MODEL = 'uma-s-1p1'  # Models: uma-s-1p1, uma-m-1p1
DEVICE = 'cpu'  # or 'cpu'

# Load the pretrained model
predictor = pretrained_mlip.get_predict_unit(MODEL, device=DEVICE) # Models: uma-s-1p1, uma-m-1p1

def uma_batch_setup(multiplicity: int, predictor, atoms: Atoms):
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

# READ INPUT
intbl = pd.read_csv('smiles_energy.csv')
mol_ids = intbl['mol_id'].tolist()
smiles_strings = intbl['smiles'].tolist()

initial_geometries = [initial_geometry(mol_id, smile) \
        for mol_id, smile in zip(mol_ids, smiles_strings)]

singlet_atomic_data = [uma_batch_setup(1, predictor, atoms) for atoms in initial_geometries]
singlet_batch = atomicdata_list_to_batch(singlet_atomic_data)
singlet_energies = predictor.predict(singlet_batch)['energy'].tolist()

triplet_atomic_data = [uma_batch_setup(3, predictor, atoms) for atoms in initial_geometries]
triplet_batch = atomicdata_list_to_batch(triplet_atomic_data)
triplet_energies = predictor.predict(triplet_batch)['energy'].tolist()

intbl['uma_st'] = [triplet - singlet for triplet, singlet in zip(triplet_energies, singlet_energies)]

intbl.to_csv('smiles_energy_uma.csv', index=False)
