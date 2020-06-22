"""
File Name:          get_random_mol.py
Project:            bcgraph

File Description:

"""
from random import randint
from os.path import abspath, join

from rdkit.Chem import Mol, SDMolSupplier


# should move the directory strings into a global place
PROCESSED_DATA_DIR = abspath('./bcgraph/faker/data')
PROCESSED_SDF_PATH = join(PROCESSED_DATA_DIR, f'pubchem_mols.sdf')


def get_random_mol() -> Mol:

    _mol_supplier = SDMolSupplier(PROCESSED_SDF_PATH)
    _index = randint(0, len(_mol_supplier) - 1)
    assert _mol_supplier[_index]

    return _mol_supplier[_index]
