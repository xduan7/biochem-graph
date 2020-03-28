"""
File Name:          molecule.py
Project:            biochem-graph

File Description:

"""
from typing import Optional, List, Sequence, Callable

from rdkit.Chem import Mol
from dgl import DGLGraph


def mol(
        predicates: Optional[Sequence[Callable]] = None,
        random_state: int = 0,
) -> Mol:
    pass


def mol_smiles(
        predicates: Optional[Sequence[Callable]] = None,
        random_state: int = 0,
) -> str:
    pass


def mol_graph(
        master_atom: bool = True,
        master_bond: bool = True,
        atom_feat_list: Optional[List[str]] = None,
        bond_feat_list: Optional[List[str]] = None,
        predicates: Optional[Sequence[Callable]] = None,
        random_state: int = 0,
) -> DGLGraph:
    pass
