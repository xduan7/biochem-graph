"""
File Name:          convert_mol_to_graph.py
Project:            bcgraph

File Description:

"""
import logging
from typing import Optional, List, Sequence, Callable, Any, Set

from rdkit import RDLogger
from rdkit.Chem import Mol

from bcgraph.utils import one_hot_encode, RDKitAtomFeature, RDKitBondFeature


# suppress RDKit warnings and errors
RDLogger.logger().setLevel(RDLogger.CRITICAL)

_LOGGER = logging.getLogger(__name__)


def convert_mol_to_graph(
        mol: Mol,
        include_Hs: bool,
        master_atom: bool,
        master_bond: bool,
        atom_feat_list: Sequence[RDKitAtomFeature],
        bond_feat_list: Sequence[RDKitBondFeature],
):
    pass
