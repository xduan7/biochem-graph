"""
File Name:          convert_mol_to_graph.py
Project:            bcgraph

File Description:

"""
import logging
from typing import Optional, Sequence, Dict

import torch
from rdkit import RDLogger
from rdkit.Chem import Mol, Conformer
from torch_geometric.data import Data
from torch_geometric.utils import to_undirected

from bcgraph.utils import RDKitFeature, convert_mol_to_generic_graph


# suppress RDKit warnings and errors
RDLogger.logger().setLevel(RDLogger.CRITICAL)

_LOGGER = logging.getLogger(__name__)


def convert_mol_to_graph(
        mol: Mol,
        conformer: Optional[Conformer],
        atom_rdkit_features: Sequence[RDKitFeature],
        bond_rdkit_features: Sequence[RDKitFeature],
        one_hot_encoding: bool = True,
        master_node: bool = True,
) -> Data:

    _graph_dict: Dict[str, torch.Tensor] = convert_mol_to_generic_graph(
        mol=mol,
        conformer=conformer,
        atom_rdkit_features=atom_rdkit_features,
        bond_rdkit_features=bond_rdkit_features,
        one_hot_encoding=one_hot_encoding,
        master_node=master_node,
    )

    # make the edges undirected (symmetric)
    edge_index = _graph_dict['edge_index'].transpose(0, -1)
    edge_index = to_undirected(edge_index)

    return Data(
        x=_graph_dict['node_attr'],
        edge_index=edge_index,
        edge_attr=torch.cat(2 * [_graph_dict['edge_attr']]),
        pos=_graph_dict['node_pos'],
    )
