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
from dgl import DGLGraph

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
) -> DGLGraph:

    _graph_dict: Dict[str, torch.Tensor] = convert_mol_to_generic_graph(
        mol=mol,
        conformer=conformer,
        atom_rdkit_features=atom_rdkit_features,
        bond_rdkit_features=bond_rdkit_features,
        one_hot_encoding=one_hot_encoding,
        master_node=master_node,
    )

    dgl_graph = DGLGraph()
    dgl_graph.add_nodes(
        num=len(_graph_dict['node_attr']),
    )
    dgl_graph.nodes.data['attr'] = _graph_dict['node_attr']
    dgl_graph.nodes.data['pos'] = _graph_dict['node_pos']

    # all DGL graphs are directional, so need to add edges twice
    # ref: https://docs.dgl.ai/api/python/graph.html
    dgl_graph.add_edges(
        _graph_dict['edge_index'][:, 0],
        _graph_dict['edge_index'][:, 1],
        _graph_dict['edge_attr'],
    )
    dgl_graph.add_edges(
        _graph_dict['edge_index'][:, 1],
        _graph_dict['edge_index'][:, 0],
        _graph_dict['edge_attr'],
    )

    return dgl_graph
