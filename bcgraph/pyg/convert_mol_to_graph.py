"""
File Name:          convert_mol_to_graph.py
Project:            biochem-graph

File Description:

"""
import logging
from typing import Optional, Sequence, Callable

from rdkit.Chem import Mol

from biochem_graph import utils



def convert_mol_to_graph(
        mol: Mol,
        # include Hs?
        master_atom: bool = True,
        master_bond: bool = True,
        atom_feat_list: Optional[Sequence[RDKitAtomFeature]] = None,
        bond_feat_list: Optional[Sequence[RDKitBondFeature]] = None,
        predicates: Optional[Sequence[Callable]] = None,
):

    """
    This implementation is based on:
        https://github.com/HIPS/neural-fingerprint/
        neuralfingerprint/features.py
    which is the git repo for https://arxiv.org/pdf/1509.09292.pdf
    And
        https://github.com/deepchem/deepchem/
        deepchem/feat/graph_features.py
    which is the git repo for DeepChem
    """

    # Sanity check for graph size
    num_atoms = mol.GetNumAtoms() + master_atom
    if (num_atoms > max_num_atoms) and (max_num_atoms >=0):
        logger.warning(f'Number of atoms for {Chem.MolToSmiles(mol)} '
                       f'exceeds the maximum number of atoms {max_num_atoms}')
        return None

    if atom_feat_list is None:
        atom_feat_list = DEFAULT_ATOM_FEAT_LIST
    if bond_feat_list is None:
        bond_feat_list = DEFAULT_BOND_FEAT_LIST

    # Process the graph in the way that aligns with PyG
    # Returning (node_attr=[N, F], edge_index=[2, M], edge_attr=[M, E])
    # TODO: add position information for the atoms?

    # Prepare features for atoms/nodes
    node_attr = []
    for atom in mol.GetAtoms():

        single_node_attr = []
        for atom_feat_name in atom_feat_list:

            feat_func: callable = ATOM_FEAT_FUNC_DICT[atom_feat_name]
            feat = feat_func(atom)

            # If the feature is not numeric, then one-hot encoding
            if type(feat) in [int, float, bool]:
                single_node_attr.append(feat)
            else:
                possible_values = DEFAULT_FEAT_VALUE_DICT[atom_feat_name] \
                    if atom_feat_name in DEFAULT_FEAT_VALUE_DICT else None
                attr = one_hot_encode(feat, possible_values)
                single_node_attr.extend(list(attr))

        node_attr.append(single_node_attr)

    # Prepare features for bonds/edges
    edge_index, edge_attr = [], []
    for bond in mol.GetBonds():

        # edge_index.append([bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()])
        single_edge_attr = [1, ] if master_bond else []
        for bond_feat_name in bond_feat_list:

            feat_func: callable = BOND_FEAT_FUNC_DICT[bond_feat_name]
            feat = feat_func(bond)

            # If the feature is not numeric, then one-hot encoding
            if type(feat) in [int, float, bool]:
                single_edge_attr.append(feat)
            else:
                possible_values = DEFAULT_FEAT_VALUE_DICT[bond_feat_name] \
                    if bond_feat_name in DEFAULT_FEAT_VALUE_DICT else None
                attr = one_hot_encode(feat, possible_values)
                single_edge_attr.extend(list(attr))

        # Note that in molecules, bonds are always mutually shared
        edge_index.append([bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()])
        edge_index.append([bond.GetEndAtomIdx(), bond.GetBeginAtomIdx()])
        edge_attr.append(single_edge_attr)
        edge_attr.append(single_edge_attr)

    node_attr = np.array(node_attr, dtype=np.float32)
    edge_index = np.transpose(np.array(edge_index, dtype=np.int64))
    edge_attr = np.array(edge_attr, dtype=np.float32)

    return Data(x=torch.from_numpy(node_attr),
                edge_index=torch.from_numpy(edge_index),
                edge_attr=torch.from_numpy(edge_attr))