"""
File Name:          get_random_mol.py
Project:            bcgraph

File Description:

"""
import logging
from dataclasses import dataclass
from collections import namedtuple
from typing import Optional, Sequence, Union, Dict

import torch
import numpy as np
from torch.nn.functional import pad
from rdkit.Chem import Atom, ChiralType, HybridizationType, \
    Bond, BondDir, BondType, BondStereo, Mol, Conformer

from bcgraph.utils import one_hot_encode


_LOGGER = logging.getLogger(__name__)

RDKitFeature: type = namedtuple(
    'RDKitFeature',
    [
        # RDKit function for feature extraction
        'rdkit_function',
        # indicator for returned data type (categorical or numeric)
        # If the returned data is categorical, then it's a tuple of all
        # possible returned values, otherwise (if numeric), then it's the
        # actual data type (e.g. float, int, etc.)
        'returned_dtype',
    ],
)


@dataclass(frozen=True)
class RDKitAtomFeatures:
    atomic_number = RDKitFeature(
        rdkit_function=Atom.GetAtomicNum,
        returned_dtype=int,
    )
    chiral_tag = RDKitFeature(
        rdkit_function=Atom.GetChiralTag,
        returned_dtype=(
            # https://www.rdkit.org/docs/source/rdkit.Chem.rdchem.html
            # #rdkit.Chem.rdchem.ChiralType
            ChiralType.CHI_UNSPECIFIED,
            ChiralType.CHI_TETRAHEDRAL_CW,
            ChiralType.CHI_TETRAHEDRAL_CCW,
            ChiralType.CHI_OTHER,
        ),
    )
    degree = RDKitFeature(
        rdkit_function=Atom.GetDegree,
        returned_dtype=int,
    )
    explicit_valence = RDKitFeature(
        rdkit_function=Atom.GetExplicitValence,
        returned_dtype=int,
    )
    formal_charge = RDKitFeature(
        rdkit_function=Atom.GetFormalCharge,
        returned_dtype=int,
    )
    hybridization = RDKitFeature(
        rdkit_function=Atom.GetHybridization,
        returned_dtype=(
            # https://www.rdkit.org/docs/source/rdkit.Chem.rdchem.html
            # #rdkit.Chem.rdchem.HybridizationType
            HybridizationType.UNSPECIFIED,
            HybridizationType.S,
            HybridizationType.SP,
            HybridizationType.SP2,
            HybridizationType.SP3,
            HybridizationType.SP3D,
            HybridizationType.SP3D2,
            HybridizationType.OTHER,
        ),
    )
    implicit_valence = RDKitFeature(
        rdkit_function=Atom.GetImplicitValence,
        returned_dtype=int,
    )
    is_aromatic = RDKitFeature(
        rdkit_function=Atom.GetIsAromatic,
        returned_dtype=bool,
    )
    is_in_ring = RDKitFeature(
        rdkit_function=Atom.IsInRing,
        returned_dtype=bool,
    )
    mass = RDKitFeature(
        rdkit_function=Atom.GetMass,
        returned_dtype=float,
    )
    no_implicit = RDKitFeature(
        rdkit_function=Atom.GetNoImplicit,
        returned_dtype=bool,
    )
    num_explicit_hs = RDKitFeature(
        rdkit_function=Atom.GetNumExplicitHs,
        returned_dtype=int,
    )
    num_implicit_hs = RDKitFeature(
        rdkit_function=Atom.GetNumImplicitHs,
        returned_dtype=int,
    )
    num_radical_electrons = RDKitFeature(
        rdkit_function=Atom.GetNumRadicalElectrons,
        returned_dtype=int,
    )
    total_degree = RDKitFeature(
        rdkit_function=Atom.GetTotalDegree,
        returned_dtype=int,
    )
    total_num_hs = RDKitFeature(
        rdkit_function=Atom.GetTotalNumHs,
        returned_dtype=int,
    )
    total_valence = RDKitFeature(
        rdkit_function=Atom.GetTotalValence,
        returned_dtype=int,
    )
    # RDKit atom functions that are not included:
    # - not related to atom: GetIsotope, GetOwningMol
    # - not applicable to all cases: GetMonomerInfo, GetPDBResidueInfo
    # - symbolic: GetSmarts, GetSymbol


@dataclass(frozen=True)
class RDKitBondFeatures:
    bond_dir = RDKitFeature(
        rdkit_function=Bond.GetBondDir,
        returned_dtype=(
            # https://www.rdkit.org/docs/source/rdkit.Chem.rdchem.html
            # #rdkit.Chem.rdchem.BondDir
            BondDir.NONE,
            BondDir.BEGINWEDGE,
            BondDir.BEGINDASH,
            BondDir.ENDDOWNRIGHT,
            BondDir.ENDUPRIGHT,
            BondDir.EITHERDOUBLE,
            BondDir.UNKNOWN,
        ),
    )
    bond_type = RDKitFeature(
        rdkit_function=Bond.GetBondType,
        returned_dtype=(
            # https://www.rdkit.org/docs/source/rdkit.Chem.rdchem.html
            # #rdkit.Chem.rdchem.BondType
            BondType.UNSPECIFIED,
            BondType.SINGLE,
            BondType.DOUBLE,
            BondType.TRIPLE,
            BondType.QUADRUPLE,
            BondType.QUINTUPLE,
            BondType.HEXTUPLE,
            BondType.ONEANDAHALF,
            BondType.TWOANDAHALF,
            BondType.THREEANDAHALF,
            BondType.FOURANDAHALF,
            BondType.FIVEANDAHALF,
            BondType.AROMATIC,
            BondType.IONIC,
            BondType.HYDROGEN,
            BondType.THREECENTER,
            BondType.DATIVEONE,
            BondType.DATIVE,
            BondType.DATIVEL,
            BondType.DATIVER,
            BondType.OTHER,
            BondType.ZERO,
        ),
    )
    bond_type_as_double = RDKitFeature(
        rdkit_function=Bond.GetBondTypeAsDouble,
        returned_dtype=float,
    )
    is_aromatic = RDKitFeature(
        rdkit_function=Bond.GetIsAromatic,
        returned_dtype=bool,
    )
    is_conjugated = RDKitFeature(
        rdkit_function=Bond.GetIsConjugated,
        returned_dtype=bool,
    )
    is_in_ring = RDKitFeature(
        rdkit_function=Bond.IsInRing,
        returned_dtype=bool,
    )
    stereo = RDKitFeature(
        rdkit_function=Bond.GetStereo,
        returned_dtype=(
            # https://www.rdkit.org/docs/source/rdkit.Chem.rdchem.html
            # #rdkit.Chem.rdchem.BondStereo
            BondStereo.STEREONONE,
            BondStereo.STEREOANY,
            BondStereo.STEREOZ,
            BondStereo.STEREOE,
            BondStereo.STEREOCIS,
            BondStereo.STEREOTRANS,
        ),
    )
    valence_contrib = RDKitFeature(
        rdkit_function=Bond.GetValenceContrib,
        returned_dtype=float,
    )


def check_conformer(
        mol: Mol,
        conformer: Conformer,
) -> bool:

    # throw warnings if
    # - the conformer is actually 2D (useless Z coordinate in graph)
    # - some atoms have all-zero coordinates, which implies bad conformer
    _positions: np.array = conformer.GetPositions()
    if not _positions[:, 2].any():
        _warning_msg = f'Conformer has no Z coordinates. Continuing ...'
        _LOGGER.warning(_warning_msg)
    if not np.array([_p.any() for _p in _positions]).all():
        _warning_msg = f'Conformer has atom(s) with invalid coordinates ' \
                       f'(0.0, 0.0, 0.0). Continuing ...'
        _LOGGER.warning(_warning_msg)

    # make sure that the molecule and conformer are of the same size
    return len(mol.GetAtoms()) == len(mol.GetConformer().GetPositions())


def convert_mol_to_generic_graph(
        mol: Mol,
        conformer: Optional[Conformer],
        atom_rdkit_features: Sequence[RDKitFeature],
        bond_rdkit_features: Sequence[RDKitFeature],
        one_hot_encoding: bool = True,
        master_node: bool = True,
) -> Dict[str, torch.Tensor]:
    """
    decisions to make before calling this function:
    - include/exclude hydrogen atoms using AddHs/RemoveHs with:
        _mol: Mol = AddHs(mol) if include_Hs else RemoveHs(mol)
    - get the conformer if the atom positions are part of the atom features

    """

    # get the positions of atoms if conformer is given
    if conformer:
        assert check_conformer(mol, conformer)
        node_pos: np.array = conformer.GetPositions()
    else:
        node_pos: np.array = np.zeros(shape=(len(mol.GetAtoms()), 3))

    # overall graph node attributes/features
    node_attr = []
    for _atom in mol.GetAtoms():

        _single_node_attr = []
        _atom_rdkit_feature: RDKitFeature
        for _atom_rdkit_feature in atom_rdkit_features:

            # get atom feature function, return type, and actual value
            _atom_feature_function: callable = \
                _atom_rdkit_feature.rdkit_function
            _atom_feature_type = _atom_rdkit_feature.returned_dtype
            _atom_feature: Union[int, float, bool, tuple] = \
                _atom_feature_function(_atom)

            # one-hot encoding if the feature is not numeric/binary AND the
            # one_hot_encoding argument is set to True
            if type(_atom_feature_type) in (int, float, bool):
                _single_node_attr.append(_atom_feature)
            elif one_hot_encoding:
                _atom_feature_type: tuple
                _single_node_attr.extend(
                    one_hot_encode(_atom_feature, _atom_feature_type))
            else:
                _atom_feature_type: tuple
                _single_node_attr.append(
                    _atom_feature_type.index(_atom_feature))

        # append single node attributes/features to the overall graph node
        # attributes/features
        node_attr.append(_single_node_attr)

    # overall graph edge indexes and attributes/features
    # this segment of code assumes that the bonds are NOT DIRECTIONAL
    edge_index, edge_attr = [], []
    for _bond in mol.GetBonds():

        _single_edge_attr = []
        for _bond_rdkit_feature in bond_rdkit_features:

            # get bond feature function, return type, and actual value
            _bond_feature_function: callable = \
                _bond_rdkit_feature.rdkit_function
            _bond_feature_type = _bond_rdkit_feature.returned_dtype
            _bond_feature: Union[int, float, bool, tuple] = \
                _bond_feature_function(_bond)

            # one-hot encoding if the feature is not numeric/binary AND the
            # one_hot_encoding argument is set to True
            if type(_bond_feature_type) in (int, float, bool):
                _single_edge_attr.append(_bond_feature)
            elif one_hot_encoding:
                _bond_feature_type: tuple
                _single_edge_attr.extend(
                    one_hot_encode(_bond_feature, _bond_feature_type))
            else:
                _bond_feature_type: tuple
                _single_edge_attr.append(
                    _bond_feature_type.index(_bond_feature))

        # append single edge index (start) attributes/features to the overall
        # graph edge indexes and attributes/features
        edge_index.append([_bond.GetBeginAtomIdx(), _bond.GetEndAtomIdx()])
        edge_attr.append(_single_edge_attr)

    # TODO: make this a seperate function?
    if master_node:
        # add master node if the master_node argument is set to True
        # the master node
        # - has its own node index, which is the same as the number of atoms
        _master_node_index = len(node_pos)
        # - has position coordinates of (0, 0, 0)
        node_pos = np.concatenate((node_pos, [[0., 0., 0.]]))
        # - has a indication digit in node attributes/features for master node
        node_attr = [(_node_attr + [0.]) for _node_attr in node_attr]
        _master_node_attr = [0.] * len(node_attr[0])
        _master_node_attr[-1] = 1.
        node_attr.append(_master_node_attr)
        # - has connections with all other atoms
        edge_index.extend(
            [[_i, _master_node_index] for _i in range(_master_node_index)])
        # - has a indication digit in edge attributes/features for master node
        edge_attr = [(_edge_attr + [0.]) for _edge_attr in edge_attr]
        _master_edge_attr = [0.] * len(edge_attr[0])
        _master_edge_attr[-1] = 1.
        _master_edge_attr = [_master_edge_attr] * _master_node_index
        edge_attr.extend(_master_edge_attr)

    node_pos = torch.FloatTensor(node_pos)
    node_attr = torch.FloatTensor(node_attr)
    edge_index = torch.LongTensor(edge_index)
    edge_attr = torch.FloatTensor(edge_attr)

    # TODO: make a data structure of generic graph?
    return {
        'node_pos': node_pos,
        'node_attr': node_attr,
        'edge_index': edge_index,
        'edge_attr': edge_attr,
    }
