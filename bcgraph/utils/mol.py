"""
File Name:          get_random_mol.py
Project:            bcgraph

File Description:

"""
import logging
from enum import Enum
from collections import namedtuple

import numpy as np
from rdkit.Chem import Atom, ChiralType, HybridizationType, \
    Bond, BondDir, BondType, BondStereo, Mol, Conformer


_LOGGER = logging.getLogger(__name__)

_RDKitFeature: type = namedtuple(
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


class RDKitAtomFeature(Enum):
    atomic_number = _RDKitFeature(
        rdkit_function=Atom.GetAtomicNum,
        returned_dtype=int,
    ),
    chiral_tag = _RDKitFeature(
        rdkit_function=Atom.GetChiralTag,
        returned_dtype=(
            # https://www.rdkit.org/docs/source/rdkit.Chem.rdchem.html
            # #rdkit.Chem.rdchem.ChiralType
            ChiralType.CHI_UNSPECIFIED,
            ChiralType.CHI_TETRAHEDRAL_CW,
            ChiralType.CHI_TETRAHEDRAL_CCW,
            ChiralType.CHI_OTHER,
        ),
    ),
    degree = _RDKitFeature(
        rdkit_function=Atom.GetDegree,
        returned_dtype=int,
    ),
    explicit_valence = _RDKitFeature(
        rdkit_function=Atom.GetExplicitValence,
        returned_dtype=int,
    ),
    formal_charge = _RDKitFeature(
        rdkit_function=Atom.GetFormalCharge,
        returned_dtype=int,
    ),
    hybridization = _RDKitFeature(
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
    ),
    implicit_valence = _RDKitFeature(
        rdkit_function=Atom.GetImplicitValence,
        returned_dtype=int,
    ),
    is_aromatic = _RDKitFeature(
        rdkit_function=Atom.GetIsAromatic,
        returned_dtype=bool,
    ),
    is_in_ring = _RDKitFeature(
        rdkit_function=Atom.IsInRing,
        returned_dtype=bool,
    ),
    mass = _RDKitFeature(
        rdkit_function=Atom.GetMass,
        returned_dtype=float,
    ),
    no_implicit = _RDKitFeature(
        rdkit_function=Atom.GetNoImplicit,
        returned_dtype=bool,
    ),
    num_explicit_hs = _RDKitFeature(
        rdkit_function=Atom.GetNumExplicitHs,
        returned_dtype=int,
    ),
    num_implicit_hs = _RDKitFeature(
        rdkit_function=Atom.GetNumImplicitHs,
        returned_dtype=int,
    ),
    num_radical_electrons = _RDKitFeature(
        rdkit_function=Atom.GetNumRadicalElectrons,
        returned_dtype=int,
    ),
    total_degree = _RDKitFeature(
        rdkit_function=Atom.GetTotalDegree,
        returned_dtype=int,
    ),
    total_num_hs = _RDKitFeature(
        rdkit_function=Atom.GetTotalNumHs,
        returned_dtype=int,
    ),
    total_valence = _RDKitFeature(
        rdkit_function=Atom.GetTotalValence,
        returned_dtype=int,
    ),
    # RDKit atom functions that are not included:
    # - not related to atom: GetIsotope, GetOwningMol
    # - not applicable to all cases: GetMonomerInfo, GetPDBResidueInfo
    # - symbolic: GetSmarts, GetSymbol


class RDKitBondFeature(Enum):
    bond_dir = _RDKitFeature(
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
    ),
    bond_type = _RDKitFeature(
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
    ),
    bond_type_as_double = _RDKitFeature(
        rdkit_function=Bond.GetBondTypeAsDouble,
        returned_dtype=float,
    ),
    is_aromatic = _RDKitFeature(
        rdkit_function=Bond.GetIsAromatic,
        returned_dtype=bool,
    ),
    is_conjugated = _RDKitFeature(
        rdkit_function=Bond.GetIsConjugated,
        returned_dtype=bool,
    ),
    is_in_ring = _RDKitFeature(
        rdkit_function=Bond.IsInRing,
        returned_dtype=bool,
    ),
    stereo = _RDKitFeature(
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
    ),
    valence_contrib = _RDKitFeature(
        rdkit_function=Bond.GetValenceContrib,
        returned_dtype=float,
    ),



def validate_conformer(
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
