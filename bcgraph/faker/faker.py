"""
File Name:          faker.py
Project:            bcgraph

File Description:

"""
import logging
from typing import Any, Optional, Sequence, Callable

from rdkit.Chem import Mol, MolToSmiles

from bcgraph.faker import get_random_mol


_LOGGER = logging.getLogger(__name__)
_DEFAULT_MAX_NUM_TRAILS = 10


def _check_predicates(
        obj: Any,
        obj_str: Optional[str] = None,
        predicates: Optional[Sequence[Callable]] = None,
) -> bool:

    if not predicates:
        predicates = []

    for _predicate in predicates:
        try:
            assert _predicate(obj)
        except AssertionError as e:
            _debug_msg = \
                f'Generated object {obj_str if obj_str else obj} ' \
                f'failed to pass a faker predicate {_predicate.__name__}.'
            _LOGGER.debug(_debug_msg)
            return False
    return True


class Faker:
    def __init__(
            self,
            max_num_trails: Optional[int] = None,
    ):
        self.max_num_trails = max_num_trails if max_num_trails \
            else _DEFAULT_MAX_NUM_TRAILS

    def mol(
            self,
            predicates: Optional[Sequence[Callable]] = None,
    ) -> Optional[Mol]:

        _num_trails = 0
        while _num_trails < self.max_num_trails:
            _mol = get_random_mol()
            if _check_predicates(_mol, MolToSmiles(_mol), predicates):
                return _mol
            _num_trails += 1

        _warning_msg = \
            f'Cannot find a molecule that passes all {len(predicates)} ' \
            f'predicates in {self.max_num_trails} trails. Returning None ...'
        _LOGGER.warning(_warning_msg)

        return None

    def mol_smiles(
            self,
            predicates: Optional[Sequence[Callable]] = None,
    ) -> Optional[str]:

        _num_trails = 0
        while _num_trails < self.max_num_trails:
            _mol = get_random_mol()
            _smiles = MolToSmiles(_mol)
            if _check_predicates(_smiles, _smiles, predicates):
                return _smiles
            _num_trails += 1

        _warning_msg = \
            f'Cannot find a molecule SMILES string that passes all' \
            f' {len(predicates)} predicates in {self.max_num_trails} trails. ' \
            f'Returning None ...'
        _LOGGER.warning(_warning_msg)

        return None
