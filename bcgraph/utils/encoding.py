"""
File Name:          encoding.py
Project:            bcgraph

File Description:

"""
import logging
from typing import Any, Sequence, List

_LOGGER = logging.getLogger(__name__)


def one_hot_encode(
        value: Any,
        possible_values: Sequence[Any] = None,
) -> List[int]:

    ret_enc_feat = [0] * len(possible_values)
    try:
        ret_enc_feat[possible_values.index(value)] = 1
    except ValueError:
        _warning_msg = \
            f'Feature value {value} is not one of ' \
            f'all possible values: {possible_values}.'
        _LOGGER.warning(_warning_msg)

    return ret_enc_feat
