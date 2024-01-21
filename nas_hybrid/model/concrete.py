"""Definitions and utils for concrete specification of architectures."""

from __future__ import annotations

import copy
import dataclasses
import enum
from typing import Any, Dict, Sequence, Union

from nas_hybrid import utils


@dataclasses.dataclass
class Op:
    """The python operations for layers in network"""

    class OpType(enum.Enum):
        """Types of ops."""

        # synthesis helpers
        NONE = 0
        IDENTITY = 1
        # linear
        DENSE = 2
        DENSE_GENERAL = 3  # compat with big_vision/models/vit
        CONV = 4
        ADD = 5
        SCALAR_ADD = 6
        MUL = 25
        SCALAR_MUL = 7
        DOT_GENERAL = 8
        SELF_ATTENTION = 9  # compat with big_vision/models/vit
        EINSUM = 10
        # activation functions
        RELU = 11
        GELU = 12
        SWISH = 26
        SIGMOID = 28
        SOFTMAX = 13
        # normalization layers
        BATCH_NORM = 14
        LAYER_NORM = 15
        GROUP_NORM = 16
        # shape layers
        RESHAPE = 17
        FLATTEN = 18
        TRANSPOSE = 19
        # stochastic layers
        DROPOUT = 20
        STOCH_DEPTH = 27
        # pooling layers
        AVG_POOL = 21
        MAX_POOL = 22
        MEAN = 23
        # new param
        PARAM = 24

    name: str
    type: OpType
    input_names: Sequence[str]
    input_kwargs: Dict[str, Any] = dataclasses.field(default_factory=dict)
    op_kwargs: Dict[str, Any] = dataclasses.field(default_factory=dict)
    num_outputs: int = 1


@dataclasses.dataclass
class Graph(object):
    """The python equivalent of the graph_pb2.Graph."""
    input_names: Sequence[str]
    output_names: Sequence[str]
    ops: Sequence[Op]


OpType = Op.OpType


def new_op(op_name,
           op_type,
           input_names,
           input_kwargs=None,
           op_kwargs=None,
           num_outputs=1):
    """Returns a new Op object."""

    # restricted characters are used as delimiters when parsing symbolic values
    # see _resolve_symbolic in ../model.py
    for r in [":", "*", "%"]:
        if r in op_name:
            raise ValueError(f"op_name {op_name} contains restricted char ``{r}''.")

    input_kwargs = input_kwargs if input_kwargs is not None else {}
    op_kwargs = op_kwargs if op_kwargs is not None else {}

    input_names = [utils.canonicalize_tensor_name(i) for i in input_names]
    return Op(
        name=op_name,
        type=op_type,
        input_names=input_names,
        input_kwargs=copy.deepcopy(input_kwargs),
        op_kwargs=copy.deepcopy(op_kwargs),
        num_outputs=num_outputs)


def new_graph(input_names, output_names,
              ops):
    """Returns a new Graph object."""
    for output_name in output_names:
        found = False
        for op in ops:
            if op.num_outputs == 1 and op.name == output_name:
                found = True
                break
            for idx in range(op.num_outputs):
                if f"{op.name}:{idx}" == output_name:
                    found = True
                    break
            if found:
                break
        if not found:
            raise ValueError(f"Required output {output_name} not found in ops")

    return Graph(input_names=copy.deepcopy(input_names),
                 output_names=copy.deepcopy(output_names),
                 ops=copy.deepcopy(ops))
