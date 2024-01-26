"""Tests for OpPairings."""

import numpy as np

from absl.testing import absltest as test
from nas_hybrid.abstract.linear import OpPairings
from nas_hybrid.model.concrete import new_op
from nas_hybrid.model.concrete import OpType


class OpPairingsTest(test.TestCase):

    def test_get_conv(self):
        op = new_op(
            op_name="output",
            op_type=OpType.CONV,
            op_kwargs={
                "features": 10,
                "kernel_size": [3, 3]
            },
            input_names=["input"])

        input_shapes = [(5, 5, 5)]

        pairings = OpPairings()

        my_pairings = pairings.get(op, input_shapes, 0, 0)

        self.assertEqual(my_pairings.in_dims, 3)
        self.assertEqual(my_pairings.out_dims, 3)
        self.assertEqual(np.shape(my_pairings.mappings), (3, 3))
