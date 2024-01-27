"""Tests for sequential."""

import jax.numpy as jnp

from absl.testing import absltest as test
from nas_hybrid.abstract import shape
from nas_hybrid.model.concrete import new_op
from nas_hybrid.model.concrete import OpType
from nas_hybrid.model.subgraph import replace_subgraph
from nas_hybrid.model.subgraph import SubgraphModel
from nas_hybrid.model.subgraph import SubgraphNode
from nas_hybrid.synthesis import sequential
from nas_hybrid.zoo import cnn


class TestSequentialSynthesizer(sequential.AbstractSequentialSynthesizer):

    def synthesize(self):
        pass


class SequentialTest(test.TestCase):

    def test_abstract_sequential_synthesizer(self):
        graph, constants, _ = cnn.CifarNet()
        subgraph_spec = [
            SubgraphNode(
                op=new_op(
                    op_name="conv_layer1/conv/1",
                    op_type=OpType.CONV,
                    op_kwargs={
                        "features": 64,
                        "kernel_size": [1, 1]
                    },
                    input_names=["conv_layer0/avg_pool"]), ),
            SubgraphNode(
                op=new_op(
                    op_name="conv_layer1/gelu/1",
                    op_type=OpType.GELU,
                    input_names=["conv_layer1/conv/1"]),
                output_names=["conv_layer1/relu"])
        ]
        subgraph = SubgraphModel(graph, constants, None,
                                 {"input": jnp.zeros((5, 32, 32, 10))},
                                 subgraph_spec)
        TestSequentialSynthesizer([(subgraph, [])], 0)

    def test_abstract_sequential_synthesizer_fail(self):
        graph, constants, _ = cnn.CifarNet()
        subgraph_spec = [
            SubgraphNode(
                op=new_op(
                    op_name="conv_layer1/conv/1",
                    op_type=OpType.CONV,
                    op_kwargs={
                        "features": 64,
                        "kernel_size": [1, 1]
                    },
                    input_names=["conv_layer0/avg_pool"]),
                output_names=["conv_layer1/conv"]),
            SubgraphNode(
                op=new_op(
                    op_name="conv_layer1/gelu/1",
                    op_type=OpType.GELU,
                    input_names=["conv_layer1/conv"]),
                output_names=["conv_layer1/relu"])
        ]
        subgraph = SubgraphModel(graph, constants, None,
                                 {"input": jnp.zeros((5, 32, 32, 10))},
                                 subgraph_spec)
        self.assertRaisesRegex(ValueError, ".*exactly one input.*",
                               TestSequentialSynthesizer, [(subgraph, [])], 0)

    def test_abstract_sequential_synthesizer_output_features(self):
        graph, constants, _ = cnn.CifarNet()
        subgraph_spec = [
            SubgraphNode(
                op=new_op(
                    op_name="conv_layer1/conv",
                    op_type=OpType.CONV,
                    op_kwargs={
                        "features": "S:-1*2",
                        "kernel_size": [1, 1]
                    },
                    input_names=["conv_layer0/avg_pool"]), ),
            SubgraphNode(
                op=new_op(
                    op_name="conv_layer1/relu",
                    op_type=OpType.RELU,
                    input_names=["conv_layer1/conv"]),
                output_names=["conv_layer1/relu"])
        ]
        subgraph = replace_subgraph(graph, subgraph_spec)
        subgraph_model = SubgraphModel(subgraph, constants, None,
                                       {"input": jnp.zeros((5, 32, 32, 10))},
                                       subgraph_spec)
        sp = shape.ShapeProperty().infer(subgraph_model)
        syn = TestSequentialSynthesizer([(subgraph_model, [sp])], 0)
        self.assertEqual(syn.output_features_mul, 2)
        self.assertEqual(syn.output_features_div, 1)


if __name__ == "__main__":
    test.main()
