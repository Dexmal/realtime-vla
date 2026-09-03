import copy

import numpy as np
import pytest

from openpi_lora import merge_lora_weights


def leaf(value):
    return {"value": np.asarray(value, dtype=np.float32)}


def test_merges_attention_weights_across_scanned_layers():
    base = np.arange(24, dtype=np.float32).reshape(2, 2, 2, 3)
    lora_a = np.arange(16, dtype=np.float32).reshape(2, 2, 2, 2) / 10
    lora_b = np.arange(24, dtype=np.float32).reshape(2, 2, 2, 3) / 20
    params = {
        "PaliGemma": {
            "llm": {
                "attn": {
                    "q_einsum": {
                        "w": leaf(base.copy()),
                        "lora_a": leaf(lora_a),
                        "lora_b": leaf(lora_b),
                    }
                }
            }
        }
    }

    merged = merge_lora_weights(params)

    actual = params["PaliGemma"]["llm"]["attn"]["q_einsum"]["w"]["value"]
    np.testing.assert_allclose(actual, base + lora_a @ lora_b)
    assert merged == ["params/PaliGemma/llm/attn/q_einsum/w"]


def test_merges_both_feed_forward_weights_with_scaling():
    gating = np.ones((2, 3, 4), dtype=np.float32)
    linear = np.full((4, 3), 2.0, dtype=np.float32)
    gating_a = np.arange(12, dtype=np.float32).reshape(2, 3, 2)
    gating_b = np.arange(16, dtype=np.float32).reshape(2, 2, 4)
    linear_a = np.arange(8, dtype=np.float32).reshape(4, 2)
    linear_b = np.arange(6, dtype=np.float32).reshape(2, 3)
    params = {
        "mlp": {
            "gating_einsum": leaf(gating.copy()),
            "gating_einsum_lora_a": leaf(gating_a),
            "gating_einsum_lora_b": leaf(gating_b),
            "linear": leaf(linear.copy()),
            "linear_lora_a": leaf(linear_a),
            "linear_lora_b": leaf(linear_b),
        }
    }

    merged = merge_lora_weights(params, scaling=0.5)

    np.testing.assert_allclose(
        params["mlp"]["gating_einsum"]["value"], gating + 0.5 * (gating_a @ gating_b)
    )
    np.testing.assert_allclose(
        params["mlp"]["linear"]["value"], linear + 0.5 * (linear_a @ linear_b)
    )
    assert merged == ["params/mlp/gating_einsum", "params/mlp/linear"]


def test_dense_checkpoint_is_unchanged():
    params = {"attn": {"w": leaf(np.eye(3))}, "mlp": {"linear": leaf(np.ones((3, 2)))}}
    before = copy.deepcopy(params)

    assert merge_lora_weights(params) == []

    np.testing.assert_equal(params["attn"]["w"]["value"], before["attn"]["w"]["value"])
    np.testing.assert_equal(
        params["mlp"]["linear"]["value"], before["mlp"]["linear"]["value"]
    )


@pytest.mark.parametrize(
    "node",
    [
        {"w": leaf(np.zeros((2, 3))), "lora_a": leaf(np.zeros((2, 1)))},
        {
            "linear": leaf(np.zeros((2, 3))),
            "linear_lora_a": leaf(np.zeros((2, 1))),
            "linear_lora_b": leaf(np.zeros((2, 4))),
        },
    ],
)
def test_rejects_incomplete_or_incompatible_adapters(node):
    with pytest.raises(ValueError):
        merge_lora_weights({"layer": node})
