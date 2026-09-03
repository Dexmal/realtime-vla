"""Utilities for folding OpenPI LoRA parameters into dense weights."""


def _leaf_value(node, path):
    if (
        not isinstance(node, dict)
        or "value" not in node
        or isinstance(node["value"], dict)
    ):
        raise ValueError(f"Expected an array leaf at {path}")
    return node["value"]


def _merge_pair(node, base_key, a_key, b_key, path, scaling):
    base = _leaf_value(node[base_key], f"{path}/{base_key}")
    lora_a = _leaf_value(node[a_key], f"{path}/{a_key}")
    lora_b = _leaf_value(node[b_key], f"{path}/{b_key}")

    if base.ndim < 2 or lora_a.ndim != base.ndim or lora_b.ndim != base.ndim:
        raise ValueError(
            f"LoRA tensors at {path} must have the same rank as the dense weight: "
            f"{base.shape}, {lora_a.shape}, {lora_b.shape}"
        )
    if lora_a.shape[:-2] != base.shape[:-2] or lora_b.shape[:-2] != base.shape[:-2]:
        raise ValueError(
            f"LoRA batch dimensions at {path} do not match the dense weight: "
            f"{base.shape}, {lora_a.shape}, {lora_b.shape}"
        )
    if (
        lora_a.shape[-2] != base.shape[-2]
        or lora_b.shape[-1] != base.shape[-1]
        or lora_a.shape[-1] != lora_b.shape[-2]
    ):
        raise ValueError(
            f"LoRA matrix dimensions at {path} are incompatible: "
            f"{base.shape}, {lora_a.shape}, {lora_b.shape}"
        )

    delta = lora_a @ lora_b
    if scaling != 1.0:
        delta = delta * scaling
    node[base_key]["value"] = base + delta


def merge_lora_weights(params, scaling=1.0):
    """Fold OpenPI LoRA leaves into their dense weights in place.

    OpenPI stores attention adapters beside ``w`` as ``lora_a`` and
    ``lora_b``. Feed-forward adapters use sibling names such as
    ``gating_einsum_lora_a`` and ``gating_einsum_lora_b``. Leading axes
    (including the scanned transformer-layer axis) are preserved while the
    final two matrix axes are contracted.

    Official Pi0 and Pi0.5 LoRA presets use alpha equal to rank, hence the
    default scale of one. ``scaling`` is exposed for checkpoints trained with
    a different attention LoRA scale.

    Returns the paths of all dense weights that were updated.
    """
    merged = []

    def visit(node, path):
        if not isinstance(node, dict):
            return

        has_a = "lora_a" in node
        has_b = "lora_b" in node
        if has_a or has_b:
            if not (has_a and has_b and "w" in node):
                raise ValueError(f"Incomplete attention LoRA parameters at {path}")
            _merge_pair(node, "w", "lora_a", "lora_b", path, scaling)
            merged.append(f"{path}/w")

        for key in tuple(node):
            if not key.endswith("_lora_a"):
                continue
            base_key = key[: -len("_lora_a")]
            b_key = f"{base_key}_lora_b"
            if base_key not in node or b_key not in node:
                raise ValueError(
                    f"Incomplete feed-forward LoRA parameters at {path}/{base_key}"
                )
            _merge_pair(node, base_key, key, b_key, path, scaling)
            merged.append(f"{path}/{base_key}")

        for key, value in node.items():
            visit(value, f"{path}/{key}")

    visit(params, "params")
    return merged
