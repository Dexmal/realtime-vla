import os
import numpy as np
import torch
import torch.nn as nn
import jax
import pickle
import orbax.checkpoint as ocp
import math
import torch.nn.functional as F
from transformers import AutoTokenizer

def convert_weights_pi05(weights, dump_weights, time_embeds):
    num_steps = time_embeds.shape[0]
    # vision encoder weights
    weights['vision_patch_embedding_w'].copy_(torch.tensor(dump_weights['PaliGemma']['img']['embedding']['kernel']['value'], dtype=torch.bfloat16, device="cpu"))
    weights['vision_patch_embedding_b'].copy_(torch.tensor(dump_weights['PaliGemma']['img']['embedding']['bias']['value'], dtype=torch.bfloat16, device="cpu"))
    weights['vision_position_embedding'].copy_(torch.tensor(dump_weights['PaliGemma']['img']['pos_embedding']['value'], dtype=torch.bfloat16, device="cpu").squeeze())

    vision_attn_qkv_w = torch.cat([
        torch.tensor(dump_weights['PaliGemma']['img']['Transformer']['encoderblock']['MultiHeadDotProductAttention_0']['query']['kernel']['value'], dtype=torch.bfloat16, device="cpu").flatten(2),
        torch.tensor(dump_weights['PaliGemma']['img']['Transformer']['encoderblock']['MultiHeadDotProductAttention_0']['key']['kernel']['value'], dtype=torch.bfloat16, device="cpu").flatten(2),
        torch.tensor(dump_weights['PaliGemma']['img']['Transformer']['encoderblock']['MultiHeadDotProductAttention_0']['value']['kernel']['value'], dtype=torch.bfloat16, device="cpu").flatten(2),
    ], dim=2)
    weights['vision_attn_qkv_w'].copy_(vision_attn_qkv_w)
    vision_attn_qkv_b = torch.cat([
        torch.tensor(dump_weights['PaliGemma']['img']['Transformer']['encoderblock']['MultiHeadDotProductAttention_0']['query']['bias']['value'], dtype=torch.bfloat16, device="cpu").flatten(1),
        torch.tensor(dump_weights['PaliGemma']['img']['Transformer']['encoderblock']['MultiHeadDotProductAttention_0']['key']['bias']['value'], dtype=torch.bfloat16, device="cpu").flatten(1),
        torch.tensor(dump_weights['PaliGemma']['img']['Transformer']['encoderblock']['MultiHeadDotProductAttention_0']['value']['bias']['value'], dtype=torch.bfloat16, device="cpu").flatten(1),
    ], dim=1)
    weights['vision_attn_qkv_b'].copy_(vision_attn_qkv_b)
    weights['vision_attn_o_w'].copy_(torch.tensor(dump_weights['PaliGemma']['img']['Transformer']['encoderblock']['MultiHeadDotProductAttention_0']['out']['kernel']['value'], dtype=torch.bfloat16, device="cpu").flatten(1, -2))
    weights['vision_attn_o_b'].copy_(torch.tensor(dump_weights['PaliGemma']['img']['Transformer']['encoderblock']['MultiHeadDotProductAttention_0']['out']['bias']['value'], dtype=torch.bfloat16, device="cpu").flatten(1))

    weights['vision_ffn_up_w'].copy_(torch.tensor(dump_weights['PaliGemma']['img']['Transformer']['encoderblock']['MlpBlock_0']['Dense_0']['kernel']['value'], dtype=torch.bfloat16, device="cpu"))
    weights['vision_ffn_up_b'].copy_(torch.tensor(dump_weights['PaliGemma']['img']['Transformer']['encoderblock']['MlpBlock_0']['Dense_0']['bias']['value'], dtype=torch.bfloat16, device="cpu"))
    weights['vision_ffn_down_w'].copy_(torch.tensor(dump_weights['PaliGemma']['img']['Transformer']['encoderblock']['MlpBlock_0']['Dense_1']['kernel']['value'], dtype=torch.bfloat16, device="cpu"))
    weights['vision_ffn_down_b'].copy_(torch.tensor(dump_weights['PaliGemma']['img']['Transformer']['encoderblock']['MlpBlock_0']['Dense_1']['bias']['value'], dtype=torch.bfloat16, device="cpu"))

    weights['vision_pre_attn_norm_w'].copy_(torch.tensor(dump_weights['PaliGemma']['img']['Transformer']['encoderblock']['LayerNorm_0']['scale']['value'], dtype=torch.bfloat16, device="cpu"))
    weights['vision_pre_attn_norm_b'].copy_(torch.tensor(dump_weights['PaliGemma']['img']['Transformer']['encoderblock']['LayerNorm_0']['bias']['value'], dtype=torch.bfloat16, device="cpu"))
    weights['vision_pre_ffn_norm_w'].copy_(torch.tensor(dump_weights['PaliGemma']['img']['Transformer']['encoderblock']['LayerNorm_1']['scale']['value'], dtype=torch.bfloat16, device="cpu"))
    weights['vision_pre_ffn_norm_b'].copy_(torch.tensor(dump_weights['PaliGemma']['img']['Transformer']['encoderblock']['LayerNorm_1']['bias']['value'], dtype=torch.bfloat16, device="cpu"))
    weights['vision_final_norm_w'].copy_(torch.tensor(dump_weights['PaliGemma']['img']['Transformer']['encoder_norm']['scale']['value'], dtype=torch.bfloat16, device="cpu"))
    weights['vision_final_norm_b'].copy_(torch.tensor(dump_weights['PaliGemma']['img']['Transformer']['encoder_norm']['bias']['value'], dtype=torch.bfloat16, device="cpu"))

    # encoder weights
    weights['encoder_multi_modal_projector_w'].copy_(torch.tensor(dump_weights['PaliGemma']['img']['head']['kernel']['value'], dtype=torch.bfloat16, device="cpu"))
    weights['encoder_multi_modal_projector_b'].copy_(torch.tensor(dump_weights['PaliGemma']['img']['head']['bias']['value'], dtype=torch.bfloat16, device="cpu"))
    w_scale = dump_weights['PaliGemma']['llm']['layers']['pre_attention_norm']['scale']['value'].astype('float32')

    w_q = dump_weights['PaliGemma']['llm']['layers']['attn']['q_einsum']['w']['value'].astype('float32')
    w_q = w_q.transpose((0, 2, 1, 3)).reshape((18, 2048, 8 * 256))
    w_k = dump_weights['PaliGemma']['llm']['layers']['attn']['kv_einsum']['w']['value'][:, 0, 0].astype('float32')
    w_v = dump_weights['PaliGemma']['llm']['layers']['attn']['kv_einsum']['w']['value'][:, 1, 0].astype('float32')
    w_q *= (1 + w_scale[:, :, None])
    w_k *= (1 + w_scale[:, :, None])
    w_v *= (1 + w_scale[:, :, None])
    w_q = w_q.reshape((18, 2048, 8, 2, 128)).transpose((0, 1, 2, 4, 3)).reshape((18, 2048, 2048))
    w_k = w_k.reshape((18, 2048, 2, 128)).transpose((0, 1, 3, 2)).reshape((18, 2048, 256))

    weights['encoder_attn_qkv_w'].copy_(torch.tensor(
        np.concatenate([w_q, w_k, w_v], axis=2),
        dtype=torch.bfloat16, device="cpu"
    ))

    w_attn_o = dump_weights['PaliGemma']['llm']['layers']['attn']['attn_vec_einsum']['w']['value'].reshape((18, 8 * 256, 2048)).astype('float32')
    weights['encoder_attn_o_w'].copy_(torch.tensor(
        w_attn_o,
        dtype=torch.bfloat16, device="cpu"
    ))

    rms_norm = dump_weights['PaliGemma']['llm']['layers']['pre_ffw_norm']['scale']['value'].astype('float32')
    w_gate = dump_weights['PaliGemma']['llm']['layers']['mlp']['gating_einsum']['value'][:, 0].astype('float32')
    w_up = dump_weights['PaliGemma']['llm']['layers']['mlp']['gating_einsum']['value'][:, 1].astype('float32')
    w_down = dump_weights['PaliGemma']['llm']['layers']['mlp']['linear']['value'].astype('float32')
    w_gate *= (1 + rms_norm[:, :, None])
    w_up *= (1 + rms_norm[:, :, None])

    weights['encoder_ffn_gate_w'].copy_(torch.tensor(
        w_gate,
        dtype=torch.bfloat16, device="cpu"
    ))
    weights['encoder_ffn_up_w'].copy_(torch.tensor(
        w_up,
        dtype=torch.bfloat16, device="cpu"
    ))
    weights['encoder_ffn_down_w'].copy_(torch.tensor(
        w_down,
        dtype=torch.bfloat16, device="cpu"
    ))

    # decoder weights
    w_q = dump_weights['PaliGemma']['llm']['layers']['attn']['q_einsum_1']['w']['value'].astype('float32')
    w_q = w_q.transpose((0, 2, 1, 3)).reshape((18, 1024, 8 * 256))
    w_k = dump_weights['PaliGemma']['llm']['layers']['attn']['kv_einsum_1']['w']['value'][:, 0, 0].astype('float32')
    w_v = dump_weights['PaliGemma']['llm']['layers']['attn']['kv_einsum_1']['w']['value'][:, 1, 0].astype('float32')
    w_q = w_q.reshape((18, 1024, 8, 2, 128)).transpose((0, 1, 2, 4, 3)).reshape((18, 1024, 2048))
    w_k = w_k.reshape((18, 1024, 2, 128)).transpose((0, 1, 3, 2)).reshape((18, 1024, 256))
    decoder_attn_qkv_w = torch.tensor(
        np.concatenate([w_q, w_k, w_v], axis=2),
        dtype=torch.float32, device="cpu")


    w_attn_o = dump_weights['PaliGemma']['llm']['layers']['attn']['attn_vec_einsum_1']['w']['value'].reshape((18, 8 * 256, 1024)).astype('float32')
    weights['decoder_attn_o_w'].copy_(torch.tensor(
        w_attn_o,
        dtype=torch.bfloat16, device="cpu"
    ))

    w_gate = dump_weights['PaliGemma']['llm']['layers']['mlp_1']['gating_einsum']['value'][:, 0].astype('float32')
    w_up = dump_weights['PaliGemma']['llm']['layers']['mlp_1']['gating_einsum']['value'][:, 1].astype('float32')
    w_down = dump_weights['PaliGemma']['llm']['layers']['mlp_1']['linear']['value'].astype('float32')
    attn_adarms_kernel = torch.tensor(dump_weights['PaliGemma']['llm']['layers']['pre_attention_norm_1']['Dense_0']['kernel']['value'], dtype=torch.bfloat16, device="cpu")
    attn_adarms_bias = torch.tensor(dump_weights['PaliGemma']['llm']['layers']['pre_attention_norm_1']['Dense_0']['bias']['value'], dtype=torch.bfloat16, device="cpu")

    ffn_adarms_kernel = torch.tensor(dump_weights['PaliGemma']['llm']['layers']['pre_ffw_norm_1']['Dense_0']['kernel']['value'], dtype=torch.bfloat16, device="cpu")
    ffn_adarms_bias = torch.tensor(dump_weights['PaliGemma']['llm']['layers']['pre_ffw_norm_1']['Dense_0']['bias']['value'], dtype=torch.bfloat16, device="cpu")


    time_mlp_in_kernel = torch.tensor(dump_weights['time_mlp_in']['kernel']['value'], dtype=torch.bfloat16, device="cpu")
    time_mlp_in_bias = torch.tensor(dump_weights['time_mlp_in']['bias']['value'], dtype=torch.bfloat16, device="cpu")
    time_mlp_out_kernel = torch.tensor(dump_weights['time_mlp_out']['kernel']['value'], dtype=torch.bfloat16, device="cpu")
    time_mlp_out_bias = torch.tensor(dump_weights['time_mlp_out']['bias']['value'], dtype=torch.bfloat16, device="cpu")

    action_in_proj_w = torch.tensor(
        dump_weights['action_in_proj']['kernel']['value'],
        dtype=torch.bfloat16, device="cpu"
    )
    action_in_proj_b = torch.tensor(
        dump_weights['action_in_proj']['bias']['value'],
        dtype=torch.bfloat16, device="cpu"
    )

    weights['decoder_action_in_proj_w'].copy_(action_in_proj_w)
    weights['decoder_action_in_proj_b'].copy_(action_in_proj_b)

    action_out_kernel = torch.tensor(
        dump_weights['action_out_proj']['kernel']['value'],
        dtype=torch.float32, device="cpu"
    )
    action_out_bias = torch.tensor(
        dump_weights['action_out_proj']['bias']['value'],
        dtype=torch.float32, device="cpu"
    )
    action_out_kernel *= -1.0 / num_steps
    action_out_bias *= -1.0 / num_steps
    final_norm_1 = dump_weights['PaliGemma']['llm']['final_norm_1']
    final_adarms_kernel = torch.tensor(final_norm_1['Dense_0']['kernel']['value'], dtype=torch.bfloat16, device="cpu")
    final_adarms_bias = torch.tensor(final_norm_1['Dense_0']['bias']['value'], dtype=torch.bfloat16, device="cpu")


    decoder_time_emb = []
    for step in range(num_steps):
        decoder_time_emb = F.silu(  F.silu((time_embeds[step].view(1,-1) @ time_mlp_in_kernel + time_mlp_in_bias)) @ time_mlp_out_kernel + time_mlp_out_bias  ) 
        for i in range(18):
            decoder_style_attn = decoder_time_emb @ attn_adarms_kernel[i] + attn_adarms_bias[i]
            scale, shift, weights['decoder_gate_attn'][step, i] = torch.chunk(decoder_style_attn, 3, dim=-1)
            scale = scale.to(dtype=torch.float32)
            shift = shift.to(dtype=torch.float32)          
            fused_scale = (1 + scale).view(-1, 1)
            attn_w_fused = fused_scale * decoder_attn_qkv_w[i] 
            attn_b_fused = shift @ decoder_attn_qkv_w[i]
            weights['decoder_attn_fused_qkv_w'][step, i].copy_(attn_w_fused.squeeze(0).to(dtype=torch.bfloat16))
            weights['decoder_attn_fused_qkv_b'][step, i].copy_(attn_b_fused.squeeze(0).to(dtype=torch.bfloat16))

            decoder_style_ffn = decoder_time_emb @ ffn_adarms_kernel[i] + ffn_adarms_bias[i]
            scale, shift, weights['decoder_gate_ffn'][step, i] = torch.chunk(decoder_style_ffn, 3, dim=-1)
            scale = scale.to(dtype=torch.float32)
            shift = shift.to(dtype=torch.float32)
            gate_w = torch.tensor(w_gate[i], dtype=torch.float32, device="cpu")
            up_w = torch.tensor(w_up[i], dtype=torch.float32, device="cpu")
            fused_scale = (1 + scale).view(-1, 1)
            gate_w_fused = fused_scale * gate_w
            up_w_fused = fused_scale * up_w
            gate_b_fused = shift @ gate_w
            up_b_fused = shift @ up_w
            weights['decoder_ffn_fused_gate_w'][step, i].copy_(gate_w_fused.to(dtype=torch.bfloat16))
            weights['decoder_ffn_fused_up_w'][step, i].copy_(up_w_fused.to(dtype=torch.bfloat16))
            weights['decoder_ffn_fused_gate_b'][step, i].copy_(gate_b_fused.squeeze(0).to(dtype=torch.bfloat16))
            weights['decoder_ffn_fused_up_b'][step, i].copy_(up_b_fused.squeeze(0).to(dtype=torch.bfloat16))

        decoder_style_final = decoder_time_emb @ final_adarms_kernel + final_adarms_bias
        scale, shift, _ = torch.chunk(decoder_style_final, 3, dim=-1)
        scale = scale.to(dtype=torch.float32)
        shift = shift.to(dtype=torch.float32)
        fused_scale = (1 + scale).view(-1, 1)
        action_w_fused = fused_scale * action_out_kernel 
        action_b_fused = shift @ action_out_kernel + action_out_bias
        weights['decoder_action_fused_out_proj_w'][step].copy_(action_w_fused.squeeze(0).to(dtype=torch.bfloat16))
        weights['decoder_action_fused_out_proj_b'][step].copy_(action_b_fused.squeeze(0).to(dtype=torch.bfloat16))

    weights['decoder_ffn_down_w'].copy_(torch.tensor(
        w_down,
        dtype=torch.bfloat16, device="cpu"
    ))
def load_jax_weights(jax_path: str):
    params_path = os.path.join(jax_path, "params")
    print(f"Loading jax weights from {params_path}")

    single_device_sharding = jax.sharding.SingleDeviceSharding(jax.devices("cpu")[0])
    
    with ocp.PyTreeCheckpointer() as ckptr:
        metadata = ckptr.metadata(params_path)
        params = ckptr.restore(
            params_path,
            ocp.args.PyTreeRestore(
                item=metadata,
                restore_args=jax.tree.map(
                    lambda _: ocp.ArrayRestoreArgs(
                        restore_type=jax.Array,
                        sharding=single_device_sharding,
                    ),
                    metadata,
                ),
                transforms={},
            ),
        )["params"]
        print(f"Loaded JAX params keys: {params.keys()}")

    def _ensure_value_dict(node):
        if isinstance(node, dict):
            new_node = {}
            for key, value in node.items():
                if key == "value":
                    if isinstance(value, dict):
                        new_node[key] = _ensure_value_dict(value)
                    else:
                        new_node[key] = value
                else:
                    new_node[key] = _ensure_value_dict(value)
            return new_node
        else:
            return {"value": node}

    return _ensure_value_dict(params)

def prepare_adarms_cond(num_steps):
    dt = -1.0 / num_steps
    time = torch.tensor(1.0, dtype=torch.float32, device="cpu")
    min_period = 4e-3
    max_period = 4.0
    embedding_dim = 1024
    fraction = torch.linspace(0.0, 1.0, embedding_dim // 2, device="cpu")
    period = min_period * (max_period / min_period) ** fraction
    time_emb = []
    for step in range(num_steps):
        sinusoid_input = time.unsqueeze(-1) * (1.0 / period).unsqueeze(0) * 2 * math.pi
        time_emb.append(torch.cat([torch.sin(sinusoid_input), torch.cos(sinusoid_input)], dim=-1).to(torch.bfloat16))
        time = time + dt
    time_emb = torch.cat(time_emb, dim=0)
    return time_emb

def prepare_prompt(prompt: str, embedding_weight, tokenizer_path):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    embedding_weight_torch = torch.tensor(
        embedding_weight,
        dtype=torch.bfloat16, device="cpu"
    )
    num_embeddings, embedding_dim = embedding_weight_torch.shape
    language_embedding = nn.Embedding(num_embeddings=num_embeddings, embedding_dim=embedding_dim).bfloat16()
    with torch.no_grad():
        language_embedding.weight.copy_(embedding_weight_torch)
    prompt = [prompt.strip().replace("_", " ") + "\n"]
    language_tokens = tokenizer(
        prompt,
        max_length=48,
        return_tensors="pt",
    )["input_ids"].squeeze(0)
    language_embeds = language_embedding(language_tokens)
    language_embeds *= language_embeds.shape[-1] ** 0.5
    return language_embeds, language_embeds.shape[0]

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Convert JAX Pi0.5 weights to PyTorch")
    parser.add_argument('--jax_path', type=str, required=True)
    parser.add_argument('--output', type=str, required=True)
    parser.add_argument('--prompt', type=str, required=True)
    parser.add_argument('--tokenizer_path', type=str, required=True)
    args = parser.parse_args()
    dump_weights = load_jax_weights(args.jax_path)
    embedding_weight = dump_weights['PaliGemma']['llm']['embedder']['input_embedding']['value']
    embedding_weight_torch = torch.tensor(
        embedding_weight,
        dtype=torch.bfloat16, device="cpu"
    )
    language_embeds, prompt_len = prepare_prompt(args.prompt, embedding_weight, args.tokenizer_path)
    num_steps = 10
    time_embeds = prepare_adarms_cond(num_steps = num_steps)
    weights = {
        
        "embedding_weight":                   torch.zeros(257152,2048,            dtype=torch.bfloat16, device="cpu"),
        
        "vision_patch_embedding_w":           torch.zeros(14, 14, 3, 1152,        dtype=torch.bfloat16, device="cpu"),
        "vision_patch_embedding_b":           torch.zeros(1152,                   dtype=torch.bfloat16, device="cpu"),
        "vision_position_embedding":          torch.zeros(256, 1152,              dtype=torch.bfloat16, device="cpu"),
        "vision_attn_qkv_w":                  torch.zeros(27, 1152, 3 * 1152,     dtype=torch.bfloat16, device="cpu"),
        "vision_attn_qkv_b":                  torch.zeros(27, 3 * 1152,           dtype=torch.bfloat16, device="cpu"),
        "vision_attn_o_w":                    torch.zeros(27, 1152, 1152,         dtype=torch.bfloat16, device="cpu"),
        "vision_attn_o_b":                    torch.zeros(27, 1152,               dtype=torch.bfloat16, device="cpu"),
        "vision_ffn_up_w":                    torch.zeros(27, 1152, 4304,         dtype=torch.bfloat16, device="cpu"),
        "vision_ffn_up_b":                    torch.zeros(27, 4304,               dtype=torch.bfloat16, device="cpu"),
        "vision_ffn_down_w":                  torch.zeros(27, 4304, 1152,         dtype=torch.bfloat16, device="cpu"),
        "vision_ffn_down_b":                  torch.zeros(27, 1152,               dtype=torch.bfloat16, device="cpu"),
        "vision_pre_attn_norm_w":             torch.zeros(27, 1152,               dtype=torch.bfloat16, device="cpu"),
        "vision_pre_attn_norm_b":             torch.zeros(27, 1152,               dtype=torch.bfloat16, device="cpu"),
        "vision_pre_ffn_norm_w":              torch.zeros(27, 1152,               dtype=torch.bfloat16, device="cpu"),
        "vision_pre_ffn_norm_b":              torch.zeros(27, 1152,               dtype=torch.bfloat16, device="cpu"),
        "vision_final_norm_w":                torch.zeros(1152,                   dtype=torch.bfloat16, device="cpu"),
        "vision_final_norm_b":                torch.zeros(1152,                   dtype=torch.bfloat16, device="cpu"),


        "encoder_multi_modal_projector_w":    torch.zeros(1152, 2048,             dtype=torch.bfloat16, device="cpu"),
        "encoder_multi_modal_projector_b":    torch.zeros(2048,                   dtype=torch.bfloat16, device="cpu"),
        "encoder_attn_qkv_w":                 torch.zeros(18, 2048, 2560,         dtype=torch.bfloat16, device="cpu"),
        "encoder_attn_o_w":                   torch.zeros(18, 2048, 2048,         dtype=torch.bfloat16, device="cpu"),
        "encoder_ffn_gate_w":                 torch.zeros(18, 2048, 16384,        dtype=torch.bfloat16, device="cpu"),
        "encoder_ffn_up_w":                   torch.zeros(18, 2048, 16384,        dtype=torch.bfloat16, device="cpu"),
        "encoder_ffn_down_w":                 torch.zeros(18, 16384, 2048,        dtype=torch.bfloat16, device="cpu"),

        "decoder_attn_fused_qkv_w":                 torch.zeros(num_steps, 18, 1024, 2560,         dtype=torch.bfloat16, device="cpu"),
        "decoder_attn_fused_qkv_b":                 torch.zeros(num_steps, 18, 2560,               dtype=torch.bfloat16, device="cpu"),
        
        "decoder_attn_o_w":                   torch.zeros(18, 2048, 1024,         dtype=torch.bfloat16, device="cpu"),
        "decoder_ffn_fused_gate_w":                 torch.zeros(num_steps, 18, 1024, 4096,         dtype=torch.bfloat16, device="cpu"),
        "decoder_ffn_fused_gate_b":                 torch.zeros(num_steps, 18, 4096,               dtype=torch.bfloat16, device="cpu"),
        "decoder_ffn_fused_up_w":                   torch.zeros(num_steps, 18, 1024, 4096,         dtype=torch.bfloat16, device="cpu"),
        "decoder_ffn_fused_up_b":                   torch.zeros(num_steps, 18, 4096,               dtype=torch.bfloat16, device="cpu"),
        "decoder_ffn_down_w":                 torch.zeros(18, 4096, 1024,         dtype=torch.bfloat16, device="cpu"),
        "decoder_action_in_proj_w":           torch.zeros(32, 1024,               dtype=torch.bfloat16, device="cpu"),
        "decoder_action_in_proj_b":           torch.zeros(1024,                   dtype=torch.bfloat16, device="cpu"),

        "decoder_action_fused_out_proj_w":    torch.zeros(num_steps, 1024, 32,               dtype = torch.bfloat16, device = "cpu"),
        "decoder_action_fused_out_proj_b":    torch.zeros(num_steps, 32,                     dtype = torch.bfloat16, device = "cpu"),


        "decoder_gate_attn":               torch.zeros((num_steps, 18, 1024),      dtype = torch.bfloat16, device = "cpu"), 
        
        "decoder_gate_ffn":               torch.zeros((num_steps, 18, 1024),      dtype = torch.bfloat16, device = "cpu"), 

        
        "language_embeds": torch.zeros(prompt_len, 2048, dtype=torch.bfloat16, device="cpu"),
    }

    convert_weights_pi05(weights, dump_weights, time_embeds)
    weights['embedding_weight'].copy_(embedding_weight_torch)
    weights['language_embeds'].copy_(language_embeds)
    with open(args.output, 'wb') as f:
        pickle.dump(weights, f)
    print(f"Successfully converted Pi0.5 weights to {args.output}")
