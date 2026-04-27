import torch
import triton
import triton.language as tl
import numpy as np
import torch.nn as nn
from transformers import AutoTokenizer
from pi0_infer import (
    vision_encoder,
    layer_norm_matmul_n256_1152_2048_bias,
    rms_matmul_n_2048_2560_qkv_rope,
    matmul_n_2048_2048_res,
    matmul_n_16384_2048_res,
    rms_matmul_n_2048_16384_gate,
    matmul_small_bias,
    matmul_k8_n_256,
    rmsnorm_factor_kernel,
    matmul_abT_scale,
    scaled_matmul_small_bias_res,
)
@triton.jit
def matmul_small_res_gate(
    inp_ptr, weight_ptr, out_ptr, res_ptr, gate_ptr, 
    seq_len : tl.constexpr, features : tl.constexpr, hidden : tl.constexpr,
    BLOCK_SIZE_N : tl.constexpr, BLOCK_SIZE_M : tl.constexpr, BLOCK_SIZE_K : tl.constexpr
):
    pid = tl.program_id(0)
    psize = tl.num_programs(0)
    grid_i = tl.cdiv(seq_len, BLOCK_SIZE_N)
    grid_j = tl.cdiv(hidden, BLOCK_SIZE_M)
    for p in range(pid, grid_i * grid_j, psize):
        i = (p // grid_j) * BLOCK_SIZE_N
        j = (p % grid_j) * BLOCK_SIZE_M
        
        acc = tl.load(
            res_ptr + (i + tl.arange(0, BLOCK_SIZE_N))[:, None] * hidden + (j + tl.arange(0, BLOCK_SIZE_M))[None, :],
            mask = ((i + tl.arange(0, BLOCK_SIZE_N))[:, None] < seq_len) & ((j + tl.arange(0, BLOCK_SIZE_M))[None, :] < hidden),
            other = 0.0
        ).to(tl.float32)
        matmul_acc = tl.zeros((BLOCK_SIZE_N, BLOCK_SIZE_M), dtype=tl.float32)
        for k in range(0, features, BLOCK_SIZE_K):
            x = tl.load(
                inp_ptr + (i + tl.arange(0, BLOCK_SIZE_N))[:, None] * features + (k + tl.arange(0, BLOCK_SIZE_K))[None, :],
                mask = ((i + tl.arange(0, BLOCK_SIZE_N))[:, None] < seq_len) & ((k + tl.arange(0, BLOCK_SIZE_K))[None, :] < features),
                other = 0.0
            )
            w = tl.load(
                weight_ptr + (k + tl.arange(0, BLOCK_SIZE_K))[:, None] * hidden + (j + tl.arange(0, BLOCK_SIZE_M))[None, :],
                mask = ((k + tl.arange(0, BLOCK_SIZE_K))[:, None] < features) & ((j + tl.arange(0, BLOCK_SIZE_M))[None, :] < hidden),
                other = 0.0
            )
            matmul_acc = tl.dot(x, w, matmul_acc)
        
        gate = tl.load(
            gate_ptr + (i + tl.arange(0, BLOCK_SIZE_N))[:, None] * hidden + (j + tl.arange(0, BLOCK_SIZE_M))[None, :],
            mask = ((i + tl.arange(0, BLOCK_SIZE_N))[:, None] < seq_len) & ((j + tl.arange(0, BLOCK_SIZE_M))[None, :] < hidden),
            other = 0.0
        ).to(tl.float32)

        acc += matmul_acc * gate
        
        tl.store(
            out_ptr + (i + tl.arange(0, BLOCK_SIZE_N))[:, None] * hidden + (j + tl.arange(0, BLOCK_SIZE_M))[None, :],
            acc.to(tl.bfloat16),
            mask = ((i + tl.arange(0, BLOCK_SIZE_N))[:, None] < seq_len) & ((j + tl.arange(0, BLOCK_SIZE_M))[None, :] < hidden)
        )

def matmul_k_32_1024_bias(x, weight, bias, out):
    seq_len = x.shape[0]
    matmul_small_bias[((seq_len + 31) // 32) * (1024 // 32),] (
        x, weight, out, bias,
        seq_len = seq_len,
        features = 32,
        hidden = 1024,
        BLOCK_SIZE_N = 32,
        BLOCK_SIZE_M = 32,
        BLOCK_SIZE_K = 32
    )

@triton.jit
def scaled_matmul_bias_rope_qkv(
    inp_ptr, inp_norm_factor_ptr,
    weight_qkv_ptr, bias_ptr, rope_weights_ptr, q_ptr, k_ptr, v_ptr,
    seq_len: tl.constexpr, features: tl.constexpr, head_dim: tl.constexpr, num_heads: tl.constexpr,
    BLOCK_SIZE_M : tl.constexpr = 64, BLOCK_SIZE_N : tl.constexpr = 32, BLOCK_SIZE_K : tl.constexpr = 64,
):
    pid = tl.program_id(axis=0)
    psize = tl.num_programs(axis=0)

    grid_m = triton.cdiv(seq_len, BLOCK_SIZE_M)
    grid_n = triton.cdiv((num_heads + 2) * head_dim, BLOCK_SIZE_N)

    assert head_dim % BLOCK_SIZE_N == 0, f"head_dim {head_dim} must be divisible by BLOCK_SIZE_N {BLOCK_SIZE_N}"

    while pid < grid_m * grid_n:
        pid_m = pid // grid_n
        pid_n = pid % grid_n
        start_i = pid_m * BLOCK_SIZE_M
        start_j = pid_n * BLOCK_SIZE_N
        offs_i = start_i + tl.arange(0, BLOCK_SIZE_M)[:, None]
        offs_j = start_j + tl.arange(0, BLOCK_SIZE_N)[None, :]

        norm_factor = tl.load(
            inp_norm_factor_ptr + start_i + tl.arange(0, BLOCK_SIZE_M),
            mask = start_i + tl.arange(0, BLOCK_SIZE_M) < seq_len, other=0
        )

        accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
        accumulator += tl.load(
            bias_ptr + (start_j + tl.arange(0, BLOCK_SIZE_N))[None, :],
            mask = ((start_j + tl.arange(0, BLOCK_SIZE_N))[None, :] < (num_heads + 2) * head_dim),
            other = 0.0
        )
        for k in range(0, features, BLOCK_SIZE_K):
            offs_k = k + tl.arange(0, BLOCK_SIZE_K)
            x = tl.load(
                inp_ptr + offs_i * features + offs_k[None, :],
                mask = (offs_i < seq_len) & (offs_k[None, :] < features), other = 0
            )
            x = x * norm_factor[:, None]
            w = tl.load(
                weight_qkv_ptr + offs_k[:, None] * ((num_heads + 2) * head_dim) + offs_j,
                mask = (offs_k[:, None] < features) & (offs_j < (num_heads + 2) * head_dim), other = 0
            )
            accumulator = tl.dot(x, w, accumulator)

        if start_j < (num_heads + 1) * head_dim:
            x0, x1 = tl.split(accumulator.reshape(BLOCK_SIZE_M, BLOCK_SIZE_N // 2, 2))
            x_cossin = tl.load(rope_weights_ptr + offs_i * head_dim + offs_j % head_dim, mask = offs_i < seq_len, other = 0)
            x_cos, x_sin = tl.split(x_cossin.reshape(BLOCK_SIZE_M, BLOCK_SIZE_N // 2, 2))
            x0_ = x0 * x_cos - x1 * x_sin
            x1_ = x1 * x_cos + x0 * x_sin
            accumulator = tl.interleave(x0_, x1_)

        accumulator = accumulator.to(tl.bfloat16)

        if start_j < num_heads * head_dim:
            out_ptr = q_ptr
            out_stride = num_heads * head_dim
        elif start_j < (num_heads + 1) * head_dim:
            out_ptr = k_ptr
            out_stride = head_dim
        else:
            out_ptr = v_ptr
            out_stride = head_dim
        tl.store(
            out_ptr + offs_i * out_stride + offs_j % out_stride,
            accumulator,
            mask = (offs_i < seq_len) & (offs_j < (num_heads + 2) * head_dim)
        )
        pid += psize

def adarms_matmul_k_1024_2560_qkv_rope(x, weight_qkv, bias_ptr, rope_weight, Q, K, V, x_norm_factor):
    head_dim = 256
    num_heads = 8
    seq_len = x.shape[0]
    rmsnorm_factor_kernel[(128,)](x, x_norm_factor, seq_len, 1024, eps=1e-6, BLOCK_SIZE=1024)
    scaled_matmul_bias_rope_qkv[(128,)](
        x, x_norm_factor,
        weight_qkv, bias_ptr, rope_weight, 
        Q, K, V,
        seq_len = seq_len, 
        features = 1024, 
        head_dim = head_dim,
        num_heads = num_heads,
    )
@triton.jit
def scaled_matmul_small_bias_gate(
    inp_ptr, inp_norm_factor_ptr, weight1_ptr, bias1_ptr, weight2_ptr, bias2_ptr, out_ptr, 
    seq_len : tl.constexpr, features : tl.constexpr, hidden: tl.constexpr, 
    BLOCK_SIZE_N : tl.constexpr = 64, BLOCK_SIZE_M : tl.constexpr = 64, BLOCK_SIZE_K : tl.constexpr = 64
):

    pid = tl.program_id(0)
    psize = tl.num_programs(0)
    grid_i = tl.cdiv(seq_len, BLOCK_SIZE_N)
    grid_j = tl.cdiv(hidden, BLOCK_SIZE_M)
    for p in range(pid, grid_i * grid_j, psize):
        i = (p // grid_j) * BLOCK_SIZE_N
        j = (p % grid_j) * BLOCK_SIZE_M
        norm_factor = tl.load(
            inp_norm_factor_ptr + i + tl.arange(0, BLOCK_SIZE_N),
            mask = i + tl.arange(0, BLOCK_SIZE_N) < seq_len, other=0
        )
        acc = tl.zeros((BLOCK_SIZE_N, BLOCK_SIZE_M), dtype=tl.float32)
        acc += tl.load(
            bias1_ptr + (j + tl.arange(0, BLOCK_SIZE_M))[None, :],
            mask = ((j + tl.arange(0, BLOCK_SIZE_M))[None, :] < hidden),
            other = 0.0
        )
        acc2 = tl.zeros((BLOCK_SIZE_N, BLOCK_SIZE_M), dtype=tl.float32)
        acc2 += tl.load(
            bias2_ptr + (j + tl.arange(0, BLOCK_SIZE_M))[None, :],
            mask = ((j + tl.arange(0, BLOCK_SIZE_M))[None, :] < hidden),
            other = 0.0
        )
        for k in range(0, features, BLOCK_SIZE_K):
            x = tl.load(
                inp_ptr + (i + tl.arange(0, BLOCK_SIZE_N)[:, None]) * features + k + tl.arange(0, BLOCK_SIZE_K), 
                mask = i + tl.arange(0, BLOCK_SIZE_N)[:, None] < seq_len,
                other = 0.0
            )
            x = x * norm_factor[:, None]
            w = tl.load(weight1_ptr + (k + tl.arange(0, BLOCK_SIZE_K)[:, None]) * hidden + j + tl.arange(0, BLOCK_SIZE_M))
            acc = tl.dot(x, w, acc)
            w2 = tl.load(weight2_ptr + (k + tl.arange(0, BLOCK_SIZE_K)[:, None]) * hidden + j + tl.arange(0, BLOCK_SIZE_M))
            acc2 = tl.dot(x, w2, acc2)
        acc = acc * tl.sigmoid(1.5957691216057308 * acc * (1 + 0.044715 * acc * acc))
        acc = (acc * acc2).to(tl.bfloat16)
        tl.store(out_ptr + (i + tl.arange(0, BLOCK_SIZE_N)[:, None]) * hidden + j + tl.arange(0, BLOCK_SIZE_M), acc, mask = i + tl.arange(0, BLOCK_SIZE_N)[:, None] < seq_len)

def adarms_matmul_k_1024_4096_gate(x, weight1, bias1, weight2, bias2, out, x_norm_factor):
    seq_len = x.shape[0]
    hidden = 4096
    rmsnorm_factor_kernel[(128,)](x, x_norm_factor, seq_len, 1024, eps=1e-6, BLOCK_SIZE=1024)
    scaled_matmul_small_bias_gate[(128,)] (
        x, x_norm_factor, weight1, bias1, weight2, bias2, out,
        seq_len = seq_len,
        features = 1024,
        hidden = hidden,
        BLOCK_SIZE_N = 64,
        BLOCK_SIZE_M = 64,
        BLOCK_SIZE_K = 64
    )
    
def adarms_matmul_k_1024_32_bias_res(x, weight, bias, out, x_norm_factor):
    seq_len = x.shape[0]
    rmsnorm_factor_kernel[(seq_len,)](x, x_norm_factor, seq_len, 1024, eps=1e-6, BLOCK_SIZE=1024)
    scaled_matmul_small_bias_res[((seq_len + 15) // 16) * (32 // 16),] (
        x, x_norm_factor, weight, out, bias, out,
        seq_len = seq_len,
        features = 1024,
        hidden = 32,
        BLOCK_SIZE_N = 16,
        BLOCK_SIZE_M = 16,
        BLOCK_SIZE_K = 256
    )

def matmul_k_2048_1024_gate(x, weight, out, gate):
    seq_len = x.shape[0]
    matmul_small_res_gate[(128,)](
        x,
        weight,
        out,
        out, 
        gate,
        seq_len = seq_len,
        features = 2048,
        hidden = 1024,
        BLOCK_SIZE_N = 32,
        BLOCK_SIZE_M = 32,
        BLOCK_SIZE_K = 128
    )

def matmul_k_4096_1024_gate(x, weight, out, gate):
    seq_len = x.shape[0]
    matmul_small_res_gate[(((seq_len + 15) // 16) * (1024 // 32),)](
        x,
        weight,
        out,
        out,
        gate,
        seq_len = seq_len,
        features = 4096,
        hidden = 1024,
        BLOCK_SIZE_N = 16,
        BLOCK_SIZE_M = 32,
        BLOCK_SIZE_K = 256
    )

@triton.jit
def softmax_kernel_masklen(
    inp_ptr,
    queries: tl.constexpr,
    keys: tl.constexpr,
    valid_keys_len_ptr,
    out_ptr,
    BLOCK_SIZE_M: tl.constexpr = 4,
    BLOCK_SIZE: tl.constexpr = 1024,
):
    pid = tl.program_id(axis=0)
    psize = tl.num_programs(axis=0)
    big_neg = -2.3819763e38
    assert BLOCK_SIZE >= keys, f"BLOCK_SIZE must be >= keys, got {BLOCK_SIZE} < {keys}"

    valid_keys_len = tl.load(valid_keys_len_ptr).to(tl.int32)
    valid_keys_len = tl.maximum(0, tl.minimum(valid_keys_len, keys))

    for i in range(pid * BLOCK_SIZE_M, queries, psize * BLOCK_SIZE_M):
        offs_i = i + tl.arange(0, BLOCK_SIZE_M)[:, None]
        offs_j = tl.arange(0, BLOCK_SIZE)[None, :]
        attn_mask = (offs_i < queries) & (offs_j < keys) & (offs_j < valid_keys_len)
        vals = tl.load(inp_ptr + offs_i * keys + offs_j, mask=attn_mask, other=big_neg)
        vals = tl.exp(vals - tl.max(vals, axis=1, keep_dims=True))
        vsum = tl.sum(vals, axis=1, keep_dims=True, dtype=tl.float32)
        vals = vals / vsum
        tl.store(out_ptr + offs_i * keys + offs_j, vals.to(tl.bfloat16),
                 mask=(offs_i < queries) & (offs_j < keys))

def transformer_encoder(weights, buffers, encoder_seq_len):
    layer_norm_matmul_n256_1152_2048_bias(
        buffers['vision_x'],
        weights['vision_final_norm_w'],
        weights['vision_final_norm_b'],
        weights['encoder_multi_modal_projector_w'],
        weights['encoder_multi_modal_projector_b'],
        buffers['encoder_x'],
        buffers['vision_x_norm']
    )
    for i in range(18):
        rms_matmul_n_2048_2560_qkv_rope(
            buffers['encoder_x'],
            weights['encoder_attn_qkv_w'][i],
            buffers['encoder_rope_weights'],
            buffers['encoder_Q'],
            buffers['encoder_K'][i, :encoder_seq_len],
            buffers['encoder_V'][i, :encoder_seq_len],
            buffers['encoder_x_norm']
        )
        if i != 17:
            scale = 1.0 / (256 ** 0.5)
            total_queries = buffers['encoder_Q'].shape[0]
            total_keys = encoder_seq_len
            matmul_abT_scale[(((total_queries + 31) // 32) * ((total_keys + 31) // 32),)](
                buffers['encoder_Q'],
                buffers['encoder_K'][i, :encoder_seq_len],
                buffers['encoder_logits_buf'],
                total_queries,
                total_keys,
                256,
                scale,
                BLOCK_SIZE_M=32,
                BLOCK_SIZE_N=32,
                BLOCK_SIZE_K=64,
            )
            softmax_kernel_masklen[((total_queries + 3) // 4,)](
                buffers['encoder_logits_buf'],
                total_queries,
                total_keys,
                buffers['valid_encoder_len'],
                buffers['encoder_attn_buf'],
                BLOCK_SIZE_M=4,
                BLOCK_SIZE=1024,
            )
            matmul_k8_n_256(
                buffers['encoder_attn_buf'],
                buffers['encoder_V'][i, :encoder_seq_len],
                buffers['encoder_ctx_buf'],
            )
            
            matmul_n_2048_2048_res(
                buffers['encoder_ctx_buf'].view(-1, 2048),
                weights['encoder_attn_o_w'][i],
                buffers['encoder_x']
            )
        
            rms_matmul_n_2048_16384_gate(
                buffers['encoder_x'],
                weights['encoder_ffn_gate_w'][i],
                weights['encoder_ffn_up_w'][i],
                buffers['encoder_hidden'],
                buffers['encoder_x_norm']
            )

            matmul_n_16384_2048_res(
                buffers['encoder_hidden'],
                weights['encoder_ffn_down_w'][i],
                buffers['encoder_x']
            )

@triton.jit
def softmax_kernel_prefix_suffix(
    inp_ptr,
    queries: tl.constexpr,
    keys_prefix: tl.constexpr,
    keys_suffix: tl.constexpr,
    valid_prefix_len_ptr,
    out_ptr,
    BLOCK_SIZE_M: tl.constexpr = 4,
    BLOCK_SIZE: tl.constexpr = 1024,
):
    pid = tl.program_id(axis=0)
    psize = tl.num_programs(axis=0)
    big_neg = -2.3819763e38
    total_keys: tl.constexpr = keys_prefix + keys_suffix
    assert BLOCK_SIZE >= total_keys, f"BLOCK_SIZE must be >= total_keys, got {BLOCK_SIZE} < {total_keys}"

    valid_prefix_len = tl.load(valid_prefix_len_ptr).to(tl.int32)
    valid_prefix_len = tl.maximum(0, tl.minimum(valid_prefix_len, keys_prefix))

    for i in range(pid * BLOCK_SIZE_M, queries, psize * BLOCK_SIZE_M):
        offs_i = i + tl.arange(0, BLOCK_SIZE_M)[:, None]
        offs_j = tl.arange(0, BLOCK_SIZE)[None, :]

        in_bounds = (offs_i < queries) & (offs_j < total_keys)
        is_prefix = offs_j < keys_prefix
        prefix_ok = is_prefix & (offs_j < valid_prefix_len)
        suffix_ok = (~is_prefix)
        attn_mask = in_bounds & (prefix_ok | suffix_ok)

        vals = tl.load(inp_ptr + offs_i * total_keys + offs_j, mask=attn_mask, other=big_neg)
        vals = tl.exp(vals - tl.max(vals, axis=1, keep_dims=True))
        vsum = tl.sum(vals, axis=1, keep_dims=True, dtype=tl.float32)
        vals = vals / vsum
        tl.store(out_ptr + offs_i * total_keys + offs_j, vals.to(tl.bfloat16), mask=in_bounds)

def transformer_decoder(weights, buffers, encoder_seq_len, num_steps=10):
    for step in range(num_steps):
        matmul_k_32_1024_bias(
            buffers['diffusion_noise'],
            weights['decoder_action_in_proj_w'],
            weights['decoder_action_in_proj_b'],
            buffers['decoder_x']
        )
        seq_len = buffers['decoder_x'].shape[0]
        for i in range(18):
            adarms_matmul_k_1024_2560_qkv_rope(
                buffers['decoder_x'], 
                weights['decoder_attn_fused_qkv_w'][step, i],
                weights['decoder_attn_fused_qkv_b'][step, i],
                buffers['decoder_rope_weights'],
                buffers['decoder_q_buf'],
                buffers['encoder_K'][i, encoder_seq_len:encoder_seq_len + seq_len],
                buffers['encoder_V'][i, encoder_seq_len:encoder_seq_len + seq_len],
                buffers['decoder_norm_factor_buf'],
            )
            total_queries = buffers['decoder_q_buf'].shape[0]
            prefix_keys = encoder_seq_len
            suffix_keys = seq_len
            total_keys = prefix_keys + suffix_keys

            matmul_abT_scale[(((total_queries + 31) // 32) * ((total_keys + 31) // 32),)](
                buffers['decoder_q_buf'],
                buffers['encoder_K'][i, :encoder_seq_len + seq_len],
                buffers['decoder_logits_buf'],
                total_queries,
                total_keys,
                256,
                256 ** -0.5,
                BLOCK_SIZE_M=32,
                BLOCK_SIZE_N=32,
                BLOCK_SIZE_K=64,
            )

            softmax_kernel_prefix_suffix[((total_queries + 3) // 4,)](
                buffers['decoder_logits_buf'],
                total_queries,
                prefix_keys,
                suffix_keys,
                buffers['valid_encoder_len'],
                buffers['decoder_attn_buf'],
                BLOCK_SIZE_M=4,
                BLOCK_SIZE=1024,
            )

            matmul_k8_n_256(
                buffers['decoder_attn_buf'],
                buffers['encoder_V'][i, :encoder_seq_len + seq_len],
                buffers['decoder_q_buf'],
            )
            matmul_k_2048_1024_gate(
                buffers['decoder_q_buf'].view(-1, 2048),
                weights['decoder_attn_o_w'][i],
                buffers['decoder_x'],
                weights['decoder_gate_attn'][step, i],
            )
            adarms_matmul_k_1024_4096_gate(
                buffers['decoder_x'],
                weights['decoder_ffn_fused_gate_w'][step, i],
                weights['decoder_ffn_fused_gate_b'][step, i],
                weights['decoder_ffn_fused_up_w'][step, i],
                weights['decoder_ffn_fused_up_b'][step, i],
                buffers['decoder_hidden'],
                buffers['decoder_norm_factor_buf'],
            )
            matmul_k_4096_1024_gate(
                buffers['decoder_hidden'],
                weights['decoder_ffn_down_w'][i],
                buffers['decoder_x'],
                weights['decoder_gate_ffn'][step, i],
            )

        adarms_matmul_k_1024_32_bias_res(
            buffers['decoder_x'],
            weights['decoder_action_fused_out_proj_w'][step],
            weights['decoder_action_fused_out_proj_b'][step],
            buffers['diffusion_noise'],
            buffers['decoder_norm_factor_buf'],
        )

def pi05_model(weights, buffers, num_views, encoder_seq_len, num_steps=10):
    vision_encoder(weights, buffers, num_views)
    transformer_encoder(weights, buffers, encoder_seq_len)
    transformer_decoder(weights, buffers, encoder_seq_len, num_steps)

class Pi05Inference:
    def __init__(
        self,
        checkpoint,
        num_views,
        chunk_size,
        tokenizer_path: str | None = None,
        max_tokenize_len: int = 200,
        discrete_state_input: bool = True,
        max_prompt_text: str | None = None,
        state_dim_for_max_prompt: int | None = None,
        num_steps: int = 10,
    ):
        self.discrete_state_input = discrete_state_input
        self.tokenizer_path = tokenizer_path
        self.checkpoint = checkpoint
        self.num_views = num_views
        self.chunk_size = chunk_size
        self.max_tokenize_len = int(max_tokenize_len)
        if discrete_state_input:
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
            if max_prompt_text is not None and state_dim_for_max_prompt is not None:
                self.max_prompt_len = self.estimate_max_prompt_len(
                    tokenizer=self.tokenizer,
                    task_prompt=max_prompt_text,
                    state_dim=int(state_dim_for_max_prompt),
                    max_tokenize_len=self.max_tokenize_len,
                    state_token_value=255,
                )
            else:
                self.max_prompt_len = self.max_tokenize_len
        else:
            self.max_prompt_len = len(checkpoint['language_embeds'])
        print(f"max_prompt_len: {self.max_prompt_len}, max_tokenize_len: {self.max_tokenize_len}")

        encoder_seq_len = num_views * 256 + self.max_prompt_len
        decoder_seq_len = chunk_size 

        self.weights = {
            "vision_patch_embedding_w":           torch.empty(14, 14, 3, 1152,        dtype = torch.bfloat16, device = "cuda"),
            "vision_patch_embedding_b":           torch.empty(1152,                   dtype = torch.bfloat16, device = "cuda"),
            "vision_position_embedding":          torch.empty(256, 1152,              dtype = torch.bfloat16, device = "cuda"),
            "vision_attn_qkv_w":                  torch.empty(27, 1152, 3 * 1152,     dtype = torch.bfloat16, device = "cuda"),
            "vision_attn_qkv_b":                  torch.empty(27, 3 * 1152,           dtype = torch.bfloat16, device = "cuda"),
            "vision_attn_o_w":                    torch.empty(27, 1152, 1152,         dtype = torch.bfloat16, device = "cuda"),
            "vision_attn_o_b":                    torch.empty(27, 1152,               dtype = torch.bfloat16, device = "cuda"),
            "vision_ffn_up_w":                    torch.empty(27, 1152, 4304,         dtype = torch.bfloat16, device = "cuda"),
            "vision_ffn_up_b":                    torch.empty(27, 4304,               dtype = torch.bfloat16, device = "cuda"),
            "vision_ffn_down_w":                  torch.empty(27, 4304, 1152,         dtype = torch.bfloat16, device = "cuda"),
            "vision_ffn_down_b":                  torch.empty(27, 1152,               dtype = torch.bfloat16, device = "cuda"),
            "vision_pre_attn_norm_w":             torch.empty(27, 1152,               dtype = torch.bfloat16, device = "cuda"),
            "vision_pre_attn_norm_b":             torch.empty(27, 1152,               dtype = torch.bfloat16, device = "cuda"),
            "vision_pre_ffn_norm_w":              torch.empty(27, 1152,               dtype = torch.bfloat16, device = "cuda"),
            "vision_pre_ffn_norm_b":              torch.empty(27, 1152,               dtype = torch.bfloat16, device = "cuda"),
            "vision_final_norm_w":                torch.empty(1152,                   dtype = torch.bfloat16, device = "cuda"),
            "vision_final_norm_b":                torch.empty(1152,                   dtype = torch.bfloat16, device = "cuda"),

            "encoder_multi_modal_projector_w":    torch.empty(1152, 2048,             dtype = torch.bfloat16, device = "cuda"),
            "encoder_multi_modal_projector_b":    torch.empty(2048,                   dtype = torch.bfloat16, device = "cuda"),
            "encoder_attn_qkv_w":                 torch.empty(18, 2048, 2560,         dtype = torch.bfloat16, device = "cuda"),
            "encoder_attn_o_w":                   torch.empty(18, 2048, 2048,         dtype = torch.bfloat16, device = "cuda"),
            "encoder_ffn_gate_w":                 torch.empty(18, 2048, 16384,        dtype = torch.bfloat16, device = "cuda"),
            "encoder_ffn_up_w":                   torch.empty(18, 2048, 16384,        dtype = torch.bfloat16, device = "cuda"),
            "encoder_ffn_down_w":                 torch.empty(18, 16384, 2048,        dtype = torch.bfloat16, device = "cuda"),
            
            "decoder_gate_attn":               torch.empty((num_steps, 18, decoder_seq_len, 1024),      dtype = torch.bfloat16, device = "cuda"), 
            "decoder_gate_ffn":               torch.empty((num_steps, 18, decoder_seq_len, 1024),      dtype = torch.bfloat16, device = "cuda"), 

            "decoder_action_in_proj_w":           torch.empty(32, 1024,                      dtype = torch.bfloat16, device = "cuda"),
            "decoder_action_in_proj_b":           torch.empty(1024,                          dtype = torch.bfloat16, device = "cuda"),

            "decoder_attn_fused_qkv_w":                 torch.empty(num_steps, 18, 1024, 2560,         dtype = torch.bfloat16, device = "cuda"),
            "decoder_attn_fused_qkv_b":                 torch.empty(num_steps, 18, 2560,               dtype = torch.bfloat16, device = "cuda"),


            "decoder_attn_o_w":                   torch.empty(18, 2048, 1024,         dtype = torch.bfloat16, device = "cuda"),
            "decoder_ffn_fused_gate_w":                 torch.empty(num_steps, 18, 1024, 4096,         dtype = torch.bfloat16, device = "cuda"),
            "decoder_ffn_fused_gate_b":                 torch.empty(num_steps, 18, 4096,               dtype = torch.bfloat16, device = "cuda"),
            "decoder_ffn_fused_up_w":                   torch.empty(num_steps, 18, 1024, 4096,         dtype = torch.bfloat16, device = "cuda"),
            "decoder_ffn_fused_up_b":                   torch.empty(num_steps, 18, 4096,               dtype = torch.bfloat16, device = "cuda"),
            "decoder_ffn_down_w":                 torch.empty(18, 4096, 1024,         dtype = torch.bfloat16, device = "cuda"),

            "decoder_action_fused_out_proj_w":    torch.empty(num_steps, 1024, 32,               dtype = torch.bfloat16, device = "cuda"),
            "decoder_action_fused_out_proj_b":    torch.empty(num_steps, 32,                     dtype = torch.bfloat16, device = "cuda"),

            "language_embeds":                    torch.empty(len(checkpoint['language_embeds']), 2048,  dtype = torch.bfloat16, device = "cuda"),   
        }


        self.buffers = {
            'observation_images_normalized':      torch.empty(num_views, 224, 224,3,           dtype=torch.bfloat16,   device = "cuda"),
            'diffusion_noise':                    torch.empty(chunk_size, 32,                  dtype = torch.bfloat16, device = "cuda"),
            'vision_x':                           torch.empty(num_views, 256, 1152,            dtype = torch.bfloat16, device = "cuda"),
            'vision_x_norm':                      torch.empty(num_views, 256, 1152,            dtype = torch.bfloat16, device = "cuda"),
            'vision_QKV':                         torch.empty(num_views, 256, 3 * 1152,        dtype = torch.bfloat16, device = "cuda"),
            'vision_hidden':                      torch.empty(num_views, 256, 4304,            dtype = torch.bfloat16, device = "cuda"),
            'vision_x_split_k_buf':               torch.empty((num_views * 256 * 1152 * 4,),   dtype = torch.float32, device = "cuda"),
            'encoder_rope_weights':               torch.empty(encoder_seq_len, 256,            dtype = torch.bfloat16, device = "cuda"),
            'encoder_x':                          torch.empty(encoder_seq_len, 2048,           dtype = torch.bfloat16, device = "cuda"),
            'encoder_x_norm':                     torch.empty(encoder_seq_len, 2048,           dtype = torch.bfloat16, device = "cuda"),
            'encoder_K':                          torch.empty(18, encoder_seq_len + decoder_seq_len, 256,   dtype = torch.bfloat16, device = "cuda"),
            'encoder_V':                          torch.empty(18, encoder_seq_len + decoder_seq_len, 256,   dtype = torch.bfloat16, device = "cuda"),
            'encoder_Q':                          torch.empty(encoder_seq_len * 8, 256,        dtype = torch.bfloat16, device = "cuda"),
            'encoder_hidden':                     torch.empty(encoder_seq_len, 16384,          dtype = torch.bfloat16, device = "cuda"),
            'valid_encoder_len':                  torch.empty((1,),                           dtype = torch.int32, device = "cuda"),
            'encoder_logits_buf':                 torch.empty((encoder_seq_len * 8, encoder_seq_len), dtype=torch.float32,  device="cuda"),
            'encoder_attn_buf':                   torch.empty((encoder_seq_len * 8, encoder_seq_len), dtype=torch.bfloat16, device="cuda"),
            'encoder_ctx_buf':                    torch.empty((encoder_seq_len * 8, 256),     dtype=torch.bfloat16, device="cuda"),
            'decoder_rope_weights':               torch.empty(decoder_seq_len, 256,            dtype = torch.bfloat16, device = "cuda"),
            'decoder_x':                          torch.empty((decoder_seq_len, 1024),         dtype = torch.bfloat16, device = "cuda"),
            'decoder_q_buf':                      torch.empty((decoder_seq_len * 8, 256),      dtype = torch.bfloat16, device = "cuda"),
            'decoder_logits_buf':                 torch.empty((decoder_seq_len * 8, encoder_seq_len + decoder_seq_len),         dtype=torch.float32, device="cuda"),
            'decoder_attn_buf':                   torch.empty((decoder_seq_len * 8, encoder_seq_len + decoder_seq_len),         dtype = torch.bfloat16, device = "cuda"),
            'decoder_hidden':                     torch.empty((decoder_seq_len, 4096),         dtype = torch.bfloat16, device = "cuda"),
            'decoder_norm_factor_buf':            torch.empty((decoder_seq_len,),              dtype = torch.bfloat16, device = "cuda"),
            'x_normed_buf':                       torch.empty((decoder_seq_len, 1024),         dtype = torch.bfloat16, device = "cuda"),

        }

        prefix_alloc = self.num_views * 256 + self.max_prompt_len
        max_pos = (self.num_views * 256 + self.max_prompt_len - 1) + self.chunk_size
        position_ids = torch.arange(max_pos + 1, device="cuda")
        inv_freq = 1.0 / (10000 ** (torch.arange(0, 256, 2, dtype=torch.float32, device="cuda") / 256))
        k_phase = inv_freq[None, :] * position_ids[:, None]
        k_cos = torch.cos(k_phase).to(torch.bfloat16)
        k_sin = torch.sin(k_phase).to(torch.bfloat16)
        self._rope_table = torch.cat([k_cos[:, :, None], k_sin[:, :, None]], 2).view(-1, 256)
        self.buffers['encoder_rope_weights'].copy_(self._rope_table[:prefix_alloc])

        self.buffers['valid_encoder_len'].fill_(self.num_views * 256 + 1)
        for k, v in checkpoint.items():
            if k == "embedding_weight":
                continue
            if k in ("decoder_gate_attn", "decoder_gate_ffn"):
                v = v[:, :, None, :].expand(-1, -1, self.weights[k].shape[2], -1)
            self.weights[k].copy_(v)

        self.prompt_embedding = None
        self._prompt_embed_scale = None
        if self.discrete_state_input:
            if "embedding_weight" not in checkpoint:
                raise KeyError("checkpoint must contain 'embedding_weight' when discrete_state_input=True")
            emb_w = checkpoint["embedding_weight"]
            if isinstance(emb_w, np.ndarray):
                emb_w_t = torch.from_numpy(emb_w)
            else:
                emb_w_t = emb_w
            emb_w_t = emb_w_t.to(device="cuda", dtype=torch.bfloat16, non_blocking=True)
            self.prompt_embedding = nn.Embedding(
                num_embeddings=emb_w_t.shape[0],
                embedding_dim=emb_w_t.shape[1],
                device="cuda",
                dtype=torch.bfloat16,
            )
            with torch.no_grad():
                self.prompt_embedding.weight.copy_(emb_w_t)
            self._prompt_embed_scale = float(emb_w_t.shape[1] ** 0.5)
        self.encoder_seq_len = encoder_seq_len

        self.infer_graph = torch.cuda.CUDAGraph()
        self.record_infer_graph()

    def estimate_max_prompt_len(
        self,
        tokenizer: AutoTokenizer,
        task_prompt: str,
        state_dim: int,
        max_tokenize_len: int = 200,
        state_token_value: int = 255,
    ) -> int:
        task_prompt = task_prompt.strip().replace("_", " ")
        state_str = " ".join([str(int(state_token_value))] * int(state_dim))
        full_prompt = f"Task: {task_prompt}, State: {state_str};\nAction: "
        token_ids = tokenizer(
            full_prompt,
            return_tensors="pt",
            truncation=True,
            max_length=int(max_tokenize_len),
            padding=False,
        )["input_ids"][0]
        return int(token_ids.shape[0])

    def build_prompt_embeds(
        self,
        task_prompt: str,
        state_tokens: np.ndarray
    ) -> tuple[torch.Tensor, int]:
        task_prompt = task_prompt.strip().replace("_", " ")
        state_str = " ".join(map(str, state_tokens.tolist()))
        full_prompt = f"Task: {task_prompt}, State: {state_str};\nAction: "
        token_ids = self.tokenizer(
            full_prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_tokenize_len,
            padding=False,
        )["input_ids"][0].to(device="cuda", non_blocking=True)
        embeds = self.prompt_embedding(token_ids) * self._prompt_embed_scale
        return embeds, int(embeds.shape[0])

    def get_decoder_rope_weights(self, prompt_len: int) -> torch.Tensor:
        start = self.num_views * 256 + prompt_len
        end = start + self.chunk_size
        return self._rope_table[start:end]

    def record_run(self):
        pi05_model(self.weights, self.buffers, self.num_views, self.encoder_seq_len)

    def record_infer_graph(self):
        for _ in range(3):
            self.record_run()
        stream = torch.cuda.Stream()
        with torch.cuda.stream(stream):
            self.infer_graph.capture_begin()
            self.record_run()
            self.infer_graph.capture_end()

    def forward(
        self,
        observation_images_normalized: torch.Tensor,
        diffusion_noise: torch.Tensor,
        task_prompt: str = None,
        state_tokens: np.ndarray = None,
    ) -> torch.Tensor:
        if self.discrete_state_input:
            prompt_embeds, prompt_len = self.build_prompt_embeds(
                task_prompt=task_prompt,
                state_tokens=state_tokens
            )
        else:
            prompt_embeds = self.weights['language_embeds']
            prompt_len = self.weights['language_embeds'].shape[0]
        start = self.num_views * 256
        self.buffers['encoder_x'][start : start + prompt_len].copy_(prompt_embeds)
        self.buffers['valid_encoder_len'].fill_(start + prompt_len)
        self.buffers['decoder_rope_weights'].copy_(self.get_decoder_rope_weights(prompt_len))
        self.buffers['observation_images_normalized'].copy_(observation_images_normalized)
        self.buffers['diffusion_noise'].copy_(diffusion_noise)
        self.infer_graph.replay()
        return self.buffers['diffusion_noise']