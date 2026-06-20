from __future__ import annotations

try:
    from flash_attn_interface import flash_attn_func as _fa3_flash_attn_func
    from flash_attn_interface import flash_attn_varlen_func as _fa3_flash_attn_varlen_func
except ImportError as exc:  # pragma: no cover - exercised inside the Docker image.
    raise ImportError(
        "flash-attn-3 is required for this compatibility shim. "
        "Rebuild the qwen image so flash-attn-3 is installed."
    ) from exc


def flash_attn_func(*args, dropout_p: float = 0.0, **kwargs):
    if dropout_p not in (0, 0.0):
        raise ValueError("flash-attn-3 inference shim only supports dropout_p=0.")
    return _fa3_flash_attn_func(*args, **kwargs)


def flash_attn_varlen_func(
    q,
    k,
    v,
    cu_seqlens_q,
    cu_seqlens_k,
    max_seqlen_q,
    max_seqlen_k,
    dropout_p: float = 0.0,
    softmax_scale=None,
    causal: bool = False,
    window_size=(-1, -1),
    return_attn_probs: bool = False,
    **kwargs,
):
    if dropout_p not in (0, 0.0):
        raise ValueError("flash-attn-3 inference shim only supports dropout_p=0.")

    allowed_kwargs = {
        key: value
        for key, value in kwargs.items()
        if key
        in {
            "attention_chunk",
            "deterministic",
            "k_descale",
            "num_splits",
            "pack_gqa",
            "q_descale",
            "qv",
            "seqused_k",
            "seqused_q",
            "sm_margin",
            "softcap",
            "v_descale",
        }
    }
    return _fa3_flash_attn_varlen_func(
        q,
        k,
        v,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        softmax_scale=softmax_scale,
        causal=causal,
        window_size=window_size,
        return_attn_probs=return_attn_probs,
        **allowed_kwargs,
    )


flash_attn_unpadded_func = flash_attn_varlen_func
