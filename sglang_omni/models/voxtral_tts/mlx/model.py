# SPDX-License-Identifier: Apache-2.0

import glob
import os
import re

import mlx.core as mx
import torch
from mlx_lm.models import ministral3
from safetensors import safe_open

from sglang_omni.models.voxtral_tts.model_config import VoxtralTextConfig
from sglang_omni.models.voxtral_tts.voxtral_tts_audio_generation import (
    interleave_qk_weight,
)

LAYER_KEY_PATTERN = re.compile("^layers\.(\d+)\.(.+)$")
LAYER_KEY_MAP: dict[str, str] = {
    "attention.wq.weight": "self_attn.q_proj.weight",
    "attention.wk.weight": "self_attn.k_proj.weight",
    "attention.wv.weight": "self_attn.v_proj.weight",
    "attention.wo.weight": "self_attn.o_proj.weight",
    "attention_norm.weight": "input_layernorm.weight",
    "feed_forward.w1.weight": "mlp.gate_proj.weight",
    "feed_forward.w2.weight": "mlp.down_proj.weight",
    "feed_forward.w3.weight": "mlp.up_proj.weight",
    "ffn_norm.weight": "post_attention_layernorm.weight",
}
GLOBAL_KEY_MAP = {
    "norm.weight": "norm.weight",
    "mm_audio_embeddings.tok_embeddings.weight": "embed_tokens.weight",
}
AUDIO_EMBEDDING_KEY = "mm_audio_embeddings.audio_codebook_embeddings.embeddings.weight"


def to_mlx_bfloat16(tensor: torch.Tensor) -> mx.array:
    return mx.array(tensor.to(torch.float32).numpy()).astype(mx.bfloat16)


def find_checkpoint_shards(checkpoint_dir: str) -> list[str]:
    shard_paths = sorted(glob.glob(os.path.join(checkpoint_dir, "*.safetensors")))
    if not shard_paths:
        raise FileNotFoundError(
            f"Voxtral-TTS checkpoint has no safetensors in {checkpoint_dir}"
        )
    else:
        return shard_paths


def remap_language_model_key(checkpoint_key: str) -> str | None:
    """Map a checkpoint key to the mlx_lm parameter path, or None if not a backbone key."""
    layer_match = LAYER_KEY_PATTERN.match(checkpoint_key)
    if checkpoint_key in GLOBAL_KEY_MAP:
        return GLOBAL_KEY_MAP[checkpoint_key]
    elif layer_match is None:
        return None
    elif layer_match.group(2) not in LAYER_KEY_MAP:
        return None
    else:
        return f"layers.{layer_match.group(1)}.{LAYER_KEY_MAP[layer_match.group(2)]}"


def load_language_model(
    checkpoint_dir: str,
    language_model: ministral3.LanguageModel,
    text_config: VoxtralTextConfig,
) -> mx.array:
    """Load backbone weights and return the audio codebook table, which lives outside the backbone."""
    language_model_weights = []
    audio_embedding_weight = None
    for shard_path in find_checkpoint_shards(checkpoint_dir):
        with safe_open(shard_path, framework="pt", device="cpu") as shard:
            for checkpoint_key in shard.keys():
                mlx_key = remap_language_model_key(checkpoint_key)
                if checkpoint_key == AUDIO_EMBEDDING_KEY:
                    audio_embedding_weight = to_mlx_bfloat16(
                        shard.get_tensor(checkpoint_key)
                    )
                elif mlx_key is None:
                    pass
                else:
                    tensor = shard.get_tensor(checkpoint_key)
                    if "attention.wq." in checkpoint_key:
                        tensor = interleave_qk_weight(
                            tensor, text_config.n_heads, text_config.head_dim
                        )
                    elif "attention.wk." in checkpoint_key:
                        tensor = interleave_qk_weight(
                            tensor, text_config.n_kv_heads, text_config.head_dim
                        )
                    else:
                        pass
                    language_model_weights.append((mlx_key, to_mlx_bfloat16(tensor)))

    if audio_embedding_weight is None:
        raise RuntimeError(f"Voxtral-TTS checkpoint is missing {AUDIO_EMBEDDING_KEY}")
    else:
        language_model.load_weights(audio_embedding_weight, strict=True)
        return audio_embedding_weight
