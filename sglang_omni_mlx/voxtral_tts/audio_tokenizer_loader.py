# SPDX-License-Identifier: Apache-2.0
"""Checkpoint loading for the Voxtral MLX audio decoder."""

from __future__ import annotations

import mlx.core as mx

from sglang_omni_mlx.voxtral_tts.audio_tokenizer import (
    AudioTokenizerDecoder,
    CausalConvTranspose1d,
)
from sglang_omni_mlx.voxtral_tts.config import AudioTokenizerConfig, VoxtralConfig


def load_audio_tokenizer(config: VoxtralConfig) -> AudioTokenizerDecoder:
    tokenizer = AudioTokenizerDecoder(AudioTokenizerConfig.from_voxtral_config(config))
    source_weights: dict[str, mx.array] = {}
    for shard_path in sorted(config.checkpoint_directory.glob("*.safetensors")):
        for checkpoint_key, checkpoint_weight in mx.load(str(shard_path)).items():
            if checkpoint_key.startswith("audio_tokenizer."):
                source_weights[checkpoint_key] = checkpoint_weight
            else:
                pass
    checkpoint_weights: dict[str, mx.array] = {}
    for checkpoint_key, checkpoint_weight in source_weights.items():
        decoder_key = checkpoint_key.removeprefix("audio_tokenizer.")
        if decoder_key == "quantizer.semantic_codebook.embedding_sum":
            checkpoint_weights["semantic_embedding_sum"] = checkpoint_weight
        elif decoder_key == "quantizer.semantic_codebook.cluster_usage":
            checkpoint_weights["semantic_cluster_usage"] = checkpoint_weight
        elif decoder_key.endswith(".parametrizations.weight.original0"):
            continue
        elif decoder_key.endswith(".parametrizations.weight.original1"):
            module_key = decoder_key.removesuffix(".parametrizations.weight.original1")
            scale_key = (
                f"audio_tokenizer.{module_key}.parametrizations.weight.original0"
            )
            weight_norm = checkpoint_weight.astype(mx.float32)
            scale = source_weights[scale_key].astype(mx.float32)
            norm = mx.sqrt(
                mx.sum(weight_norm * weight_norm, axis=(1, 2), keepdims=True)
            )
            effective_weight = (weight_norm * scale / mx.maximum(norm, 1e-12)).astype(
                checkpoint_weight.dtype
            )
            if module_key.startswith("decoder_blocks."):
                block_index = int(module_key.split(".")[1])
                decoder_block = tokenizer.decoder_blocks[block_index]
                if isinstance(decoder_block, CausalConvTranspose1d):
                    effective_weight = effective_weight.transpose(1, 2, 0)
                else:
                    effective_weight = effective_weight.transpose(0, 2, 1)
            else:
                effective_weight = effective_weight.transpose(0, 2, 1)
            checkpoint_weights[f"{module_key}.weight"] = effective_weight
        elif decoder_key.endswith(".conv.weight"):
            if decoder_key.startswith("decoder_blocks."):
                block_index = int(decoder_key.split(".")[1])
                decoder_block = tokenizer.decoder_blocks[block_index]
                if isinstance(decoder_block, CausalConvTranspose1d):
                    effective_weight = checkpoint_weight.transpose(1, 2, 0)
                else:
                    effective_weight = checkpoint_weight.transpose(0, 2, 1)
            else:
                effective_weight = checkpoint_weight.transpose(0, 2, 1)
            checkpoint_weights[decoder_key] = effective_weight
        elif decoder_key.startswith("quantizer."):
            continue
        else:
            checkpoint_weights[decoder_key] = checkpoint_weight
    tokenizer.load_weights(list(checkpoint_weights.items()), strict=True)
    mx.eval(tokenizer.parameters())
    return tokenizer
