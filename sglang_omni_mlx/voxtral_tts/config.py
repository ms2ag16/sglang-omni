# SPDX-License-Identifier: Apache-2.0
"""Voxtral TTS checkpoint configuration for the MLX runtime."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, kw_only=True)
class TextConfig:
    dim: int
    n_layers: int
    head_dim: int
    hidden_dim: int
    n_heads: int
    n_kv_heads: int
    vocab_size: int
    rope_theta: float
    norm_eps: float
    max_seq_len: int
    tied_embeddings: bool


@dataclass(frozen=True, kw_only=True)
class AcousticConfig:
    input_dim: int
    dim: int
    n_layers: int
    head_dim: int
    hidden_dim: int
    n_heads: int
    n_kv_heads: int
    use_biases: bool
    norm_eps: float
    sigma: float


@dataclass(frozen=True, kw_only=True)
class AudioModelConfig:
    semantic_codebook_size: int
    acoustic_codebook_size: int
    n_acoustic_codebook: int
    acoustic_transformer: AcousticConfig
    audio_token_id: int
    begin_audio_token_id: int


@dataclass(frozen=True, kw_only=True)
class VoxtralConfig:
    text: TextConfig
    audio: AudioModelConfig
    audio_tokenizer: dict[str, object]
    voices: dict[str, int]
    checkpoint_directory: Path


def read_voxtral_config(checkpoint_directory: Path) -> VoxtralConfig:
    params_path = checkpoint_directory / "params.json"
    params = json.loads(params_path.read_text())
    if not isinstance(params, dict):
        raise ValueError(f"Voxtral params must be a JSON object: {params_path}")
    else:
        pass
    multimodal = params.get("multimodal", {})
    if not isinstance(multimodal, dict):
        raise ValueError("Voxtral params.multimodal must be an object")
    else:
        pass
    audio_values = multimodal.get("audio_model_args", {})
    tokenizer_values = multimodal.get("audio_tokenizer_args", {})
    if not isinstance(audio_values, dict) or not isinstance(tokenizer_values, dict):
        raise ValueError("Voxtral audio configuration values must be objects")
    else:
        pass
    acoustic_values = audio_values.get("acoustic_transformer_args", {})
    if not isinstance(acoustic_values, dict):
        raise ValueError("Voxtral acoustic_transformer_args must be an object")
    else:
        pass
    tokenizer_values = tokenizer_values.copy()
    voice_values = tokenizer_values.pop("voice", {})
    if not isinstance(voice_values, dict):
        raise ValueError("Voxtral audio_tokenizer_args.voice must be an object")
    else:
        pass
    return VoxtralConfig(
        text=TextConfig(
            dim=int(params.get("dim", 3072)),
            n_layers=int(params.get("n_layers", 26)),
            head_dim=int(params.get("head_dim", 128)),
            hidden_dim=int(params.get("hidden_dim", 9216)),
            n_heads=int(params.get("n_heads", 32)),
            n_kv_heads=int(params.get("n_kv_heads", 8)),
            vocab_size=int(params.get("vocab_size", 131072)),
            rope_theta=float(params.get("rope_theta", 1_000_000.0)),
            norm_eps=float(params.get("norm_eps", 1e-5)),
            max_seq_len=int(params.get("max_seq_len", 65536)),
            tied_embeddings=bool(params.get("tied_embeddings", True)),
        ),
        audio=AudioModelConfig(
            semantic_codebook_size=int(audio_values.get("semantic_codebook_size", 8192)),
            acoustic_codebook_size=int(audio_values.get("acoustic_codebook_size", 21)),
            n_acoustic_codebook=int(audio_values.get("n_acoustic_codebook", 36)),
            acoustic_transformer=AcousticConfig(
                input_dim=int(acoustic_values.get("input_dim", 3072)),
                dim=int(acoustic_values.get("dim", 768)),
                n_layers=int(acoustic_values.get("n_layers", 3)),
                head_dim=int(acoustic_values.get("head_dim", 128)),
                hidden_dim=int(acoustic_values.get("hidden_dim", 2048)),
                n_heads=int(acoustic_values.get("n_heads", 6)),
                n_kv_heads=int(acoustic_values.get("n_kv_heads", 2)),
                use_biases=bool(acoustic_values.get("use_biases", False)),
                norm_eps=float(acoustic_values.get("norm_eps", 1e-5)),
                sigma=float(acoustic_values.get("sigma", 1e-5)),
            ),
            audio_token_id=int(audio_values.get("audio_token_id", 24)),
            begin_audio_token_id=int(audio_values.get("begin_audio_token_id", 25)),
        ),
        audio_tokenizer=dict(tokenizer_values),
        voices={str(name): int(token_id) for name, token_id in voice_values.items()},
        checkpoint_directory=checkpoint_directory,
    )
