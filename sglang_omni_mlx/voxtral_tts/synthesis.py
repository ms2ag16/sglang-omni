# SPDX-License-Identifier: Apache-2.0
"""Autoregressive Voxtral semantic and acoustic code generation in MLX."""

from __future__ import annotations

import mlx.core as mx

from sglang_omni_mlx.voxtral_tts.acoustic_head import END_AUDIO_TOKEN_ID
from sglang_omni_mlx.voxtral_tts.model import VoxtralMLXModel


class VoxtralSynthesizer:
    def __init__(
        self,
        model: VoxtralMLXModel,
        audio_token_id: int,
        max_audio_frames: int,
    ) -> None:
        if max_audio_frames < 1:
            raise ValueError(f"max_audio_frames must be positive, got {max_audio_frames}")
        else:
            pass
        self.model = model
        self.audio_token_id = audio_token_id
        self.max_audio_frames = max_audio_frames

    def generate_codes(
        self,
        prompt_token_ids: list[int],
        voice_embedding: mx.array | None,
    ) -> mx.array:
        if not prompt_token_ids:
            raise ValueError("Voxtral prompt must contain at least one token")
        else:
            pass
        input_ids = mx.array([prompt_token_ids], dtype=mx.int32)
        prompt_embeddings = self.model.text_model.embed_tokens(input_ids)
        placeholder_positions = mx.where(input_ids[0] == self.audio_token_id)[0]
        if voice_embedding is not None and placeholder_positions.size > 0:
            voice_embedding = voice_embedding.reshape(-1, voice_embedding.shape[-1])
            voice_frame_count = min(placeholder_positions.size, voice_embedding.shape[0])
            prompt_embeddings[0, placeholder_positions[:voice_frame_count], :] = (
                voice_embedding[:voice_frame_count].astype(prompt_embeddings.dtype)
            )
        else:
            pass

        caches = self.model.new_caches()
        language_hidden_states = self.model.text_model(prompt_embeddings, caches)
        generated_codes: list[mx.array] = []
        for _ in range(self.max_audio_frames):
            audio_codes = self.model.acoustic_head(language_hidden_states)
            semantic_code = int(audio_codes[0, 0].item())
            if semantic_code == END_AUDIO_TOKEN_ID:
                break
            else:
                generated_codes.append(audio_codes[0])
                feedback_embedding = self.model.embed_audio_codes(audio_codes)[None, None, :]
                language_hidden_states = self.model.text_model(feedback_embedding, caches)
                mx.eval(language_hidden_states)
        if not generated_codes:
            return mx.zeros(
                (0, self.model.acoustic_head.acoustic_codebook_count + 1),
                dtype=mx.int32,
            )
        else:
            return mx.stack(generated_codes).astype(mx.int32)
