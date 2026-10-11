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
            raise ValueError(
                f"max_audio_frames must be positive, got {max_audio_frames}"
            )
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
            voice_frame_count = min(
                placeholder_positions.size, voice_embedding.shape[0]
            )
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
                feedback_embedding = self.model.embed_audio_codes(audio_codes)[
                    None, None, :
                ]
                language_hidden_states = self.model.text_model(
                    feedback_embedding, caches
                )
                mx.eval(language_hidden_states)
        if not generated_codes:
            return mx.zeros(
                (0, self.model.acoustic_head.acoustic_codebook_count + 1),
                dtype=mx.int32,
            )
        else:
            return mx.stack(generated_codes).astype(mx.int32)

    def synthesize(
        self,
        prompt_token_ids: list[int],
        voice_embedding: mx.array | None,
    ) -> mx.array:
        if self.model.audio_tokenizer is None:
            raise RuntimeError("Voxtral audio tokenizer was not loaded")
        else:
            pass
        audio_codes = self.generate_codes(prompt_token_ids, voice_embedding)
        if audio_codes.shape[0] == 0:
            raise RuntimeError("Voxtral TTS generated no audio codes")
        else:
            pass
        audio_codes = audio_codes - 2
        warmup_codes = mx.repeat(audio_codes[:1], 2, axis=0)
        decoder_codes = mx.concatenate([warmup_codes, audio_codes], axis=0)
        waveform = self.model.audio_tokenizer(
            decoder_codes.transpose(1, 0)[None, :, :]
        )[0, 0]
        trim_samples = 2 * self.model.audio_tokenizer.downsample_factor
        waveform = waveform[trim_samples:]
        fade_samples = self.model.audio_tokenizer.sampling_rate // 100
        fade = mx.linspace(0.0, 1.0, fade_samples, dtype=waveform.dtype)
        fade_length = min(fade_samples, waveform.shape[0])
        waveform[:fade_length] = waveform[:fade_length] * fade[:fade_length]
        return waveform
