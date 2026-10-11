# SPDX-License-Identifier: Apache-2.0
"""Serialized Voxtral MLX loading and speech generation."""

from __future__ import annotations

import asyncio
import io
import threading
import wave
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import mlx.core as mx
import numpy as np

from sglang_omni_mlx.voxtral_tts.config import VoxtralConfig, read_voxtral_config
from sglang_omni_mlx.voxtral_tts.model import VoxtralMLXModel, load_voxtral_model
from sglang_omni_mlx.voxtral_tts.synthesis import VoxtralSynthesizer
from sglang_omni_mlx.voxtral_tts.tokenizer import (
    VoxtralSpeechTokenizer,
    load_voice_embeddings,
)

DEFAULT_VOICE = "cheerful_female"
DEFAULT_MAX_NEW_TOKENS = 4096


@dataclass(frozen=True, kw_only=True)
class SpeechResult:
    audio: bytes
    sample_rate: int


class VoxtralTTSWorker:
    def __init__(self, checkpoint_directory: Path) -> None:
        self.executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="voxtral-mlx"
        )
        self.config, self.model, self.tokenizer, self.voice_embeddings = (
            self.executor.submit(self.load, checkpoint_directory).result()
        )
        self.state_lock = threading.Lock()
        self.queued_requests = 0
        self.running_requests = 0

    @staticmethod
    def load(
        checkpoint_directory: Path,
    ) -> tuple[
        VoxtralConfig,
        VoxtralMLXModel,
        VoxtralSpeechTokenizer,
        dict[str, mx.array],
    ]:
        config = read_voxtral_config(checkpoint_directory)
        model = load_voxtral_model(config)
        tokenizer = VoxtralSpeechTokenizer(checkpoint_directory)
        voice_embeddings = load_voice_embeddings(config)
        return config, model, tokenizer, voice_embeddings

    def request_states(self) -> dict[str, int]:
        with self.state_lock:
            return {
                "queued": self.queued_requests,
                "running": self.running_requests,
            }

    def run(self, text: str, voice: str, max_new_tokens: int) -> SpeechResult:
        with self.state_lock:
            self.queued_requests -= 1
            self.running_requests += 1
        try:
            if voice not in self.voice_embeddings:
                raise ValueError(f"Unknown Voxtral voice: {voice}")
            else:
                pass
            prompt_token_ids = self.tokenizer.encode(text, voice)
            synthesizer = VoxtralSynthesizer(
                self.model,
                self.config.audio.audio_token_id,
                max_new_tokens,
            )
            waveform = synthesizer.synthesize(
                prompt_token_ids, self.voice_embeddings[voice]
            )
            audio_tokenizer = self.model.audio_tokenizer
            assert audio_tokenizer is not None
            mx.eval(waveform)
            waveform_samples = np.asarray(waveform.astype(mx.float32))
            pcm_samples = (
                np.clip(waveform_samples, -1.0, 1.0) * 32767.0
            ).astype("<i2")
            audio_buffer = io.BytesIO()
            with wave.open(audio_buffer, "wb") as wav_writer:
                wav_writer.setnchannels(1)
                wav_writer.setsampwidth(2)
                wav_writer.setframerate(audio_tokenizer.sampling_rate)
                wav_writer.writeframes(pcm_samples.tobytes())
            audio_bytes = audio_buffer.getvalue()
            return SpeechResult(
                audio=audio_bytes,
                sample_rate=audio_tokenizer.sampling_rate,
            )
        finally:
            with self.state_lock:
                self.running_requests -= 1

    async def synthesize(
        self, text: str, voice: str, max_new_tokens: int
    ) -> SpeechResult:
        with self.state_lock:
            self.queued_requests += 1
        future = self.executor.submit(self.run, text, voice, max_new_tokens)
        return await asyncio.wrap_future(future)
