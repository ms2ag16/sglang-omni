# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import io
import wave
from dataclasses import dataclass, field

import numpy as np
import pytest

pytest.importorskip("mlx.core")

from starlette.testclient import TestClient  # noqa: E402

from sglang_omni_mlx.voxtral_tts.audio import encode_wav  # noqa: E402
from sglang_omni_mlx.voxtral_tts.server import build_app  # noqa: E402
from sglang_omni_mlx.voxtral_tts.worker import SpeechResult  # noqa: E402

MODEL_NAME = "voxtral-test"


@dataclass
class FakeWorker:
    voice_embeddings: dict[str, np.ndarray] = field(
        default_factory=lambda: {"cheerful_female": np.zeros((1, 4), dtype=np.float32)}
    )
    calls: list[tuple[str, str, int]] = field(default_factory=list)

    def request_states(self) -> dict[str, int]:
        return {"queued": 0, "running": 0}

    async def synthesize(
        self, text: str, voice: str, max_new_tokens: int
    ) -> SpeechResult:
        self.calls.append((text, voice, max_new_tokens))
        audio = encode_wav(np.zeros(240, dtype=np.float32), 24000)
        return SpeechResult(audio=audio)


def test_encode_wav_writes_clipped_pcm16_mono() -> None:
    encoded = encode_wav(np.array([-1.0, 0.0, 0.5, 1.0], dtype=np.float32), 24000)
    with wave.open(io.BytesIO(encoded), "rb") as audio_file:
        assert audio_file.getnchannels() == 1
        assert audio_file.getsampwidth() == 2
        assert audio_file.getframerate() == 24000
        assert (
            audio_file.readframes(4)
            == np.array([-32768, 0, 16384, 32767], dtype="<i2").tobytes()
        )


def test_speech_route_returns_wav_and_forwards_options() -> None:
    worker = FakeWorker()
    client = TestClient(build_app(worker, MODEL_NAME))
    response = client.post(
        "/v1/audio/speech",
        json={
            "model": MODEL_NAME,
            "input": "Hello there.",
            "voice": "cheerful_female",
            "response_format": "wav",
            "max_new_tokens": 24,
        },
    )
    assert response.status_code == 200
    assert response.headers["content-type"] == "audio/wav"
    assert response.content[:4] == b"RIFF"
    assert worker.calls == [("Hello there.", "cheerful_female", 24)]


@pytest.mark.parametrize(
    ("fields", "status_code"),
    [
        ({"input": "Hello", "voice": "unknown"}, 400),
        ({"input": "Hello", "response_format": "mp3"}, 400),
        ({"input": "Hello", "model": "another-model"}, 404),
        ({"input": "", "voice": "cheerful_female"}, 400),
    ],
)
def test_invalid_speech_requests_do_not_reach_worker(
    fields: dict[str, str], status_code: int
) -> None:
    worker = FakeWorker()
    response = TestClient(build_app(worker, MODEL_NAME)).post(
        "/v1/audio/speech", json=fields
    )
    assert response.status_code == status_code
    assert worker.calls == []
