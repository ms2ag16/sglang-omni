# SPDX-License-Identifier: Apache-2.0
"""Voxtral TTS HTTP server backed by the standalone MLX implementation.

    python -m sglang_omni_mlx.voxtral_tts.server --model-path DIR --model-name NAME --port 8000
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Literal

import mlx.core as mx
import uvicorn
from pydantic import BaseModel, Field, ValidationError
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from starlette.routing import Route

from sglang_omni_mlx.voxtral_tts.worker import (
    DEFAULT_MAX_NEW_TOKENS,
    DEFAULT_VOICE,
    VoxtralTTSWorker,
)

logger = logging.getLogger(__name__)


class SpeechBody(BaseModel):
    input: str = Field(min_length=1)
    voice: str = DEFAULT_VOICE
    model: str | None = None
    response_format: Literal["wav"] = "wav"
    max_new_tokens: int = Field(default=DEFAULT_MAX_NEW_TOKENS, ge=1)


def build_app(worker: VoxtralTTSWorker, model_name: str) -> Starlette:
    async def health(request: Request) -> JSONResponse:
        return JSONResponse(
            {
                "status": "healthy",
                "running": True,
                "request_states": worker.request_states(),
            }
        )

    async def models(request: Request) -> JSONResponse:
        return JSONResponse(
            {"object": "list", "data": [{"id": model_name, "object": "model"}]}
        )

    async def speech(request: Request) -> Response:
        try:
            body = SpeechBody.model_validate(await request.json())
        except ValidationError as error:
            return JSONResponse(
                {"detail": error.errors(include_input=False)}, status_code=400
            )
        except ValueError as error:
            return JSONResponse({"detail": str(error)}, status_code=400)
        if body.model is not None and body.model != model_name:
            return JSONResponse(
                {"detail": f"Unknown model: {body.model}"}, status_code=404
            )
        else:
            pass
        try:
            result = await worker.synthesize(
                body.input,
                body.voice,
                body.max_new_tokens,
            )
        except ValueError as error:
            return JSONResponse({"detail": str(error)}, status_code=400)
        return Response(
            result.audio,
            media_type="audio/wav",
            headers={"Content-Disposition": 'attachment; filename="speech.wav"'},
        )

    return Starlette(
        routes=[
            Route("/health", health),
            Route("/v1/models", models),
            Route("/v1/audio/speech", speech, methods=["POST"]),
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, required=True)
    arguments = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    mx.set_cache_limit(0)
    worker = VoxtralTTSWorker(Path(arguments.model_path))
    logger.info(f"Loaded Voxtral TTS from {arguments.model_path}")
    uvicorn.run(
        build_app(worker, arguments.model_name),
        host=arguments.host,
        port=arguments.port,
        log_level="warning",
    )


if __name__ == "__main__":
    main()
