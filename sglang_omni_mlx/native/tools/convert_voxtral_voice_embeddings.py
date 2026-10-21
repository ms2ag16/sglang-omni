# SPDX-License-Identifier: Apache-2.0
"""Convert Voxtral voice tensors to the safetensors format used by native MLX."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from safetensors.torch import save_file


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model_path", type=Path)
    arguments = parser.parse_args()
    voice_directory = arguments.model_path / "voice_embedding"
    if not voice_directory.is_dir():
        raise FileNotFoundError(
            f"Voxtral voice embeddings are missing: {voice_directory}"
        )
    else:
        pass
    voice_paths = sorted(voice_directory.glob("*.pt"))
    if not voice_paths:
        raise FileNotFoundError(
            f"Voxtral voice directory has no .pt files: {voice_directory}"
        )
    else:
        pass
    for voice_path in voice_paths:
        embedding = torch.load(voice_path, map_location="cpu", weights_only=True)
        if not isinstance(embedding, torch.Tensor):
            raise ValueError(f"Voxtral voice embedding is not a tensor: {voice_path}")
        else:
            pass
        output_path = voice_path.with_suffix(".safetensors")
        save_file(
            {"voice_embedding": embedding.detach().to(torch.float32).contiguous()},
            str(output_path),
        )


if __name__ == "__main__":
    main()
