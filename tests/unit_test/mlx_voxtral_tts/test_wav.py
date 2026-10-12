# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import numpy as np
import pytest

from sglang_omni_mlx.voxtral_tts.audio import encode_wav


@pytest.mark.parametrize("sample", [np.nan, np.inf, -np.inf])
def test_encode_wav_rejects_non_finite_samples(sample: float) -> None:
    with pytest.raises(ValueError, match="samples must be finite"):
        encode_wav(np.array([sample], dtype=np.float32), sample_rate_hz=24000)
