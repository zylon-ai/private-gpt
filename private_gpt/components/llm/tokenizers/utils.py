from typing import Any

import numpy as np
from PIL import Image

from private_gpt.components.llm.tokenizers.tokenizer_base import (
    AudioLike,
    ImageLike,
    TextLike,
)


def build_minimal_messages(
    texts: TextLike | None = None,
    images: ImageLike | None = None,
    audios: AudioLike | None = None,
) -> list[dict[str, Any]]:
    """Build minimal messages for token estimation."""
    content: list[dict[str, Any]] = []

    if images:
        for img_b in images:
            img_b.seek(0)
            image_pil = Image.open(img_b)  # type: ignore
            content.append(
                {
                    "type": "image",
                    "image": image_pil,
                }
            )

    if audios:
        for audio_b in audios:
            audio_b.seek(0)
            audio_bytes = audio_b.read()
            num_floats = len(audio_bytes) // 4
            audio_array = np.frombuffer(audio_bytes[: num_floats * 4], dtype=np.float32)
            content.append(
                {
                    "type": "audio",
                    "audio": audio_array,
                }
            )

    if texts:
        if isinstance(texts, str):
            content.append({"type": "text", "text": texts})
        else:
            for text in texts:
                content.append({"type": "text", "text": text})
    else:
        content.append({"type": "text", "text": ""})

    return [{"role": "user", "content": content}]
