#!/usr/bin/env python3
"""Embed the radar model in the reserved ESP32-S3 flash region."""

import os
import sys
import tempfile
from pathlib import Path

FLASH_SIZE = 0x400000
MODEL_OFFSET = 0x380000
MODEL_PARTITION_SIZE = 0x80000


def main() -> int:
    if len(sys.argv) != 4:
        raise SystemExit("usage: add_model_to_image.py IMAGE MODEL OFFSET")

    image_path = Path(sys.argv[1])
    model_path = Path(sys.argv[2])
    offset = int(sys.argv[3], 0)

    if not image_path.is_file():
        raise SystemExit(f"flash image not found: {image_path}")
    if not model_path.is_file():
        raise SystemExit(f"model file not found: {model_path}")
    if offset != MODEL_OFFSET:
        raise SystemExit(f"model offset must match partition table: 0x{MODEL_OFFSET:x}")
    if offset < 0 or offset >= FLASH_SIZE:
        raise SystemExit("model offset is outside the 4 MB flash")

    image = bytearray(image_path.read_bytes())
    model = model_path.read_bytes()
    if not model:
        raise SystemExit("model file is empty")
    if len(model) > MODEL_PARTITION_SIZE:
        raise SystemExit(
            f"model is too large: {len(model)} bytes; partition is {MODEL_PARTITION_SIZE} bytes"
        )

    end = offset + len(model)
    if end > FLASH_SIZE:
        raise SystemExit("model would extend beyond the 4 MB flash")
    if len(image) < offset:
        image.extend(b"\xff" * (offset - len(image)))
    if len(image) < end:
        image.extend(b"\xff" * (end - len(image)))

    existing = bytes(image[offset:end])
    if existing != b"\xff" * len(model) and existing != model:
        raise SystemExit("flash image already uses the reserved model region")

    image[offset:end] = model
    if bytes(image[offset:end]) != model:
        raise SystemExit("model verification failed")

    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", dir=str(image_path.parent), prefix=image_path.name + ".",
            suffix=".tmp", delete=False
        ) as stream:
            temp_path = Path(stream.name)
            stream.write(image)
            stream.flush()
            os.fsync(stream.fileno())
        if temp_path.stat().st_size < end:
            raise SystemExit("temporary flash image is incomplete")
        os.replace(temp_path, image_path)
    finally:
        if temp_path is not None and temp_path.exists():
            temp_path.unlink()

    print(f"model added at 0x{offset:x}: {len(model)} bytes")
    print(f"merged flash image: {image_path.stat().st_size} bytes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
