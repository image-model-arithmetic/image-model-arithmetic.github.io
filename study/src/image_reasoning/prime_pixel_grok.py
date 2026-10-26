"""Pixel-only modular arithmetic over a prime-sized visual vocabulary."""
from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from image_reasoning.pixel_flow_grok import (
    CompactBlock,
    CompactLayerNorm,
    MasterLinear,
    STYLE_FONTS,
    TRAIN_FONTS,
    _mlx_imports,
    font_path,
)

MODULUS = 23
CANVAS_HEIGHT = 32
CANVAS_WIDTH = 96
LEFT_CROP = (0, 0, 24, 32)
RIGHT_CROP = (31, 0, 55, 32)
OUTPUT_HEIGHT = 26
OUTPUT_WIDTH = 24


def answer(pair: tuple[int, int]) -> int:
    return (pair[0] + pair[1]) % MODULUS


def pair_split(seed: int) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    """Answer-balanced directed split with every operand preserved in train."""
    rng = random.Random(f"prime-pixel-split:{seed}")
    by_answer = {
        value: [
            (left, right)
            for left in range(MODULUS)
            for right in range(MODULUS)
            if answer((left, right)) == value
        ]
        for value in range(MODULUS)
    }
    for _ in range(1000):
        heldout = []
        for value, pairs in by_answer.items():
            choices = list(pairs)
            rng.shuffle(choices)
            heldout.extend(choices[:11])
        heldout = sorted(heldout)
        train = sorted(
            set((a, b) for a in range(MODULUS) for b in range(MODULUS))
            - set(heldout)
        )
        if (
            {a for a, _ in train} == set(range(MODULUS))
            and {b for _, b in train} == set(range(MODULUS))
        ):
            return train, heldout
    raise RuntimeError("could not construct prime split")


def seed_for(base: int, pair: tuple[int, int], variant: int, tag: str) -> int:
    payload = f"{base}:{tag}:{pair[0]}:{pair[1]}:{variant}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % (2**31)


def _draw_centered(draw, text, center, font, rng) -> None:
    bounds = draw.textbbox((0, 0), text, font=font)
    width = bounds[2] - bounds[0]
    height = bounds[3] - bounds[1]
    draw.text(
        (
            center[0] - width // 2 - bounds[0] + rng.randint(-1, 1),
            center[1] - height // 2 - bounds[1] + rng.randint(-1, 1),
        ),
        text,
        fill=0,
        font=font,
    )


def render_input(pair: tuple[int, int], seed: int, pattern: str | None = None) -> np.ndarray:
    rng = random.Random(f"prime-render:{seed}")
    font = ImageFont.truetype(font_path(pattern or rng.choice(TRAIN_FONTS)), rng.choice((13, 14, 15)))
    image = Image.new("L", (CANVAS_WIDTH, CANVAS_HEIGHT), 255)
    draw = ImageDraw.Draw(image)
    _draw_centered(draw, str(pair[0]), (12, 16), font, rng)
    _draw_centered(draw, "+", (27, 16), font, rng)
    _draw_centered(draw, str(pair[1]), (43, 16), font, rng)
    _draw_centered(draw, "=", (61, 16), font, rng)
    return np.asarray(image, dtype=np.float32) / 255.0


def canonical_numeral(value: int) -> np.ndarray:
    image = Image.new("L", (OUTPUT_WIDTH, OUTPUT_HEIGHT), 255)
    draw = ImageDraw.Draw(image)
    font = ImageFont.truetype(font_path("DejaVu Sans:style=Bold"), 15)
    bounds = draw.textbbox((0, 0), str(value), font=font)
    width = bounds[2] - bounds[0]
    height = bounds[3] - bounds[1]
    draw.text(
        (
            OUTPUT_WIDTH // 2 - width // 2 - bounds[0],
            OUTPUT_HEIGHT // 2 - height // 2 - bounds[1],
        ),
        str(value),
        fill=0,
        font=font,
    )
    return np.asarray(image, dtype=np.float32) / 255.0


def build_dataset(output: Path, *, split_seed: int, variants: int) -> dict:
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    train, heldout = pair_split(split_seed)
    heldout_set = set(heldout)
    prototypes = np.stack([canonical_numeral(v) for v in range(MODULUS)])
    inputs = []
    targets = []
    pairs_out = []
    heldout_out = []
    variant_out = []
    for split_flag, pairs in ((False, train), (True, heldout)):
        for pair in pairs:
            for variant in range(variants):
                inputs.append(render_input(pair, seed_for(split_seed, pair, variant, "main")))
                targets.append(prototypes[answer(pair)])
                pairs_out.append(pair)
                heldout_out.append(split_flag)
                variant_out.append(variant)
    style_inputs = []
    style_pairs = []
    for font_index, pattern in enumerate(STYLE_FONTS):
        for pair in train + heldout:
            style_inputs.append(
                render_input(pair, seed_for(split_seed, pair, font_index, "style"), pattern)
            )
            style_pairs.append(pair)
    strict_heldout = [pair for pair in heldout if (pair[1], pair[0]) in heldout_set]
    np.savez_compressed(
        output / "dataset.npz",
        inputs=np.stack(inputs)[..., None].astype(np.float16),
        targets=np.stack(targets)[..., None].astype(np.float16),
        pairs=np.asarray(pairs_out, dtype=np.int8),
        heldout=np.asarray(heldout_out, dtype=bool),
        variants=np.asarray(variant_out, dtype=np.int16),
        prototypes=prototypes[..., None].astype(np.float32),
        style_inputs=np.stack(style_inputs)[..., None].astype(np.float16),
        style_pairs=np.asarray(style_pairs, dtype=np.int8),
        strict_heldout_pairs=np.asarray(strict_heldout, dtype=np.int8),
    )
    manifest = {
        "modulus": MODULUS,
        "split_seed": split_seed,
        "variants_per_pair": variants,
        "train_pairs": [list(pair) for pair in train],
        "heldout_pairs": [list(pair) for pair in heldout],
        "strict_double_heldout_pairs": [list(pair) for pair in strict_heldout],
        "train_pair_count": len(train),
        "heldout_pair_count": len(heldout),
        "strict_double_heldout_pair_count": len(strict_heldout),
        "heldout_answer_counts": {
            str(value): sum(answer(pair) == value for pair in heldout)
            for value in range(MODULUS)
        },
        "maximum_constant_accuracy": 1 / MODULUS,
        "input_has_answer_pixels": False,
        "input_shape": [CANVAS_HEIGHT, CANVAS_WIDTH, 1],
        "target_shape": [OUTPUT_HEIGHT, OUTPUT_WIDTH, 1],
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def classify(outputs: np.ndarray, prototypes: np.ndarray) -> np.ndarray:
    distances = np.mean(
        (outputs[:, None].astype(np.float32) - prototypes[None].astype(np.float32)) ** 2,
        axis=(2, 3, 4),
    )
    return np.argmin(distances, axis=1)


try:
    import mlx.nn as _mlx_nn

    _ModuleBase = _mlx_nn.Module
except ImportError:  # pragma: no cover
    _ModuleBase = object


class SharedNumeralReader(_ModuleBase):
    def __init__(self, width: int):
        super().__init__()
        self.input = MasterLinear(32 * 24, width)
        self.output = MasterLinear(width, width)

    def __call__(self, crop):
        _, nn = _mlx_imports()
        ink = 1.0 - crop[..., 0]
        return self.output(nn.gelu(self.input(ink.reshape(ink.shape[0], -1))))


class PrimeRasterCompleter(_ModuleBase):
    def __init__(self, *, width: int, layers: int, heads: int, mlp_ratio: int):
        super().__init__()
        mx, _ = _mlx_imports()
        self.reader = SharedNumeralReader(width)
        self.query = (mx.random.normal((1, width)) * 0.02).astype(mx.float32)
        self.position = (mx.random.normal((3, width)) * 0.02).astype(mx.float32)
        self.blocks = [CompactBlock(width, heads, mlp_ratio) for _ in range(layers)]
        self.norm = CompactLayerNorm(width)
        self.writer_hidden = MasterLinear(width, width * 2)
        self.writer_output = MasterLinear(width * 2, OUTPUT_HEIGHT * OUTPUT_WIDTH)

    def __call__(self, images):
        mx, nn = _mlx_imports()
        left = images[:, :, LEFT_CROP[0] : LEFT_CROP[2], :]
        right = images[:, :, RIGHT_CROP[0] : RIGHT_CROP[2], :]
        left_token = self.reader(left)
        right_token = self.reader(right)
        query = mx.broadcast_to(self.query.astype(mx.bfloat16), left_token.shape)
        tokens = mx.stack([left_token, right_token, query], axis=1)
        tokens = tokens + self.position.astype(mx.bfloat16)[None]
        for block in self.blocks:
            tokens = block(tokens)
        hidden = nn.gelu(self.writer_hidden(self.norm(tokens[:, 2])))
        output = mx.sigmoid(self.writer_output(hidden).astype(mx.float32))
        return output.reshape(images.shape[0], OUTPUT_HEIGHT, OUTPUT_WIDTH, 1)
