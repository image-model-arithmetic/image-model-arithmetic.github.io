"""Seven-segment raster-only modular addition without a class output head."""
from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np

from image_reasoning.pixel_flow_grok import CompactBlock, CompactLayerNorm, MasterLinear, _mlx_imports
from image_reasoning.prime_pixel_grok import MODULUS, answer, seed_for

HEIGHT = 32
NUMERAL_WIDTH = 28
CANVAS_WIDTH = 104
LEFT_CROP = (0, 28)
RIGHT_CROP = (40, 68)
SEGMENTS = {
    0: "abcedf",  # sets are normalized below
    1: "bc",
    2: "abdeg",
    3: "abcdg",
    4: "bcfg",
    5: "acdfg",
    6: "acdefg",
    7: "abc",
    8: "abcdefg",
    9: "abcdfg",
}
SEGMENTS[0] = "abcdef"


def segment_pair_split(
    seed: int, heldout_per_answer: int = 7
) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    rng = random.Random(f"segment-pixel-split:{seed}")
    heldout = []
    for value in range(MODULUS):
        pairs = [
            (left, right)
            for left in range(MODULUS)
            for right in range(MODULUS)
            if answer((left, right)) == value
        ]
        rng.shuffle(pairs)
        heldout.extend(pairs[:heldout_per_answer])
    heldout = sorted(heldout)
    train = sorted(
        set((a, b) for a in range(MODULUS) for b in range(MODULUS))
        - set(heldout)
    )
    return train, heldout


def _digit(canvas: np.ndarray, value: int, x: int, *, thickness: int, jitter_y: int) -> None:
    y = 3 + jitter_y
    coordinates = {
        "a": (x + 2, y, x + 9, y + thickness),
        "b": (x + 9 - thickness, y + 2, x + 9, y + 12),
        "c": (x + 9 - thickness, y + 13, x + 9, y + 23),
        "d": (x + 2, y + 23 - thickness, x + 9, y + 23),
        "e": (x, y + 13, x + thickness, y + 23),
        "f": (x, y + 2, x + thickness, y + 12),
        "g": (x + 2, y + 11, x + 9, y + 11 + thickness),
    }
    for name in SEGMENTS[value]:
        left, top, right, bottom = coordinates[name]
        canvas[top:bottom, left:right] = 0.0


def numeral(value: int, *, thickness: int = 2, jitter_y: int = 0) -> np.ndarray:
    canvas = np.ones((HEIGHT, NUMERAL_WIDTH), dtype=np.float32)
    if value >= 10:
        _digit(canvas, value // 10, 3, thickness=thickness, jitter_y=jitter_y)
    _digit(canvas, value % 10, 16, thickness=thickness, jitter_y=jitter_y)
    return canvas


def render_input(pair: tuple[int, int], seed: int) -> np.ndarray:
    rng = random.Random(f"segment:{seed}")
    image = np.ones((HEIGHT, CANVAS_WIDTH), dtype=np.float32)
    thickness = rng.choice((2, 2, 3))
    jitter = rng.choice((-1, 0, 0, 1))
    image[:, LEFT_CROP[0] : LEFT_CROP[1]] = numeral(pair[0], thickness=thickness, jitter_y=jitter)
    image[:, RIGHT_CROP[0] : RIGHT_CROP[1]] = numeral(pair[1], thickness=thickness, jitter_y=jitter)
    # Plus and equals are fixed visual syntax, not a task-information channel.
    image[14:17, 31:38] = 0.0
    image[11:20, 33:36] = 0.0
    image[12:15, 72:81] = 0.0
    image[18:21, 72:81] = 0.0
    return image


def build_dataset(output: Path, *, split_seed: int, variants: int) -> dict:
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    train, heldout = segment_pair_split(split_seed)
    heldout_set = set(heldout)
    prototypes = np.stack([numeral(v) for v in range(MODULUS)])[..., None]
    inputs, targets, pairs_out, heldout_out, variant_out = [], [], [], [], []
    for flag, split in ((False, train), (True, heldout)):
        for pair in split:
            for variant in range(variants):
                inputs.append(render_input(pair, seed_for(split_seed, pair, variant, "segment")))
                targets.append(prototypes[answer(pair)])
                pairs_out.append(pair)
                heldout_out.append(flag)
                variant_out.append(variant)
    strict = [pair for pair in heldout if (pair[1], pair[0]) in heldout_set]
    np.savez_compressed(
        output / "dataset.npz",
        inputs=np.stack(inputs)[..., None].astype(np.float16),
        targets=np.stack(targets).astype(np.float16),
        pairs=np.asarray(pairs_out, dtype=np.int8),
        heldout=np.asarray(heldout_out, dtype=bool),
        variants=np.asarray(variant_out, dtype=np.int16),
        prototypes=prototypes.astype(np.float32),
        strict_heldout_pairs=np.asarray(strict, dtype=np.int8),
    )
    manifest = {
        "modulus": MODULUS,
        "split_seed": split_seed,
        "variants_per_pair": variants,
        "train_pairs": [list(pair) for pair in train],
        "heldout_pairs": [list(pair) for pair in heldout],
        "strict_double_heldout_pairs": [list(pair) for pair in strict],
        "heldout_answer_counts": {
            str(v): sum(answer(pair) == v for pair in heldout)
            for v in range(MODULUS)
        },
        "maximum_constant_accuracy": 1 / MODULUS,
        "input_has_answer_pixels": False,
        "representation": "seven-segment numeral rasters",
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def classify(outputs: np.ndarray, prototypes: np.ndarray) -> np.ndarray:
    return np.argmin(
        np.mean((outputs[:, None] - prototypes[None]) ** 2, axis=(2, 3, 4)), axis=1
    )


try:
    import mlx.nn as _mlx_nn
    _Base = _mlx_nn.Module
except ImportError:  # pragma: no cover
    _Base = object


class SegmentReader(_Base):
    def __init__(self, width: int):
        super().__init__()
        self.input = MasterLinear(HEIGHT * NUMERAL_WIDTH, width)
        self.output = MasterLinear(width, width)

    def __call__(self, crop):
        _, nn = _mlx_imports()
        return self.output(nn.gelu(self.input((1.0 - crop[..., 0]).reshape(crop.shape[0], -1))))


class SegmentRasterCompleter(_Base):
    def __init__(self, *, width: int, layers: int, heads: int, mlp_ratio: int):
        super().__init__()
        mx, _ = _mlx_imports()
        self.reader = SegmentReader(width)
        self.query = (mx.random.normal((1, width)) * 0.02).astype(mx.float32)
        self.position = (mx.random.normal((3, width)) * 0.02).astype(mx.float32)
        self.blocks = [CompactBlock(width, heads, mlp_ratio) for _ in range(layers)]
        self.norm = CompactLayerNorm(width)
        self.writer_hidden = MasterLinear(width, width * 2)
        self.writer_output = MasterLinear(width * 2, HEIGHT * NUMERAL_WIDTH)

    def __call__(self, images):
        mx, nn = _mlx_imports()
        left = self.reader(images[:, :, LEFT_CROP[0] : LEFT_CROP[1], :])
        right = self.reader(images[:, :, RIGHT_CROP[0] : RIGHT_CROP[1], :])
        query = mx.broadcast_to(self.query.astype(mx.bfloat16), left.shape)
        tokens = mx.stack([left, right, query], axis=1) + self.position.astype(mx.bfloat16)[None]
        for block in self.blocks:
            tokens = block(tokens)
        hidden = nn.gelu(self.writer_hidden(self.norm(tokens[:, 2])))
        output = mx.sigmoid(self.writer_output(hidden).astype(mx.float32))
        return output.reshape(images.shape[0], HEIGHT, NUMERAL_WIDTH, 1)
