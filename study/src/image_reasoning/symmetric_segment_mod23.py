"""Symmetric raster model and frozen train/dev/test split for addition modulo 23."""
from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np

from image_reasoning.pixel_flow_grok import CompactBlock, CompactLayerNorm, MasterLinear, _mlx_imports
from image_reasoning.prime_pixel_grok import MODULUS, answer, seed_for
from image_reasoning.segment_pixel_grok import (
    HEIGHT,
    LEFT_CROP,
    NUMERAL_WIDTH,
    RIGHT_CROP,
    SegmentReader,
    _Base,
    numeral,
    render_input,
)


def unordered_pair_split(seed: int) -> dict[str, list[tuple[int, int]]]:
    """Hold out two development and two test pairs for every answer."""
    splits = {"train": [], "development": [], "test": []}
    for value in range(MODULUS):
        pairs = [
            (left, right)
            for left in range(MODULUS)
            for right in range(left, MODULUS)
            if answer((left, right)) == value
        ]
        assert len(pairs) == 12
        random.Random(f"symmetric-segment:{seed}:{value}").shuffle(pairs)
        splits["development"].extend(pairs[:2])
        splits["test"].extend(pairs[2:4])
        splits["train"].extend(pairs[4:])
    return {name: sorted(pairs) for name, pairs in splits.items()}


def build_dataset(output: Path, *, split_seed: int, variants: int) -> dict:
    if output.exists() or output.is_symlink():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    splits = unordered_pair_split(split_seed)
    prototypes = np.stack([numeral(value) for value in range(MODULUS)])[..., None]
    inputs, targets, pairs_out, split_out, variant_out = [], [], [], [], []
    for split_index, name in enumerate(("train", "development", "test")):
        for pair in splits[name]:
            for variant in range(variants):
                inputs.append(render_input(pair, seed_for(split_seed, pair, variant, name)))
                targets.append(prototypes[answer(pair)])
                pairs_out.append(pair)
                split_out.append(split_index)
                variant_out.append(variant)
    np.savez_compressed(
        output / "dataset.npz",
        inputs=np.stack(inputs)[..., None].astype(np.float16),
        targets=np.stack(targets).astype(np.float16),
        pairs=np.asarray(pairs_out, dtype=np.int8),
        splits=np.asarray(split_out, dtype=np.int8),
        variants=np.asarray(variant_out, dtype=np.int16),
        prototypes=prototypes.astype(np.float32),
    )
    manifest = {
        "schema": 1,
        "modulus": MODULUS,
        "split_seed": split_seed,
        "variants_per_pair": variants,
        "split_ids": {"train": 0, "development": 1, "test": 2},
        "pairs": {name: [list(pair) for pair in pairs] for name, pairs in splits.items()},
        "pair_counts": {name: len(pairs) for name, pairs in splits.items()},
        "answer_counts": {
            name: {
                str(value): sum(answer(pair) == value for pair in pairs)
                for value in range(MODULUS)
            }
            for name, pairs in splits.items()
        },
        "input_has_answer_pixels": False,
        "representation": "seven-segment numeral rasters",
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


class SymmetricSegmentRasterCompleter(_Base):
    """The answer is unchanged when the two operand images exchange places."""

    def __init__(self, *, width: int, layers: int, heads: int, mlp_ratio: int):
        super().__init__()
        mx, _ = _mlx_imports()
        self.reader = SegmentReader(width)
        self.query = (mx.random.normal((1, width)) * 0.02).astype(mx.float32)
        self.position = (mx.random.normal((2, width)) * 0.02).astype(mx.float32)
        self.blocks = [CompactBlock(width, heads, mlp_ratio) for _ in range(layers)]
        self.norm = CompactLayerNorm(width)
        self.writer_hidden = MasterLinear(width, width * 2)
        self.writer_output = MasterLinear(width * 2, HEIGHT * NUMERAL_WIDTH)

    def __call__(self, images):
        mx, nn = _mlx_imports()
        left = self.reader(images[:, :, LEFT_CROP[0] : LEFT_CROP[1], :])
        right = self.reader(images[:, :, RIGHT_CROP[0] : RIGHT_CROP[1], :])
        operand_position = self.position[0].astype(mx.bfloat16)
        query_position = self.position[1].astype(mx.bfloat16)
        query = mx.broadcast_to(self.query.astype(mx.bfloat16), left.shape)
        tokens = mx.stack(
            [left + operand_position, right + operand_position, query + query_position], axis=1
        )
        for block in self.blocks:
            tokens = block(tokens)
        hidden = nn.gelu(self.writer_hidden(self.norm(tokens[:, 2])))
        output = mx.sigmoid(self.writer_output(hidden).astype(mx.float32))
        return output.reshape(images.shape[0], HEIGHT, NUMERAL_WIDTH, 1)
