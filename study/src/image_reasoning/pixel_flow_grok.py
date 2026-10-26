"""Compact pixel-only arithmetic inpainting experiment and MLX model.

Dataset construction and auditing are NumPy/Pillow-only. MLX is imported only
when the model is instantiated so dataset tests do not require Apple Silicon.
"""
from __future__ import annotations

import hashlib
import json
import math
import random
import subprocess
import sys
from functools import cache
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

MODULUS = 10
CANVAS_WIDTH = 64
CANVAS_HEIGHT = 32
ANSWER_BOX = (48, 4, 64, 30)
TRAIN_FONTS = (
    "DejaVu Sans:style=Bold",
    "DejaVu Serif:style=Bold",
    "Liberation Sans:style=Bold",
    "Liberation Serif:style=Bold",
)
STYLE_FONTS = ("DejaVu Sans Mono", "Liberation Sans:style=Italic")
DARWIN_FONTS = {
    "DejaVu Sans:style=Bold": "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
    "DejaVu Serif:style=Bold": "/System/Library/Fonts/Supplemental/Times New Roman Bold.ttf",
    "Liberation Sans:style=Bold": "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
    "Liberation Serif:style=Bold": "/System/Library/Fonts/Supplemental/Times New Roman Bold.ttf",
    "DejaVu Sans Mono": "/System/Library/Fonts/Supplemental/Courier New.ttf",
    "Liberation Sans:style=Italic": "/System/Library/Fonts/Supplemental/Arial Italic.ttf",
}


@cache
def font_path(pattern: str) -> str:
    if sys.platform == "darwin":
        return DARWIN_FONTS[pattern]
    result = subprocess.run(
        ["fc-match", "-f", "%{file}", pattern],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if not result:
        raise RuntimeError(f"font not found: {pattern}")
    return result


def answer(pair: tuple[int, int]) -> int:
    return (pair[0] + pair[1]) % MODULUS


def balanced_pair_split(seed: int) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    """Select exactly three held-out pairs per answer while preserving operands."""
    rng = random.Random(f"compact-pixel-grok-split:{seed}")
    by_answer = {
        value: [(left, right) for left in range(10) for right in range(10) if answer((left, right)) == value]
        for value in range(10)
    }
    for _ in range(10000):
        heldout: list[tuple[int, int]] = []
        for value in range(10):
            choices = list(by_answer[value])
            rng.shuffle(choices)
            heldout.extend(choices[:3])
        heldout = sorted(heldout)
        train = sorted(set((a, b) for a in range(10) for b in range(10)) - set(heldout))
        if {a for a, _ in train} == set(range(10)) and {b for _, b in train} == set(range(10)):
            return train, heldout
    raise RuntimeError("could not construct balanced pair split")


def rendering_seed(base_seed: int, pair: tuple[int, int], variant: int, tag: str) -> int:
    payload = f"{base_seed}:{tag}:{pair[0]}:{pair[1]}:{variant}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % (2**31)


def _draw_centered(
    draw: ImageDraw.ImageDraw,
    text: str,
    center: tuple[int, int],
    font: ImageFont.FreeTypeFont,
    rng: random.Random,
) -> None:
    bounds = draw.textbbox((0, 0), text, font=font)
    width = bounds[2] - bounds[0]
    height = bounds[3] - bounds[1]
    x = center[0] - width // 2 - bounds[0] + rng.randint(-1, 1)
    y = center[1] - height // 2 - bounds[1] + rng.randint(-1, 1)
    draw.text((x, y), text, fill=0, font=font)


def render_equation(
    pair: tuple[int, int],
    seed: int,
    *,
    include_answer: bool = True,
    font_pattern: str | None = None,
) -> np.ndarray:
    """Render a compact grayscale equation. No label metadata enters the pixels."""
    rng = random.Random(f"compact-render:{seed}")
    pattern = font_pattern or rng.choice(TRAIN_FONTS)
    font = ImageFont.truetype(font_path(pattern), rng.choice((13, 14, 15)))
    image = Image.new("L", (CANVAS_WIDTH, CANVAS_HEIGHT), 255)
    draw = ImageDraw.Draw(image)
    _draw_centered(draw, str(pair[0]), (6, 16), font, rng)
    _draw_centered(draw, "+", (17, 16), font, rng)
    _draw_centered(draw, str(pair[1]), (28, 16), font, rng)
    _draw_centered(draw, "=", (39, 16), font, rng)
    if include_answer:
        _draw_centered(draw, str(answer(pair)), (55, 16), font, rng)
    return np.asarray(image, dtype=np.float32) / 255.0


def context_from_completed(images: np.ndarray) -> np.ndarray:
    contexts = np.array(images, copy=True)
    left, top, right, bottom = ANSWER_BOX
    contexts[..., top:bottom, left:right, :] = 1.0
    return contexts


def answer_mask() -> np.ndarray:
    mask = np.zeros((CANVAS_HEIGHT, CANVAS_WIDTH, 1), dtype=np.float32)
    left, top, right, bottom = ANSWER_BOX
    mask[top:bottom, left:right, 0] = 1.0
    return mask


def build_dataset(output: Path, *, split_seed: int, variants: int) -> dict:
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    train_pairs, heldout_pairs = balanced_pair_split(split_seed)
    rows: list[dict] = []
    images: list[np.ndarray] = []
    for split, pairs in (("train", train_pairs), ("heldout", heldout_pairs)):
        for pair in pairs:
            for variant in range(variants):
                seed = rendering_seed(split_seed, pair, variant, "train-style")
                images.append(render_equation(pair, seed, include_answer=True))
                rows.append(
                    {
                        "split": split,
                        "left": pair[0],
                        "right": pair[1],
                        "answer": answer(pair),
                        "variant": variant,
                        "seed": seed,
                    }
                )
    completed = np.stack(images)[..., None].astype(np.float16)
    pairs_array = np.asarray([[row["left"], row["right"]] for row in rows], dtype=np.int8)
    split_array = np.asarray([row["split"] == "heldout" for row in rows], dtype=bool)
    variant_array = np.asarray([row["variant"] for row in rows], dtype=np.int16)

    template_stack = []
    for value in range(10):
        class_crops = []
        for variant in range(max(variants, 16)):
            pair = (0, value)
            seed = rendering_seed(split_seed + 1, pair, variant, "template")
            image = render_equation(pair, seed, include_answer=True)
            left, top, right, bottom = ANSWER_BOX
            class_crops.append(normalize_crop(image[top:bottom, left:right]))
        template_stack.append(np.mean(class_crops, axis=0))
    templates = np.stack(template_stack).astype(np.float32)

    style_images = []
    style_pairs = []
    for font_index, pattern in enumerate(STYLE_FONTS):
        for pair in sorted(train_pairs + heldout_pairs):
            seed = rendering_seed(split_seed, pair, font_index, "style-heldout")
            style_images.append(
                render_equation(pair, seed, include_answer=True, font_pattern=pattern)
            )
            style_pairs.append(pair)

    np.savez_compressed(
        output / "dataset.npz",
        completed=completed,
        pairs=pairs_array,
        heldout=split_array,
        variants=variant_array,
        templates=templates,
        style_completed=np.stack(style_images)[..., None].astype(np.float16),
        style_pairs=np.asarray(style_pairs, dtype=np.int8),
        mask=answer_mask(),
    )
    manifest = {
        "split_seed": split_seed,
        "variants_per_pair": variants,
        "train_pairs": [list(pair) for pair in train_pairs],
        "heldout_pairs": [list(pair) for pair in heldout_pairs],
        "train_examples": len(train_pairs) * variants,
        "heldout_examples": len(heldout_pairs) * variants,
        "heldout_answer_counts": {
            str(value): sum(answer(pair) == value for pair in heldout_pairs)
            for value in range(10)
        },
        "train_answer_counts": {
            str(value): sum(answer(pair) == value for pair in train_pairs)
            for value in range(10)
        },
        "answer_box": list(ANSWER_BOX),
        "canvas": [CANVAS_WIDTH, CANVAS_HEIGHT],
        "train_fonts": list(TRAIN_FONTS),
        "style_fonts": list(STYLE_FONTS),
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def normalize_crop(crop: np.ndarray) -> np.ndarray:
    """Center and resize visible ink for an answer-only audit readout."""
    crop = np.clip(crop, 0.0, 1.0)
    ink = crop < 0.75
    rows = np.flatnonzero(ink.any(axis=1))
    columns = np.flatnonzero(ink.any(axis=0))
    normalized = np.ones((ANSWER_BOX[3] - ANSWER_BOX[1], ANSWER_BOX[2] - ANSWER_BOX[0]), dtype=np.float32)
    if not len(rows) or not len(columns):
        return normalized
    glyph = crop[rows[0] : rows[-1] + 1, columns[0] : columns[-1] + 1]
    resized = np.asarray(
        Image.fromarray((glyph * 255).round().astype("uint8")).resize((12, 22), Image.Resampling.BILINEAR),
        dtype=np.float32,
    ) / 255.0
    normalized[2:24, 2:14] = resized
    return normalized


def classify_crops(images: np.ndarray, templates: np.ndarray) -> np.ndarray:
    left, top, right, bottom = ANSWER_BOX
    crops = np.clip(images[..., top:bottom, left:right, 0], 0.0, 1.0)
    normalized = np.stack([normalize_crop(crop) for crop in crops])
    distances = np.mean(
        (normalized[:, None, :, :] - templates[None, :, :, :]) ** 2, axis=(2, 3)
    )
    return np.argmin(distances, axis=1)


def dataset_audit(dataset_path: Path) -> dict:
    manifest = json.loads((dataset_path / "manifest.json").read_text(encoding="utf-8"))
    data = np.load(dataset_path / "dataset.npz")
    heldout_pairs = [tuple(pair) for pair in manifest["heldout_pairs"]]
    heldout_answers = np.asarray([answer(pair) for pair in heldout_pairs])
    constant = max(np.mean(heldout_answers == value) for value in range(10))
    completed = data["completed"].astype(np.float32)
    labels = (data["pairs"][:, 0].astype(int) + data["pairs"][:, 1].astype(int)) % 10
    template_predictions = classify_crops(completed, data["templates"])
    context_predictions = classify_crops(context_from_completed(completed), data["templates"])
    return {
        "examples": int(len(completed)),
        "train_pairs": len(manifest["train_pairs"]),
        "heldout_pairs": len(heldout_pairs),
        "heldout_answer_counts": manifest["heldout_answer_counts"],
        "constant_heldout_accuracy": float(constant),
        "completed_template_accuracy": float(np.mean(template_predictions == labels)),
        "blank_context_template_accuracy": float(np.mean(context_predictions == labels)),
        "pair_disjoint": not bool(
            set(map(tuple, manifest["train_pairs"])) & set(heldout_pairs)
        ),
    }


# MLX model is intentionally below the pure dataset utilities.
def _mlx_imports():
    import mlx.core as mx
    import mlx.nn as nn

    return mx, nn


try:
    import mlx.nn as _mlx_nn

    _ModuleBase = _mlx_nn.Module
except ImportError:  # pragma: no cover - exercised on non-Apple CI
    _ModuleBase = object


class MasterLinear(_ModuleBase):
    def __init__(self, input_dims: int, output_dims: int, bias: bool = True):
        super().__init__()
        mx, _ = _mlx_imports()
        scale = math.sqrt(1.0 / input_dims)
        self.weight = mx.random.uniform(
            low=-scale, high=scale, shape=(output_dims, input_dims)
        ).astype(mx.float32)
        if bias:
            self.bias = mx.zeros((output_dims,), dtype=mx.float32)

    def __call__(self, x):
        mx, _ = _mlx_imports()
        result = mx.matmul(x.astype(mx.bfloat16), self.weight.T.astype(mx.bfloat16))
        if "bias" in self:
            result = result + self.bias.astype(mx.bfloat16)
        return result

    def __contains__(self, name: str) -> bool:
        return hasattr(self, name)


class CompactLayerNorm(_ModuleBase):
    def __init__(self, width: int, eps: float = 1e-5):
        super().__init__()
        mx, _ = _mlx_imports()
        self.weight = mx.ones((width,), dtype=mx.float32)
        self.bias = mx.zeros((width,), dtype=mx.float32)
        self.eps = eps

    def __call__(self, x):
        mx, _ = _mlx_imports()
        source = x.astype(mx.float32)
        mean = mx.mean(source, axis=-1, keepdims=True)
        variance = mx.mean((source - mean) ** 2, axis=-1, keepdims=True)
        normalized = (source - mean) * mx.rsqrt(variance + self.eps)
        return (
            normalized.astype(mx.bfloat16) * self.weight.astype(mx.bfloat16)
            + self.bias.astype(mx.bfloat16)
        )


class CompactAttention(_ModuleBase):
    def __init__(self, width: int, heads: int):
        super().__init__()
        self.heads = heads
        self.head_dim = width // heads
        if width % heads:
            raise ValueError("width must be divisible by heads")
        self.to_qkv = MasterLinear(width, width * 3)
        self.to_out = MasterLinear(width, width)

    def __call__(self, x):
        mx, _ = _mlx_imports()
        batch, tokens, width = x.shape
        qkv = self.to_qkv(x).reshape(batch, tokens, 3, self.heads, self.head_dim)
        q, k, v = (qkv[:, :, index].transpose(0, 2, 1, 3) for index in range(3))
        attended = mx.fast.scaled_dot_product_attention(
            q, k, v, scale=self.head_dim**-0.5
        )
        attended = attended.transpose(0, 2, 1, 3).reshape(batch, tokens, width)
        return self.to_out(attended)


class CompactBlock(_ModuleBase):
    def __init__(self, width: int, heads: int, mlp_ratio: int):
        super().__init__()
        self.norm1 = CompactLayerNorm(width)
        self.attention = CompactAttention(width, heads)
        self.norm2 = CompactLayerNorm(width)
        self.up = MasterLinear(width, width * mlp_ratio)
        self.down = MasterLinear(width * mlp_ratio, width)

    def __call__(self, x):
        mx, nn = _mlx_imports()
        x = x + self.attention(self.norm1(x))
        x = x + self.down(nn.gelu(self.up(self.norm2(x))))
        return x


class CompactPixelDenoiser(_ModuleBase):
    def __init__(
        self,
        *,
        width: int,
        layers: int,
        heads: int,
        patch_size: int,
        mlp_ratio: int = 4,
    ):
        super().__init__()
        mx, _ = _mlx_imports()
        if CANVAS_HEIGHT % patch_size or CANVAS_WIDTH % patch_size:
            raise ValueError("patch size must divide the canvas")
        self.width = width
        self.layers_count = layers
        self.patch_size = patch_size
        self.patch_rows = CANVAS_HEIGHT // patch_size
        self.patch_columns = CANVAS_WIDTH // patch_size
        self.patch_input = MasterLinear(patch_size * patch_size, width)
        self.position = (
            mx.random.normal((self.patch_rows * self.patch_columns, width)) * 0.02
        ).astype(mx.float32)
        time_width = max(32, width // 2)
        self.time_frequencies = mx.exp(
            -math.log(10000.0)
            * mx.arange(0, time_width // 2, dtype=mx.float32)
            / max(time_width // 2 - 1, 1)
        )
        self.time_in = MasterLinear(time_width, width)
        self.time_out = MasterLinear(width, width)
        self.blocks = [CompactBlock(width, heads, mlp_ratio) for _ in range(layers)]
        self.final_norm = CompactLayerNorm(width)
        self.patch_output = MasterLinear(width, patch_size * patch_size)

    def _patchify(self, images):
        batch = images.shape[0]
        p = self.patch_size
        return (
            images[..., 0]
            .reshape(batch, self.patch_rows, p, self.patch_columns, p)
            .transpose(0, 1, 3, 2, 4)
            .reshape(batch, self.patch_rows * self.patch_columns, p * p)
        )

    def _unpatchify(self, patches):
        batch = patches.shape[0]
        p = self.patch_size
        return (
            patches.reshape(batch, self.patch_rows, self.patch_columns, p, p)
            .transpose(0, 1, 3, 2, 4)
            .reshape(batch, CANVAS_HEIGHT, CANVAS_WIDTH, 1)
        )

    def __call__(self, images, sigma):
        mx, nn = _mlx_imports()
        time = sigma.astype(mx.float32)[:, None] * self.time_frequencies[None, :]
        time = mx.concatenate([mx.sin(time), mx.cos(time)], axis=-1)
        time = self.time_out(nn.silu(self.time_in(time)))[:, None, :]
        tokens = self.patch_input(self._patchify(images))
        tokens = tokens + self.position.astype(tokens.dtype)[None] + time
        for block in self.blocks:
            tokens = block(tokens)
        return self._unpatchify(self.patch_output(self.final_norm(tokens))).astype(
            mx.float32
        )
