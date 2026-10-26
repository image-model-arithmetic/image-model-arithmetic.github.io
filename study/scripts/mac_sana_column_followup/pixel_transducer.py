"""A small image-to-image full adder trained from scratch with pixel MSE only.

No autoencoder, pretrained weights, classifier, or answer reader is used.
The three input crops have shared visual features; a learned decoder emits RGB.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
from PIL import Image
import torch
from torch import nn

INPUT_BOXES = ((80, 48, 176, 208), (208, 48, 304, 208), (336, 48, 432, 208))
OUTPUT_BOXES = ((144, 320, 240, 480), (272, 320, 368, 480))


class PixelTransducer(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=2, padding=1), nn.SiLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.SiLU(),
            nn.Conv2d(32, 32, 3, stride=2, padding=1), nn.SiLU(),
            nn.AdaptiveAvgPool2d((5, 3)))
        self.combine = nn.Sequential(nn.Linear(3 * 32 * 5 * 3, 128), nn.SiLU(),
                                     nn.Linear(128, 2 * 32 * 5 * 3), nn.SiLU())
        channels = [32, 32, 16, 16, 8, 3]
        layers = []
        for i, (a, b) in enumerate(zip(channels, channels[1:])):
            layers.extend([nn.ConvTranspose2d(a, b, 4, stride=2, padding=1),
                           nn.Tanh() if i == 4 else nn.SiLU()])
        self.decoder = nn.Sequential(*layers)

    def forward(self, patches):
        batch = patches.shape[0]
        assert patches.shape[1:] == (3, 3, 160, 96)
        features = self.encoder(patches.reshape(batch * 3, 3, 160, 96))
        output = self.combine(features.reshape(batch, -1)).reshape(batch * 2, 32, 5, 3)
        return self.decoder(output).reshape(batch, 2, 3, 160, 96)


def crop_tensor(image, boxes):
    return torch.from_numpy(np.stack([
        np.asarray(image.crop(box), dtype=np.float32).transpose(2, 0, 1) / 127.5 - 1
        for box in boxes]))


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def train(root):
    from safetensors.torch import save_file
    sys.path.insert(0, str(root / 'source'))
    from common import render
    protocol = root / 'source/train_protocol.json'
    cfg = json.loads(protocol.read_text())
    assert cfg['updates'] == 2000 and cfg['batch_size'] == 16
    assert cfg['learning_rate'] == .001 and cfg['input_noise_std'] == .04
    assert cfg['loss'] == 'foreground-weighted RGB MSE' and cfg['pretrained_weights'] is False
    assert sys.platform == 'darwin' and os.geteuid() == 502
    assert digest(Path(__file__)) == cfg['model_source_sha256']
    torch.set_num_threads(4)
    torch.mps.set_per_process_memory_fraction(.65)
    torch.manual_seed(cfg['seed'])
    rng = torch.Generator().manual_seed(cfg['seed'] + 1)
    font = root / 'fonts/train.ttf'
    assert digest(font) == cfg['font_sha256']
    inputs, targets = [], []
    for a in range(2):
        for b in range(2):
            for c in range(2):
                total = a + b + c
                target = crop_tensor(render(0, 0, 0, font, answer=10 * (total // 2) + total % 2), OUTPUT_BOXES)
                for variant in range(4):
                    inputs.append(crop_tensor(render(a, b, c, font, variant=variant), INPUT_BOXES))
                    targets.append(target)
    inputs, targets = torch.stack(inputs), torch.stack(targets)
    model = PixelTransducer().to('mps')
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg['learning_rate'], weight_decay=.0001,
                                  foreach=False)
    folder = root / 'training'
    folder.mkdir(exist_ok=False)
    parameters = sum(p.numel() for p in model.parameters())
    data_hash = hashlib.sha256(inputs.numpy().tobytes() + targets.numpy().tobytes()).hexdigest()
    initial_hash = hashlib.sha256()
    for name, parameter in sorted(model.named_parameters()):
        initial_hash.update(name.encode())
        initial_hash.update(parameter.detach().cpu().numpy().tobytes())
    with (folder / 'initialization.json').open('x') as stream:
        json.dump({'parameters': parameters, 'initial_parameters_sha256': initial_hash.hexdigest(),
                   'training_tensors_sha256': data_hash, 'protocol_sha256': digest(protocol)}, stream, indent=2)
    started = time.time()
    with (folder / 'metrics.jsonl').open('x') as log:
        for step in range(1, cfg['updates'] + 1):
            indices = torch.randint(len(inputs), (cfg['batch_size'],), generator=rng)
            x, y = inputs[indices], targets[indices].to('mps')
            x = (x + cfg['input_noise_std'] * torch.randn(x.shape, generator=rng)).clamp(-1, 1).to('mps')
            output = model(x)
            weights = 1 + 4 * (1 - y) / 2
            loss = ((output - y).square() * weights).sum() / weights.sum()
            assert torch.isfinite(loss).item()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            assert torch.isfinite(norm).item()
            optimizer.step()
            if step == 1 or step % 10 == 0:
                row = {'step': step, 'loss': float(loss.detach()),
                       'elapsed_seconds': time.time() - started}
                log.write(json.dumps(row) + '\n')
                log.flush()
                print(json.dumps(row), flush=True)
    model.to('cpu').eval()
    path = folder / 'model.safetensors'
    save_file(model.state_dict(), str(path))
    result = {'complete': True, 'updates': cfg['updates'], 'seed': cfg['seed'],
              'parameters': parameters, 'training_rasters': len(inputs),
              'checkpoint_sha256': digest(path), 'checkpoint_bytes': path.stat().st_size,
              'protocol_sha256': digest(protocol), 'model_source_sha256': digest(Path(__file__)),
              'elapsed_seconds': time.time() - started, 'selection': 'final fixed update',
              'evaluation_performed': False}
    with (folder / 'training-summary.json').open('x') as stream:
        json.dump(result, stream, indent=2)
    print(json.dumps(result), flush=True)


def load_predictor(parent, cfg):
    from safetensors.torch import load_file
    assert digest(Path(__file__)) == cfg['pixel_model_source_sha256']
    device = cfg.get('device', 'mps')
    assert device in {'cpu', 'mps', 'cuda'}
    threads = cfg.get('threads', 4)
    assert type(threads) is int and 1 <= threads <= 4
    torch.set_num_threads(threads)
    if device == 'mps':
        torch.mps.set_per_process_memory_fraction(.65)
    model = PixelTransducer()
    model.load_state_dict(load_file(parent / cfg['checkpoint_file']))
    model.to(device).eval().requires_grad_(False)

    def predict(reference, seed):
        # This baseline is deterministic: no sampling noise or hidden carry state.
        with torch.no_grad():
            output = model(crop_tensor(reference, INPUT_BOXES).unsqueeze(0).to(device))[0].cpu().numpy()
        result = reference.copy()
        for value, box in zip(output, OUTPUT_BOXES):
            pixels = np.clip((value.transpose(1, 2, 0) + 1) * 127.5, 0, 255).round().astype('uint8')
            result.paste(Image.fromarray(pixels), box[:2])
        return result

    return predict


if __name__ == '__main__':
    os.umask(0o077)
    os.environ.setdefault('PYTORCH_ENABLE_MPS_FALLBACK', '0')
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    train(args.root.resolve())
