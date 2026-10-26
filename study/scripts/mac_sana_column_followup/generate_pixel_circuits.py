"""Execute frozen image circuits with raw RGB wires and no answer reader."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

from pixel_circuits import INPUT_BOXES, OUTPUT_BOXES, validate
from generate_long_binary import digest, write


def initial_wires(root, circuit, font, render):
    from PIL import Image
    wires = {}
    for key, value in circuit['leaves'].items():
        external = circuit.get('externals', {}).get(key)
        if external is None:
            patch = render(value, 0, 0, font).crop(INPUT_BOXES[0])
        else:
            relative = Path(external['file'])
            assert not relative.is_absolute() and '..' not in relative.parts
            path = root / relative
            assert digest(path) == external['sha256']
            image = Image.open(path)
            assert image.mode == 'RGB' and image.size == (512, 512)
            patch = image.crop(OUTPUT_BOXES[external['port']])
            if external['transform'] == 'binary-128':
                patch = patch.convert('L').point(lambda x: 255 if x >= 128 else 0).convert('RGB')
        wires[key] = patch
    return wires


def load_generator(cfg):
    parent = Path(cfg['parent'])
    assert digest(parent / 'source-manifest.json') == cfg['parent_manifest_sha256']
    for row in json.loads((parent / 'source-manifest.json').read_text())['files']:
        p = parent / row['path']
        assert p.stat().st_size == row['bytes'] and digest(p) == row['sha256']
    checkpoint = parent / cfg.get('checkpoint_file', 'training/best.pt')
    parent_protocol = parent / cfg.get('parent_protocol_file', 'source/mac_sana_pixel_v1.json')
    assert digest(checkpoint) == cfg['checkpoint_sha256']
    assert digest(parent_protocol) == cfg['parent_protocol_sha256']
    sys.path.insert(0, str(parent / 'source'))
    if cfg.get('backend', 'latent') == 'pixel':
        from pixel_transducer import load_predictor
        return load_predictor(parent, cfg)
    assert cfg.get('backend', 'latent') == 'latent'
    from common import pixels, pil
    from train_pixel import load_pixel_model, sample_pixel
    from serial_image_inference import generate_serial
    import torch
    from diffusers import AutoencoderDC

    torch.set_num_threads(4)
    torch.mps.set_per_process_memory_fraction(.65)
    model, _ = load_pixel_model(parent)
    state = torch.load(checkpoint, map_location='cpu', weights_only=False)
    assert state['pixel_protocol_sha256'] == cfg['parent_protocol_sha256']
    assert set(state['trainable']) == {n for n, p in model.named_parameters() if p.requires_grad}
    assert not model.load_state_dict(state['trainable'], strict=False).unexpected_keys
    model.eval()
    del state
    base = json.loads((parent / 'model.json').read_text())
    assert base['revision'] == cfg['base_revision']
    snapshot = Path(base['snapshot'])
    for row in base['files']:
        p = snapshot / row['path']
        assert p.stat().st_size == row['bytes'] and digest(p) == row['sha256']
    vae = AutoencoderDC.from_pretrained(snapshot, subfolder='vae', variant='fp16',
                                       torch_dtype=torch.float32).eval().requires_grad_(False)

    def generate(reference, seed):
        return generate_serial(model, vae, [reference], seed, sample_pixel, pixels, pil, 1)[0]

    return generate


def main():
    os.umask(0o077)
    os.environ.setdefault('PYTORCH_ENABLE_MPS_FALLBACK', '0')
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    cfg = json.loads((root / 'protocol.json').read_text())
    parent = Path(cfg['parent'])
    admission = json.loads((root / 'started.json').read_text())
    assert admission['admission']['passed']
    assert digest(root / 'registration.json') == admission['registration_sha256']
    registration = json.loads((root / 'registration.json').read_text())
    for row in registration['files']:
        p = root / row['path']
        assert p.stat().st_size == row['bytes'] and digest(p) == row['sha256']
    circuits = json.loads((root / 'circuits.json').read_text())
    for circuit in circuits:
        validate(circuit)
    assert sum(len(c['nodes']) for c in circuits) == cfg['image_count']

    models = cfg.get('models', [cfg])
    assert 1 <= len(models) <= 2
    if len(models) == 2:
        # This study has exactly one latent generator and one independent pixel model.
        assert [m.get('backend', 'latent') for m in models] == ['latent', 'pixel']
    assert models[0]['parent'] == cfg['parent']
    for model_cfg in models:
        peer = Path(model_cfg['parent'])
        assert digest(peer / 'source/common.py') == digest(parent / 'source/common.py')
        assert digest(peer / 'fonts/train.ttf') == digest(parent / 'fonts/train.ttf')
    generators = [load_generator(model_cfg) for model_cfg in models]
    from common import render
    font = parent / 'fonts/train.ttf'
    out = root / 'evaluation'
    out.mkdir(exist_ok=False)
    identity = {'registration_sha256': digest(root / 'registration.json'),
                'checkpoint_sha256': cfg['checkpoint_sha256'], 'font_sha256': digest(font)}
    if 'models' in cfg:
        identity['model_checkpoints'] = [m['checkpoint_sha256'] for m in models]
    write(out / 'identity.json', identity)
    completed = 0
    for circuit in circuits:
        folder = out / circuit['id']
        folder.mkdir()
        wires = initial_wires(root, circuit, font, render)
        for index, entry in enumerate(circuit['nodes']):
            assert time.time() < admission['deadline_unix'] - 60
            reference = render(0, 0, 0, font)
            wire_hashes = []
            for box, wire in zip(INPUT_BOXES, entry['inputs']):
                patch = wires[wire]
                assert patch.mode == 'RGB' and patch.size == (96, 160)
                reference.paste(patch, box[:2])
                wire_hashes.append(hashlib.sha256(patch.tobytes()).hexdigest())
            seed = entry.get('noise_seed', cfg['noise_seed'] + completed)
            model_index = entry.get('model_index', 0)
            assert type(model_index) is int and 0 <= model_index < len(generators)
            image = generators[model_index](reference, seed)
            path = folder / (entry['id'] + '.png')
            temporary = path.with_suffix('.tmp.png')
            image.save(temporary)
            temporary.replace(path)
            record = {'id': entry['id'], 'seed': seed,
                'inputs': entry['inputs'], 'wire_rgb_sha256': wire_hashes,
                'input_rgb_sha256': hashlib.sha256(reference.tobytes()).hexdigest(),
                'image_sha256': digest(path), 'image_bytes': path.stat().st_size}
            if 'models' in cfg:
                record['model_index'] = model_index
            write(path.with_suffix('.json'), record)
            for port, box in OUTPUT_BOXES.items():
                wires[entry['id'] + ':' + port] = image.crop(box)
            completed += 1
            write(root / 'progress.json', {'completed_images': completed,
                'total_images': cfg['image_count'], 'updated_unix': time.time()})
        print(json.dumps({'circuit': circuit['id'], 'completed_images': completed}), flush=True)
    assert completed == cfg['image_count']
    write(out / 'generation-summary.json', {'complete': True, 'image_count': completed,
        'answer_scoring_performed': False, 'finished_unix': time.time()})


if __name__ == '__main__':
    main()
