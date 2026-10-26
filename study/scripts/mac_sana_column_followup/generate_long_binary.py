"""Run the frozen binary model for longer sums, with resumable raw-pixel state."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

os.environ.setdefault('PYTORCH_ENABLE_MPS_FALLBACK', '0')
sys.dont_write_bytecode = True


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(4*1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


def write(path, data):
    temporary = path.with_name(path.stem+'.tmp.json')
    temporary.write_text(json.dumps(data, indent=2, sort_keys=True)+'\n')
    temporary.replace(path)


def load_committed_stage(path, record_path, expected):
    from PIL import Image
    saved = json.loads(record_path.read_text())
    assert set(saved) == set(expected) | {'image_sha256', 'image_bytes'}
    assert all(saved[k] == v for k, v in expected.items()), expected['id']
    assert path.is_file() and path.stat().st_size == saved['image_bytes']
    assert digest(path) == saved['image_sha256'], expected['id']
    image = Image.open(path)
    assert image.size == (512, 512) and image.mode == 'RGB'
    return image, saved


def preserve_uncommitted(root, label, candidates, expected):
    paths = [p for p in candidates if p.exists()]
    if not paths:
        return
    recovery = root/'recovery'/f'{label}-{time.time_ns()}'
    recovery.mkdir(parents=True, exist_ok=False)
    inventory = [{'path': p.name, 'bytes': p.stat().st_size, 'sha256': digest(p)} for p in paths]
    write(recovery/'receipt.json', {'reason': 'Uncommitted files preserved before deterministic regeneration',
                                    'expected': expected, 'files': inventory})
    for path in paths:
        path.rename(recovery/path.name)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--parent', type=Path, required=True)
    args = parser.parse_args()
    root, parent = args.root.resolve(), args.parent.resolve()
    config_path = root/'source/protocol.json'
    config = json.loads(config_path.read_text())
    admission = json.loads((root/'admission.json').read_text())
    assert admission['passed'] and admission['protocol_sha256'] == digest(config_path)
    assert admission['parent_protocol_sha256'] == config['parent_protocol_sha256']
    assert digest(parent/'source/mac_sana_pixel_v1.json') == config['parent_protocol_sha256']
    for manifest_root, filename in [(root, 'source-manifest.json'), (parent, 'source-manifest.json')]:
        entries = json.loads((manifest_root/filename).read_text())['files']
        for row in entries:
            path = manifest_root/row['path']
            assert path.stat().st_size == row['bytes'] and digest(path) == row['sha256'], row['path']
    checkpoint = parent/'training/best.pt'
    assert digest(checkpoint) == admission['checkpoint_sha256']
    for name in ['status.json', 'finish-status.json']:
        status = json.loads((parent/name).read_text())
        assert status['complete'] and status['phase'] == 'complete'
    specs_path = root/'data/programs.json'
    assert digest(specs_path) == admission['programs_sha256']
    specs = json.loads(specs_path.read_text())
    from long_binary_protocol import programs, stage_seed
    assert specs == programs(config)

    sys.path.insert(0, str(parent/'source'))
    from common import CARRY_OUT, pil, pixels, render
    from train_pixel import load_pixel_model, sample_pixel
    from serial_image_inference import generate_serial
    import torch
    from diffusers import AutoencoderDC

    torch.set_num_threads(4)
    torch.mps.set_per_process_memory_fraction(.65)
    model, _ = load_pixel_model(parent)
    state = torch.load(checkpoint, map_location='cpu', weights_only=False)
    assert state['pixel_protocol_sha256'] == config['parent_protocol_sha256']
    assert set(state['trainable']) == {n for n, p in model.named_parameters() if p.requires_grad}
    assert not model.load_state_dict(state['trainable'], strict=False).unexpected_keys
    model.eval()
    del state
    base = json.loads((parent/'model.json').read_text())
    snapshot = Path(base['snapshot'])
    assert base['revision'] == config['base_revision']
    for entry in base['files']:
        path = snapshot/entry['path']
        assert path.stat().st_size == entry['bytes'] and digest(path) == entry['sha256']
    vae = AutoencoderDC.from_pretrained(snapshot, subfolder='vae', variant='fp16',
                                      torch_dtype=torch.float32).eval().requires_grad_(False)
    out = root/'evaluation'
    out.mkdir(exist_ok=True)
    identity = {'protocol_sha256': digest(config_path), 'checkpoint_sha256': admission['checkpoint_sha256'],
                'programs_sha256': digest(specs_path), 'script_sha256': digest(Path(__file__)),
                'parent_source_manifest_sha256': digest(parent/'source-manifest.json'),
                'font_sha256': digest(parent/'fonts/train.ttf')}
    identity_path = out/'identity.json'
    if identity_path.exists():
        assert json.loads(identity_path.read_text()) == identity
    else:
        assert not list(out.iterdir())
        write(identity_path, identity)
    completed = 0
    started = time.monotonic()
    for spec in specs:
        folder = out/spec['id']
        folder.mkdir(exist_ok=True)
        carry = None
        records = []
        for j, (a, b) in enumerate(zip(spec['a_bits'], spec['b_bits'])):
            assert time.monotonic()-started < config['resources']['maximum_controller_hours']*3600
            reference = render(a, b, 0, parent/'fonts/train.ttf', carry_pixels=carry)
            input_sha = hashlib.sha256(reference.tobytes()).hexdigest()
            seed = stage_seed(config, spec, j)
            path = folder/f'step-{j:02d}.png'
            record_path = folder/f'step-{j:02d}.json'
            expected = {'id': f"{spec['id']}/step-{j:02d}", 'program_id': spec['id'],
                        'stage': j, 'seed': seed, 'input_rgb_sha256': input_sha,
                        'image_path': path.relative_to(out).as_posix()}
            if record_path.exists():
                image, saved = load_committed_stage(path, record_path, expected)
            else:
                # Preserve an image whose process ended before its commit receipt.
                # No later receipt may exist beyond a missing stage.
                assert not any((folder/f'step-{k:02d}.json').exists() for k in range(j+1, spec['length']))
                preserve_uncommitted(root, f'{spec["id"]}-{j:02d}',
                    [path, path.with_name(path.stem+'.tmp.png'), record_path.with_name(record_path.stem+'.tmp.json')], expected)
                image = generate_serial(model, vae, [reference], seed, sample_pixel, pixels, pil, 1)[0]
                temporary = path.with_name(path.stem+'.tmp.png')
                image.save(temporary)
                assert temporary.is_file()
                temporary.replace(path)
                saved = {**expected, 'image_sha256': digest(path), 'image_bytes': path.stat().st_size}
                write(record_path, saved)
            carry = image.crop(CARRY_OUT)
            records.append(saved)
            completed += 1
            write(root/'progress.json', {'completed_images': completed, 'total_images': 3264,
                                        'program': spec['id'], 'completed_stage': j, 'updated_unix': time.time()})
        program_record = {'spec': spec, 'stages': records}
        if (folder/'program.json').exists():
            assert json.loads((folder/'program.json').read_text()) == program_record, spec['id']
        else:
            preserve_uncommitted(root, spec['id'], [folder/'program.tmp.json'], {'program': spec['id']})
            write(folder/'program.json', program_record)
        print(json.dumps({'program': spec['id'], 'completed_images': completed, 'total_images': 3264}), flush=True)
    assert completed == 3264
    write(out/'generation-summary.json', {**identity, 'complete': True, 'program_count': len(specs),
        'image_count': completed, 'answer_scoring_performed': False, 'finished_unix': time.time()})


if __name__ == '__main__':
    main()
