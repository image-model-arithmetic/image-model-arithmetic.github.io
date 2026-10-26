"""Prepare and execute the fixed SANA curriculum without editing Python source.

Preparation is CPU-only. Execution is a single sequential, finite Mac process.
Historical runs supply immutable data and protocols, never trained parameters.
"""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import resource
import shutil
import signal
import stat
import subprocess
import sys
import time


def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def write(path, value):
    """Private, atomic receipts. Callers decide whether replacement is allowed."""
    path = Path(path)
    temporary = path.with_name(path.name + '.tmp')
    with open(temporary, 'x', opener=lambda p, f: os.open(p, f, 0o600)) as stream:
        json.dump(value, stream, indent=2)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def validate(recipe):
    assert recipe['schema'] == 1
    stages = recipe['stages']
    assert len(stages) == 7 and len({s['name'] for s in stages}) == 7
    assert [s['updates'] for s in stages] == [800, 900, 1200, 2000, 600, 200, 1200]
    assert sum(s['updates'] for s in stages) == recipe['total_updates'] == 6900
    assert type(recipe['seed']) is int and recipe['seed'] > 0
    assert 0 < recipe['wall_seconds'] <= 48 * 3600
    assert recipe['runtime_free_floor_bytes'] >= 40 * 2**30
    # Seven retained endpoints plus the largest temporary checkpoint require
    # 5,101,959,736 bytes in the observed architecture. Reserve 6 GiB for these
    # files and the much smaller evaluation outputs, above the 40 GiB floor.
    assert recipe['minimum_free_bytes'] >= recipe['runtime_free_floor_bytes'] + 6 * 2**30
    assert set(recipe['evaluation_seeds']) == {'local', 'program', 'program_noise'}
    assert all(type(v) is int and v > 0 for v in recipe['evaluation_seeds'].values())
    for index, stage in enumerate(stages):
        assert re.fullmatch(r'[a-z][a-z0-9-]+', stage['name'])
        assert stage['trainer'] == ('train.py' if index == 0 else 'train_pixel.py')
        assert re.fullmatch('[0-9a-f]{64}', stage['protocol_sha256'])
        assert re.fullmatch('[0-9a-f]{64}', stage['data_manifest_sha256'])
        assert not Path(stage['protocol']).is_absolute() and '..' not in Path(stage['protocol']).parts
        assert Path(stage['data_root']).is_absolute()
    return recipe


def stage_protocol(original, recipe, index):
    """The scientific amendment is explicit data, never a source-code patch."""
    config = json.loads(json.dumps(original))
    stage = recipe['stages'][index]
    seed = recipe['seed'] + index
    config.update(seed=seed, updates=stage['updates'],
                  development_endpoints=[stage['updates']],
                  checkpoint_selection=recipe['selection'],
                  claim_boundary=recipe['claim_boundary'])
    if index == 0:
        config['training']['seed'] = seed
        config['training']['max_updates'] = stage['updates']
    if 'critic' in config:
        config['critic_sha256'] = recipe['critic_sha256']
        config['critic_validation_protocol_sha256'] = recipe['critic_validation_protocol_sha256']
    if 'binary' in config:
        config['binary'].pop('early_stop', None)
        config['binary']['development_noise_seed'] = recipe['seed'] + 100
        config['evaluation_seeds'] = recipe['evaluation_seeds']
    # A new preceding checkpoint is bound after it finishes; inherited hashes
    # describe the old run and must never authorize a new initializer.
    config.pop('initial_checkpoint_sha256', None)
    config.pop('initial_checkpoint_step', None)
    config.pop('trigger', None)
    config['initial_adapter'] = 'Fresh public-weight start, then the preceding recipe stage'
    return config


def prepare(repo, recipe_path, output):
    recipe = validate(json.loads(recipe_path.read_text()))
    assert not output.exists() and not output.is_symlink()
    assert output.parent.is_dir() and output.parent.resolve() == output.parent.absolute()
    base = repo / 'scripts/mac_sana_column'
    follow = repo / 'scripts/mac_sana_column_followup'
    sources = {name: base / name for name in ['common.py', 'train.py']}
    sources.update({name: follow / name for name in [
        'pixel_contrast.py', 'ocr_critic.py', 'binary_protocol.py',
        'serial_image_inference.py', 'serial_inference_control.py']})
    sources['train_pixel.py'] = follow / 'train_curriculum.py'
    sources['evaluate_binary.py'] = repo / 'runs/mac-sana-binary-font-grid-v1/bundle/source/evaluate_binary.py'
    sources['audit_binary.py'] = repo / 'runs/mac-sana-binary-public-v1/reconnect-local-source-r1/scripts/mac_sana_column_followup/audit_binary.py'
    sources['recipe.py'] = Path(__file__).resolve()
    configs = []
    for index, stage in enumerate(recipe['stages']):
        path = repo / stage['protocol']
        assert path.is_file() and not path.is_symlink()
        assert digest(path) == stage['protocol_sha256']
        configs.append(stage_protocol(json.loads(path.read_text()), recipe, index))
    for path in sources.values():
        assert path.is_file() and not path.is_symlink() and path.stat().st_size < 100_000
        compile(path.read_text(), str(path), 'exec')
    assert sum(p.stat().st_size for p in sources.values()) < 250_000
    output.mkdir(mode=0o700)
    # Failed preparation remains a classified partial with a terminal receipt.
    try:
        (output / 'source').mkdir()
        (output / 'protocols').mkdir()
        for name, path in sources.items():
            with open(output / 'source' / name, 'xb') as stream:
                stream.write(path.read_bytes())
        write(output / 'recipe.json', recipe)
        for stage, config in zip(recipe['stages'], configs):
            write(output / 'protocols' / (stage['name'] + '.json'), config)
        members = [output / 'recipe.json', *sorted((output / 'source').iterdir()),
                   *sorted((output / 'protocols').iterdir())]
        write(output / 'manifest.json', {'files': [
            {'path': str(p.relative_to(output)), 'bytes': p.stat().st_size,
             'sha256': digest(p), 'mode': stat.S_IMODE(p.stat().st_mode)} for p in members]})
    except BaseException as error:
        write(output / 'prepare-failure.json', {'error_kind': type(error).__name__})
        raise
    return {'prepared': True, 'stages': len(configs), 'updates': recipe['total_updates'],
            'manifest_sha256': digest(output / 'manifest.json'), 'training_started': False}


def verify_bundle(root):
    manifest = json.loads((root / 'manifest.json').read_text())
    assert 0 < len(manifest['files']) < 40
    for row in manifest['files']:
        relative = Path(row['path'])
        assert not relative.is_absolute() and '..' not in relative.parts
        path = root / relative
        assert path.is_file() and not path.is_symlink()
        assert path.stat().st_size == row['bytes'] and digest(path) == row['sha256']
        assert stat.S_IMODE(path.stat().st_mode) == row['mode']
    return validate(json.loads((root / 'recipe.json').read_text()))


def input_links(root, recipe):
    """Preflight the complete immutable input table before creating any links."""
    links = []
    model = Path(recipe['model_root']) / 'model.json'
    model_info = json.loads(model.read_text())
    assert model_info['revision'] == '83d7a190bfd1fd070570a793d2dab5c7a3231b9d'
    assert 0 < len(model_info['files']) <= 10
    for row in model_info['files']:
        relative = Path(row['path'])
        assert not relative.is_absolute() and '..' not in relative.parts
        path = Path(model_info['snapshot']) / relative
        assert path.stat().st_size == row['bytes'] and digest(path) == row['sha256']
    for stage in recipe['stages']:
        old = Path(stage['data_root'])
        destination = root / 'stages' / stage['name']
        manifest = old / 'data/manifest.json'
        assert digest(manifest) == stage['data_manifest_sha256']
        rows = json.loads(manifest.read_text())['rows']
        assert 0 < len(rows) <= 1040
        assert len({r['file'] for r in rows}) == len(rows)
        members = [(manifest, destination / 'data/manifest.json'), (model, destination / 'model.json')]
        for row in rows:
            assert Path(row['file']).name == row['file'] and row['file'].endswith('.npz')
            source = old / 'data' / row['file']
            assert digest(source) == row['sha256']
            members.append((source, destination / 'data' / row['file']))
        font_root = (Path(recipe['model_root']) if stage['name'].startswith('decimal-') else old) / 'fonts'
        fonts = list(font_root.iterdir())
        assert 2 <= len(fonts) <= 12
        assert all(p.suffix in {'.ttf', '.otf'} for p in fonts)
        members.extend((p, destination / 'fonts' / p.name) for p in fonts)
        config = json.loads((root / 'protocols' / (stage['name'] + '.json')).read_text())
        if 'critic' in config:
            assert digest(Path(recipe['critic_root']) / 'critic/model.safetensors') == recipe['critic_sha256']
            for name in ['model.safetensors', 'validation.json']:
                members.append((Path(recipe['critic_root']) / 'critic' / name, destination / 'critic' / name))
        members.extend((p, destination / 'source' / p.name) for p in (root / 'source').iterdir())
        for name in ['mac_sana_column_v1.json', 'mac_sana_pixel_v1.json']:
            members.append((root / 'protocols' / (stage['name'] + '.json'), destination / 'source' / name))
        for source, target in members:
            # Resolve only the specifically declared immutable input alias.
            source = source.resolve(strict=True)
            info = source.stat()
            assert stat.S_ISREG(info.st_mode) and info.st_dev == root.stat().st_dev
            assert not target.exists() and not target.is_symlink()
            links.append((source, target))
    assert len(links) < 6000 and len({target for _, target in links}) == len(links)
    return links


def admission(root, recipe):
    """Read the actual Mac state before linking data or loading any model."""
    assert sys.platform == 'darwin' and os.geteuid() == 502
    tools = {name: shutil.which(name) for name in ['ps', 'sysctl', 'memory_pressure', 'caffeinate']}
    assert all(tools.values())
    limits = {name: list(resource.getrlimit(getattr(resource, name)))
              for name in ['RLIMIT_CPU', 'RLIMIT_AS', 'RLIMIT_DATA']}
    assert all(pair == [resource.RLIM_INFINITY, resource.RLIM_INFINITY] for pair in limits.values())
    result = subprocess.run([tools['sysctl'], '-n', 'hw.memsize'], capture_output=True, text=True, timeout=10)
    assert result.returncode == 0 and int(result.stdout.strip()) >= 16 * 2**30
    result = subprocess.run([tools['ps'], '-axo', 'pid=,ppid=,comm='], capture_output=True, text=True, timeout=10)
    assert result.returncode == 0
    rows = [line.split(None, 2) for line in result.stdout.splitlines()]
    assert rows and all(len(row) == 3 and row[0].isdigit() and row[1].isdigit() for row in rows)
    parents = {int(row[0]): int(row[1]) for row in rows}
    ancestors = {os.getpid()}
    current = os.getpid()
    while parents.get(current, 0) > 0 and parents[current] not in ancestors:
        current = parents[current]
        ancestors.add(current)
    other_python = [int(row[0]) for row in rows if 'python' in Path(row[2]).name.lower() and int(row[0]) not in ancestors]
    assert not other_python, 'Another Python job is active; run sequentially'
    samples = []
    for _ in range(2):
        result = subprocess.run([tools['memory_pressure']], capture_output=True, text=True, timeout=10)
        match = re.search(r'System-wide memory free percentage: (\d+)%', result.stdout)
        assert result.returncode == 0 and match
        samples.append({'unix': time.time(), 'memory_free_percent': int(match[1]),
                        'disk_free_bytes': shutil.disk_usage(root).free})
        time.sleep(1)
    assert all(s['memory_free_percent'] >= 20 and s['disk_free_bytes'] >= recipe['minimum_free_bytes'] for s in samples)
    return {'passed': True, 'euid': os.geteuid(), 'python': sys.executable,
            'hard_limits': limits, 'samples': samples, 'other_python_processes': len(other_python), 'tools': tools}


def install_links(links):
    created, directories = [], []
    try:
        for source, target in links:
            parents = []
            parent = target.parent
            while not parent.exists():
                parents.append(parent)
                parent = parent.parent
            assert parent.is_dir() and not parent.is_symlink()
            for parent in reversed(parents):
                parent.mkdir(mode=0o700)
                directories.append(parent)
            os.link(source, target)
            created.append(target)
    except BaseException:
        for target in reversed(created):
            target.unlink()
        for directory in reversed(directories):
            directory.rmdir()
        raise


def execute(argv, directory, deadline, floor, cancelled):
    """One child process group, private streams, terminal evidence on every exit."""
    assert time.time() < deadline - 30 and not cancelled
    directory.mkdir(mode=0o700)
    child = None
    error_kind = None
    try:
        with open(directory / 'stdout', 'xb') as stdout, open(directory / 'stderr', 'xb') as stderr:
            child = subprocess.Popen(argv, stdin=subprocess.DEVNULL, stdout=stdout, stderr=stderr,
                                     start_new_session=True, umask=0o077,
                                     env=dict(os.environ, PYTHONDONTWRITEBYTECODE='1'))
            low = 0
            while child.poll() is None:
                if cancelled or time.time() >= deadline - 30:
                    raise TimeoutError('deadline-or-signal')
                if shutil.disk_usage(directory).free < floor:
                    raise RuntimeError('disk-floor')
                if sys.platform == 'darwin':
                    result = subprocess.run(['/usr/bin/memory_pressure'], capture_output=True,
                                            text=True, timeout=min(10, max(1, deadline-time.time()-20)))
                    match = re.search(r'System-wide memory free percentage: (\d+)%', result.stdout)
                    assert result.returncode == 0 and match
                    low = low + 1 if int(match[1]) < 15 else 0
                    if low >= 2:
                        raise RuntimeError('memory-pressure')
                time.sleep(.1 if sys.platform != 'darwin' else 5)
    except BaseException as error:
        error_kind = type(error).__name__
        raise
    finally:
        if child is not None and child.poll() is None:
            os.killpg(child.pid, signal.SIGTERM)
            try:
                child.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)
                child.wait(timeout=10)
        write(directory / 'terminal.json', {'exit_code': None if child is None else child.returncode,
              'error_kind': error_kind, 'child_quiescent': child is None or child.poll() is not None,
              'argv': list(map(str, argv)), 'deadline_unix': deadline})
    assert child.returncode == 0, 'child-failed'


def run(root):
    recipe = verify_bundle(root)
    assert not (root / 'started.json').exists(), 'Preserve previous attempts; do not overwrite their evidence'
    admitted = admission(root, recipe)
    write(root / 'admission.json', admitted)
    links = input_links(root, recipe)
    write(root / 'input-preflight.json', {'links': [
        {'source': str(s), 'target': str(t), 'bytes': s.stat().st_size,
         'sha256': digest(s)} for s, t in links]})
    install_links(links)
    deadline = time.time() + recipe['wall_seconds']
    write(root / 'started.json', {'deadline_unix': deadline, 'hard_limits': admitted['hard_limits'],
          'python': sys.executable, 'manifest_sha256': digest(root / 'manifest.json')})
    cancelled = []
    previous_handlers = {signum: signal.signal(signum, lambda n, f: cancelled.append(n))
                         for signum in [signal.SIGTERM, signal.SIGINT]}
    generation_complete = False
    error_kind = None
    try:
        previous = None
        for stage in recipe['stages']:
            path = root / 'stages' / stage['name']
            if previous is not None:
                checkpoint = previous / 'training/resume.pt'
                receipt = json.loads((previous / 'training/checkpoint.json').read_text())
                assert digest(checkpoint) == receipt['sha256']
                assert not (path / 'initial.pt').exists()
                os.link(checkpoint, path / 'initial.pt')
                write(path / 'initial-checkpoint.json', receipt)
            write(root / 'status.json', {'complete': False, 'phase': stage['name']})
            argv = [sys.executable, '-B', str(path / 'source' / stage['trainer']),
                    '--root', str(path), '--end', str(stage['updates']), '--skip-development']
            execute(argv, path / 'execution', deadline, recipe['runtime_free_floor_bytes'], cancelled)
            receipt = json.loads((path / 'training/checkpoint.json').read_text())
            assert receipt['step'] == stage['updates'] and digest(path / 'training/resume.pt') == receipt['sha256']
            write(path / 'complete.json', receipt)
            previous = path
        os.link(previous / 'training/resume.pt', previous / 'training/best.pt')
        write(previous / 'training/selection.json', {**receipt, 'selection': recipe['selection']})
        execute([sys.executable, '-B', str(previous / 'source/evaluate_binary.py'), '--root', str(previous)],
                previous / 'evaluation-execution', deadline, recipe['runtime_free_floor_bytes'], cancelled)
        generation_complete = True
    except BaseException as error:
        error_kind = type(error).__name__
        raise
    finally:
        for signum, handler in previous_handlers.items():
            signal.signal(signum, handler)
        write(root / 'terminal.json', {'generation_complete': generation_complete, 'error_kind': error_kind,
              'independent_audit_complete': False, 'received_signals': cancelled,
              'finished_unix': time.time()})
        # The frozen OCR reader runs on Linux after a verified artifact transfer.
        # Generation success alone must never count as an audited experiment.
        write(root / 'status.json', {'complete': False,
              'phase': 'awaiting-independent-audit' if generation_complete else 'stopped'})


def audit(root):
    """Use the previously validated independent PNG reader with all cases retained."""
    result = subprocess.run(['tesseract', '--version'], capture_output=True, text=True, timeout=10)
    assert result.returncode == 0 and result.stdout.splitlines()[0] == 'tesseract 5.5.3'
    sys.path.insert(0, str(root / 'source'))
    spec = importlib.util.spec_from_file_location('frozen_binary_auditor', root / 'source/audit_binary.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    protocol = root / 'source/mac_sana_pixel_v1.json'
    result = module.audit(root / 'evaluation', [root / 'fonts/train.ttf', root / 'fonts/transfer.ttf'],
                          json.loads(protocol.read_text()), expected_protocol_sha256=digest(protocol))
    assert not (root / 'independent-audit.json').exists()
    write(root / 'independent-audit.json', result)
    assert result['integrity_passed']
    print(json.dumps({key: result[key] for key in ['integrity_passed', 'audited_success', 'local', 'programs']}))


def main():
    os.umask(0o077)
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    p = commands.add_parser('prepare')
    p.add_argument('--recipe', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p = commands.add_parser('run')
    p.add_argument('--root', type=Path, required=True)
    p = commands.add_parser('audit')
    p.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    if args.command == 'prepare':
        print(json.dumps(prepare(Path(__file__).resolve().parents[2], args.recipe.resolve(), args.output.absolute())))
    elif args.command == 'run':
        run(args.root.resolve())
    else:
        audit(args.root.resolve())


if __name__ == '__main__':
    main()
