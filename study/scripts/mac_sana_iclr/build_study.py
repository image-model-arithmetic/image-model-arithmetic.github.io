import ast
import hashlib
import json
import os
from pathlib import Path
import shutil
import time


REPO = Path(__file__).resolve().parents[2]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def replace_definition(source, name, replacement):
    tree = ast.parse(source)
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name]
    assert len(nodes) == 1
    node = nodes[0]
    lines = source.splitlines(keepends=True)
    result = ''.join(lines[:node.lineno-1])+replacement+'\n'+''.join(lines[node.end_lineno:])
    compile(result, name, 'exec')
    return result


def main():
    root = REPO/'runs/mac-sana-iclr-controls-v1'
    assert not root.exists()
    source = REPO/'runs/mac-sana-binary-public-v1/bundle/source'
    config = json.loads((source/'mac_sana_pixel_v1.json').read_text())
    candidates = {}
    for p in source.iterdir():
        assert p.is_file() and not p.is_symlink() and p.stat().st_size < 20000
        candidates[p.name] = p.read_text()
    for name in ['study_models.py', 'initialize_study.py']:
        candidates[name] = (Path(__file__).parent/name).read_text()
    arms = []
    for seed in [20260924, 20260925, 20260926]:
        for kind in ['latent-mlp', 'restricted-random', 'public']:
            if seed == 20260924 and kind == 'public':
                continue
            arms.append({'name': kind+'-'+str(seed), 'condition': kind, 'seed': seed})
    plan = {'registered_unix': time.time(), 'model_execution_started': False,
            'collector_sha256': digest(REPO/'runs/mac-sana-binary-public-v1/reconnect-local-source-r1/scripts/mac_sana_column_followup/collect_completed.py'),
            'reference': 'mac-sana-binary-public-v1', 'reference_seed': 20260924,
            'new_arms': arms, 'maximum_queue_hours': 72, 'maximum_arm_hours': 12,
            'data': 'Exact existing public-run 960 training / 80 development rasters and VAE latents',
            'evaluation': 'Same frozen facts, programs, font files and evaluation noise for every condition and seed',
            'selection': 'Original earliest-best addition then copy development selection; original perfect-development early stop at 600',
            'outcome_policy': 'Report every seed, failure and infrastructure interruption; never select a condition from final test scores',
            'interpretation': 'Restricted random SANA tests the original adaptation regime, not a fully trained scratch SANA. Latent MLP shares the pretrained VAE. Neither is a from-scratch pixel-only baseline.',
            'small_baseline': {'architecture': '12800-512-512-8192 GELU MLP predicting the same one-step latent velocity',
                'trainable_parameters': 11019264, 'learning_rate': .0003, 'tuning_trials': 1,
                'limit': 'One fixed optimization recipe; failure does not establish the best attainable small-model performance'},
            'remaining_studies': ['visible addition/subtraction', 'visual-state stability', 'family-disjoint appearance panel', 'SANA causal localization', 'pixel-only baseline', 'denoising schedule comparison if needed by the final claim']}
    compiled = []
    for arm in arms:
        texts = dict(candidates)
        arm_config = dict(config)
        arm_config.update(experiment='mac-sana-iclr-'+arm['name'], seed=arm['seed'], study_condition=arm['condition'],
                          status='Prospective fixed initialization comparison; no outputs inspected',
                          trigger='Complete clean-public experiment and independent audit; sequential resource admission',
                          claim_boundary=plan['interpretation'])
        if arm['condition'] == 'latent-mlp':
            arm_config.update(learning_rate=.0003, final_feedforward_learning_rate=.0003,
                initial_adapter='Fresh latent-MLP parameters; no arithmetic checkpoint',
                trainable='All 11,019,264 latent-MLP parameters; original VAE and digit critic frozen',
                conditioning='Flattened noisy latent and top nine rows of answer-free context latents',
                inference='One latent velocity prediction and original frozen VAE decode; no candidate selection or OCR during generation')
            arm_config['initialization'] = dict(arm_config['initialization'], kind='random-latent-mlp')
            texts['train_pixel.py'] = replace_definition(texts['train_pixel.py'], 'load_pixel_model',
                "def load_pixel_model(root):\n    from study_models import small_model\n    return small_model(root)\n")
            texts['train_pixel.py'] = replace_definition(texts['train_pixel.py'], 'predict_pixel',
                "def predict_pixel(model, x, context):\n    with torch.autocast('mps', dtype=torch.bfloat16):\n        result = model(x, context)\n    return result.float()\n")
        if arm['condition'] == 'restricted-random':
            old = "model=SanaTransformer2DModel.from_pretrained(snapshot,subfolder='transformer',variant='fp16',torch_dtype=torch.float32).requires_grad_(False)"
            assert texts['train.py'].count(old) == 1
            texts['train.py'] = texts['train.py'].replace(old, "from study_models import random_base\n model=random_base(root,snapshot)")
            arm_config['initialization'] = dict(arm_config['initialization'], kind='restricted-random-denoiser')
            arm_config['initial_adapter'] = 'Native randomly initialized SANA, rounded to FP16 then promoted; original adaptation mask; frozen pretrained VAE'
        for name in ['mac_sana_pixel_v1.json', 'mac_sana_column_v1.json']:
            texts[name] = json.dumps(arm_config, indent=2, sort_keys=True)+'\n'
        for name, text in texts.items():
            if name.endswith('.py'):
                compile(text, name, 'exec')
            else:
                json.loads(text)
        compiled.append((arm, texts))
    for name in ['run_study_arm.py', 'study_rpc.py', 'watch_study.py', 'archive_checkpoints.py', 'checkpoint_rpc.py']:
        compile((Path(__file__).parent/name).read_text(), name, 'exec')
    root.mkdir(mode=0o700)
    (root/'plan.json').write_text(json.dumps(plan, indent=2)+'\n')
    for arm, texts in compiled:
        directory = root/'arms'/arm['name']/'source'
        directory.mkdir(parents=True)
        for name, text in texts.items():
            (directory/name).write_text(text)
        (directory.parent/'source-manifest.json').write_text(json.dumps({'files': [
            {'path': 'source/'+name, 'bytes': (directory/name).stat().st_size, 'sha256': digest(directory/name)} for name in sorted(texts)]}, indent=2)+'\n')
        control = directory.parent/'control-source'
        control.mkdir()
        shutil.copyfile(REPO/'runs/mac-sana-binary-public-v1/bundle/control-source/inventory_evidence.py', control/'inventory_evidence.py')
    for name in ['run_study_arm.py', 'study_rpc.py', 'watch_study.py', 'archive_checkpoints.py', 'checkpoint_rpc.py']:
        shutil.copyfile(Path(__file__).parent/name, root/name)
    files = [p for p in root.rglob('*') if p.is_file()]
    assert len(files) < 200 and sum(p.stat().st_size for p in files) < 2000000
    (root/'transfer-manifest.json').write_text(json.dumps({'files': [
        {'path': p.relative_to(root).as_posix(), 'bytes': p.stat().st_size, 'sha256': digest(p)} for p in sorted(files)]}, indent=2)+'\n')
    print(json.dumps({'new_arms': len(arms), 'source_files': len(files), 'registered': True}))


if __name__ == '__main__':
    main()
