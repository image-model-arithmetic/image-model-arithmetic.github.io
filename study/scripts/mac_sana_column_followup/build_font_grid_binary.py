"""Register richer input appearances after the failed four-font transfer result."""
import ast
import json
from pathlib import Path
import shutil
import time
from collect_completed import digest, write

REPO = Path(__file__).resolve().parents[2]
ROOT = REPO/'runs/mac-sana-binary-font-grid-v1'
PARENT = REPO/'runs/mac-sana-binary-style-v1'


def replace(path, old, new):
    source = path.read_text()
    assert source.count(old) == 1, (path, old)
    path.write_text(source.replace(old, new))


def main():
    assert not ROOT.exists()
    ROOT.mkdir()
    bundle = ROOT/'bundle'; bundle.mkdir()
    for name in ['source-manifest.json', 'post-source-manifest.json', 'control-source-manifest.json']:
        for row in json.loads((PARENT/'bundle'/name).read_text())['files']:
            source = PARENT/'bundle'/row['path']
            assert source.stat().st_size == row['bytes'] and digest(source) == row['sha256']
            target = bundle/row['path']; target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
    font_specs = []
    inputs = [('train.ttf', '/usr/share/fonts/liberation/LiberationSans-Bold.ttf'),
        ('input-sans-regular.ttf', '/usr/share/fonts/liberation/LiberationSans-Regular.ttf'),
        ('input-serif-regular.ttf', '/usr/share/fonts/liberation/LiberationSerif-Regular.ttf'),
        ('input-free-sans.otf', '/usr/share/fonts/gnu-free/FreeSans.otf'),
        ('input-liberation-mono.ttf', '/usr/share/fonts/liberation/LiberationMono-Regular.ttf'),
        ('input-free-mono.otf', '/usr/share/fonts/gnu-free/FreeMono.otf'),
        ('input-free-serif.otf', '/usr/share/fonts/gnu-free/FreeSerif.otf'),
        ('input-adwaita-mono.ttf', '/usr/share/fonts/Adwaita/AdwaitaMono-Regular.ttf'),
        ('transfer.ttf', '/usr/share/fonts/TTF/IosevkaNerdFont-Regular.ttf')]
    for name, source in inputs:
        original = Path(source); assert original.is_file()
        shutil.copy2(original, bundle/'fonts'/name)
        font_specs.append({'path': 'fonts/'+name, 'bytes': original.stat().st_size, 'sha256': digest(original)})
    config = json.loads((PARENT/'bundle/source/mac_sana_pixel_v1.json').read_text())
    config.pop('input_style_curriculum')
    config.update(experiment='mac-sana-binary-font-grid-v1', seed=20260921,
        status='Prospective font and appearance grid; no model execution at registration',
        initial_adapter='Exact selected G update-200 checkpoint, with fresh optimizer and random generators',
        initial_checkpoint_sha256='c8236ab7c5f6c1a7381e7daf3dfdf2d49a1f6d75e1ce250fb94e6bb30e83b71e',
        initial_checkpoint_step=200,
        evaluation_seeds={'local': 2200921, 'program': 3300921, 'program_noise': 4400921},
        trigger='After the complete G independent integrity audit, G export/demo completion, and the earlier conditional long/carry decisions; regardless of G mathematical failure',
        checkpoint_selection='Addition then copying exactness on the fixed 80-image appearance development split; earliest tie. Stop at update 600 only if all 64 additions and 16 copies are template correct.',
        require_stage_seed_receipts=True,
        claim_boundary='Binary arithmetic after a broader supervised input-font and appearance curriculum on the same pretrained SANA. All eight elementary facts are supervised. New noise and programs, and one held-out Iosevka input font. Earlier decimal and font-transfer failures remain failures. No pretraining advantage, withheld-fact inference, autonomous scheduling, or small-total-training claim.')
    config['font_grid'] = {'training_fonts': font_specs[:8], 'heldout_font': font_specs[8],
        'training_positions': [[0, 0], [-3, 2], [3, -2], [1, 3]], 'training_stroke_widths': [0, 1, 2],
        'development_position': [5, -5], 'development_stroke_width': 1,
        'training_images': 960, 'development_images': 80,
        'assignment': 'Full Cartesian product of 8 fonts, 4 positions, 3 stroke widths, and the same 8 addition plus 2 copy facts. Development has every font/fact at a separate position; development rows are excluded from every optimizer sample.',
        'output': 'Every answer is rendered in the original LiberationSans-Bold convention. Only input glyphs are augmented; target context exactly equals the answer-free reference.',
        'prior_failures': 'The two previously tested monospace fonts are explicitly training inputs in this new experiment. Their previous evaluations remain failures. Iosevka is excluded from training and development.',
        'heldout_use': 'First Iosevka model outputs occur in the frozen independent evaluation. Font raster portability checks perform no model inference.'}
    config['binary'].update(development_all_variants=False, development_noise_seed=1100921,
        training_rows=960, development_rows=80,
        early_stop={'step': 600, 'addition_correct': 64, 'copy_correct': 16},
        variants=12)
    config_path = REPO/'configs/mac_sana_binary_font_grid_v1.json'
    assert not config_path.exists(); write(config_path, config)
    for name in ['mac_sana_pixel_v1.json', 'mac_sana_column_v1.json']:
        shutil.copy2(config_path, bundle/'source'/name)
    shutil.copy2(Path(__file__).with_name('font_grid_binary_protocol.py'), bundle/'source/font_grid_binary_protocol.py')
    prepare = bundle/'source/prepare_binary.py'
    replace(prepare, 'from style_binary_protocol import style_rows,render_pair',
        'from font_grid_binary_protocol import training_and_development_rows,render_pair,audit_rasters')
    replace(prepare, "audit['protocol_sha256']=digest(root/'source/mac_sana_pixel_v1.json')",
        "audit['font_grid']=audit_rasters(root,config)\naudit['elementary_reference_training_rows']=audit.pop('training_rows')\naudit['training_rows']=960\naudit['development_rows']=80\naudit['protocol_sha256']=digest(root/'source/mac_sana_pixel_v1.json')")
    replace(prepare, 'rows=style_rows(root,config);checks=[]', 'rows=training_and_development_rows(root,config);checks=[]')
    replace(prepare, "'total':40", "'total':len(rows)")
    trainer = bundle/'source/train_pixel.py'
    replace(trainer, 'import numpy as np\n', 'import numpy as np\nfrom font_grid_binary_protocol import optimization_indices, development_indices\n')
    replace(trainer, "copy_ids = [i for i, row in enumerate(rows) if row['copy']]", 'copy_ids = optimization_indices(rows, True)')
    replace(trainer, "add_ids = [i for i, row in enumerate(rows) if not row['copy']]", 'add_ids = optimization_indices(rows, False)')
    replace(trainer, 'ids = list(range(len(rows)))', 'ids = development_indices(rows)')
    replace(trainer, "record = {'step': step,", "record = {'step': step, 'example_ids': [rows[i]['id'] for i in ids],")
    initial = bundle/'source/initial_binary.py'
    replace(initial, 'from train_pixel import load_pixel_model,sample_pixel',
        'from train_pixel import load_pixel_model,sample_pixel\nfrom font_grid_binary_protocol import development_indices')
    replace(initial, "contexts=torch.from_numpy(np.stack([np.load(root/'data'/r['file'])['context'] for r in rows]))",
        "ids=development_indices(rows)\ncontexts=torch.from_numpy(np.stack([np.load(root/'data'/rows[i]['file'])['context'] for i in ids]))")
    replace(initial, 'indices=np.arange(len(rows))', 'indices=np.array(ids)')
    baseline = bundle/'post-source/baseline_pixel.py'
    replace(baseline, 'ids = list(range(len(rows)))', "ids = [i for i, row in enumerate(rows) if row['split'] == 'development']")
    orchestrator = bundle/'source/orchestrate_binary.py'
    replace(orchestrator, "assert metrics['add']['count']==32 and metrics['copy']['count']==8",
        "assert metrics['add']['count']==64 and metrics['copy']['count']==16")
    replace(orchestrator, 'if end==600 and score==(32,8):', 'if end==600 and score==(64,16):')
    evaluator = bundle/'source/evaluate_binary.py'
    replace(evaluator, 'folder.mkdir();carry=None;images=[];input_digests=[]',
        'folder.mkdir();carry=None;images=[];input_digests=[];stage_seeds=[]')
    replace(evaluator, "  im=generate(ref,seed);im.save(folder/f'step-{j:02d}.png');",
        "  stage_seeds.append(seed)\n  im=generate(ref,seed);im.save(folder/f'step-{j:02d}.png');")
    replace(evaluator, '  r.update(input_rgb_sha256=input_digests[j],image_sha256=',
        '  r.update(seed=stage_seeds[j],input_rgb_sha256=input_digests[j],image_sha256=')
    for row in json.loads((PARENT/'local-source-manifest.json').read_text())['files']:
        if row['path'].endswith('watch_style_binary.py'): continue
        source = PARENT/row['path']
        assert source.stat().st_size == row['bytes'] and digest(source) == row['sha256']
        target = ROOT/row['path']; target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    auditor = ROOT/'local-source/scripts/mac_sana_column_followup/audit_binary.py'
    replace(auditor, "            stage_id=f'{identifier}/step-{j:02d}'",
        "            stage_id=f'{identifier}/step-{j:02d}'\n            required_seed=config['evaluation_seeds']['program_noise']+spec['length']*10000+int(identifier.rsplit('-',1)[1])*100+j\n            require(stage.get('seed')==required_seed,stage_id,'prospective stage noise seed')")
    collector = ROOT/'local-source/scripts/mac_sana_column_followup/collect_completed.py'
    replace(collector, 'check=True, timeout=600)', 'check=True, timeout=1200)')
    for path in [collector, ROOT/'local-source/scripts/mac_sana_column_followup/finish_collection.py']:
        source = path.read_text().replace('8 * 3600', '12 * 3600'); path.write_text(source)
    for path in ROOT.rglob('*.py'): ast.parse(path.read_text(), filename=str(path))
    for directory, name in [('source', 'source-manifest.json'), ('post-source', 'post-source-manifest.json'), ('control-source', 'control-source-manifest.json')]:
        paths = list((bundle/directory).rglob('*'))
        if directory == 'source': paths += list((bundle/'fonts').iterdir())
        entries = [{'path': p.relative_to(bundle).as_posix(), 'bytes': p.stat().st_size, 'sha256': digest(p)} for p in paths if p.is_file()]
        write(bundle/name, {'files': sorted(entries, key=lambda r: r['path'])})
    entries = [{'path': p.relative_to(ROOT).as_posix(), 'bytes': p.stat().st_size, 'sha256': digest(p)}
               for p in (ROOT/'local-source').rglob('*') if p.is_file()]
    write(ROOT/'local-source-manifest.json', {'files': sorted(entries, key=lambda r: r['path'])})
    write(ROOT/'registration.json', {'registered_unix': time.time(), 'protocol_sha256': digest(config_path),
        'model_execution_started': False, 'parent_template_results': {'familiar': [200, 200], 'unfamiliar': [121, 200]},
        'parent_independent_audit_complete_at_registration': False,
        'parent_checkpoint_sha256': config['initial_checkpoint_sha256'],
        'source_manifests': {name: digest(bundle/name) for name in ['source-manifest.json', 'post-source-manifest.json', 'control-source-manifest.json']},
        'local_source_manifest_sha256': digest(ROOT/'local-source-manifest.json')})
    print(json.dumps({'root': str(ROOT), 'protocol_sha256': digest(config_path)}))


if __name__ == '__main__': main()
