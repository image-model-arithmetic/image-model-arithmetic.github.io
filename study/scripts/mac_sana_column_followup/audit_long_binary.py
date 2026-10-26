"""Independently score every stage of the conditional long binary evaluation."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from PIL import Image
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'mac_sana_column'))
from common import CARRY_OUT, answer_templates, digest, read_answer, render
from audit_binary import ocr
from long_binary_protocol import programs, stage_seed


def audit(directory, config, specs, fonts, parent_report, protocol_sha, ocr_reader=ocr, expected_identity=None):
    failures = []
    results = []
    expected_files = {'identity.json', 'generation-summary.json'}

    def require(condition, identifier, invariant):
        if not condition:
            failures.append({'id': identifier, 'invariant': invariant})

    require(parent_report['complete'] and parent_report['integrity_passed'] and parent_report['audited_success'],
            'parent', 'completed successful parent independent evaluation')
    require(parent_report['numerical_control']['passed'], 'parent', 'parent numerical control')
    require(specs == programs(config), 'programs', 'exact prospective corpus')
    require(config['gates'] == {
        'random_correct_per_length_minimum': 29, 'random_count_per_length': 32, 'stress_correct_required': 4,
        'program_exactness': 'A program counts as correct only if every intermediate subtotal and the final answer are numerically correct and every generated output is valid under both unchanged readers.',
        'both_readers_must_satisfy_every_gate': True, 'independent_integrity_audit_required': True},
        'protocol', 'unchanged mathematical gates')
    identity = json.loads((directory/'identity.json').read_text())
    generated = json.loads((directory/'generation-summary.json').read_text())
    require(identity['protocol_sha256'] == protocol_sha, 'identity', 'protocol identity')
    require(identity['checkpoint_sha256'] == parent_report['checkpoint']['full_checkpoint_sha256'],
            'identity', 'same frozen parent weights')
    require(identity['font_sha256'] == digest(fonts[0]), 'identity', 'training input font')
    for key, value in (expected_identity or {}).items():
        require(identity[key] == value, 'identity', key)
    for key, value in identity.items():
        require(generated[key] == value, 'generation-summary', key)
    require(generated['complete'] and generated['program_count'] == 68 and generated['image_count'] == 3264,
            'generation-summary', 'complete fixed inventory')
    require(generated['answer_scoring_performed'] is False, 'generation-summary', 'scoring absent from generation')
    banks = [answer_templates(font) for font in fonts]
    for spec in specs:
        identifier = spec['id']
        folder = directory/identifier
        expected_files.add(identifier+'/program.json')
        saved_program = json.loads((folder/'program.json').read_text())
        require(saved_program['spec'] == spec, identifier, 'program fields')
        require(len(saved_program['stages']) == spec['length'], identifier, 'complete stage receipts')
        raw_carry = None
        expected_carry = 0
        stages = []
        for j, (a, b) in enumerate(zip(spec['a_bits'], spec['b_bits'])):
            stage_id = f'{identifier}/step-{j:02d}'
            image_path = stage_id+'.png'
            record_path = stage_id+'.json'
            expected_files.update([image_path, record_path])
            saved = json.loads((directory/record_path).read_text())
            require(set(saved) == {'id', 'program_id', 'stage', 'seed', 'input_rgb_sha256',
                                   'image_path', 'image_sha256', 'image_bytes'}, stage_id, 'stage schema')
            require(saved['id'] == stage_id and saved['program_id'] == identifier and saved['stage'] == j,
                    stage_id, 'stable identifiers')
            require(saved['seed'] == stage_seed(config, spec, j), stage_id, 'prospective seed')
            require(saved['image_path'] == image_path, stage_id, 'expected image path')
            if j < len(saved_program['stages']):
                require(saved_program['stages'][j] == saved, stage_id, 'program receipt agrees')
            reference = render(a, b, 0, fonts[0], carry_pixels=raw_carry)
            require(hashlib.sha256(reference.tobytes()).hexdigest() == saved['input_rgb_sha256'],
                    stage_id, 'unmodified generated carry transport and answer-free input')
            path = directory/image_path
            require(path.stat().st_size == saved['image_bytes'] and digest(path) == saved['image_sha256'],
                    stage_id, 'image bytes and SHA-256')
            image = Image.open(path)
            format_valid = image.size == (512, 512) and image.mode == 'RGB'
            require(format_valid, stage_id, 'RGB 512-pixel image')
            image = image.convert('RGB')
            readings = [read_answer(image, bank) for bank in banks]
            reading = min(readings, key=lambda r: r['carry']['mse']+r['digit']['mse'])
            template_valid = format_valid and reading['valid'] and reading['digit']['prediction'] in [0, 1]
            prediction = 2*reading['carry']['prediction']+reading['digit']['prediction'] if template_valid else None
            text, other = ocr_reader(image)
            expected = a+b+expected_carry
            expected_carry = expected//2
            template_correct = prediction == expected
            other_correct = format_valid and other == expected
            stages.append({'id': stage_id, 'expected_subtotal': expected,
                'template_prediction': prediction, 'template_valid': template_valid,
                'ocr_text': text, 'ocr_prediction': other,
                'template_correct': template_correct, 'ocr_correct': other_correct,
                'consensus_correct': template_correct and other_correct})
            raw_carry = image.crop(CARRY_OUT)

        def assemble(values):
            if len(values) != spec['length'] or any(value is None for value in values):
                return None
            return sum((value % 2) << j for j, value in enumerate(values)) + ((values[-1]//2) << len(values))

        expected_sum = spec['left']+spec['right']
        template_answer = assemble([r['template_prediction'] for r in stages])
        other_answer = assemble([r['ocr_prediction'] for r in stages])
        template_correct = template_answer == expected_sum and all(r['template_correct'] for r in stages)
        other_correct = other_answer == expected_sum and all(r['ocr_correct'] for r in stages)
        results.append({'id': identifier, 'kind': spec['kind'], 'length': spec['length'],
            'left': spec['left'], 'right': spec['right'], 'expected_sum': expected_sum,
            'template_answer': template_answer, 'ocr_answer': other_answer,
            'template_correct': template_correct, 'ocr_correct': other_correct,
            'consensus_correct': template_correct and other_correct, 'stages': stages})
        print(json.dumps({'audited_program': identifier, 'completed': len(results), 'total': len(specs)}), flush=True)
    actual = {p.relative_to(directory).as_posix() for p in directory.rglob('*') if p.is_file()}
    require(actual == expected_files, 'inventory', 'exact complete output file set')
    random_metrics = {}
    for length in config['lengths']:
        rows = [r for r in results if r['kind'] == 'random' and r['length'] == length]
        random_metrics[str(length)] = {'count': len(rows),
            **{key: sum(r[key] for r in rows) for key in ['template_correct', 'ocr_correct', 'consensus_correct']}}
    stress = [r for r in results if r['kind'] == 'stress']
    stress_metrics = {'count': len(stress),
        **{key: sum(r[key] for r in stress) for key in ['template_correct', 'ocr_correct', 'consensus_correct']}}
    success = not failures and all(r['count'] == 32 and r['consensus_correct'] >= 29 for r in random_metrics.values())
    success = success and stress_metrics['count'] == 4 and stress_metrics['consensus_correct'] == 4
    return {'integrity_passed': not failures, 'audited_success': success,
            'random': random_metrics, 'stress': stress_metrics, 'programs': results,
            'failures': failures, 'checkpoint_sha256': identity['checkpoint_sha256'],
            'protocol_sha256': protocol_sha, 'claim_boundary': config['claim_boundary']}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--directory', type=Path, required=True)
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--programs', type=Path, required=True)
    parser.add_argument('--fonts', type=Path, nargs=2, required=True)
    parser.add_argument('--parent-report', type=Path, required=True)
    parser.add_argument('--generation-source', type=Path, required=True)
    parser.add_argument('--parent-source-manifest', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    version = subprocess.check_output(['tesseract', '--version'], text=True).splitlines()[0]
    assert version == 'tesseract 5.5.3'
    expected_identity = {'programs_sha256': digest(args.programs), 'script_sha256': digest(args.generation_source),
                         'parent_source_manifest_sha256': digest(args.parent_source_manifest)}
    result = audit(args.directory, json.loads(args.protocol.read_text()), json.loads(args.programs.read_text()),
                   args.fonts, json.loads(args.parent_report.read_text()), digest(args.protocol),
                   expected_identity=expected_identity)
    identity = json.loads((args.directory/'identity.json').read_text())
    assert identity['programs_sha256'] == digest(args.programs)
    result.update(auditor_sha256=digest(Path(__file__)), tesseract_version=version,
                  parent_report_sha256=digest(args.parent_report), programs_sha256=digest(args.programs),
                  reader_sha256=digest(Path(__file__).with_name('audit_binary.py')),
                  common_sha256=digest(Path(__file__).resolve().parents[1]/'mac_sana_column/common.py'),
                  font_sha256=[digest(f) for f in args.fonts])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    assert not args.output.exists()
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True)+'\n')
    print(json.dumps({k: result[k] for k in ['integrity_passed', 'audited_success', 'random', 'stress']}))
    assert result['integrity_passed'], result['failures']
