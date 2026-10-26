"""Recompute completed evaluation receipts from saved PNGs, including independent OCR.

This program is strictly post-generation. It does not select checkpoints or repair answers.
"""
import argparse
from collections import Counter
import io
import json
from pathlib import Path
import re
import subprocess

import numpy as np
from PIL import Image
from common import answer_templates, atomic_json, digest, read_answer


def ocr(im):
    stream = io.BytesIO()
    im.crop((128, 300, 384, 488)).save(stream, format='PNG')
    result = subprocess.run(
        ['tesseract', 'stdin', 'stdout', '--psm', '7', '-c',
         'tessedit_char_whitelist=0123456789'],
        input=stream.getvalue(), capture_output=True, check=True, timeout=20)
    text = re.sub(r'\s+', '', result.stdout.decode())
    return text, int(text) if re.fullmatch(r'\d{2}', text) else None


def records(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def audit(root, fonts, program_seed):
    reported = json.loads((root / 'summary.json').read_text())
    local = records(root / 'local-records.jsonl')
    programs = records(root / 'program-records.jsonl')
    banks = [answer_templates(f) for f in fonts]
    errors, details, inventory = [], [], []

    def require(condition, identifier, invariant):
        if not condition:
            errors.append({'id': identifier, 'invariant': invariant})

    def read(path, identifier):
        if not path.is_file():
            errors.append({'id': identifier, 'invariant': 'PNG exists'})
            return None, None
        im = Image.open(path).convert('RGB')
        require(im.size == (512, 512), identifier, '512 by 512 RGB image')
        bank_readings = [read_answer(im, bank) for bank in banks]
        result = min(bank_readings, key=lambda r: r['carry']['mse'] + r['digit']['mse'])
        text, number = ocr(im)
        inventory.append({'path': str(path.relative_to(root)), 'bytes': path.stat().st_size,
                          'sha256': digest(path)})
        return result, {'text': text, 'prediction': number}

    def check_readout(saved, fresh, identifier):
        require(saved['prediction'] == fresh['prediction'], identifier, 'prediction recomputation')
        require(saved['valid'] == fresh['valid'], identifier, 'validity recomputation')
        for component in ['carry', 'digit']:
            for key in ['prediction', 'valid', 'mse', 'ink']:
                left, right = saved[component][key], fresh[component][key]
                equal = abs(left - right) < 1e-6 if key in ['mse', 'ink'] else left == right
                require(equal, identifier, f'{component}.{key} recomputation')

    expected_ids = {f'{style}-{a}-{b}-{c}' for style in ['familiar', 'unfamiliar']
                    for a in range(10) for b in range(10) for c in range(2)}
    counts = Counter(r['id'] for r in local)
    require(set(counts) == expected_ids, 'local-inventory', 'exact 400-case identifier set')
    for identifier, count in counts.items():
        require(count == 1, identifier, 'unique local identifier')
    for row in local:
        identifier = row['id']
        require(identifier == f"{row['style']}-{row['a']}-{row['b']}-{row['c']}",
                identifier, 'identifier matches operands and style')
        expected = row['a'] + row['b'] + row['c']
        require(row['expected'] == expected, identifier, 'sum recomputation')
        result, independent = read(root / row['style'] / (identifier + '.png'), identifier)
        if result is None:
            continue
        check_readout(row, result, identifier)
        correct = result['valid'] and result['prediction'] == expected
        require(row['correct'] == correct, identifier, 'exactness recomputation')
        details.append({'id': identifier, 'style': row['style'], 'a': row['a'], 'b': row['b'],
                        'c': row['c'], 'expected': expected, 'valid': result['valid'],
                        'prediction': result['prediction'], 'correct': correct,
                        'ocr': independent, 'ocr_correct': independent['prediction'] == expected,
                        'consensus_correct': correct and independent['prediction'] == expected})
    recomputed_local = {}
    independent_local = {}
    for style in ['familiar', 'unfamiliar']:
        rows = [r for r in details if r['style'] == style]
        by = {(r['a'], r['b'], r['c']): r for r in rows}
        pairs = [(by[(a, b, 0)], by[(a, b, 1)]) for a in range(10) for b in range(10)
                 if (a, b, 0) in by and (a, b, 1) in by]
        correct = sum(r['correct'] for r in rows)
        recomputed_local[style] = {
            'correct': correct, 'count': len(rows), 'accuracy': correct / len(rows) if rows else 0,
            'valid': sum(r['valid'] for r in rows),
            'carry_pairs_joint_exact': sum(a['correct'] and b['correct'] for a, b in pairs),
            'carry_pairs': len(pairs)}
        independent_local[style] = {
            'ocr_correct': sum(r['ocr_correct'] for r in rows),
            'consensus_correct': sum(r['consensus_correct'] for r in rows), 'count': len(rows),
            'valid_carry_edits_change_answer': sum(a['valid'] and b['valid'] and
                                                   a['prediction'] != b['prediction'] for a, b in pairs),
            'expected_distribution': dict(sorted(Counter(r['expected'] for r in rows).items())),
            'valid_prediction_distribution': dict(sorted(Counter(r['prediction'] for r in rows
                                                                 if r['valid']).items()))}
    require(recomputed_local == reported['local'], 'local-summary', 'all reported local metrics')

    program_details = []
    expected_program_ids = {f'program-{length}-{k:03d}' for length in [2, 4, 8] for k in range(32)}
    rng = np.random.default_rng(program_seed)
    prospective_operands = {}
    for length in [2, 4, 8]:
        for k in range(32):
            aa = [int(rng.integers(0, 10)) for _ in range(length)]
            bb = [int(rng.integers(0, 10)) for _ in range(length)]
            aa[-1], bb[-1] = int(rng.integers(1, 10)), int(rng.integers(1, 10))
            prospective_operands[f'program-{length}-{k:03d}'] = (
                int(''.join(map(str, reversed(aa)))), int(''.join(map(str, reversed(bb)))))
    counts = Counter(r['id'] for r in programs)
    require(set(counts) == expected_program_ids, 'program-inventory', 'exact 96-program identifier set')
    for identifier, count in counts.items():
        require(count == 1, identifier, 'unique program identifier')
    for row in programs:
        identifier, length = row['id'], row['length']
        left, right = row['left'], row['right']
        require(prospective_operands.get(identifier) == (left, right), identifier,
                'operands match prospective program seed')
        require(len(str(left)) == len(str(right)) == length, identifier, 'operand lengths')
        require(row['expected'] == left + right, identifier, 'full sum recomputation')
        require(len(row['stages']) == length, identifier, 'stage count')
        results, independent = [], []
        all_stages_correct = True
        all_ocr_stages_correct = True
        expected_carry = 0
        for j in range(length):
            stage_id = f'{identifier}/step-{j:02d}'
            fresh, other = read(root / identifier / f'step-{j:02d}.png', stage_id)
            if fresh is None:
                all_stages_correct = all_ocr_stages_correct = False
                continue
            check_readout(row['stages'][j], fresh, stage_id)
            expected = (left // 10**j) % 10 + (right // 10**j) % 10 + expected_carry
            all_stages_correct &= fresh['valid'] and fresh['prediction'] == expected
            all_ocr_stages_correct &= other['prediction'] == expected
            expected_carry = expected // 10
            results.append(fresh)
            independent.append({'id': stage_id, 'expected': expected, **other})
        if len(results) != length:
            continue
        valid = all(r['valid'] for r in results)
        prediction = int(str(results[-1]['carry']['prediction']) +
                         ''.join(str(r['digit']['prediction']) for r in reversed(results)))
        final_exact = valid and prediction == left + right
        correct = final_exact and all_stages_correct
        for key, value in [('valid', valid), ('prediction', prediction), ('correct', correct),
                           ('all_stages_correct', all_stages_correct), ('final_answer_correct', final_exact)]:
            require(row[key] == value, identifier, key + ' program recomputation')
        program_details.append({'id': identifier, 'length': length, 'left': left, 'right': right,
                                'correct': correct, 'consensus_correct': correct and all_ocr_stages_correct,
                                'ocr_all_stages_correct': all_ocr_stages_correct, 'ocr_stages': independent})
    recomputed_programs, independent_programs = {}, {}
    for length in [2, 4, 8]:
        rows = [r for r in program_details if r['length'] == length]
        correct = sum(r['correct'] for r in rows)
        recomputed_programs[str(length)] = {
            'correct': correct, 'count': len(rows), 'accuracy': correct / len(rows) if rows else 0}
        independent_programs[str(length)] = {
            'count': len(rows), 'unique_problems': len({(r['left'], r['right']) for r in rows}),
            'ocr_all_stages_correct': sum(r['ocr_all_stages_correct'] for r in rows),
            'consensus_correct': sum(r['consensus_correct'] for r in rows)}
    require(recomputed_programs == reported['programs'], 'program-summary', 'all reported program metrics')
    passed = (recomputed_local['familiar']['accuracy'] >= .95 and
              recomputed_local['unfamiliar']['accuracy'] >= .8 and
              all(v['accuracy'] >= .9 for v in recomputed_programs.values()))
    require(passed == reported['passed'], 'summary', 'all prospective numerical gates')
    consensus_passed = (
        independent_local['familiar']['count'] > 0 and
        independent_local['unfamiliar']['count'] > 0 and
        independent_local['familiar']['consensus_correct']/independent_local['familiar']['count'] >= .95 and
        independent_local['unfamiliar']['consensus_correct']/independent_local['unfamiliar']['count'] >= .8 and
        all(value['count'] > 0 and value['consensus_correct']/value['count'] >= .9
            for value in independent_programs.values()))
    actual_pngs = {str(p.relative_to(root)) for p in root.rglob('*.png')}
    require(actual_pngs == {r['path'] for r in inventory}, 'image-inventory', 'no unaccounted PNGs')
    return {'integrity_passed': not errors, 'primary_gates_passed': passed and not errors,
            'consensus_gates_passed': consensus_passed and not errors,
            'audited_success': passed and consensus_passed and not errors,
            'failures': errors, 'local': recomputed_local, 'programs': recomputed_programs,
            'independent_local': independent_local, 'independent_programs': independent_programs,
            'tesseract_version': subprocess.check_output(['tesseract', '--version'], text=True).splitlines()[0],
            'font_inventory': [{'path': str(f), 'sha256': digest(f)} for f in fonts],
            'image_inventory': sorted(inventory, key=lambda r: r['path']),
            'local_records': details, 'program_records': program_details,
            'reported_summary_sha256': digest(root / 'summary.json')}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('directory', type=Path)
    parser.add_argument('--fonts', nargs=2, type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--program-seed', type=int, required=True)
    parser.add_argument('--readout-protocol', type=Path, required=True)
    args = parser.parse_args()
    protocol = json.loads(args.readout_protocol.read_text())
    assert protocol['consensus_thresholds'] == {'familiar': .95, 'unfamiliar': .8, 'each_program_length': .9}
    assert protocol['crop'] == [128, 300, 384, 488]
    version = subprocess.check_output(['tesseract', '--version'], text=True).splitlines()[0]
    assert version == protocol['independent_reader'], (version, protocol['independent_reader'])
    result = audit(args.directory, args.fonts, args.program_seed)
    result['readout_protocol_sha256'] = digest(args.readout_protocol)
    result['auditor_sha256'] = digest(Path(__file__))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(args.output, result)
    print(json.dumps({k: v for k, v in result.items() if k not in
                      ['image_inventory', 'local_records', 'program_records']}))
    raise SystemExit(0 if result['integrity_passed'] else 1)
