"""Rebuild every RGB input and independently read every generated circuit node."""
import argparse
from collections import defaultdict
import hashlib
import json
import os
from pathlib import Path
import stat
import subprocess
import sys

from PIL import Image
from pixel_circuits import INPUT_BOXES, OUTPUT_BOXES, evaluate

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'mac_sana_column'))
from common import answer_templates, digest, read_answer, render
from audit_binary import ocr


def audit(root, fonts, ocr_reader=ocr):
    cfg = json.loads((root / 'protocol.json').read_text())
    circuits = json.loads((root / 'circuits.json').read_text())
    registration = json.loads((root / 'registration.json').read_text())
    errors, records, programs = [], [], []

    def require(predicate, identifier, check):
        if not predicate:
            errors.append({'id': identifier, 'check': check})

    for row in registration['files']:
        p = root / row['path']
        require(p.is_file() and digest(p) == row['sha256'] and p.stat().st_size == row['bytes'],
                row['path'], 'registered source bytes')
    out = root / 'evaluation'
    identity = json.loads((out / 'identity.json').read_text())
    summary = json.loads((out / 'generation-summary.json').read_text())
    expected_identity = {'registration_sha256': digest(root / 'registration.json'),
                         'checkpoint_sha256': cfg['checkpoint_sha256'],
                         'font_sha256': digest(fonts[0])}
    if 'models' in cfg:
        expected_identity['model_checkpoints'] = [m['checkpoint_sha256'] for m in cfg['models']]
    require(identity == expected_identity, 'identity', 'frozen model and source')
    require(summary['complete'] is True and summary['answer_scoring_performed'] is False and
            summary['image_count'] == cfg['image_count'], 'summary', 'complete unscored generation')
    banks = [answer_templates(font) for font in fonts]
    expected_paths = {'identity.json': 'file', 'generation-summary.json': 'file'}
    completed = 0
    for circuit in circuits:
        expected, output_bits = evaluate(circuit)
        folder = out / circuit['id']
        expected_paths[circuit['id']] = 'directory'
        wires = {key: render(value, 0, 0, fonts[0]).crop(INPUT_BOXES[0])
                 for key, value in circuit['leaves'].items()}
        for key, external in circuit.get('externals', {}).items():
            relative = Path(external['file'])
            require(not relative.is_absolute() and '..' not in relative.parts, key, 'relative donor path')
            path = root / relative
            require(digest(path) == external['sha256'], key, 'frozen external donor bytes')
            image = Image.open(path)
            require(image.mode == 'RGB' and image.size == (512, 512), key, 'external donor format')
            patch = image.crop(OUTPUT_BOXES[external['port']])
            if external['transform'] == 'binary-128':
                # Independently implement the fixed quantizer using a lookup table.
                patch = patch.convert('L').point([0] * 128 + [255] * 128).convert('RGB')
            wires[key] = patch
        predicted = {reader: dict(circuit['leaves']) for reader in ['template', 'ocr']}
        details = []
        conditions = {r['id']: r['condition'] for r in circuit.get('groups', [])}
        for entry in circuit['nodes']:
            identifier = circuit['id'] + '/' + entry['id']
            png, receipt = folder / (entry['id'] + '.png'), folder / (entry['id'] + '.json')
            expected_paths[identifier + '.png'] = 'file'
            expected_paths[identifier + '.json'] = 'file'
            if not png.is_file() or not receipt.is_file() or any(w not in wires for w in entry['inputs']):
                require(False, identifier, 'all committed inputs and outputs exist')
                completed += 1
                continue
            reference = render(0, 0, 0, fonts[0])
            hashes = []
            for box, wire in zip(INPUT_BOXES, entry['inputs']):
                patch = wires[wire]
                reference.paste(patch, box[:2])
                hashes.append(hashlib.sha256(patch.tobytes()).hexdigest())
            record = json.loads(receipt.read_text())
            wanted = {'id': entry['id'], 'seed': entry.get('noise_seed', cfg['noise_seed'] + completed),
                'inputs': entry['inputs'], 'wire_rgb_sha256': hashes,
                'input_rgb_sha256': hashlib.sha256(reference.tobytes()).hexdigest(),
                'image_sha256': digest(png), 'image_bytes': png.stat().st_size}
            if 'models' in cfg:
                wanted['model_index'] = entry.get('model_index', 0)
            require(record == wanted, identifier, 'exact input, wire, seed, output commit')
            image = Image.open(png)
            require(image.mode == 'RGB' and image.size == (512, 512), identifier, 'RGB image format')
            readings = [read_answer(image, bank) for bank in banks]
            reading = min(readings, key=lambda r: r['carry']['mse'] + r['digit']['mse'])
            valid = reading['valid'] and reading['digit']['prediction'] in [0, 1]
            template = [reading['carry']['prediction'], reading['digit']['prediction']] if valid else None
            text, number = ocr_reader(image)
            other = [number // 2, number % 2] if number is not None else None
            detail = {'id': identifier, 'expected': expected[entry['id']],
                'condition': conditions.get(entry['id'], 'donor' if circuit['kind'] == 'ports' else 'multiplication'),
                'template_prediction': template, 'ocr_prediction': other, 'ocr_text': text,
                'template_correct': template == expected[entry['id']],
                'ocr_correct': other == expected[entry['id']],
                'consensus_correct': template == other == expected[entry['id']]}
            details.append(detail)
            records.append(detail)
            for index, (port, box) in enumerate(OUTPUT_BOXES.items()):
                wire = entry['id'] + ':' + port
                wires[wire] = image.crop(box)
                predicted['template'][wire] = None if template is None else template[index]
                predicted['ocr'][wire] = None if other is None else other[index]
            completed += 1
        if circuit['kind'] == 'multiplication':
            expected_product = circuit['a'] * circuit['b']
            require(sum(v << i for i, v in enumerate(output_bits)) == expected_product,
                    circuit['id'], 'independent multiplication oracle')
            final = {reader: [values.get(w) for w in circuit['outputs']]
                     for reader, values in predicted.items()}
            programs.append({'id': circuit['id'], 'width': circuit['width'], 'a': circuit['a'], 'b': circuit['b'],
                'expected': expected_product, 'nodes': len(details), 'expected_nodes': len(circuit['nodes']),
                'template_final_correct': final['template'] == output_bits,
                'ocr_final_correct': final['ocr'] == output_bits,
                'consensus_final_correct': final['template'] == final['ocr'] == output_bits,
                'template_all_nodes_correct': len(details) == len(circuit['nodes']) and
                                               all(r['template_correct'] for r in details),
                'ocr_all_nodes_correct': len(details) == len(circuit['nodes']) and
                                          all(r['ocr_correct'] for r in details),
                'consensus_all_nodes_correct': len(details) == len(circuit['nodes']) and
                                                all(r['consensus_correct'] for r in details)})
    actual = {}
    total_bytes = 0
    stack = [(out, 0)]
    while stack:
        folder, depth = stack.pop()
        for entry in os.scandir(folder):
            assert len(actual) < 2 * cfg['image_count'] + 100 and depth <= 1
            p = Path(entry.path)
            info = p.lstat()
            kind = 'directory' if stat.S_ISDIR(info.st_mode) else 'file' if stat.S_ISREG(info.st_mode) else 'other'
            actual[p.relative_to(out).as_posix()] = kind
            if kind == 'directory':
                stack.append((p, depth + 1))
            elif kind == 'file':
                assert info.st_size <= 2_000_000
                total_bytes += info.st_size
                assert total_bytes <= 400_000_000
    require(actual == expected_paths, 'inventory', 'complete exact path and type set')
    require(len(records) == completed == cfg['image_count'], 'count', 'all nodes independently scored')
    groups = defaultdict(list)
    for row in records:
        groups[row['condition']].append(row)
    metrics = {name: {'count': len(rows), **{key: sum(r[key] for r in rows)
               for key in ['template_correct', 'ocr_correct', 'consensus_correct']}}
               for name, rows in groups.items()}
    return {'integrity_passed': not errors, 'failures': errors, 'image_count': len(records),
        'groups': metrics, 'programs': programs, 'records': records,
        'all_products_correct': bool(programs) and all(p['consensus_final_correct'] for p in programs),
        'all_product_nodes_correct': bool(programs) and all(p['consensus_all_nodes_correct'] for p in programs),
        'registration_sha256': digest(root / 'registration.json'), 'checkpoint_sha256': cfg['checkpoint_sha256'],
        'reader_source_sha256': digest(Path(__file__)), 'font_sha256': [digest(p) for p in fonts]}


if __name__ == '__main__':
    os.umask(0o077)
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--fonts', type=Path, nargs=2, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    assert subprocess.check_output(['tesseract', '--version'], text=True).splitlines()[0] == 'tesseract 5.5.3'
    result = audit(args.root, args.fonts)
    with args.output.open('x') as stream:
        json.dump(result, stream, indent=2)
        stream.write('\n')
    print(json.dumps({k: result[k] for k in ['integrity_passed', 'image_count', 'groups',
                                           'all_products_correct', 'all_product_nodes_correct']}))
    assert result['integrity_passed']
