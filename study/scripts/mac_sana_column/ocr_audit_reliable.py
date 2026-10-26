import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import re
import resource
import subprocess

from PIL import Image


def read_panel(path, execute=subprocess.run):
    image = Image.open(path).convert('RGB').crop((128, 300, 384, 488))
    data = io.BytesIO()
    image.save(data, format='PNG')
    command = ['tesseract', 'stdin', 'stdout', '--psm', '7', '-c', 'tessedit_char_whitelist=0123456789']
    try:
        process = execute(command, input=data.getvalue(), capture_output=True, timeout=20)
        stdout, stderr, code = process.stdout, process.stderr, process.returncode
        failure = None if code == 0 else {'kind': 'process-exit', 'returncode': code}
    except subprocess.TimeoutExpired as error:
        stdout, stderr = error.stdout or b'', error.stderr or b''
        failure = {'kind': 'timeout', 'seconds': 20}
    text = re.sub(r'\s+', '', stdout.decode('utf-8', errors='replace'))
    valid = failure is None and bool(re.fullmatch(r'\d{2}', text))
    return text, valid, int(text) if valid else None, failure, stdout, stderr


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('directory', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    assert not args.output.exists()
    records = json.loads((args.directory / 'summary.json').read_text())['records']
    assert len(records) in [40, 80] and len({r['id'] for r in records}) == len(records)
    results = []
    for index, row in enumerate(records):
        path = args.directory / (row['id'] + '.png')
        text, valid, prediction, failure, stdout, stderr = read_panel(path)
        result = {'id': row['id'], 'copy': row['copy'], 'expected': row['expected'],
                  'template_correct': row['correct'], 'ocr_text': text, 'ocr_valid': valid,
                  'ocr_prediction': prediction, 'ocr_correct': valid and prediction == row['expected'],
                  'consensus_correct': row['correct'] and valid and prediction == row['expected'],
                  'image_sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
        if failure is not None:
            folder = args.output.with_suffix('.reader-failures')
            folder.mkdir(mode=0o700, exist_ok=True)
            for name, raw in [('stdout', stdout), ('stderr', stderr)]:
                with open(folder / f'{index:03d}.{name}', 'xb', opener=lambda p, f: os.open(p, f, 0o600)) as stream:
                    stream.write(raw)
            result['reader_failure'] = {**failure, 'stderr_sha256': hashlib.sha256(stderr).hexdigest(),
                                        'stderr_bytes': len(stderr), 'stdout_bytes': len(stdout)}
        results.append(result)
    metrics = {}
    for name, rows in [('copy', [r for r in results if r['copy']]), ('add', [r for r in results if not r['copy']])]:
        metrics[name] = {'count': len(rows), 'template_correct': sum(r['template_correct'] for r in rows),
                         'ocr_correct': sum(r['ocr_correct'] for r in rows),
                         'consensus_correct': sum(r['consensus_correct'] for r in rows)}
    value = {'tesseract_version': subprocess.check_output(['tesseract', '--version'], text=True).splitlines()[0],
             'command': 'tesseract stdin stdout --psm 7 -c tessedit_char_whitelist=0123456789',
             'crop': [128, 300, 384, 488], 'parser': 'exactly two decimal digits after removing whitespace',
             'metrics': metrics, 'records': results,
             'reader_failure_policy': 'Count process failures and timeouts as incorrect and invalid; retain every case and raw error stream.'}
    with open(args.output, 'x', opener=lambda p, f: os.open(p, f, 0o600)) as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write('\n')
    print(json.dumps({'metrics': metrics, 'reader_failures': sum('reader_failure' in r for r in results)}))


if __name__ == '__main__':
    main()
