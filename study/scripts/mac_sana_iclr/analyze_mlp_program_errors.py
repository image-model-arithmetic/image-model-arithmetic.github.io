import argparse
import hashlib
import json
import os
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--study', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    assert args.output.is_absolute() and not args.output.exists()
    os.umask(0o077)
    rows = []
    bindings = []
    for seed in [20260924, 20260925, 20260926]:
        root = args.study/f'arms/latent-mlp-{seed}/completed-evidence'
        audit_path = root/'independent-audit.json'
        audit = json.loads(audit_path.read_text())
        assert audit['integrity_passed']
        path = root/'evaluation/program-records.jsonl'
        raw = path.read_bytes()
        assert len(raw) < 2000000
        records = [json.loads(line) for line in raw.splitlines()]
        assert len(records) == 96 and len({r['id'] for r in records}) == 96
        for p in [path, audit_path]:
            bindings.append({'path': str(p.relative_to(args.study)),
                             'sha256': hashlib.sha256(p.read_bytes()).hexdigest()})
        for length in [4, 8, 16]:
            selected = [r for r in records if r['length'] == length]
            assert len(selected) == 32
            invalid_steps = 0
            wrong_nearest = 0
            nearest_complete = 0
            for record in selected:
                assert record['expected'] == record['left']+record['right']
                assert len(record['stages']) == length
                carry = 0
                all_nearest = True
                for i, step in enumerate(record['stages']):
                    expected = record['a_bits'][i]+record['b_bits'][i]+carry
                    carry = expected//2
                    invalid_steps += not step['valid']
                    wrong_nearest += step['numeric_prediction'] != expected
                    all_nearest &= step['numeric_prediction'] == expected
                nearest_complete += all_nearest
            audited = [r for r in audit['program_records'] if r['length'] == length]
            assert len(audited) == 32
            row = {'seed': seed, 'length': length, 'count': 32,
                   'strict_template_correct': sum(r['correct'] for r in audited),
                   'ocr_correct': sum(r['ocr_correct'] for r in audited),
                   'consensus_correct': sum(r['consensus_correct'] for r in audited),
                   'invalid_template_steps': invalid_steps,
                   'nearest_template_label_wrong_steps': wrong_nearest,
                   'all_nearest_template_labels_correct_programs': nearest_complete}
            assert row['strict_template_correct'] == sum(r['correct'] for r in selected)
            rows.append(row)
    result = {'scope': 'Post hoc program failure diagnostic for all three MLP seeds. Primary validity '
              'and consensus gates are unchanged. Unthresholded nearest-template labels are classifier '
              'outputs, not independently established visual correctness.',
              'rows': rows, 'bindings': bindings,
              'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    with open(args.output, 'x') as stream:
        json.dump(result, stream, indent=2)
        stream.write('\n')


if __name__ == '__main__':
    main()
