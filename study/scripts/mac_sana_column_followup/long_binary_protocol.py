"""Frozen cases for the conditional long binary composition evaluation."""
from collections import Counter
import numpy as np


def programs(config):
    assert config['lengths'] == [32, 64]
    assert config['random_programs_per_length'] == 32
    assert config['stress_programs_per_length'] == ['maximum-plus-one', 'maximum-plus-maximum']
    rng = np.random.default_rng(config['program_seed'])
    rows = []
    for length in config['lengths']:
        seen = set()
        while len(seen) < config['random_programs_per_length']:
            aa = [int(rng.integers(0, 2)) for _ in range(length)]
            bb = [int(rng.integers(0, 2)) for _ in range(length)]
            aa[-1] = bb[-1] = 1
            left = sum(bit << i for i, bit in enumerate(aa))
            right = sum(bit << i for i, bit in enumerate(bb))
            if (left, right) in seen:
                continue
            index = len(seen)
            seen.add((left, right))
            rows.append({'id': f'program-{length}-{index:03d}', 'kind': 'random', 'index': index,
                         'length': length, 'left': left, 'right': right, 'a_bits': aa, 'b_bits': bb})
        maximum = (1 << length)-1
        for index, (name, right) in enumerate([('maximum-plus-one', 1), ('maximum-plus-maximum', maximum)], 100):
            assert (maximum, right) not in seen
            rows.append({'id': f'program-{length}-{name}', 'kind': 'stress', 'index': index,
                         'length': length, 'left': maximum, 'right': right,
                         'a_bits': [1]*length, 'b_bits': [(right >> i) & 1 for i in range(length)]})
    return rows


def stage_seed(config, program, stage):
    assert 0 <= stage < program['length']
    return config['noise_seed'] + program['length']*10000 + program['index']*1000 + stage


def audit_corpus(config):
    rows = programs(config)
    assert len(rows) == len({r['id'] for r in rows}) == 68
    assert len({(r['left'], r['right']) for r in rows}) == 68
    seeds = [stage_seed(config, row, j) for row in rows for j in range(row['length'])]
    assert len(seeds) == len(set(seeds)) == 3264
    summary = {}
    for length in config['lengths']:
        subset = [r for r in rows if r['length'] == length]
        assert Counter(r['kind'] for r in subset) == {'random': 32, 'stress': 2}
        by_kind = {}
        for kind in ['random', 'stress']:
            group = [r for r in subset if r['kind'] == kind]
            ignored = copy_left = copy_right = 0
            carry_counts = Counter()
            for row in group:
                assert len(row['a_bits']) == len(row['b_bits']) == length
                assert sum(v << i for i, v in enumerate(row['a_bits'])) == row['left']
                assert sum(v << i for i, v in enumerate(row['b_bits'])) == row['right']
                expected = row['left']+row['right']
                carry = 0
                for a, b in zip(row['a_bits'], row['b_bits']):
                    carry_counts[carry] += 1
                    carry = (a+b+carry)//2
                prediction = (row['left'] ^ row['right']) | (((row['a_bits'][-1]+row['b_bits'][-1])//2) << length)
                ignored += prediction == expected
                copy_left += row['left'] == expected
                copy_right += row['right'] == expected
            by_kind[kind] = {'count': len(group), 'ignore_carry_correct': ignored,
                'copy_left_correct': copy_left, 'copy_right_correct': copy_right,
                'carry_input_counts': dict(sorted(carry_counts.items()))}
        summary[str(length)] = by_kind
    return {'passed': True, 'program_count': len(rows), 'image_count': len(seeds),
            'unique_stage_seeds': len(set(seeds)), 'by_length': summary,
            'scope': 'Prospective corpus and shortcut audit, not a model result'}
