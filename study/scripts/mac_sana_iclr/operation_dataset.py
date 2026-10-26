import hashlib
from collections import Counter
from binary_operations import OPERATIONS, TRAIN_OPERATOR_GEOMETRIES, local_target, draw_operator


def rows(config):
    result = []
    for split in ('train', 'development'):
        appearances = [(v, s) for v in range(4) for s in (0, 1, 2)] if split == 'train' else [(4, 1)]
        for font_index, font in enumerate(config['font_grid']['training_fonts']):
            for variant, stroke in appearances:
                for operation in ('copy',)+OPERATIONS:
                    for geometry in ([None] if operation == 'copy' else range(4)):
                        for a in range(2):
                            for b, state in ([(0, 0)] if operation == 'copy' else [(b, c) for b in range(2) for c in range(2)]):
                                carry, digit = (0, a) if operation == 'copy' else local_target(operation, a, b, state)
                                result.append({'id': f'{split}-f{font_index}-{operation}-g{geometry}-{a}-{b}-{state}-v{variant}-s{stroke}',
                                    'a': a, 'b': b, 'c': state, 'copy': operation == 'copy', 'operation': operation,
                                    'operator_geometry': geometry, 'variant': variant, 'stroke': stroke, 'split': split,
                                    'input_font': font['path'], 'input_font_sha256': font['sha256'], 'answer': 10*carry+digit})
    assert len(result) == len({r['id'] for r in result}) == 6864
    assert Counter(r['split'] for r in result) == {'train': 6336, 'development': 528}
    return result


def render_pair(root, row, font_grid_render_pair):
    reference, canonical = font_grid_render_pair(root, row)
    if not row['copy']:
        reference = draw_operator(reference, row['operation'], TRAIN_OPERATOR_GEOMETRIES[row['operator_geometry']])
    target = reference.copy()
    target.paste(canonical.crop((0, 288, 512, 512)), (0, 288))
    return reference, target


def audit(root, config, font_grid_render_pair):
    manifest = []
    banks = {'train': set(), 'development': set()}
    counts = Counter()
    for row in rows(config):
        reference, target = render_pair(root, row, font_grid_render_pair)
        assert reference.crop((0, 288, 512, 512)).getextrema() == ((255, 255),)*3
        assert reference.crop((0, 0, 512, 288)).tobytes() == target.crop((0, 0, 512, 288)).tobytes()
        sha = hashlib.sha256(reference.tobytes()).hexdigest()
        assert sha not in banks[row['split']]
        banks[row['split']].add(sha)
        counts[(row['split'], row['operation'])] += 1
        manifest.append({**row, 'reference_rgb_sha256': sha, 'target_rgb_sha256': hashlib.sha256(target.tobytes()).hexdigest()})
    assert not banks['train'] & banks['development']
    return {'passed': True, 'rows': manifest, 'counts': [
        {'split': split, 'operation': operation, 'count': count} for (split, operation), count in sorted(counts.items())],
        'training_images': len(banks['train']), 'development_images': len(banks['development']),
        'scope': 'Rendered dataset audit only; no model generation or training.'}


def selection_score(records):
    counts = Counter()
    correct = Counter()
    for row in records:
        assert row['operation'] in ('copy',)+OPERATIONS and type(row['correct']) is bool
        counts[row['operation']] += 1
        correct[row['operation']] += row['correct']
    assert counts == {'copy': 16, 'add': 256, 'subtract': 256}
    return min(correct['add'], correct['subtract']), correct['add']+correct['subtract'], correct['copy']
