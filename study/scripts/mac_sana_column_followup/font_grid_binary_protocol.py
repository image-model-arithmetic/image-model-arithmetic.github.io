"""Cross input fonts with position and stroke variations, holding outputs fixed."""
from collections import Counter
import hashlib
from PIL import ImageDraw, ImageFont
from common import digest, render
from binary_protocol import display_label


def training_and_development_rows(root, config):
    fonts = config['font_grid']['training_fonts']
    assert len(fonts) == 8 and len({s['sha256'] for s in fonts}) == 8
    heldout = config['font_grid']['heldout_font']
    assert heldout['sha256'] not in {s['sha256'] for s in fonts}
    for spec in [*fonts, heldout]:
        assert (root/spec['path']).stat().st_size == spec['bytes']
        assert digest(root/spec['path']) == spec['sha256'], spec['path']
    rows = []
    for split in ['train', 'development']:
        for index, font in enumerate(fonts):
            appearances = [(variant, stroke) for variant in range(4) for stroke in [0, 1, 2]] if split == 'train' else [(4, 1)]
            for variant, stroke in appearances:
                for copy in [True, False]:
                    for a in range(2):
                        for b, c in ([(0, 0)] if copy else [(b, c) for b in range(2) for c in range(2)]):
                            rows.append({'id': f'{split}-f{index}-{ "copy" if copy else "add"}-{a}-{b}-{c}-v{variant}-s{stroke}',
                                'a': a, 'b': b, 'c': c, 'copy': copy, 'variant': variant, 'stroke': stroke,
                                'split': split, 'input_font': font['path'], 'input_font_sha256': font['sha256'],
                                'answer': a if copy else display_label(a+b+c)})
    assert len(rows) == len({r['id'] for r in rows}) == 1040
    assert Counter(r['split'] for r in rows) == {'train': 960, 'development': 80}
    return rows


def render_pair(root, row):
    # All changes apply to input glyphs only. Keep the existing header and line renderer.
    font_path = root/row['input_font']
    reference = render(row['a'], row['b'], row['c'], font_path, copy=row['copy'])
    draw = ImageDraw.Draw(reference)
    font = ImageFont.truetype(str(font_path), 104, layout_engine=ImageFont.Layout.BASIC)
    dx, dy = [(0, 0), (-3, 2), (3, -2), (1, 3), (5, -5)][row['variant']]
    for x, value in [(128, row['a']), (256, row['b']), (384, row['c'])]:
        if row['copy'] and x != 128:
            continue
        draw.rectangle((x-48, 48, x+47, 207), fill='white')
        draw.text((x+dx, 128+dy), str(value), font=font, fill='black', anchor='mm',
                  stroke_width=row['stroke'], stroke_fill='black')
    target = reference.copy()
    canonical = render(0, 0, 0, root/'fonts/train.ttf', answer=row['answer'])
    target.paste(canonical.crop((0, 288, 512, 512)), (0, 288))
    return reference, target


def optimization_indices(rows, copy):
    return [i for i, row in enumerate(rows) if row['split'] == 'train' and row['copy'] == copy]


def development_indices(rows):
    return [i for i, row in enumerate(rows) if row['split'] == 'development']


def audit_rasters(root, config):
    rows = training_and_development_rows(root, config)
    digests = {'train': set(), 'development': set()}
    pairs = []
    for row in rows:
        reference, target = render_pair(root, row)
        assert reference.crop((0, 288, 512, 512)).getextrema() == ((255, 255),)*3, row['id']
        assert reference.crop((0, 0, 512, 288)).tobytes() == target.crop((0, 0, 512, 288)).tobytes(), row['id']
        reference_sha = hashlib.sha256(reference.tobytes()).hexdigest()
        digests[row['split']].add(reference_sha)
        pairs.append({'id': row['id'], 'reference': reference_sha,
                      'target': hashlib.sha256(target.tobytes()).hexdigest()})
    assert len(digests['train']) == 960 and len(digests['development']) == 80
    assert not digests['train'] & digests['development']
    return {'passed': True, 'training_images': 960, 'development_images': 80,
            'training_additions': sum(not r['copy'] and r['split'] == 'train' for r in rows),
            'training_copies': sum(r['copy'] and r['split'] == 'train' for r in rows),
            'development_additions': 64, 'development_copies': 16,
            'pairs': pairs, 'claim': 'Appearance split only; all eight arithmetic facts are supervised'}
