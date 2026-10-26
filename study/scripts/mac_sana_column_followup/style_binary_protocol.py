"""Four supervised input styles; one unchanged output convention."""
from pathlib import Path
from common import render, digest
from binary_protocol import training_rows


def style_rows(root, config):
    styles = config['input_style_curriculum']['training_fonts']
    assert len(styles) == 4 and len({r['path'] for r in styles}) == 4
    heldout = config['input_style_curriculum']['heldout_font']
    assert heldout['sha256'] not in {r['sha256'] for r in styles}
    for spec in [*styles, heldout]:
        assert digest(root/spec['path']) == spec['sha256'], spec['path']
    rows = training_rows()
    for row in rows:
        spec = styles[row['variant']]
        row.update(input_font=spec['path'], input_font_sha256=spec['sha256'])
    return rows


def render_pair(root, row):
    reference = render(row['a'], row['b'], row['c'], root/row['input_font'],
                       row['variant'], row['copy'])
    canonical = render(row['a'], row['b'], row['c'], root/'fonts/train.ttf',
                       row['variant'], row['copy'], row['answer'])
    target = reference.copy()
    target.paste(canonical.crop((0, 288, 512, 512)), (0, 288))
    return reference, target
