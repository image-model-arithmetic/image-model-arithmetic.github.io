"""Replay a multiplication with a scratch checkpoint and raw RGB wires.

This small entrypoint uses the same model loader, renderer, and static compiler
as the experiment. It needs no Mac account, SSH, source bundle, or latent model.
It performs no answer reading. Use audit_pixel_circuits.py for study scoring.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

from pixel_circuits import INPUT_BOXES, OUTPUT_BOXES, multiplication


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--font', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--width', type=int, default=2)
    parser.add_argument('--left', type=int, default=3)
    parser.add_argument('--right', type=int, default=3)
    parser.add_argument('--device', choices=['cpu', 'mps', 'cuda'], default='cpu')
    parser.add_argument('--threads', type=int, choices=range(1, 5), default=1)
    args = parser.parse_args()
    os.umask(0o077)
    checkpoint, font, output = args.checkpoint.resolve(), args.font.resolve(), args.output.absolute()
    assert checkpoint.is_file() and font.is_file()
    assert not os.path.lexists(output) and output.parent.resolve() == output.parent
    circuit = multiplication(args.width, args.left, args.right)
    from pixel_transducer import digest, load_predictor
    import pixel_transducer
    # The repository layout and the self-contained source artifact are supported.
    sibling = Path(__file__).resolve().parent.parent / 'mac_sana_column'
    if sibling.is_dir():
        sys.path.insert(0, str(sibling))
    from common import render
    cfg = {'checkpoint_file': checkpoint.name, 'pixel_model_source_sha256': digest(pixel_transducer.__file__),
           'device': args.device, 'threads': args.threads}
    predict = load_predictor(checkpoint.parent, cfg)
    output.mkdir(mode=0o700)
    manifest = {'checkpoint_sha256': digest(checkpoint), 'font_sha256': digest(font),
                'model_source_sha256': cfg['pixel_model_source_sha256'],
                'entrypoint_sha256': digest(__file__), 'device': args.device, 'threads': args.threads,
                'circuit': circuit, 'nodes': [], 'answer_scoring_performed': False}
    wires = {name: render(value, 0, 0, font).crop(INPUT_BOXES[0])
             for name, value in circuit['leaves'].items()}
    for entry in circuit['nodes']:
        reference = render(0, 0, 0, font)
        for box, name in zip(INPUT_BOXES, entry['inputs']):
            reference.paste(wires[name], box[:2])
        image = predict(reference, 0)
        path = output / (entry['id'] + '.png')
        image.save(path)
        manifest['nodes'].append({'id': entry['id'], 'image_sha256': digest(path),
                                  'input_rgb_sha256': hashlib.sha256(reference.tobytes()).hexdigest()})
        for port, box in OUTPUT_BOXES.items():
            wires[entry['id'] + ':' + port] = image.crop(box)
    manifest['complete'] = len(manifest['nodes']) == len(circuit['nodes'])
    with (output / 'replay.json').open('x') as stream:
        json.dump(manifest, stream, indent=2)
        stream.write('\n')
    print(json.dumps({'complete': manifest['complete'], 'images': len(manifest['nodes']),
                      'answer_scoring_performed': False}))


if __name__ == '__main__':
    main()
