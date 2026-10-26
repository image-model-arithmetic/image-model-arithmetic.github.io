"""Static full-adder circuits. Only the independent evaluator computes wire values.

Every generated wire is a 96 x 160 RGB crop; fan-out is an unchanged copy.
Circuit schedules depend on input width, never on generated answers.
"""
from itertools import product

INPUT_BOXES = ((80, 48, 176, 208), (208, 48, 304, 208), (336, 48, 432, 208))
OUTPUT_BOXES = {'carry': (144, 320, 240, 480), 'sum': (272, 320, 368, 480)}


def node(identifier, wires):
    assert len(wires) == 3
    return {'id': identifier, 'inputs': list(wires)}


def multiplication(width, a, b):
    """Schoolbook multiplication using full adders and their AND carry output."""
    assert 1 <= width <= 16 and 0 <= a < 2**width and 0 <= b < 2**width
    nodes = []
    leaves = {'zero': 0}
    leaves.update({f'a{i}': (a >> i) & 1 for i in range(width)})
    leaves.update({f'b{i}': (b >> i) & 1 for i in range(width)})
    rows = []
    for j in range(width):
        row = ['zero'] * (2 * width)
        for i in range(width):
            identifier = f'and-{i}-{j}'
            nodes.append(node(identifier, [f'a{i}', f'b{j}', 'zero']))
            row[i + j] = identifier + ':carry'
        rows.append(row)
    accumulator = rows[0]
    for j, row in enumerate(rows[1:], 1):
        following = list(accumulator)
        carry = 'zero'
        # Lower columns cannot change; the final column collects the last carry.
        for k in range(j, width + j):
            identifier = f'add-{j}-{k}'
            nodes.append(node(identifier, [accumulator[k], row[k], carry]))
            following[k] = identifier + ':sum'
            carry = identifier + ':carry'
        following[width + j] = carry
        accumulator = following
    return {'id': f'mul-{width}-{a}-{b}', 'kind': 'multiplication',
            'width': width, 'a': a, 'b': b, 'leaves': leaves,
            'nodes': nodes, 'outputs': accumulator}


def port_probe():
    facts = list(product(range(2), repeat=3))
    nodes = [node(f'donor-{i}', [f'bit{v}' for v in abc]) for i, abc in enumerate(facts)]
    donors = {port: {value: [] for value in [0, 1]} for port in OUTPUT_BOXES}
    for i, abc in enumerate(facts):
        total = sum(abc)
        donors['carry'][total // 2].append(f'donor-{i}:carry')
        donors['sum'][total % 2].append(f'donor-{i}:sum')
    groups = []
    for condition in ['rendered', 'carry-a', 'carry-b', 'carry-c',
                      'sum-a', 'sum-b', 'sum-c', 'carry-all', 'sum-all']:
        for i, abc in enumerate(facts):
            wires = [f'bit{value}' for value in abc]
            if condition != 'rendered':
                port, target = condition.split('-')
                slots = range(3) if target == 'all' else [['a', 'b', 'c'].index(target)]
                for slot in slots:
                    wires[slot] = donors[port][abc[slot]][(i + slot) % 4]
            identifier = f'probe-{condition}-{i}'
            nodes.append(node(identifier, wires))
            groups.append({'id': identifier, 'condition': condition, 'fact': list(abc)})
    return {'id': 'ports', 'kind': 'ports', 'leaves': {'bit0': 0, 'bit1': 1},
            'nodes': nodes, 'outputs': [], 'groups': groups}


def panel():
    return [port_probe(), *[multiplication(2, a, b) for a in range(4) for b in range(4)]]


def validate(circuit):
    known = set(circuit['leaves'])
    assert known and all(type(v) is int and v in [0, 1] for v in circuit['leaves'].values())
    ids = set()
    for entry in circuit['nodes']:
        identifier = entry['id']
        assert identifier not in ids and ':' not in identifier
        assert len(entry['inputs']) == 3 and all(w in known for w in entry['inputs'])
        ids.add(identifier)
        known.update(identifier + ':' + p for p in OUTPUT_BOXES)
    assert all(w in known for w in circuit['outputs'])
    assert set(circuit.get('externals', {})) <= set(circuit['leaves'])
    for external in circuit.get('externals', {}).values():
        assert external['port'] in OUTPUT_BOXES
        assert external['transform'] in ['raw', 'binary-128']
    return circuit


def evaluate(circuit):
    """Integer oracle for tests/auditing only. Never called by generation."""
    validate(circuit)
    values = dict(circuit['leaves'])
    expected = {}
    for entry in circuit['nodes']:
        total = sum(values[w] for w in entry['inputs'])
        expected[entry['id']] = [total // 2, total % 2]
        values[entry['id'] + ':carry'], values[entry['id'] + ':sum'] = expected[entry['id']]
    return expected, [values[w] for w in circuit['outputs']]
