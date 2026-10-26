from PIL import ImageDraw


OPERATIONS = ('add', 'subtract')
TRAIN_OPERATOR_GEOMETRIES = ((8, 2, 0, 0), (9, 2, 0, 1), (8, 3, 1, 0), (9, 3, -1, 0))
TEST_OPERATOR_GEOMETRIES = ((10, 2, 2, -2), (10, 3, -2, 2), (7, 2, -2, -1), (7, 3, 2, 1))


def local_target(operation, a, b, state):
    assert operation in OPERATIONS
    assert all(type(v) is int and v in (0, 1) for v in (a, b, state))
    value = a+b+state if operation == 'add' else a-b-state
    digit = value % 2
    following = value//2 if operation == 'add' else int(value < 0)
    return following, digit


def draw_operator(reference, operation, geometry):
    assert operation in OPERATIONS and geometry in TRAIN_OPERATOR_GEOMETRIES+TEST_OPERATOR_GEOMETRIES
    result = reference.copy()
    draw = ImageDraw.Draw(result)
    draw.rectangle((0, 0, 511, 47), fill='white')
    radius, width, dx, dy = geometry
    for center in (192, 320):
        draw.rectangle((center-15, 108, center+15, 148), fill='white')
        x, y = center+dx, 128+dy
        draw.line((x-radius, y, x+radius, y), fill='black', width=width)
        if operation == 'add':
            draw.line((x, y-radius, x, y+radius), fill='black', width=width)
    return result


def render_operation(render, operation, a, b, state, font, geometry, state_pixels=None, target=False, output_font=None):
    reference = render(a, b, state, font, carry_pixels=state_pixels)
    reference = draw_operator(reference, operation, geometry)
    if target:
        assert output_font is not None and state_pixels is None
        following, digit = local_target(operation, a, b, state)
        answer = render(0, 0, 0, output_font, answer=10*following+digit)
        reference.paste(answer.crop((0, 288, 512, 512)), (0, 288))
    return reference


def expected_program(operation, left, right, width):
    assert operation in OPERATIONS and type(width) is int and 1 <= width <= 64
    assert type(left) is int and type(right) is int and 0 <= left < 2**width and 0 <= right < 2**width
    state = 0
    digits = []
    stages = []
    for index in range(width):
        state, digit = local_target(operation, (left >> index)&1, (right >> index)&1, state)
        digits.append(digit)
        stages.append((state, digit))
    value = sum(digit << index for index, digit in enumerate(digits))
    value += (state if operation == 'add' else -state)*2**width
    return value, stages
