import math
import random
import builtins

float32 = float

class ndarray:
    def __init__(self, data, shape=None):
        if isinstance(data, ndarray):
            data = data.data
            shape = data.shape if shape is None else shape
        if shape is None:
            shape = _infer_shape(data)
        self.data = data
        self.shape = tuple(shape)
        self.ndim = len(self.shape)

    def __repr__(self):
        return f"ndarray({self.data})"

    def __getitem__(self, index):
        return self.data[index]

    def reshape(self, *shape):
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            shape = tuple(shape[0])
        return reshape(self, shape)


def _infer_shape(obj):
    if isinstance(obj, list):
        if not obj:
            return (0,)
        return (len(obj),) + _infer_shape(obj[0])
    else:
        return ()


def _flatten(obj):
    if isinstance(obj, list):
        out = []
        for x in obj:
            out.extend(_flatten(x))
        return out
    else:
        return [obj]


def _reshape(flat, shape):
    if not shape:
        return flat[0]
    size = shape[0]
    sub_shape = shape[1:]
    sub_len = 1
    for s in sub_shape:
        sub_len *= s
    return [
        _reshape(flat[i * sub_len:(i + 1) * sub_len], sub_shape)
        for i in range(size)
    ]


def zeros(shape, dtype=float32):
    def build(s):
        if not s:
            return dtype(0)
        return [build(s[1:]) for _ in range(s[0])]

    data = build(list(shape))
    return ndarray(data, shape)


def ones(shape, dtype=float32):
    def build(s):
        if not s:
            return dtype(1)
        return [build(s[1:]) for _ in range(s[0])]

    data = build(list(shape))
    return ndarray(data, shape)


def array(obj, dtype=float32):
    if isinstance(obj, ndarray):
        return ndarray(obj.data, obj.shape)
    def convert(x):
        if isinstance(x, list):
            return [convert(i) for i in x]
        return dtype(x)
    data = convert(obj)
    return ndarray(data)


def arange(stop, dtype=float32):
    data = [dtype(i) for i in range(stop)]
    return ndarray(data, (stop,))


def reshape(arr, shape):
    if isinstance(arr, ndarray):
        flat = _flatten(arr.data)
    else:
        flat = _flatten(arr)
    shape = list(shape)
    if -1 in shape:
        total = len(flat)
        known = 1
        for s in shape:
            if s != -1:
                known *= s
        idx = shape.index(-1)
        shape[idx] = int(total / known)
    data = _reshape(flat, shape)
    return ndarray(data, tuple(shape))


def _elementwise(op, a, b):
    if isinstance(a, list):
        return [_elementwise(op, x, y) for x, y in zip(a, b)]
    else:
        return op(a, b)


def multiply(a, b):
    a_data = a.data if isinstance(a, ndarray) else a
    b_data = b.data if isinstance(b, ndarray) else b
    if isinstance(a_data, list) and not isinstance(b_data, list):
        return ndarray(_scalar_op(a_data, b_data, lambda x, y: x * y),
                       _infer_shape(a_data))
    if isinstance(b_data, list) and not isinstance(a_data, list):
        return ndarray(_scalar_op(b_data, a_data, lambda x, y: x * y),
                       _infer_shape(b_data))
    if isinstance(a_data, list) and isinstance(b_data, list):
        a_shape = _infer_shape(a_data)
        b_shape = _infer_shape(b_data)
        if len(a_shape) - len(b_shape) == 1 and a_shape[1:] == b_shape:
            b_data = [b_data for _ in range(a_shape[0])]
        return ndarray(_elementwise(lambda x, y: x * y, a_data, b_data),
                       _infer_shape(a_data))
    return a_data * b_data


def _reduce_axis(data, axis, op):
    if axis == 0:
        result = data[0]
        for item in data[1:]:
            result = _elementwise(op, result, item)
        return result
    else:
        return [
            _reduce_axis(sub, axis - 1, op)
            for sub in data
        ]


def sum(arr, axis=None):
    arr_data = arr.data if isinstance(arr, ndarray) else arr
    if axis is None:
        return builtin_sum(_flatten(arr_data))
    result = _reduce_axis(arr_data, axis, lambda x, y: x + y)
    shape = list(arr.shape)
    del shape[axis]
    return ndarray(result, shape)


def mean(arr, axis=None):
    arr_data = arr.data if isinstance(arr, ndarray) else arr
    if axis is None:
        flat = _flatten(arr_data)
        return builtin_sum(flat) / len(flat)
    result = sum(arr, axis)
    return multiply(result, 1.0 / arr.shape[axis])


builtin_max = builtins.max

def max(arr, axis=None):
    arr_data = arr.data if isinstance(arr, ndarray) else arr
    if axis is None:
        return builtin_max(_flatten(arr_data))
    result = _reduce_axis(arr_data, axis, lambda x, y: x if x > y else y)
    shape = list(arr.shape)
    del shape[axis]
    return ndarray(result, shape)


def _scalar_op(arr_data, scalar, op):
    if isinstance(arr_data, list):
        return [_scalar_op(x, scalar, op) for x in arr_data]
    else:
        return op(arr_data, scalar)


def add(a, b):
    a_data = a.data if isinstance(a, ndarray) else a
    b_data = b.data if isinstance(b, ndarray) else b
    return ndarray(_elementwise(lambda x, y: x + y, a_data, b_data),
                   _infer_shape(a_data))


def sqrt(x):
    return math.sqrt(x)


def mod(a, b):
    return a % b


def random_rand(*shape):
    def build(s):
        if not s:
            return random.random()
        return [build(s[1:]) for _ in range(s[0])]
    data = build(list(shape))
    return ndarray(data, shape)

random = type('random', (), {'rand': random_rand})


def allclose(a, b, tol=1e-5):
    a_data = a.data if isinstance(a, ndarray) else a
    b_data = b.data if isinstance(b, ndarray) else b
    def compare(x, y):
        if isinstance(x, list):
            return all(compare(xi, yi) for xi, yi in zip(x, y))
        return abs(x - y) <= tol
    return compare(a_data, b_data)

builtin_sum = builtins.sum
