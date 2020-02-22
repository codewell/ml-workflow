from functools import partial
import operator as op


def assert_relation(r):
    def _assert_relation(a, b, message=None):
        assert r(a, b), f"{a} {_render(_inverse(r))} {b}" + ["", f": {message}"][message is not None]
    return _assert_relation 

def _not_contains(a, b):
    return not(op.contains(a, b))

def _inverse(r):
    return {
        op.is_: op.is_not,
        op.is_not: op.is_,
        op.eq: op.ne,
        op.ne: op.eq,
        op.lt: op.ge,
        op.gt: op.lt,
        op.le: op.gt,
        op.ge: op.lt,
        op.contains: _not_contains,
        _not_contains: op.contains,
    }[r]

def _render(r):
    return {
        op.is_: 'is',
        op.is_not: 'is not',
        op.eq: '==',
        op.ne: '!=',
        op.lt: '<',
        op.gt: '>',
        op.le: '<=',
        op.ge: '>=',
        op.contains: '∋',
        _not_contains: '∌',
    }[r]


class assert_:
    is_ = assert_relation(op.is_)
    is_not = assert_relation(op.is_not)

    eq = assert_relation(op.eq)
    equal = eq

    ne = assert_relation(op.ne)
    not_equal = ne

    le = assert_relation(op.le)
    less_or_equals = le

    ge = assert_relation(op.ge)
    greater_or_equals = ge

    lt = assert_relation(op.lt)
    less_than = lt

    gt = assert_relation(op.gt)
    greater_then = gt

    contains = assert_relation(op.contains)
    not_contains = assert_relation(_not_contains)

    inverse = lambda r: assert_relation(_inverse(r))


def test_eq():
    import pytest

    with pytest.raises(AssertionError):
        assert_.eq(1, 2)

    assert_.eq(1, 1)