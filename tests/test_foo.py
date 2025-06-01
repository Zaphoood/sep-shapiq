from __future__ import annotations

from shapiq_student.foo import Foo


def test_bar():
    foo = Foo()
    assert foo.bar(2, 3) == 5
