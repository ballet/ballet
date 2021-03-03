import pytest

from ballet.util.code import blacken_code, get_source, is_valid_python


def test_is_valid_python():
    code = '1'
    result = is_valid_python(code)
    assert result


def test_is_valid_python_invalid():
    code = 'this is not valid python code'
    result = is_valid_python(code)
    assert not result


def test_blacken_code():
    input = '''\
    x = {  'a':37,'b':42,

    'c':927}
    '''.strip()

    expected = 'x = {"a": 37, "b": 42, "c": 927}'.strip()
    actual = blacken_code(input).strip()

    assert actual == expected


def test_blacken_code_nothing_changed():
    input = '1\n'
    expected = '1\n'
    actual = blacken_code(input)

    assert actual == expected


@pytest.mark.xfail
def test_get_source():
    get_source(None)
