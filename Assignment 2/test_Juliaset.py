from JuliaSet import calc_pure_python
import pytest
# import numpy as np


def test_calc_pure_python():
    assert sum(calc_pure_python(desired_width=1000, max_iterations=300)) == 33219980


@pytest.fixture
def get_test_data():
    return [(1000,300,33219980), (10,100,1298), (50,150,45704), (100,200,233136)]

def test_calc_pure_python2(get_test_data):
    for data in get_test_data:
            desired_width = data[0]
            max_iterations = data[1]
            expected = data[2]
            assert sum(calc_pure_python(desired_width, max_iterations)) == expected