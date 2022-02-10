from A2E5 import DFT
import pytest
import numpy as np


@pytest.fixture
def get_test_data():
    """Creates test vectors and computes the correct solution using numpy fft."""
    test_vectors = np.random.uniform(0,50,size=(4,10))
    test_n = test_vectors.shape[0]
    test_set = []
    for i in range(test_n):
        test_set.append((np.array(test_vectors[i,:],dtype='complex'),np.fft.fft(test_vectors[i,:]).reshape((-1,1))))
    # return [(np.array(test_vectors[0,:],dtype='complex'),np.fft.fft(test_vectors[0,:]).reshape((-1,1)))]
    return test_set

def test_DFT(get_test_data):
    """Test that the DFT implementation is correct by comparing to np.fft.fft"""
    for curr in get_test_data:
        x = curr[0]
        expected = curr[1]
        assert sum(np.isclose(DFT(x), expected))==len(x)