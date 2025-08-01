import os
import sys
import pytest

np = pytest.importorskip('numpy')

# Add src/utils to import path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src', 'utils'))

import model_utils


def test_avg_pool_shape():
    data = np.zeros((2, 6, 4, 4), dtype=np.float32)
    out = model_utils.avg_pool(data, num_split=3)
    assert out.shape == (2, 2, 4, 4)


def test_avg_pool_values():
    data = np.array([[[[1]], [[2]], [[3]], [[4]]]], dtype=np.float32)
    out = model_utils.avg_pool(data, num_split=2)
    expected = np.array([[[[(1 + 2) / 2]], [[(3 + 4) / 2]]]], dtype=np.float32)
    assert np.allclose(out, expected)


def test_max_pool_shape():
    data = np.zeros((2, 6, 4, 4), dtype=np.float32)
    out = model_utils.max_pool(data, num_split=3)
    assert out.shape == (2, 2, 4, 4)


def test_max_pool_values():
    data = np.array([[[[1]], [[5]], [[2]], [[3]]]], dtype=np.float32)
    out = model_utils.max_pool(data, num_split=2)
    expected = np.array([[[[5]], [[3]]]], dtype=np.float32)
    assert np.allclose(out, expected)


def test_randomized_pool_shape():
    data = np.zeros((2, 4, 3, 3), dtype=np.float32)
    weights = np.ones((4, 3, 3), dtype=np.float32)
    out = model_utils.randomized_pool(weights, data, num_split=2)
    assert out.shape == (2, 2, 3, 3)


def test_randomized_pool_values():
    data = np.array([[[[1]], [[2]], [[3]], [[4]]]], dtype=np.float32)
    weights = np.ones((4, 1, 1), dtype=np.float32)
    out = model_utils.randomized_pool(weights, data, num_split=2)
    expected = np.array([[[[1 + 3]], [[2 + 4]]]], dtype=np.float32)
    assert np.allclose(out, expected)

def test_flat_2d():
    data = np.arange(16, dtype=np.float32).reshape(2, 2, 2, 2)
    out = model_utils.flat_2d(data)
    assert out.shape == (2, 8)
    assert np.allclose(out[0], np.arange(8))

def test_reshape_4d():
    data = np.arange(16, dtype=np.float32).reshape(1, 16)
    out = model_utils.reshape_4d(data, (4, 2, 2))
    expected = data.reshape(1, 4, 2, 2)
    assert out.shape == expected.shape
    assert np.allclose(out, expected)
