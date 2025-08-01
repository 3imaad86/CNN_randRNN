import os
import importlib
import pytest

h5py = pytest.importorskip('h5py')

from basic_utils import DataTypes, RunSteps
from main_steps import is_suitable_level_fusion

class Dummy:
    def __init__(self, root, step, model, dtype, split):
        self.features_root = root
        self.proceed_step = step
        self.net_model = model
        self.data_type = dtype
        self.split_no = split


def test_is_suitable_level_fusion_ok(tmp_path):
    params = Dummy(str(tmp_path) + '/', RunSteps.FIX_RECURSIVE_NN, 'alexnet', DataTypes.RGB, 1)
    scores_dir = os.path.join(params.features_root, params.proceed_step, 'svm_confidence_scores')
    os.makedirs(scores_dir)
    file_path = os.path.join(scores_dir, f'{params.net_model}_{params.data_type}_split_{params.split_no}.hdf5')
    with h5py.File(file_path, 'w') as f:
        f.create_dataset('layer1', data=[1])
    assert is_suitable_level_fusion(params)


def test_is_suitable_level_fusion_missing(tmp_path):
    params = Dummy(str(tmp_path) + '/', RunSteps.FIX_RECURSIVE_NN, 'alexnet', DataTypes.RGB, 1)
    assert not is_suitable_level_fusion(params)
