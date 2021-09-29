import os
from pathlib import Path

import pandas as pd
import pytest

from ludwig.constants import META, TRAINING, VALIDATION, TEST, CHECKSUM
from ludwig.data.cache.manager import CacheManager, alphanum
from ludwig.data.dataset.pandas import PandasDatasetManager

from tests.integration_tests.utils import sequence_feature, category_feature, LocalTestBackend


@pytest.mark.parametrize('use_split', [True, False], ids=['split', 'no_split'])
@pytest.mark.parametrize('use_cache_dir', [True, False], ids=['cache_dir', 'no_cache_dir'])
def test_cache_dataset(use_cache_dir, use_split, tmpdir):
    dataset_manager = PandasDatasetManager(backend=LocalTestBackend())
    cache_dir = os.path.join(tmpdir, 'cache') if use_cache_dir else None
    manager = CacheManager(dataset_manager, cache_dir=cache_dir)

    config = {
        'input_features': [sequence_feature(reduce_output='sum')],
        'output_features': [category_feature(vocab_size=2, reduce_input='sum')],
        'combiner': {'type': 'concat', 'fc_size': 14},
        'preprocessing': {},
    }

    def touch(basename):
        path = os.path.join(tmpdir, f'{basename}.csv')
        Path(path).touch()
        return path
