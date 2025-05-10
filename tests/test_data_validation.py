import pytest
from utils.dataset_utils import data_quality_metrics

def test_data_quality():
    df = pd.DataFrame({'col1': [1, 2, None], 'col2': [0, 0, 0]})
    metrics = data_quality_metrics(df)
    assert metrics['missing_values']['col1'] == 1