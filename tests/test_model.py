import pytest
from src.infer import load_best_model

def test_model_loading():
    model = load_best_model()
    assert model is not None