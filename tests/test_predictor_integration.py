from src.inference.predictor import Predictor
import pytest

@pytest.fixture
def sample_input_length() -> list[float]:
    real_data:list[float] = [
                            298.8699951171875, 
                            298.2099914550781, 
                            300.2300109863281, 
                            297.8399963378906, 
                            298.9700012207031, 
                            302.25, 
                            304.989990234375, 
                            308.82000732421875, 
                            308.3299865722656, 
                            310.8500061035156, 
                            312.510009765625, 
                            312.05999755859375, 
                            306.30999755859375, 
                            315.20001220703125, 
                            310.260009765625, 
                            311.2300109863281, 
                            307.3399963378906, 
                            301.5400085449219, 
                            290.54998779296875, 
                            291.5799865722656, 
                            295.6300048828125, 
                            291.1300048828125, 
                            296.4200134277344, 
                            299.239990234375, 
                            295.95001220703125, 
                            298.010009765625, 
                            297.010009765625, 
                            294.29998779296875, 
                            293.0799865722656, 
                            275.1499938964844
                            ]
    return real_data
@pytest.mark.integration
def test_predictor_success(sample_input_length):
    """
    Integration test.

    Requires the MLflow server to be running and a production
    model registered under the 'production' alias.
    """
    predictor = Predictor()
    prediction = predictor.predict(sample_input_length)

    assert isinstance(prediction, float)