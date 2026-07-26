import pytest
import pandas as pd
import numpy as np

from src.data.preprocessor import TimeSeriesPreprocessor


@pytest.fixture
def sample_raw_data() -> np.ndarray:
    sample_data = np.arange(40).reshape(-1, 1)
    return sample_data

def test_run_all_three_way_split(sample_raw_data):
    sample_df = pd.DataFrame(
        sample_raw_data,
        columns=["Close"],
    )
    preprocessor = TimeSeriesPreprocessor(
        windows_size=10,
        train_split_size=0.8,
        validation_split_size=0.9,
    )
    data =  preprocessor.run_all(sample_df)
    assert preprocessor.training_end_index == 32
    assert preprocessor.validation_end_index == 36

    assert data["X_train"].shape == (22, 10, 1)
    assert data["X_val"].shape == (4, 10, 1)
    assert data["X_test"].shape == (4, 10, 1)

    assert len(data["y_train"]) == 22
    assert len(data["y_val"]) == 4
    assert len(data["y_test"]) == 4

    np.testing.assert_array_equal(
        data["y_val_real"].values, np.arange(32, 36),
    )

    np.testing.assert_array_equal(
        data["y_test_real"].values, np.arange(36, 40),
    )

def test_create_window_middle(sample_raw_data):
    preprocessor = TimeSeriesPreprocessor(30)
    
    X, y = preprocessor.create_windows(sample_raw_data)

    x5_window_function = X[5].flatten()

    x5_window_real = np.array([
        5, 6, 7, 8, 9, 10,
        11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
        21, 22, 23, 24, 25, 26, 27, 28, 29, 30,
        31, 32, 33, 34
    ])

    assert np.array_equal(x5_window_function, x5_window_real)
    assert y[5].item() == 35

def test_create_window_first(sample_raw_data):
    preprocessor = TimeSeriesPreprocessor(30)

    X, y = preprocessor.create_windows(sample_raw_data)

    first_window_function = X[0].flatten()

    first_window_real = np.array([
        0 ,1, 2 , 3, 4, 5, 6, 7, 8, 9, 10,
        11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
        21, 22, 23, 24, 25, 26, 27, 28, 29
    ])

    assert np.array_equal(first_window_function, first_window_real)
    assert y[0].item() == 30

def test_create_window_last(sample_raw_data):
    preprocessor = TimeSeriesPreprocessor(30)

    X, y = preprocessor.create_windows(sample_raw_data)

    last_window_function = X[-1].flatten()

    last_window_real = np.array([
        9 , 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 
        20, 21, 22, 23, 24, 25, 26, 27, 28, 29,
        30, 31, 32, 33, 34, 35, 36, 37, 38
    ])

    assert np.array_equal(last_window_function, last_window_real)
    assert y[-1].item() == 39

@pytest.fixture
def sample_df() -> pd.DataFrame:
    df = pd.DataFrame(np.arange(40), columns=["Close"])
    return df


def test_scaler(sample_df):
    preprocessor = TimeSeriesPreprocessor()
    preprocessor.fit_transform(sample_df)
    stock_close = sample_df["Close"]
    data = stock_close.values
    
    transformed_data =  preprocessor.scaler.transform(data[[-2]].reshape(-1, 1))

    inversed_transformed_data = preprocessor.scaler.inverse_transform(transformed_data)
    real_data = data[-2]

    assert inversed_transformed_data  == pytest.approx(real_data, abs=1e-6)

def test_scaler_array(sample_df):
    preprocessor = TimeSeriesPreprocessor()
    preprocessor.fit_transform(sample_df)
    stock_close = sample_df["Close"]
    data = stock_close.values
    real_data = data[-9:].reshape(-1, 1)
    transformed_data =  preprocessor.scaler.transform(real_data)
    inversed_transformed_data = preprocessor.scaler.inverse_transform(transformed_data)
    
    np.testing.assert_allclose(inversed_transformed_data, real_data, atol = 1e-6)
