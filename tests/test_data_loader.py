import pandas as pd
import pytest
from unittest.mock import patch

from src.data.loader import StockLoader
def test_fetch_success():
    fake_data = pd.DataFrame({
        "Open": [100.0, 101.0],
        "Close": [102.0, 103.0],        
    })

    with patch(
        "src.data.loader.yfinance.download",
        return_value=fake_data,
    ):
        loader =  StockLoader()
        data = loader.fetch()

        assert isinstance(data, pd.DataFrame)
        assert not data.empty
        assert "Close" in data.columns

def test_fetch_empty_data():
    fake_data = pd.DataFrame()

    with patch(
        "src.data.loader.yfinance.download",
        return_value=fake_data,
    ):
        loader =  StockLoader()
        with pytest.raises(
            RuntimeError,
            match = "yfinance returned no data",
        ):
            loader.fetch()

def test_fetch_flattens_multiindex_columns():
    fake_data = pd.DataFrame(
        [[100.0, 102.0], [101.0, 103.0]],
        columns=pd.MultiIndex.from_tuples(
            [
                ("Open", "AAPL"),
                ("Close", "AAPL"),
            ]
        ),
    )

    with patch(
        "src.data.loader.yfinance.download",
        return_value=fake_data,
    ):
        loader = StockLoader()
        data = loader.fetch()

    assert not isinstance(data.columns, pd.MultiIndex)
    assert list(data.columns) == ["Open", "Close"]
    assert "Close" in data.columns

