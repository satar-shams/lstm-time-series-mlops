#from datetime import datetime
import pandas as pd
import numpy as np
import yfinance

class StockLoader:
    def __init__(self, ticker:str, start_date:str, end_date:str, auto_adjust: bool = True):
        self.ticker = ticker
        self.start_date = start_date
        self.end_date = end_date
        self.auto_adjust = auto_adjust


    def fetch(self) -> pd.DataFrame:
        data = yfinance.download(
            tickers = self.ticker, 
            start = self.start_date,
             end = self.end_date, 
             auto_adjust = self.auto_adjust)
        
        if data.empty:
            raise RuntimeError(
                f"yfinance returned no data for {self.ticker}. "
                "This is usually Yahoo-side rate-limiting or blocking, not a real "
                "delisting. Check yfinance version, retry, or check network access."
            )
        
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)

        # dataset = data["Close"]
        # dataset = stock_close.values.reshape(-1, 1)
        return data 
    

# in the case of using manually:
if __name__ == "__main__":
    stock_loader = StockLoader(ticker = "AAPL", 
                               start_date= "2010-01-01",
                               end_date="2026-06-26", auto_adjust= True)
    
    dataset = stock_loader.fetch()
    print(dataset.head())
    print(type(dataset))