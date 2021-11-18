"""
My own illustration of python mock
"""
import requests
import yfinance as yf
from collections import defaultdict
import unittest
from unittest import mock
import pandas as pd

SYMBOLS = ['AAPL','MSFT']
class equity_stats:
    def __init__(self):
        self.mean = 0
        self.variance = 0

class equity_stats_calculator:
    """
    a class gets equity prices and calculate mean/variance
    """
    def __init__(self):
        self.equity_symbol = []
        self.stock_data = {}
        self.stock_stats = defaultdict(equity_stats)

    def addEquity_symbol(self, symbol):
        self.equity_symbol.append(symbol)

    def load_stock(self, ticker, period = '1y'):
        ticker_yf = yf.Ticker(ticker)
        self.stock_data[ticker] = ticker_yf.history(period)

    def calculate_mean(self):
        for ticker, data in self.stock_data.items():
            self.stock_stats[ticker].mean = data['Close'].mean()
SAMPLE_RUN = 'SAMPLE RUN'
NO_RUN = 'NO_RUN'
run_mode = NO_RUN

if run_mode == SAMPLE_RUN:
    equity_downloader = equity_stats_calculator()
    aapl = SYMBOLS[0]
    equity_downloader.load_stock(aapl)
    stock = equity_downloader.stock_data[aapl]
    equity_downloader.calculate_mean()

class equity_stats_calculator_testor(unittest.TestCase):
    def get_test_df(self):
        return pd.DataFrame()

    @mock.patch('yfinance.Ticker', return_value=mock.MagicMock(history=get_test_df))
    def test_load_stock(self, mock_stock):
        sc = equity_stats_calculator()
        stock = 'AAPL'
        sc.load_stock(stock)
        pd.testing.assert_frame_equal(sc.stock_data[stock], pd.DataFrame())
        mock_stock.assert_called_once_with(stock)

if __name__ == '__main__':
    unittest.main()