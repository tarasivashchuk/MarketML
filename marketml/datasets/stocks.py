import sqlite3
from typing import List, Optional, Union

import numpy as np
import pandas as pd

import marketml


class StockPreprocessor:
    """Preprocess any number of stock time series data into ready-to-use model input."""

    drop_columns = ["ticker", "dividend_amount", "split_coefficient"]
    replacements = [("volume", 0)]
    price_columns = ["open", "high", "low", "close"]
    value_columns = ["volume"]

    def __init__(
        self,
        db: str = "stocks.sqlite",
        table: str = "prices",
        drop_columns: Optional[List[str]] = None,
    ):
        if drop_columns is not None:
            self.drop_columns = drop_columns

        db = str(marketml.project_root.joinpath(f"datasets/db/{db}"))
        self.db_connection = sqlite3.connect(db)
        self.table = table
        self.data = []

    def __call__(self, ticker):
        df = self.load_data(ticker)
        df = self.drop(df)
        for replacement_col, replacement_val in self.replacements:
            df = self.replace_values(df, replacement_col, replacement_val)
        df = self.calculate_pct_change(df)
        df = self.normalize_columns(df, self.price_columns)
        df = self.normalize_columns(df, self.value_columns)
        self.data.append(df)

    def load_data(self, ticker):
        df = pd.read_sql_query(
            f'SELECT * FROM {self.table} WHERE ticker="{ticker}"',
            self.db_connection,
            index_col=["date"],
            parse_dates=True,
        )
        df.sort_index(inplace=True)
        return df

    def drop(self, df: pd.DataFrame):
        df.drop(columns=self.drop_columns, inplace=True)
        return df

    def replace_values(self, df: pd.DataFrame, column: str, value: Union[int, float] = 0):
        df[column].replace(to_replace=value, method="ffill", inplace=True)
        return df

    def calculate_pct_change(self, df: pd.DataFrame):
        for col in self.price_columns + self.value_columns:
            df[col] = df[col].pct_change()
        df.dropna(how="any", axis=0, inplace=True)
        return df

    def normalize_columns(self, df: pd.DataFrame, columns: List[str]):
        if len(columns) == 1:
            column = columns[0]
            min_val, max_val = df[column].min(axis=0), df[column].max(axis=0)
        else:
            min_val, max_val = min(df[columns].min(axis=0)), max(df[columns].max(axis=0))
        delta = max_val - min_val
        for column in columns:
            df[column] = (df[column] - min_val) / delta
        return df

    def split_data(self, train_size: float, test_size: float):
        self.data: pd.DataFrame
        train_df = self.data.iloc[: int(len(self.data) * train_size)]
        test_df = self.data.iloc[
            int(len(self.data) * train_size) : int(len(self.data) * (train_size + test_size))
        ]
        val_df = self.data.iloc[int(len(self.data) * (train_size + test_size)) :]
        return train_df, test_df, val_df

    @staticmethod
    def chunk_sequences(df: pd.DataFrame, sequence_length: int):
        data = df.values
        x, y = [], []
        for i in range(sequence_length, len(data)):
            x.append(data[i - sequence_length : i])
            y.append(data[:, 3][i])
        return np.array(x), np.array(y)

    def process(
        self,
        split: bool = True,
        train_size: float = 0.8,
        test_size: float = 0.1,
        sequence_length: int = 30,
    ):
        if train_size + test_size >= 1.0:
            raise ValueError("Train and test size sum must be less than 1")
        self.data = pd.concat(self.data)
        if not split:
            return self.data
        train, test, val = self.split_data(train_size, test_size)
        train, test, val = (
            self.chunk_sequences(train, sequence_length),
            self.chunk_sequences(test, sequence_length),
            self.chunk_sequences(val, sequence_length),
        )
        return train, test, val
