"""CLI program for building, training, testing, and using models."""
from typing import Dict, Any, List, Optional

import click
import logzero
import pendulum
from tensorflow.python.keras.models import Model

import marketml
from marketml.data.stocks import StockPreprocessor
from marketml.models.models import PricePredictor


logger = logzero.setup_logger(__file__)


@click.group()
def test():
    pass


def setup_transformer(predictor: PricePredictor, num_attention_layers, num_attention_heads, attention_key_size, attention_value_size, attention_dense_size, output_dense_sizes, dropout):
    params = {
        "num_attention_layers": num_attention_layers,
        "num_attention_heads": num_attention_heads,
        "attention_sizes": (attention_key_size, attention_value_size, attention_dense_size),
        "output_dense_sizes": output_dense_sizes,
        "dropout": dropout,
    }
    return predictor.build_transformer(params)


@test.command("train")
@click.option("--tickers", type=str, required=True)
@click.option("--num_attn", type=int, required=True)
@click.option("--num_heads", type=int, required=True)
@click.option("--key_sz", type=int, required=True)
@click.option("--value_sz", type=int, required=True)
@click.option("--dense_sz", type=int, required=True)
@click.option("--out_dense_szs", type=str, required=True)
@click.option("--dropout", type=float, default=0.2)
@click.option("--batch_sz", type=int, default=8)
@click.option("--epochs", type=int, default=5)
def train(tickers, num_attn: int, num_heads: int, key_sz, value_sz, dense_sz, out_dense_szs, dropout, batch_sz, epochs,) -> Model:
    tickers = tickers.split(" ")
    out_dense_szs = [int(sz) for sz in out_dense_szs.split(" ")]

    logger.debug(tickers)
    logger.debug(num_attn)
    logger.debug(num_heads)
    logger.debug(key_sz)
    logger.debug(value_sz)
    logger.debug(dense_sz)
    logger.debug(out_dense_szs)
    logger.debug(dropout)
    logger.debug(batch_sz)
    logger.debug(epochs)
    data_loader = StockPreprocessor()
    [data_loader(ticker) for ticker in tickers]
    train_data, test_data, validation_data = data_loader.process()
    print(train_data[0].shape)

    model_builder = PricePredictor((train_data[0].shape[1], train_data[0].shape[2]))
    model = setup_transformer(model_builder, num_attn, num_heads, key_sz, value_sz, dense_sz, out_dense_szs, dropout)
    model.fit(train_data[0], train_data[1], batch_size=batch_sz, epochs=epochs, validation_data=(validation_data[0], validation_data[1]), shuffle=False)
    model.save(marketml.project_dir.joinpath(f"models/{pendulum.now()}.hdf5"))

    return model

    # TODO: Add visualization
    # TODO: Add testing evaluation
    # TODO: Add logging
    # TODO: Integrate with Darts as a forecasting model

if __name__ == '__main__':
    # python models/test.py train --num_attn 5 --num_heads 12 --key_sz 128 --value_sz 64 --dense_sz 128 --out_dense_szs "128 64" --tickers "AAPL MSFT AMZN"
    test()