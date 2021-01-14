"""CLI program for building, training, testing, and using models."""

from datetime import datetime as dt
from typing import Any, Dict, List, Optional, Union

import fire
import logzero
from tensorflow.python.keras.models import Model

import marketml
from marketml.data.stocks import StockPreprocessor
from marketml.models.tf_models import PricePredictor
<<<<<<< HEAD:src/marketml/models/test.py

=======
>>>>>>> master:marketml/models/test.py

logger = logzero.setup_logger(__file__)


# @click.group()
# def test():
#     pass


def setup_transformer(
    predictor: PricePredictor,
    num_attention_layers,
    num_attention_heads,
    attention_key_size,
    attention_value_size,
    attention_dense_size,
    output_dense_sizes,
    dropout,
):
    params = {
        "num_attention_layers": num_attention_layers,
        "num_attention_heads": num_attention_heads,
        "attention_sizes": (attention_key_size, attention_value_size, attention_dense_size),
        "output_dense_sizes": output_dense_sizes,
        "dropout": dropout,
    }
    logger.debug(f"Model Parameters: {params}")
    return predictor.build_transformer(params)


# @test.command("train")
# @click.option("--num_attention", type=int, required=True)
# @click.option("--num_heads", type=int, required=True)
# @click.option("--key_size", type=int, required=True)
# @click.option("--value_size", type=int, required=True)
# @click.option("--dense_size", type=int, required=True)
# @click.option("--out_dense_sizes", type=str, required=True)
# @click.option("--dropout", type=float, default=0.2)
# @click.option("--batch_size", type=int, default=8)
# @click.option("--epochs", type=int, default=5)
def train(
    ticker: str = "A",
    num_attention: int = 12,
    num_heads: int = 8,
    key_size: int = 64,
    value_size: int = 64,
    dense_size: int = 64,
    out_dense_sizes: Union[List[int], str] = [128, 64],
    dropout: float = 0.25,
    batch_size: int = 5,
    epochs: int = 5,
) -> Model:
    out_dense_sizes = (
        [int(size) for size in out_dense_sizes.split(" ")]
        if isinstance(out_dense_sizes, str)
        else out_dense_sizes
    )

    logger.info("Setting up dataset...")
    data_loader = StockPreprocessor()
    data_loader(ticker)
    train_data, test_data, validation_data = data_loader.process()
    logger.info("Dataset built!")

    logger.info("Building model...")
    model_builder = PricePredictor((train_data[0].shape[1], train_data[0].shape[2]))
    model = setup_transformer(
        model_builder,
        num_attention,
        num_heads,
        key_size,
        value_size,
        dense_size,
        out_dense_sizes,
        dropout,
    )
    logger.info("Model built!")

    logger.info("Training model...")
    model.fit(
        train_data[0],
        train_data[1],
        batch_size=batch_size,
        epochs=epochs,
        validation_data=(validation_data[0], validation_data[1]),
        shuffle=False,
    )
    logger.info("Model trained!")

    logger.info("Evaluating performance...")
    performance = model.evaluate(
        test_data[0], test_data[1], batch_size=batch_size, return_dict=True
    )
    [logger.info(f"{metric.title()}: {value}") for metric, value in performance.items()]

    logger.info("Exporting model...")
    model_filename = f"{performance['loss']}-{dt.now()}.hdf5"
    model.save(marketml.project_root.joinpath(f"models/{model_filename}"))

    return model

    # TODO: Add visualization
    # TODO: Add testing evaluation
    # TODO: Add logging
    # TODO: Integrate with Darts as a forecasting model


if __name__ == "__main__":
    # python models/test.py train --num_attention 5 --num_heads 12 --key_size 128 --value_size 64 --dense_size 128 --out_dense_sizes "128 64"
    # fire.Fire(name="MarketML Model Tests")
    train()
