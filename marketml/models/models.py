from typing import Tuple, Dict, Union

from tensorflow.python.keras.losses import Loss
from tensorflow.python.keras.optimizers import Optimizer

from marketml.models.time import Time2Vector
from marketml.models.transformer import TransformerEncoder
from tensorflow.keras.layers import Concatenate, Dense, Dropout, GlobalAveragePooling1D, Input
from tensorflow.keras.models import Model


class PricePredictor:
    """Predict future stock prices with time series/sequence models."""

    def __init__(self, sequence_shape: Tuple[int, int], ):
        self.sequence_length, self.sequence_width = sequence_shape
        self.model = None

    def build_transformer(self, transformer: Dict, compile: bool = True):
        """Build a transformer model according to provided parameters."""
        time_embedding = Time2Vector(self.sequence_length)

        num_attention_layers = transformer["num_attention_layers"]
        num_attention_heads = transformer["num_attention_heads"]
        attention_key_size, attention_value_size, attention_dense_size = transformer["attention_sizes"]

        output_dense_sizes = transformer["output_dense_sizes"]
        dropout = transformer["dropout"]

        attention_layers = [
            TransformerEncoder(attention_key_size, attention_value_size, num_attention_heads, attention_dense_size)
            for _ in range(num_attention_layers)
        ]

        input_sequence = Input(shape=(self.sequence_length, self.sequence_width))
        x = time_embedding(input_sequence)
        x = Concatenate(axis=-1)([input_sequence, x])

        for idx in range(num_attention_layers):
            x = attention_layers[idx]((x, x, x))

        x = GlobalAveragePooling1D(data_format="channels_first")(x)

        for layer_size in range(len(output_dense_sizes)):
            if dropout is not None:
                x = Dropout(dropout)(x)
            x = Dense(layer_size, activation="relu")(x)

        if dropout is not None:
            x = Dropout(dropout)(x)
        y = Dense(1, activation="linear")(x)

        model = Model(inputs=input_sequence, outputs=y)
        if compile:
            # TODO: Add optimizer, loss params
            model = self.compile_model(model, None, None)
        self.model = model
        return model

    @staticmethod
    def compile_model(model: Model, optimizer, loss):
        optimizer = "adam" if optimizer is None else optimizer
        loss = "mse" if loss is None else loss
        model.compile(loss=loss, optimizer=optimizer, metrics=["mae", "mape"])
        return model

    def build(self):
        pass

    def train(self):
        pass

    def predict(self):
        pass
