from typing import Tuple, Dict, Union

from tensorflow.python.keras.layers import concatenate
from tensorflow.python.keras.losses import Loss
from tensorflow.python.keras.optimizers import Optimizer

from marketml.models.time import Time2Vector
from marketml.models.transformer import TransformerEncoder
from tensorflow.keras.layers import Concatenate, Dense, Dropout, GlobalAveragePooling1D, Input
from tensorflow.keras.models import Model


class PricePredictor:
    """Predict future stock prices with time series/sequence models."""

    def __init__(self, sequence_shape: Tuple[int, int], transformer: Dict[str:Union[str, int, Optimizer, Loss]]):
        self.sequence_length, self.sequence_width = sequence_shape
        self.transformer = transformer

    def build_transformer(self):
        """Build a transformer model according to provided parameters."""
        time_embedding = Time2Vector(self.sequence_length)

        num_attention_layers = self.transformer["num_attention_layers"]
        num_heads = self.transformer["num_heads"]
        key_width, value_width, ff_width = self.transformer["dense_widths"]

        dense_layers = self.transformer["dense_layers"]
        dropout = self.transformer["dropout"]

        attention_layers = [
            TransformerEncoder(key_width, value_width, num_heads, ff_width)
            for _ in range(num_attention_layers)
        ]

        """Construct model"""
        input_sequence = Input(shape=(self.sequence_length, self.sequence_width))
        x = time_embedding(input_sequence)
        x = Concatenate([input_sequence, x])
        for idx in range(num_attention_layers):
            x = attention_layers[idx]((x, x, x))
        x = GlobalAveragePooling1D(data_format="channels_first")(x)
        for layer_size in range(len(dense_layers)):
            if dropout is not None:
                x = Dropout(dropout)(x)
            x = Dense(layer_size, activation="relu")(x)
        if dropout is not None:
            x = Dropout(dropout)(x)
        y = Dense(1, activation="linear")(x)

        optimizer = self.transformer.get("optimizer")
        loss = self.transformer.get("loss")

        model = Model(inputs=input_sequence, outputs=y)
        return model

    @staticmethod
    def compile_model(model: Model, optimizer: Union[str, Optimizer, None], loss: Union[str, Loss, None]):
        optimizer = "adam" if optimizer is None else optimizer
        loss = "mse" if loss is None else loss
        model.compile(loss=loss, optimizer=optimizer, metrics=["mae", "mape"])
        return model
