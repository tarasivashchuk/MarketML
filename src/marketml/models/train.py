from marketml.models.time import Time2Vector
from marketml.models.transformer import TransformerEncoder
from tensorflow.keras.layers import (Concatenate, Dense, Dropout,
                                     GlobalAveragePooling1D, Input)
from tensorflow.keras.models import Model


def create_transformer_model(sequence_length: int, num_heads: int):
    # TODO: Refactor somewhere else
    """Initialize time and transformer layers"""
    time_embedding = Time2Vector(sequence_length)
    attn_layer1 = TransformerEncoder(128, 128, num_heads, 128)
    attn_layer2 = TransformerEncoder(128, 128, num_heads, 128)
    attn_layer3 = TransformerEncoder(128, 128, num_heads, 128)

    """Construct model"""
    in_seq = Input(shape=(sequence_length, 5))
    x = time_embedding(in_seq)
    x = Concatenate(axis=-1)([in_seq, x])
    x = attn_layer1((x, x, x))
    x = attn_layer2((x, x, x))
    x = attn_layer3((x, x, x))
    x = GlobalAveragePooling1D(data_format="channels_first")(x)
    x = Dropout(0.1)(x)
    x = Dense(64, activation="relu")(x)
    x = Dropout(0.1)(x)
    out = Dense(1, activation="linear")(x)

    model = Model(inputs=in_seq, outputs=out)
    model.compile(loss="mse", optimizer="adam", metrics=["mae", "mape"])
    return model
