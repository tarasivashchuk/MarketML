import torch
import torch.nn as nn


class SingleAttention(nn.Module):

    def __init__(self, key_size, value_size, input_shape):
        super(SingleAttention, self).__init__()
        self.key_size = key_size
        self.value_size = value_size
        self.query = nn.Linear(input_shape, key, bias=True)
        self.key = nn.Linear(input_shape, key, bias=True)
        self.value = nn.Linear(input_shape, value, bias=True)

    def forward(self, x):
        q = self.query(x[0])
        k = self.key(x[1])

        attention_weights = torch.matmul(q, k)
        attention_weights = attention_weights / torch.sqrt(self.key_size)
        attention_weights = torch.softmax(attention_weights, dim=-1)

        v = self.value(x[2])
        attention_out = torch.matmul(attention_weights, v)
        return attention_out
