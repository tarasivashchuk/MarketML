import torch
import torch.nn as nn


class SingleAttention(nn.Module):

    def __init__(self, key_size, value_size, input_shape):
        super(SingleAttention, self).__init__()
        self.key_size = key_size
        self.value_size = value_size
        self.query = nn.Linear(input_shape, self.key_size, bias=True)
        self.key = nn.Linear(input_shape, self.key_size, bias=True)
        self.value = nn.Linear(input_shape, self.value_size, bias=True)

    def forward(self, x):
        q = self.query(x[0])
        k = self.key(x[1])

        attention_weights = torch.matmul(q, k)
        attention_weights = attention_weights / torch.sqrt(self.key_size)
        attention_weights = torch.softmax(attention_weights, dim=-1)

        v = self.value(x[2])
        attention_out = torch.matmul(attention_weights, v)
        return attention_out


class MultiAttention(nn.Module):

    def __init__(self, key_size, value_size, num_heads, input_shape):
        super(MultiAttention, self).__init__()
        self.key_size = key_size
        self.value_size = value_size
        self.num_heads = num_heads
        self.attention_heads = [self.attention_heads.append(SingleAttention(self.key_size, self.value_size, input_shape)) for _ in range(self.num_heads)]
        self.linear = nn.Linear(input_shape, input_shape[0][-1])

    def forward(self, x):
        attention = [self.attention_heads[i](x) for i in range(self.num_heads)]
        concat_attention = torch.cat(attention, dim=-1)
        multi_linear = self.linear(concat_attention)
        return multi_linear
