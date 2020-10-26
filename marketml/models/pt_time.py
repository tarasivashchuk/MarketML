"""Time2Vector encoding layer built in Pytorch."""

import torch
import torch.nn as nn


class Time2Vector(nn.Module):
    def __init__(self, vector_length):
        super(Time2Vector, self).__init__()
        self.vector_length = vector_length
        self.linear_weights = nn.Parameter(
            torch.zeros(
                self.vector_length,
            )
        )
        self.linear_bias = nn.Parameter(
            torch.ones(
                self.vector_length,
            )
        )
        self.periodic_weights = nn.Parameter(
            torch.zeros(
                self.vector_length,
            )
        )
        self.periodic_bias = nn.Parameter(
            torch.ones(
                self.vector_length,
            )
        )

    def forward(self, x):
        x = torch.mean(x[:, :, :4], dim=-1)
        linear_time = self.linear_weights * x + self.linear_bias
        linear_time = torch.unsqueeze(linear_time, dim=-1)
        periodic_time = torch.sin(torch.mul(x, self.periodic_weights) + self.periodic_bias)
        periodic_time = torch.unsqueeze(periodic_time, dim=-1)
        return torch.cat([linear_time, periodic_time], dim=-1)
