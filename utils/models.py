import torch
import torch.nn


__all__ = ["Linear"]


class LinearBlock(torch.nn.Module):
    def __init__(self, hidden_size, dropout=0.5):
        super(LinearBlock, self).__init__()

        fc1 = torch.nn.Linear(hidden_size, hidden_size)
        fc2 = torch.nn.Linear(hidden_size, hidden_size)

        torch.nn.init.kaiming_normal_(fc1.weight)
        torch.nn.init.kaiming_normal_(fc2.weight)

        self.net = torch.nn.Sequential(
            fc1,
            torch.nn.BatchNorm1d(hidden_size),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            fc2,
            torch.nn.BatchNorm1d(hidden_size),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
        )

    def forward(self, batch):
        return self.net(batch) + batch


class Linear(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_lin_block=2, dropout=0.5):
        super(Linear, self).__init__()

        linear_blocks = [LinearBlock(hidden_size, dropout=dropout) for _ in range(num_lin_block)]

        self.net = torch.nn.Sequential(
            torch.nn.Linear(input_size, hidden_size),
            torch.nn.BatchNorm1d(hidden_size),
            *linear_blocks,
            torch.nn.Linear(hidden_size, output_size),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout)
        )

    def forward(self, batch):
        return self.net(batch)
