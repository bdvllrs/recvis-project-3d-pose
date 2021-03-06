import torch
import torch.nn

__all__ = ["Linear"]


class LinearBlock(torch.nn.Module):
    def __init__(self, hidden_size, dropout=0.5):
        super(LinearBlock, self).__init__()

        fc1 = torch.nn.Linear(hidden_size, hidden_size)
        fc2 = torch.nn.Linear(hidden_size, hidden_size)

        torch.nn.utils.weight_norm(fc1)
        torch.nn.utils.weight_norm(fc2)

        torch.nn.init.kaiming_normal_(fc1.weight)
        torch.nn.init.kaiming_normal_(fc2.weight)

        self.net = torch.nn.Sequential(
            fc1,
            torch.nn.BatchNorm1d(hidden_size),
            torch.nn.Dropout(dropout),
            torch.nn.ReLU(),
            fc2,
            torch.nn.BatchNorm1d(hidden_size),
            torch.nn.Dropout(dropout),
            torch.nn.ReLU()
        )

    def forward(self, batch):
        return self.net(batch) + batch


class Linear(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_lin_block=2, dropout=0.5):
        super(Linear, self).__init__()

        linear_blocks = [LinearBlock(hidden_size, dropout=dropout) for _ in range(num_lin_block)]

        in_fc = torch.nn.Linear(input_size, hidden_size)
        out_fc = torch.nn.Linear(hidden_size, output_size)

        torch.nn.init.kaiming_normal_(in_fc.weight)
        torch.nn.init.kaiming_normal_(out_fc.weight)

        self.net = torch.nn.Sequential(
            in_fc,
            torch.nn.BatchNorm1d(hidden_size),
            *linear_blocks,
            out_fc,
            # torch.nn.Dropout(dropout),
            # torch.nn.ReLU()
        )

    def forward(self, batch):
        return self.net(batch)
