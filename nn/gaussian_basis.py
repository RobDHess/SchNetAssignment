import torch
import torch.nn as nn


class GaussianBasis(nn.Module):
    def __init__(self, num_basis, d_min, d_max, gamma):
        super().__init__()
        self.num_basis = num_basis
        self.d_min = d_min
        self.d_max = d_max

        self.register_buffer("gamma", torch.Tensor([gamma]), persistent=False)
        self.register_buffer(
            "centers", torch.linspace(d_min, d_max, num_basis), persistent=False
        )

    def forward(self, dist):
        mha = dist - self.centers.unsqueeze(0)
        out = torch.exp(-self.gamma * torch.pow(mha, 2))
        return out

    def __repr__(self):
        return "{}(num_basis={}, d_min={}, d_max={}, gamma={})".format(
            self.__class__.__name__,
            self.num_basis,
            self.d_min,
            self.d_max,
            self.gamma.item(),
        )
