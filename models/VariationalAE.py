import torch
from torch import nn
from torch.autograd import Variable


class ConvBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv = nn.Conv2d(in_c, out_c, (4, 4), (2, 2), 1)
        self.bn = nn.BatchNorm2d(out_c)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return self.relu(x)

class ConvTransposeBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv_t = nn.ConvTranspose2d(in_c, out_c, (4, 4), (2, 2), (1,))
        self.bn = nn.BatchNorm2d(out_c)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv_t(x)
        x = self.bn(x)
        return self.relu(x)

class AEModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.name = "VariationalAE"
        self.z_size = 64

        self.encoder = torch.nn.Sequential(
            ConvBlock(3, 16),
            ConvBlock(16, 32),
            ConvBlock(32, 64)
        )

        self.feature_size = int(32 / (2 ** 3))
        self.feature_volume = 64 * self.feature_size ** 2

        self.q_mean = nn.Linear(self.feature_volume, self.z_size)
        self.q_logvar = nn.Linear(self.feature_volume, self.z_size)
        self.project = nn.Linear(self.z_size, self.feature_volume)

        self.decoder = nn.Sequential(
            ConvTransposeBlock(64, 32),
            ConvTransposeBlock(32, 16),
            ConvTransposeBlock(16, 3),
            nn.Sigmoid()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        flatten = encoded.view(-1, self.feature_volume)

        mean, logvar = self.q_mean(flatten), self.q_logvar(flatten)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn(std.size(), dtype=torch.float32, requires_grad=False)
        if next(self.parameters()).is_cuda: eps = eps.to("cuda")
        z = eps * std + mean

        feature = self.project(z).view(-1, 64, self.feature_size, self.feature_size)
        x_reconstructed = self.decoder(feature)
        return x_reconstructed, (mean, logvar, z)

    @classmethod
    def reconstruction_loss(cls, x_reconstructed, x):
        return nn.BCELoss()(x_reconstructed, x) * x.size(0)

    @classmethod
    def kl_divergence_loss(cls, mean, logvar):
        return torch.mean((mean ** 2 + logvar.exp() - 1 - logvar) / 2)

if __name__ == "__main__":
    _model = AEModel()

    _x = torch.randn((64, 3, 32, 32), dtype=torch.float32)
    _x_reconstructed, (_mean, _logvar, _z) = _model(_x)
    print("IN     :", _x.shape)
    print("ENCODED:", _z.shape)
    print("DECODED:", _x_reconstructed.shape)
