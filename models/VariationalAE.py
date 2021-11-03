import torch
from torch import nn

class AEModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.name = "VariationalAE"
        self.encoder = torch.nn.Sequential(
            nn.Conv2d(3, 64, (3, 3), padding=1, groups=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(64, 32, (3, 3), padding=1, groups=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(32, 16, (3, 3), padding=1, groups=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )

        self.m = nn.Linear(16 * 4 * 4, 16 * 4 * 4)
        self.s = nn.Linear(16 * 4 * 4, 16 * 4 * 4)

        self.decoder = torch.nn.Sequential(
            nn.ConvTranspose2d(16, 32, (2, 2), stride=(2, 2), groups=1),
            nn.ReLU(),

            nn.ConvTranspose2d(32, 64, (2, 2), stride=(2, 2), groups=1),
            nn.ReLU(),

            nn.ConvTranspose2d(64, 3, (2, 2), stride=(2, 2), groups=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        batch_size = x.size(0)
        x = self.encoder(x).view(batch_size, -1)
        m = self.m(x)
        s = self.s(x)
        e = torch.randn(x.shape)
        encoded = e * s + m
        encoded = encoded.view(batch_size, 16, 4, 4)
        decoded = self.decoder(encoded)
        return encoded, decoded, m, s

    @classmethod
    def criterion(cls, h, y):
        encoded, decoded, m, s = h
        h_loss = torch.mean((h - y) ** 2) * 32 * 32
        v_loss = torch.sum(torch.exp(s) - (1 - s) + torch.square(m), -1)
        return torch.mean(h_loss + v_loss)

if __name__ == "__main__":
    model = AEModel()

    x = torch.randn((64, 3, 32, 32), dtype=torch.float32)
    e, d, m, s = model(x)
    print("IN     :", x.shape, "=", x.shape[1] * x.shape[2] * x.shape[3])
    print("ENCODED:", e.shape, "=", e.shape[1] * e.shape[2] * e.shape[3])
    print("DECODED:", d.shape)
