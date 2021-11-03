import torch
from torch import nn

class AEModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.name = "ConvolutionAE"
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
        self.decoder = torch.nn.Sequential(
            nn.ConvTranspose2d(16, 32, (2, 2), stride=(2, 2), groups=1),
            nn.ReLU(),

            nn.ConvTranspose2d(32, 64, (2, 2), stride=(2, 2), groups=1),
            nn.ReLU(),

            nn.ConvTranspose2d(64, 3, (2, 2), stride=(2, 2), groups=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded

    @classmethod
    def criterion(cls, h, y):
        encoded, decoded = h
        return torch.mean((decoded - y) ** 2) * 32 * 32

if __name__ == "__main__":
    model = AEModel()

    x = torch.randn((64, 3, 32, 32), dtype=torch.float32)
    e, d = model(x)
    print("IN     :", x.shape, "=", x.shape[1] * x.shape[2] * x.shape[3])
    print("ENCODED:", e.shape, "=", e.shape[1] * e.shape[2] * e.shape[3])
    print("DECODED:", d.shape)
