import torch
from torch import nn

class ConvAutoEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = torch.nn.Sequential(
            nn.Conv2d(3, 8, (3, 3), padding=1, groups=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(8, 4, (3, 3), padding=1, groups=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(4, 2, (3, 3), padding=1, groups=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(2, 1, (3, 3), padding=1, groups=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.decoder = torch.nn.Sequential(
            nn.ConvTranspose2d(1, 2, (2, 2), stride=(2, 2), groups=1),
            nn.ReLU(),

            nn.ConvTranspose2d(2, 4, (2, 2), stride=(2, 2), groups=1),
            nn.ReLU(),

            nn.ConvTranspose2d(4, 8, (2, 2), stride=(2, 2), groups=1),
            nn.ReLU(),

            nn.ConvTranspose2d(8, 3, (2, 2), stride=(2, 2), groups=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded

    @classmethod
    def criterion(cls, h, y):
        return torch.mean((h - y) ** 2)

if __name__ == "__main__":
    model = ConvAutoEncoder()

    x = torch.randn((64, 3, 32, 32), dtype=torch.float32)
    h = model(x)
    print("IN     :", x.shape, "=", x.shape[1] * x.shape[2] * x.shape[3])
    print("ENCODED:", h[0].shape, "=", h[0].shape[1] * h[0].shape[2] * h[0].shape[3])
    print("DECODED:", h[1].shape)
