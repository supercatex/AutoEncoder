import torch
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.datasets import CIFAR10
from torchvision.transforms import ToTensor
from torchvision.utils import make_grid
import os
import shutil
from models.VariationalAE import AEModel

dev = "cuda" if torch.cuda.is_available() else "cpu"
print("PyTorch", torch.__version__, "and using", dev)

# Args
begin = 1
n_epochs = 3000
lr = 1e-4
batch_size = 128
data_root = "./data/cifar10/"

# Prepare your dataset.
print("Training ", end="")
train_data = CIFAR10(
    root=data_root, train=True, download=True,
    transform=ToTensor()
)
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)

print("Validating ", end="")
valid_data = CIFAR10(
    root=data_root, train=False, download=True,
    transform=ToTensor()
)
valid_loader = DataLoader(valid_data, batch_size=batch_size, shuffle=True)

# Prepare your model.
model = AEModel()
if os.path.exists(model.name + ".pth"):
    if begin == 1:
        os.remove(model.name + ".pth")
    else:
        model.load_state_dict(torch.load(model.name + ".pth"))
model.to(dev)
optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)

# Logging history.
log_dir = "./logs/" + model.name
if begin == 1 and os.path.exists(log_dir):
    shutil.rmtree(log_dir)
writer = SummaryWriter(log_dir)

# Prepare your training epochs.
for epoch in range(begin, begin + n_epochs, 1):
    # Training
    model.train()
    train_loss = 0.0
    for batch, data in enumerate(train_loader):
        x, _ = data
        x = x.to(dev)

        optimizer.zero_grad()
        x_hat, (mean, logvar, z) = model(x)

        reconstruction_loss = model.reconstruction_loss(x_hat, x)
        kl_divergence_loss = model.kl_divergence_loss(mean, logvar)
        loss = reconstruction_loss + kl_divergence_loss

        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    # Validating
    model.eval()
    valid_loss = 0.0
    for batch, data in enumerate(valid_loader):
        x, _ = data
        x = x.to(dev)

        x_hat, (mean, logvar, z) = model(x)

        reconstruction_loss = model.reconstruction_loss(x_hat, x)
        kl_divergence_loss = model.kl_divergence_loss(mean, logvar)
        loss = reconstruction_loss + kl_divergence_loss

        valid_loss += loss.item()

    # Saving model and logging training history.
    torch.save(model.state_dict(), model.name + ".pth")
    train_loss /= len(train_data)
    valid_loss /= len(valid_data)
    print("Epoch: %d/%d, Train_Loss: %.6f, Valid_Loss: %.6f" % (
        epoch, n_epochs,
        train_loss, valid_loss
    ))
    writer.add_scalars("loss", {
        "train_loss": train_loss,
        "valid_loss": valid_loss
    }, epoch)

    if epoch == 1 or epoch % 10 == 0:
        x, _ = next(iter(valid_loader))
        x = x.to(dev)
        x_hat, (mean, logvar, z) = model(x)

        writer.add_image("_input_images", make_grid(x), epoch)
        writer.add_image("_reconstructed_images", make_grid(x_hat), epoch)
        if epoch == 1: writer.add_graph(model, x)

import time
time.sleep(10)
