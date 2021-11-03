import torch
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.datasets import CIFAR10
from torchvision.transforms import ToTensor
from torchvision.utils import make_grid
import os
import shutil
from models.ConvAutoEncoder import ConvAutoEncoder

dev = "cuda" if torch.cuda.is_available() else "cpu"
print("PyTorch", torch.__version__, "and using", dev)

model_name = "model.pth"
data_root = "./data/cifar10/"
log_dir = "./logs/" + model_name

# Prepare your dataset.
batch_size = 64
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
model = ConvAutoEncoder()
criterion = model.loss_fn
optimizer = optim.Adam(model.parameters(), lr=0.005)

if os.path.exists(model_name):
    # model.load_state_dict(torch.load(model_name))
    os.remove(model_name)
model.to(dev)

begin = 1
n_epochs = 3000

if begin == 1 and os.path.exists(log_dir):
    shutil.rmtree(log_dir)
writer = SummaryWriter(log_dir)

for epoch in range(begin, begin + n_epochs, 1):
    # Training
    model.train()
    train_loss = 0.0
    for batch, data in enumerate(train_loader):
        x = data[0].to(dev)

        _, h = model(x)
        loss = criterion(h, x)
        train_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Validating
    model.eval()
    valid_loss = 0.0
    for batch, data in enumerate(valid_loader):
        x = data[0].to(dev)

        _, h = model(x)
        loss = criterion(h, x)
        valid_loss += loss.item()

    # Saving model and logging training history.
    torch.save(model.state_dict(), model_name)
    train_loss /= len(train_loader)
    valid_loss /= len(valid_loader)
    print("Epoch: %d/%d, Train_Loss: %.6f, Valid_Loss: %.6f" % (
        epoch, n_epochs,
        train_loss, valid_loss
    ))
    writer.add_scalars("loss", {
        "train_loss": train_loss,
        "valid_loss": valid_loss
    }, epoch)

    if epoch == 1 or epoch % 50 == 0:
        x, _ = next(iter(valid_loader))
        x = x.to(dev)
        _, h = model(x)

        writer.add_image("_input_images", make_grid(x), epoch)
        writer.add_image("_reconstructed_images", make_grid(h), epoch)
        if epoch == 1: writer.add_graph(model, x)

import time
time.sleep(10)
