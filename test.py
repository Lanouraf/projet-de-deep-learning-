"""
This script trains multiple LeNet-5 models with different configurations and saves the loss values.
It uses the MNIST dataset from torchvision and torch.utils.data.DataLoader for data loading.
The trained models include Vanilla LeNet-5, LeNet-5 with Batch Normalization, and LeNet-5 with PyTorch's BatchNorm.
The loss values are saved in separate numpy files for each model configuration.
"""



# Rest of the code...
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from fonctions import fit
from batch.architecture import LeNet, LeNetBN, LeNetStockBN
import numpy as np
import os
import numpy as np


# Get dataset
# ---------------------------------------------
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5), (0.5))])

train = datasets.MNIST(root = "dataset", train = True, transform = transform, download = True)
valid = datasets.MNIST(root = "dataset", train = False, transform = transform, download = True)

train_dataloader = DataLoader(train, batch_size = 256, shuffle = True)
valid_dataloader = DataLoader(valid, batch_size = 512, shuffle = False)


# List of file names
file_names = ['val_losses_bn3.npy', 'val_losses_vanilla2.npy', 'val_losses_stockbn.npy', 
              'val_losses_bn2.npy', 'val_losses_bn.npy', 'losses_bn3.npy', 
              'losses_vanilla2.npy', 'losses_stockbn.npy', 'losses_bn2.npy', 
              'losses_bn.npy', 'losses_vanilla.npy']


all_losses_bn3 = []
all_losses_vanilla2 = []
all_losses_stockbn = []
all_losses_bn2 = []
all_losses_bn = []
val_losses_bn3 = []
val_losses_vanilla2 = []
val_losses_stockbn = []
val_losses_bn2 = []
val_losses_bn = []
losses_bn3 = []
losses_vanilla2 = []
losses_stockbn = []
losses_bn2 = []
losses_bn = []
losses_vanilla = []

variables = [val_losses_bn3, val_losses_vanilla2, val_losses_stockbn, 
             val_losses_bn2, val_losses_bn, losses_bn3, 
             losses_vanilla2, losses_stockbn, losses_bn2, 
             losses_bn, losses_vanilla]

for i in range(len(file_names)):
    if os.path.isfile(file_names[i]):
        # Load the loss values if the file exists
        variables[i] = np.load(file_names[i])
    else:
        pass
# Train models
# ---------------------------------------------
# Vanilla LeNet-5
all_losses_vanilla = []
all_acc_vanilla = []
for i in range(3):
    model_vanilla = LeNet()
    losses_vanilla, val_losses_vanilla, acc_vanilla = fit(model_vanilla, train_dataloader, valid_dataloader,
                                                          epochs=10, lr=1e-3)
    all_losses_vanilla.append((losses_vanilla, val_losses_vanilla))
    all_acc_vanilla.append(acc_vanilla)

# BatchNorm LeNet-5
all_losses_bn = []
all_acc_bn = []
for i in range(3):
    model_bn = LeNetBN()
    losses_bn, val_losses_bn, acc_bn = fit(model_bn, train_dataloader, valid_dataloader, epochs=10, lr=1e-3)
    all_losses_bn.append((losses_bn, val_losses_bn))
    all_acc_bn.append(acc_bn)

# BatchNorm LeNet-5 with high lr
all_losses_bn2 = []
for i in range(3):
    model_bn2 = LeNetBN()
    losses_bn2, val_losses_bn2, _ = fit(model_bn2, train_dataloader, valid_dataloader, epochs=10, lr=1e-2)
    all_losses_bn2.append((losses_bn2, val_losses_bn2))

# BatchNorm LeNet-5 with even higher lr
all_losses_bn3 = []
for i in range(3):
    model_bn3 = LeNetBN()
    losses_bn3, val_losses_bn3, _ = fit(model_bn3, train_dataloader, valid_dataloader, epochs=10, lr=5e-2)
    all_losses_bn3.append((losses_bn3, val_losses_bn3))

# Vanilla LeNet-5 with even higher lr
all_losses_vanilla2 = []
for i in range(3):
    model_vanilla2 = LeNet()
    losses_vanilla2, val_losses_vanilla2, _ = fit(model_vanilla2, train_dataloader, valid_dataloader, epochs=10, lr=5e-2)
    all_losses_vanilla2.append((losses_vanilla2, val_losses_vanilla2))

# BatchNorm LeNet-5 using Pytorch's BatchNorm
all_losses_stockbn = []
for i in range(3):
    model_stockbn = LeNetStockBN()
    losses_stockbn, val_losses_stockbn, _ = fit(model_stockbn, train_dataloader, valid_dataloader, epochs=10, lr=1e-3)
    all_losses_stockbn.append((losses_stockbn, val_losses_stockbn))





# Store loss values

# Store validation loss values
np.save('val_losses_bn3.npy', val_losses_bn3)
np.save('val_losses_vanilla2.npy', val_losses_vanilla2)
np.save('val_losses_stockbn.npy', val_losses_stockbn)
np.save('val_losses_bn2.npy', val_losses_bn2)
np.save('val_losses_bn.npy', val_losses_bn)

#store training losses values
np.save('losses_bn3.npy', losses_bn3)
np.save('losses_vanilla2.npy', losses_vanilla2)
np.save('losses_stockbn.npy', losses_stockbn)
np.save('losses_bn2.npy', losses_bn2)
np.save('losses_bn.npy', losses_bn)
np.save('losses_vanilla.npy', losses_vanilla)

# Plot results
# ---------------------------------------------
print(all_losses_bn3)
