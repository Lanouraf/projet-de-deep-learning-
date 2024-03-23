"""
This script trains multiple LeNet-5 models with different configurations and saves the loss values.
It uses the MNIST dataset from torchvision and torch.utils.data.DataLoader for data loading.
The trained models include Vanilla LeNet-5, LeNet-5 with Batch Normalization, and LeNet-5 with PyTorch's BatchNorm.
The loss values are saved in separate numpy files for each model configuration.
"""

# Rest of the code...
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from fonctions import fit , plot_compare
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


# Train models
# ---------------------------------------------

min_val_loss = float('inf')  # Initialize minimum validation loss as infinity
# Vanilla LeNet-5
all_losses_vanilla = []
all_acc_vanilla = []

for i in range(3):
    model_vanilla = LeNet()
    losses_vanilla, val_losses_vanilla, acc_vanilla ,model_vanilla = fit(model_vanilla, train_dataloader, valid_dataloader,
                                                          epochs=10, lr=1e-3)
    all_losses_vanilla.append((losses_vanilla, val_losses_vanilla))
    all_acc_vanilla.append(acc_vanilla)
     
    

# Save best Vanilla LeNet-5 model
best_model_vanilla = model_vanilla
torch.save(best_model_vanilla.state_dict(), 'best_model_vanilla.pth')

# BatchNorm LeNet-5
all_losses_bn = []
all_acc_bn = []
for i in range(3):
    model_bn = LeNetBN()
    losses_bn, val_losses_bn, acc_bn ,model_bn= fit(model_bn, train_dataloader, valid_dataloader, epochs=10, lr=1e-3)
    all_losses_bn.append((losses_bn, val_losses_bn))
    all_acc_bn.append(acc_bn)
    

        # Save best Vanilla LeNet-5 model
best_model_bn = model_bn
torch.save(best_model_bn.state_dict(), 'best_model_bn.pth')

# BatchNorm LeNet-5 with high lr
all_losses_bn2 = []
for i in range(3):
    model_bn2 = LeNetBN()
    losses_bn2, val_losses_bn2, _,model_bn2 = fit(model_bn2, train_dataloader, valid_dataloader, epochs=10, lr=1e-2)
    all_losses_bn2.append((losses_bn2, val_losses_bn2))
    

        # Save best Vanilla LeNet-5 model
best_model_bn2 = model_bn2
torch.save(best_model_bn2.state_dict(), 'best_model_bn2.pth')

# BatchNorm LeNet-5 with even higher lr
all_losses_bn3 = []
for i in range(3):
    model_bn3 = LeNetBN()
    losses_bn3, val_losses_bn3, _ ,model_bn3= fit(model_bn3, train_dataloader, valid_dataloader, epochs=10, lr=5e-2)
    all_losses_bn3.append((losses_bn3, val_losses_bn3))
    


best_model_bn3 = model_bn3
torch.save(best_model_bn2.state_dict(), 'best_model_bn3.pth')

# Vanilla LeNet-5 with even higher lr
all_losses_vanilla2 = []
for i in range(3):
    model_vanilla2 = LeNet()
    losses_vanilla2, val_losses_vanilla2, _ ,model_vanilla2= fit(model_vanilla2, train_dataloader, valid_dataloader, epochs=10, lr=5e-2)
    all_losses_vanilla2.append((losses_vanilla2, val_losses_vanilla2))
    

        # Save best Vanilla LeNet-5 model
best_model_vanilla2 = model_vanilla2
torch.save(best_model_vanilla2.state_dict(), 'best_model_vanilla2.pth')

# BatchNorm LeNet-5 using Pytorch's BatchNorm
all_losses_stockbn = []
for i in range(3):
    model_stockbn = LeNetStockBN()
    losses_stockbn, val_losses_stockbn, _ ,model_stockbn= fit(model_stockbn, train_dataloader, valid_dataloader, epochs=10, lr=1e-3)
    all_losses_stockbn.append((losses_stockbn, val_losses_stockbn))
    

# Save best Vanilla LeNet-5 model
best_model_stockbn = model_stockbn
torch.save(best_model_stockbn.state_dict(), 'best_model_stockbn.pth')





# Store loss values

np.save('valeur loss-models-batch\\val_losses_bn.npy', val_losses_bn3)
np.save('valeur loss-models-batch\\val_losses_vanilla2.npy', val_losses_vanilla2)
np.save('valeur loss-models-batch\\val_losses_stockbn.npy', val_losses_stockbn)
np.save('valeur loss-models-batch\\val_losses_bn2.npy', val_losses_bn2)
np.save('valeur loss-models-batch\\val_losses_bn.npy', val_losses_bn)

#store training losses values
np.save('valeur loss-models-batch\\losses_bn3.npy', losses_bn3)
np.save('valeur loss-models-batch\\losses_vanilla2.npy', losses_vanilla2)
np.save('valeur loss-models-batch\\losses_stockbn.npy', losses_stockbn)
np.save('valeur loss-models-batch\\losses_bn2.npy', losses_bn2)
np.save('valeur loss-models-batch\\losses_bn.npy', losses_bn)
np.save('valeur loss-models-batch\\losses_vanilla.npy', losses_vanilla)

# Plot results
# ---------------------------------------------

plot_compare(all_losses_vanilla, all_losses_bn, legend_a="Vanilla LeNet-5", legend_b="LeNet-5 with BatchNorm", save_to="batch\\losses_vanilla_bn.png")
print(all_losses_bn3)
