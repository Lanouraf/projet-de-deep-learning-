import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from fonctions import plot_compare  

# Load the data from the file
losses_vanilla = np.load('valeur loss-models-batch/losses_vanilla.npy')
val_losses_vanilla = np.load('valeur loss-models-batch/val_losses_vanilla.npy')
losses_bn = np.load('valeur loss-models-batch/losses_bn.npy')
val_losses_bn = np.load('valeur loss-models-batch/val_losses_bn.npy')
losses_stockbn = np.load('valeur loss-models-batch/losses_stockbn.npy')
val_losses_stockbn = np.load('valeur loss-models-batch/val_losses_stockbn.npy')
losses_bn2 = np.load('valeur loss-models-batch/losses_bn2.npy')
val_losses_bn2 = np.load('valeur loss-models-batch/val_losses_bn2.npy')
losses_bn3 = np.load('valeur loss-models-batch/losses_bn3.npy')
val_losses_bn3 = np.load('valeur loss-models-batch/val_losses_bn3.npy')
losses_vanilla2 = np.load('valeur loss-models-batch/losses_vanilla2.npy')
val_losses_vanilla2 = np.load('valeur loss-models-batch/val_losses_vanilla2.npy')


all_losses_vanilla = []
all_losses_vanilla.append((losses_vanilla, val_losses_vanilla))

all_losses_bn = []
all_losses_bn.append((losses_bn, val_losses_bn))


all_losses_bn2 = []
all_losses_bn2.append((losses_bn2, val_losses_bn2))


all_losses_bn3 = []
all_losses_bn3.append((losses_bn3, val_losses_bn3))

all_losses_vanilla2 = []
all_losses_vanilla2.append((losses_vanilla2, val_losses_vanilla2))

all_losses_stockbn = []
all_losses_stockbn.append((losses_stockbn, val_losses_stockbn))



plot_compare(all_losses_vanilla, all_losses_bn, legend_a="Vanilla LeNet-5", legend_b="LeNet-5 with BatchNorm", save_to="Vanilla_vs_BatchNorm")

