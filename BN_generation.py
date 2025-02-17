"""
Ce script entraîne plusieurs modèles LeNet-5 avec différentes configurations et enregistre les valeurs de perte.
Il utilise l'ensemble de données MNIST de torchvision et torch.utils.data.DataLoader pour le chargement des données.
Les modèles entraînés comprennent LeNet-5 classique, LeNet-5 avec normalisation de lot (Batch Normalization) et LeNet-5 avec BatchNorm de PyTorch et LeNet-5 avec la LayerNorm de Pytorch.
Les valeurs de perte sont enregistrées dans des fichiers numpy séparés pour chaque configuration de modèle.
Ce script peut prendre du temps à s'exécuter (40minutes en moyenne) en raison de l'entraînement de plusieurs modèles et du chargement des données.
Les résultats de l'exécution de ce script ont été sauvegardés sur un Google Drive que l'on utilisera dans la suite de nos applications.
"""


import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from BNfonctions import fit , test
from batch.Batcharchitecture import LeNet, LeNetBN, LeNetStockBN , LeNetLayerNorm
import numpy as np
import os
import numpy as np


# Get dataset
# ---------------------------------------------
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5), (0.5))])

train = datasets.MNIST(root="dataset", train=True, transform=transform, download=True)
valid = datasets.MNIST(root="dataset", train=False, transform=transform, download=True)
test1 = datasets.MNIST(root="dataset", train=False, transform=transform, download=True)

train_dataloader = DataLoader(train, batch_size=256, shuffle=True)
valid_dataloader = DataLoader(valid, batch_size=512, shuffle=False)
test_dataloader = DataLoader(test1, batch_size=512, shuffle=False)


# Train models
# ---------------------------------------------


#  LeNet-5_layerNorm
all_losses_layer = []
all_acc_layer = []

for i in range(3):
    model_layer = LeNetLayerNorm()
    losses_layer, val_losses_layer, acc_layer ,model_layer = fit(model_layer, train_dataloader, valid_dataloader,
                                                          epochs=10, lr=1e-3)
    testloss_layer, testacc_layer = test(model_layer, test_dataloader)
    all_losses_layer .append((losses_layer, val_losses_layer))
    all_losses_layer .append(acc_layer)
    
    # Save best Vanilla LeNet-5 model
best_model_layer = model_layer
torch.save(best_model_layer.state_dict(), 'best_model_layer.pth')

np.save('valeur loss-models-batch\\val_losses_layer.npy', val_losses_layer)
np.save('valeur loss-models-batch\\losses_layer.npy', losses_layer)





# Vanilla LeNet-5
all_losses_vanilla = []
all_acc_vanilla = []

for i in range(3):
    model_vanilla = LeNet()
    losses_vanilla, val_losses_vanilla, acc_vanilla ,model_vanilla = fit(model_vanilla, train_dataloader, valid_dataloader,
                                                          epochs=10, lr=1e-3)
    testloss_vanilla, testacc_vanilla = test(model_vanilla, test_dataloader)
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


# Store validation loss values

np.save('valeur loss-models-batch\\val_losses_bn3.npy', val_losses_bn3)
np.save('valeur loss-models-batch\\val_losses_vanilla2.npy', val_losses_vanilla2)
np.save('valeur loss-models-batch\\val_losses_stockbn.npy', val_losses_stockbn)
np.save('valeur loss-models-batch\\val_losses_bn2.npy', val_losses_bn2)
np.save('valeur loss-models-batch\\val_losses_bn.npy', val_losses_bn)
np.save('valeur loss-models-batch\\val_losses_vanilla.npy', val_losses_vanilla)

#store training losses values
np.save('valeur loss-models-batch\\losses_bn3.npy', losses_bn3)
np.save('valeur loss-models-batch\\losses_vanilla2.npy', losses_vanilla2)
np.save('valeur loss-models-batch\\losses_stockbn.npy', losses_stockbn)
np.save('valeur loss-models-batch\\losses_bn2.npy', losses_bn2)
np.save('valeur loss-models-batch\\losses_bn.npy', losses_bn)
np.save('valeur loss-models-batch\\losses_vanilla.npy', losses_vanilla)



