"""
Ce script récupère les modèles pré-entraînés depuis Google Drive, les teste et stocke leur précision pour référence future.

Le script effectue les étapes suivantes :
1. Importe les bibliothèques et modules nécessaires.
2. Récupère les données de test.
3. Vérifie si les fichiers des modèles pré-entraînés existent. Si ce n'est pas le cas, les télécharge depuis Google Drive.
4. Charge les modèles pré-entraînés.
5. Calcule la précision de chaque modèle sur les données de test.
6. Enregistre les précisions dans un dictionnaire.
7. Enregistre le dictionnaire des précisions dans un fichier.
8. Charge le dictionnaire des précisions depuis le fichier et effectue une vérification.

Note : Le script suppose que les fichiers et répertoires nécessaires sont déjà configurés.
"""

#


import os
from batch.Batcharchitecture import LeNet, LeNetBN, LeNetStockBN, LeNetLayerNorm
from google_drive_downloader import GoogleDriveDownloader as gdd
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from Batchfonctions import test

#on recupère les données de test 
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5), (0.5))])
test1 = datasets.MNIST(root="dataset", train=False, transform=transform, download=True)
test_dataloader = DataLoader(test1, batch_size=512, shuffle=False)


# Vérifier si les fichiers de loss existent déjà
if not os.path.exists('batch/modèles_stockbn_entrainé.pth') or not os.path.exists('batch/modèle_bn_entrainé.pth') or not os.path.exists('batch/modèle_bn2_entrainé.pth') or not os.path.exists('batch/modèle_bn3_entrainé.pth') or not os.path.exists('batch/modèle_layer_entrainé.pth') or not os.path.exists('batch/modèle_vanilla_entrainé.pth') or not os.path.exists('batch/modèle_vanilla2_entrainé.pth')  :

   # Liste des IDs de vos fichiers sur Google Drive
    file_ids =['1x9EpK8hLu4obAFoy14_tN2OdsKMB_-xq', '1_LdtVZNoiwb2BzfqI6yjSAbOxDGMRTny','1I5PMSMOu5x1mp6TVWDj5jBCBXUglDjpz', '1oDAKhAFaZP-KQ3ou3MrcEEw5rQOoY-X-','1hn4Z6Occ9n9n6nv2zRTCdp9_kxMsarTd','1W5VR-mQqIAm-CpdGNJe51fwoUjYIavO4','1O33Zo1ET0xN91GJaNJpnnHItIaK0ka1C']

    # Liste des noms de fichiers de sortie
    output_filenames = ['modèles_stockbn_entrainé.pth', 'modèle_bn_entrainé.pth', 'modèle_bn2_entrainé.pth', 'modèle_bn3_entrainé.pth','modèle_layer_entrainé.pth','modèle_vanilla_entrainé.pth','modèle_vanilla2_entrainé.pth']

    # Boucle pour télécharger chaque fichier de loss depuis Google Drive
    for file_id, output_filename in zip(file_ids, output_filenames):
        url = f'https://drive.google.com/file/d/{file_id}/view?usp=sharing'
        dest_path = './batch/' + output_filename

        gdd.download_file_from_google_drive(file_id=file_id, dest_path=dest_path, overwrite=True, showsize=True)
else:
    print("Les fichiers de modèles pré-entraînés existent déjà.")



#charger les modèles 
modèle_stockbn=LeNetStockBN() 
modèles_stockbn_entrainé=torch.load('batch/modèles_stockbn_entrainé.pth')
modèle_stockbn.load_state_dict(modèles_stockbn_entrainé)


modèle_bn=LeNetBN()
modèle_bn_entrainé=torch.load('batch/modèle_bn_entrainé.pth')
modèle_bn.load_state_dict(modèle_bn_entrainé)

modèle_bn2=LeNetBN()
modèle_bn2_entrainé=torch.load('batch/modèle_bn2_entrainé.pth')
modèle_bn2.load_state_dict(modèle_bn2_entrainé)

modèle_bn3=LeNetBN()
modèle_bn3_entrainé=torch.load('batch/modèle_bn3_entrainé.pth')
modèle_bn3.load_state_dict(modèle_bn3_entrainé)

modèle_layer=LeNetLayerNorm()
modèle_layer_entrainé=torch.load('batch/modèle_layer_entrainé.pth')
modèle_layer.load_state_dict(modèle_layer_entrainé)

modèle_vanilla=LeNet()
modèle_vanilla_entrainé=torch.load('batch/modèle_vanilla_entrainé.pth')
modèle_vanilla.load_state_dict(modèle_vanilla_entrainé)

modèle_vanilla2=LeNet()
modèle_vanilla2_entrainé=torch.load('batch/modèle_vanilla2_entrainé.pth')
modèle_vanilla2.load_state_dict(modèle_vanilla2_entrainé)

accuracies = {}


# Create a dictionary to store accuracies and their associated models
accuracies = {}

# Calculate accuracy for modèle_stockbn
_,accuracy_stockbn = test(modèle_stockbn, test_dataloader)
accuracies['modèle_stockbn'] = accuracy_stockbn

# Calculate accuracy for modèle_bn
_,accuracy_bn = test(modèle_bn, test_dataloader)
accuracies['modèle_bn'] = accuracy_bn

# Calculate accuracy for modèle_bn2
_, accuracy_bn2 = test(modèle_bn2, test_dataloader)
accuracies['modèle_bn2'] = accuracy_bn2

# Calculate accuracy for modèle_bn3
_, accuracy_bn3 = test(modèle_bn3, test_dataloader)
accuracies['modèle_bn3'] = accuracy_bn3

# Calculate accuracy for modèle_layer
_, accuracy_layer = test(modèle_layer, test_dataloader)
accuracies['modèle_layer'] = accuracy_layer

# Calculate accuracy for modèle_vanilla
_, accuracy_vanilla = test(modèle_vanilla, test_dataloader)
accuracies['modèle_vanilla'] = accuracy_vanilla

# Calculate accuracy for modèle_vanilla2
_, accuracy_vanilla2 = test(modèle_vanilla2, test_dataloader)
accuracies['modèle_vanilla2'] = accuracy_vanilla2

# Save the accuracies dictionary
torch.save(accuracies, 'batch/accuracies.pth')


#verifications
# Load the accuracies dictionary
loaded_accuracies = torch.load('batch/accuracies.pth')

# Check if the accuracies dictionary is empty
if not loaded_accuracies:
    print("The accuracies dictionary is empty.")
else:
    print("The accuracies dictionary is not empty.")