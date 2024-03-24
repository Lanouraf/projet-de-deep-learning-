import os
from batch.architecture import LeNet, LeNetBN, LeNetStockBN, LeNetLayerNorm
from google_drive_downloader import GoogleDriveDownloader as gdd
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from fonctions import test

#on recupère les données de test 
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5), (0.5))])
test1 = datasets.MNIST(root="dataset", train=False, transform=transform, download=True)
test_dataloader = DataLoader(test1, batch_size=512, shuffle=False)


# Vérifier si les fichiers de loss existent déjà
if not os.path.exists('batch/modèles_stockbn_entrainé.pth') or not os.path.exists('batch/modèle_bn_entrainé.pth') :

   # Liste des IDs de vos fichiers sur Google Drive
    file_ids = ['1x9EpK8hLu4obAFoy14_tN2OdsKMB_-xq', '1_LdtVZNoiwb2BzfqI6yjSAbOxDGMRTny']

    # Liste des noms de fichiers de sortie
    output_filenames = ['modèles_stockbn_entrainé.pth', 'modèle_bn_entrainé.pth']

    # Boucle pour télécharger chaque fichier de loss depuis Google Drive
    for file_id, output_filename in zip(file_ids, output_filenames):
        url = f'https://drive.google.com/file/d/{file_id}/view?usp=sharing'
        dest_path = './batch/' + output_filename

        gdd.download_file_from_google_drive(file_id=file_id, dest_path=dest_path, overwrite=True, showsize=True)




#charger les modèles 
modèle_stockbn=LeNetStockBN() 
modèles_stockbn_entrainé=torch.load('batch/modèles_stockbn_entrainé.pth')
modèle_stockbn.load_state_dict(modèles_stockbn_entrainé)


modèle_bn=LeNetBN()
modèle_bn_entrainé=torch.load('batch/modèle_bn_entrainé.pth')
modèle_bn.load_state_dict(modèle_bn_entrainé)


accuracies = {}


# Create a dictionary to store accuracies and their associated models
accuracies = {}

# Calculate accuracy for modèle_stockbn
accuracy_stockbn = calculate_accuracy(modèle_stockbn)
accuracies['modèle_stockbn'] = accuracy_stockbn

# Calculate accuracy for modèle_bn
accuracy_bn = calculate_accuracy(modèle_bn)
accuracies['modèle_bn'] = accuracy_bn

# Save the accuracies dictionary
torch.save(accuracies, 'batch/accuracies.pth')