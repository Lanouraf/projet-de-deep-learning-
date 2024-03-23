from LNtrain import LNtest
import numpy as np
from Sequences import prep_data
from LNModule import BagOfWordsClassifier,BagOfWordsClassifierLayer,BagOfWordsClassifierLayerHM
from google_drive_downloader import GoogleDriveDownloader as gdd
import torch
import pandas as pd 

train_dataset,train_loader,test_dataset,test_loader=prep_data()

gdd.download_file_from_google_drive(file_id='1P28VRYGSueQ7DO7zghG5', dest_path='./BOW.pth')
gdd.download_file_from_google_drive(file_id='1INsBCiiYbPC7zNbP7Ufp5jfb0Ku2rTA', dest_path='./BOW_LN.pth')
gdd.download_file_from_google_drive(file_id='1TstrD3ic2NDqxy_rpFpFp4gYhyvtPZGQ', dest_path='./BOW_HM_LN.pth')
gdd.download_file_from_google_drive(file_id='1lESMtwwy_qzfWtpfpGDQYKzLOkABrIyp', dest_path='./losses.pth')
# BOW HM LN : 1TstrD3ic2NDqxy_rpFpFp4gYhyvtPZGQ
# BOW LN : 1INsBCiiYbPC7zNbP7Ufp5jfb0Ku2rTA
# BOW : 1P28VRYGSueQ7DO7zghG5
# losses : 1lESMtwwy_qzfWtpfpGDQYKzLOkABrIyp
# accuracies: 1Q_hO_bs4-ZiUGysPgDgTFuGGd_UKWjtO



vocab_size = len(train_dataset.vectorizer.vocabulary_)
hidden1 = 128
hidden2 = 64
output_shape = 1
model1 = BagOfWordsClassifier(vocab_size, hidden1, hidden2, output_shape)
model2 = BagOfWordsClassifierLayer(vocab_size, hidden1, hidden2, output_shape)
model3 = BagOfWordsClassifierLayerHM(vocab_size, hidden1, hidden2, output_shape)

model1.load_state_dict(torch.load('BOW.pth'))
model2.load_state_dict(torch.load('BOW_LN.pth'))
model3.load_state_dict(torch.load('BOW_HM_LN.pth'))

losses = torch.load('./losses.pth')
averages = {}
for model, loss in losses.items():
    averages[model] = sum(loss) / len(loss)

# Afficher les moyennes des pertes par modèle
for model, average_loss in averages.items():
    print(f"Moyenne des pertes pour {model}: {average_loss}")

acc1 = LNtest(model1, test_loader)
acc2 = LNtest(model2, test_loader)
acc3 = LNtest(model3, test_loader)

# Afficher les accuracies
print("Accuracy modèle 1:", acc1)
print("Accuracy modèle 2:", acc2)
print("Accuracy modèle 3:", acc3)

accuracies_dict = {
    "BOW": acc1,
    "BOW_LN": acc2,
    "BOW_HM_LN": acc3
}

accuracies_df = pd.DataFrame.from_dict(accuracies_dict, orient='index', columns=['Accuracy'])

# Enregistrer le DataFrame dans un fichier CSV
accuracies_df.to_csv('accuracies.csv')
