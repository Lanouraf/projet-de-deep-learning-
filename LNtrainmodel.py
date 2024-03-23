from Sequences import prep_data
from LNModule import BagOfWordsClassifierLayer,BagOfWordsClassifier,BagOfWordsClassifierLayerHM,BagOfWordsClassifierBatchNorm
from LNtrain import LNtrain,LNtest
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torch
import pandas as pd

train_dataset,train_loader,test_dataset,test_loader=prep_data()

vocab_size = len(train_dataset.vectorizer.vocabulary_)
hidden1 = 128
hidden2 = 64
output_shape = 1

criterion = nn.BCELoss()

model = BagOfWordsClassifier(vocab_size, hidden1, hidden2, output_shape)
model2=BagOfWordsClassifier(vocab_size, hidden1, hidden2, output_shape)
model3=BagOfWordsClassifierLayerHM(vocab_size, hidden1, hidden2, output_shape)
model4=BagOfWordsClassifierBatchNorm(vocab_size, hidden1, hidden2, output_shape)

optimizer = optim.Adam(model.parameters(), lr=0.001)
optimizer2 = optim.Adam(model2.parameters(), lr=0.001)
optimizer3 = optim.Adam(model3.parameters(), lr=0.001)
optimizer4 = optim.Adam(model4.parameters(), lr=0.001)

BOW_LN,loss=LNtrain(model,optimizer,criterion)
BOW,loss2=LNtrain(model2,optimizer2,criterion)
BOW_HM_LN,loss3=LNtrain(model3,optimizer3,criterion)
BOW_BN,loss4=LNtrain(model4,optimizer4,criterion)

# Créer un dictionnaire de pertes pour chaque modèle
losses_dict = {
    'BOW': loss2,
    'BOW_LN': loss,
    'BOW_HM_LN': loss3
}

losses_dict_BN = {
    'BOW_BN': loss}


torch.save(losses_dict_BN, 'loss_BN.pth')
torch.save(losses_dict, 'losses.pth')


torch.save(BOW.state_dict(), 'BOW.pth')
torch.save(BOW_LN.state_dict(), 'BOW_LN.pth')
torch.save(BOW_HM_LN.state_dict(), 'BOW_HM_LN.pth')
torch.save(BOW_BN.state_dict(), 'BOW_BN.pth')


