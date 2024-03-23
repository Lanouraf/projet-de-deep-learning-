from Sequences import prep_data
from LNModule import BagOfWordsClassifierLayer,BagOfWordsClassifier,BagOfWordsClassifierLayerHM,BagOfWordsClassifierBatchNorm
from LNtrain import LNtrain,LNtest
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torch

train_dataset,train_loader,test_dataset,test_loader=prep_data()

vocab_size = len(train_dataset.vectorizer.vocabulary_)
hidden1 = 128
hidden2 = 64
output_shape = 1

#model = BagOfWordsClassifier(vocab_size, hidden1, hidden2, output_shape)
#model2=BagOfWordsClassifier(vocab_size, hidden1, hidden2, output_shape)
#model3=BagOfWordsClassifierLayerHM(vocab_size, hidden1, hidden2, output_shape)
criterion = nn.BCELoss()

#optimizer = optim.Adam(model.parameters(), lr=0.001)
#optimizer2 = optim.Adam(model2.parameters(), lr=0.001)
#optimizer3 = optim.Adam(model3.parameters(), lr=0.001)

#BOW_LN,loss=LNtrain(model,optimizer,criterion)
#BOW,loss2=LNtrain(model2,optimizer2,criterion)
#BOW_HM_LN,loss3=LNtrain(model3,optimizer3,criterion)

model = BagOfWordsClassifierBatchNorm(vocab_size, hidden1, hidden2, output_shape)
optimizer = optim.Adam(model.parameters(), lr=0.001)
BOW_LN,loss=LNtrain(model,optimizer,criterion)

losses_dict = {
    'BOW_BN': loss
}
torch.save(BOW_BN.state_dict(), 'BOW_BN.pth')


# Enregistrer le dictionnaire des pertes dans un fichier avec torch.save()
torch.save(losses_dict, 'losses.pth')

#torch.save(BOW.state_dict(), 'BOW.pth')
#torch.save(BOW_LN.state_dict(), 'BOW_LN.pth')
#torch.save(BOW_HM_LN.state_dict(), 'BOW_HM_LN.pth')


