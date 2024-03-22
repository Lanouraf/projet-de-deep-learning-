from Sequences import prep_data
from LNModule import BagOfWordsClassifierLayer,BagOfWordsClassifier
from LNtrain import LNtrain,LNtest
import torch.nn as nn
import torch.optim as optim
import numpy as np

train_dataset,train_loader,test_dataset,test_loader=prep_data()

vocab_size = len(train_dataset.vectorizer.vocabulary_)
hidden1 = 128
hidden2 = 64
output_shape = 1

model = BagOfWordsClassifierLayer(vocab_size, hidden1, hidden2, output_shape)
model2=BagOfWordsClassifier(vocab_size, hidden1, hidden2, output_shape)
model3=BagOfWordsClassifier(vocab_size, hidden1, hidden2, output_shape)
criterion = nn.BCELoss()

optimizer = optim.Adam(model.parameters(), lr=0.001)
optimizer2 = optim.Adam(model2.parameters(), lr=0.001)
optimizer3 = optim.Adam(model3.parameters(), lr=0.001)

model,loss=LNtrain(model,optimizer,criterion)
model2,loss2=LNtrain(model2,optimizer2,criterion)
model3,loss3=LNtrain(model3,optimizer3,criterion)

acc1=LNtest(model,test_loader)
acc2=LNtest(model2,test_loader)
acc3=LNtest(model3,test_loader)
print(acc1)
print(np.mean(loss))
print(acc2)
print(np.mean(loss2))
print(acc3)
print(np.mean(loss3))
