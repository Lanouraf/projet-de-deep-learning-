from LNtrain import LNtest
import numpy as np
from Sequences import prep_data
from LNModule import BagOfWordsClassifier
from google_drive_downloader import GoogleDriveDownloader as gdd
import torch

train_dataset,train_loader,test_dataset,test_loader=prep_data()

gdd.download_file_from_google_drive(file_id='19cpOB0IOJBmgDN6ffc8jtIyPi8lcK5bo', dest_path='./BOW.pth')

vocab_size = len(train_dataset.vectorizer.vocabulary_)
hidden1 = 128
hidden2 = 64
output_shape = 1
model = BagOfWordsClassifier(vocab_size, hidden1, hidden2, output_shape)


model.load_state_dict(torch.load('BOW.pth'))



acc1=LNtest(model,test_loader)
#acc2=LNtest(model2,test_loader)
#acc3=LNtest(model3,test_loader)
print(acc1)
#print(acc2)
#print(acc3)
