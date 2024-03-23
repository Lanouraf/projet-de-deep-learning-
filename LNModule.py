import torch.nn as nn
import torch
from HOMEMADE_LN import LayerNorm

class BagOfWordsClassifier(nn.Module):
        def __init__(self, vocab_size, hidden1, hidden2, out_shape):
            super(BagOfWordsClassifier, self).__init__()
            self.layer1 = nn.Sequential(
                nn.Linear(vocab_size, hidden1),
                nn.ReLU()
            )
            self.layer2 = nn.Sequential(
                nn.Linear(hidden1, hidden2),
                nn.ReLU()
            )
            self.layer3 = nn.Sequential(
                nn.Linear(hidden2, out_shape),
            )
        def forward(self, x):
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            x = x.squeeze()
            x = nn.Sigmoid()(x)
            return x

class BagOfWordsClassifierLayer(nn.Module):
        def __init__(self, vocab_size, hidden1, hidden2, out_shape):
            super(BagOfWordsClassifierLayer, self).__init__()
            self.layer1 = nn.Sequential(
                nn.Linear(vocab_size, hidden1),
                nn.LayerNorm(hidden1),  # Utilisation de torch.nn.LayerNorm
                nn.ReLU()
            )
            self.layer2 = nn.Sequential(
                nn.Linear(hidden1, hidden2),
                nn.LayerNorm(hidden2),  # Utilisation de torch.nn.LayerNorm
                nn.ReLU()
            )
            self.layer3 = nn.Sequential(
                nn.Linear(hidden2, out_shape),
            )

        def forward(self, x):
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            x = x.squeeze()
            x = nn.Sigmoid()(x)
            return x

class BagOfWordsClassifierBatchNorm(nn.Module):
    def __init__(self, vocab_size, hidden1, hidden2, out_shape):
        super(BagOfWordsClassifierBatchNorm, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Linear(vocab_size, hidden1),
            nn.BatchNorm1d(hidden1),  # Utilisation de torch.nn.BatchNorm1d
            nn.ReLU()
        )
        self.layer2 = nn.Sequential(
            nn.Linear(hidden1, hidden2),
            nn.BatchNorm1d(hidden2),  # Utilisation de torch.nn.BatchNorm1d
            nn.ReLU()
        )
        self.layer3 = nn.Sequential(
            nn.Linear(hidden2, out_shape),
        )

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = x.squeeze()
        x = nn.Sigmoid()(x)
        return x

class BagOfWordsClassifierLayerHM(nn.Module):
        def __init__(self, vocab_size, hidden1, hidden2, out_shape):
            super(BagOfWordsClassifierLayerHM, self).__init__()
            self.layer1 = nn.Sequential(
                nn.Linear(vocab_size, hidden1),
                LayerNorm(hidden1),  # Utilisation de torch.nn.LayerNorm
                nn.ReLU()
            )
            self.layer2 = nn.Sequential(
                nn.Linear(hidden1, hidden2),
                LayerNorm(hidden2),  # Utilisation de torch.nn.LayerNorm
                nn.ReLU()
            )
            self.layer3 = nn.Sequential(
                nn.Linear(hidden2, out_shape),
            )

        def forward(self, x):
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            x = x.squeeze()
            x = nn.Sigmoid()(x)
            return x
