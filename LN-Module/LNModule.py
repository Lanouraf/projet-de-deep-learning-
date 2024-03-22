import torch
import torch.nn as nn

class SimpleRNNModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, out_shape):
        super(SimpleRNNModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.rnn = nn.RNN(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, out_shape)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        embedded = self.embedding(x)
        _, hidden = self.rnn(embedded)
        out = self.fc(hidden.squeeze(0))
        return self.sigmoid(out)

class CustomLayerNormalization(nn.Module):
    def __init__(self, input_shape):
        super(CustomLayerNormalization, self).__init__()
        self.alpha = nn.Parameter(torch.ones(input_shape[-1]))
        self.beta = nn.Parameter(torch.zeros(input_shape[-1]))

    def forward(self, inputs):
        mean = torch.mean(inputs, dim=-1, keepdim=True)
        variance = torch.var(inputs, dim=-1, keepdim=True)
        return self.alpha * (inputs - mean)/ torch.sqrt(variance + 1e-10) + self.beta

class RNNWithCustomLayerModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, out_shape):
        super(RNNWithCustomLayerModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.rnn = nn.RNN(embedding_dim, hidden_dim, batch_first=True)
        self.custom_layer = CustomLayerNormalization(hidden_dim)
        self.fc = nn.Linear(hidden_dim, out_shape)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        embedded = self.embedding(x)
        _, hidden = self.rnn(embedded)
        normalized = self.custom_layer(hidden)
        out = self.fc(normalized.squeeze(0))
        return self.sigmoid(out)

class RNNWithBuiltinLayerModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, out_shape):
        super(RNNWithBuiltinLayerModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.rnn = nn.RNN(embedding_dim, hidden_dim, batch_first=True)
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.fc = nn.Linear(hidden_dim, out_shape)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        embedded = self.embedding(x)
        _, hidden = self.rnn(embedded)
        normalized = self.layer_norm(hidden)
        out = self.fc(normalized.squeeze(0))
        return self.sigmoid(out)