import torch
from torch import nn
from torch.nn import functional as F


class LyricModel(nn.Module):
    def __init__(self, batch_size = 32, num_layers = 2, bidirectional = True, hidden_size = 128, length = 64):
        self.hidden = hidden_size
        self.batch_size = batch_size
        self.length = length
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.in_features = self.hidden * self.num_layers * (int(self.bidirectional) + 1)
        
        super(LyricModel, self).__init__()
        
        self.embedder = nn.Embedding(num_embeddings = len(char_dict), embedding_dim = self.length)
        
        self.lstm = nn.LSTM(input_size = self.length, hidden_size = self.hidden, batch_first = True,
                            num_layers = self.num_layers, bidirectional = self.bidirectional)
        self.linear1 = nn.Linear(self.in_features, self.in_features//2)
        self.linear2 = nn.Linear(self.in_features//2, 1)
        
    def forward(self, x, h):
        y = self.embedder(x)
        y, h = self.lstm(y, h)
        h_ = h[0].view(-1, self.in_features)
        y = F.leaky_relu(self.linear1(h_), .1)
        y = F.leaky_relu(self.linear2(y), .1)
        
        return torch.sigmoid(y).squeeze(), h
    
    def init_hidden_state(self, mean, stddev):
        """
        Initialize hidden state and context tensors.
        """
        h = torch.distributions.Normal(mean, stddev).sample(((int(self.bidirectional) + 1)*self.num_layers,\
                                                             self.batch_size, self.hidden))
        c = torch.distributions.Normal(mean, stddev).sample(((int(self.bidirectional) + 1)*self.num_layers, \
                                                             self.batch_size, self.hidden))
        
        return (h.to(device), c.to(device))