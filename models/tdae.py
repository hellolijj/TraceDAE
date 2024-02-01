import torch.nn as nn
import torch.nn.functional as F
import torch
from models.layers import GraphConvolution
from models.gat import GraphAttentionLayer


class TraceDAE(nn.Module):
    def __init__(self,
                 in_dim,
                 num_nodes,
                 hidden_dim=64,
                 dropout=0.):
        super(TraceDAE, self).__init__()

        self.hidden_dim = hidden_dim
        self.in_dim = in_dim
        self.fea_transform = torch.nn.Linear(in_dim, hidden_dim)
        self.encoder_lstm = nn.LSTM(in_dim, hidden_dim, num_layers=1, batch_first=True)
        self.decoder_lstm = nn.LSTM(hidden_dim, in_dim, num_layers=1, batch_first=True)

        self.att_transform2 = torch.nn.Linear(hidden_dim, in_dim)
        self.att_transform1 = torch.nn.Linear(num_nodes, hidden_dim)
        # self.transform = torch.nn.Linear(num_nodes, in_dim)

        self.gat = GraphAttentionLayer(hidden_dim, in_dim)
        self.softmax = F.softmax
        self.relu = torch.nn.ReLU()

    def forward(self, x, adj):
        structure = self.fea_transform(x)
        structure = self.relu(structure)

        emb = self.gat(structure, adj)
        # emb = self.relu(emb)

        input = x.view(len(x), 1, -1)
        encoder_lstm, (n, c) = self.encoder_lstm(input, (torch.zeros(1, len(input), self.hidden_dim), torch.zeros(1, len(input), self.hidden_dim)))
        decoder_lstm, (n, c) = self.decoder_lstm(encoder_lstm, (torch.zeros(1, len(input), self.in_dim), torch.zeros(1, len(input), self.in_dim)))
        att_adj = decoder_lstm.squeeze()

        con_adj = emb @ emb.T

        return con_adj, att_adj