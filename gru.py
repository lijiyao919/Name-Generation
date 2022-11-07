import torch
import torch.nn as nn
from utils import device

class GRU(nn.Module):
    def __init__(self, input_size, n_hidden, n_categories, output_size, n_layers):
        super(GRU, self).__init__()
        self.n_hidden = n_hidden
        self.n_layers = n_layers
        self.rnn = nn.GRU(input_size+n_categories, n_hidden, n_layers)

        self.fc = nn.Linear(n_hidden, output_size)
        self.dropout = nn.Dropout(0.1)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, category, input, hidden):
        input_combined_tensor = torch.cat((category, input), 1).unsqueeze(0)
        output, hidden = self.rnn(input_combined_tensor, hidden)
        output = output[-1, :, :]
        output = self.fc(output)
        output = self.dropout(output)
        output = self.softmax(output)
        return output, hidden.detach()

    def initHidden(self):
        return torch.zeros(self.n_layers, 1, self.n_hidden, device=device)
