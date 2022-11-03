import torch
import torch.nn as nn

class RNN(nn.Module):
    def __init__(self, input_size, n_hidden, n_categories, output_size, n_layers):
        super(RNN, self).__init__()
        self.n_hidden = n_hidden
        self.n_layers = n_layers
        self.rnn = nn.RNN(input_size+n_categories, n_hidden, n_layers)

        self.fc = nn.Linear(n_hidden, output_size)
        self.dropout = nn.Dropout(0.1)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, category, input, hidden):
        input_combined_tensor = torch.cat((category, input), 1).unsqueeze(0)
        hidden,_ = self.rnn(input_combined_tensor, hidden)
        hidden = hidden[-1, :, :]
        output = self.fc(hidden)
        output = self.dropout(output)
        output = self.softmax(output)
        return output, hidden

    def initHidden(self):
        return torch.zeros(self.n_layers, 1, self.n_hidden)
