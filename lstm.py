import torch
import torch.nn as nn
from utils import device

class LSTM(nn.Module):
    def __init__(self, input_size, n_hidden, n_categories, output_size, n_layers):
        super(LSTM, self).__init__()
        self.n_hidden = n_hidden
        self.n_layers = n_layers
        self.lstm = nn.LSTM(input_size + n_categories, n_hidden, n_layers)

        self.fc = nn.Linear(n_hidden, output_size)
        self.dropout = nn.Dropout(0.1)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, category, input, hidden, cell):
        input_combined_tensor = torch.cat((category, input), 1).unsqueeze(0)
        output, (hidden, cell) = self.lstm(input_combined_tensor, (hidden, cell))
        output = output[-1, :, :]
        output = self.fc(output)
        output = self.dropout(output)
        output = self.softmax(output)
        return output, hidden.detach(), cell.detach()

    def initHidden(self):
        return torch.zeros(self.n_layers, 1, self.n_hidden, device=device)

    def initCell(self):
        return torch.zeros(self.n_layers, 1, self.n_hidden, device=device)