import torch
import time
import matplotlib.pyplot as plt
from string import ascii_uppercase
from learner import Learner
from rnn import RNN
from lstm import LSTM
from gru import GRU

from utils import all_letters, n_letters, all_categories, categoryTensor, inputTensor, category_names
from utils import randomTrainingExample, create_dataset
from utils import timeSince
from utils import device

MAX_LEN = 20

class Trainer:
    def __init__(self, model, learning_rate, name):
        self.model = model
        self.learner = Learner(model, learning_rate)
        self.n_iters = 100000
        self.all_losses = []
        self.all_accuracy = []
        self.print_every = 5000
        self.plot_every = 5000
        self.name = name

    def train(self):
        print(f"Training the {self.name}:")
        start = time.time()
        total_loss = 0
        for iter in range(1, self.n_iters+1):
            category_tensor, input_name_tensor, target_name_tensor = randomTrainingExample()
            target_name_tensor = target_name_tensor.unsqueeze(1)
            hidden_tensor = self.model.initHidden()
            cell_tensor = None
            if self.name=="LSTM":
                cell_tensor = self.model.initCell()
            loss = 0
            for i in range(len(input_name_tensor)):
                _, hidden_tensor,cell_tensor, l = self.learner.update(input_name_tensor[i], category_tensor, target_name_tensor[i], hidden_tensor, cell_tensor)
                loss+=l
            loss = loss/input_name_tensor.size()[0]
            total_loss += loss

            if iter%self.print_every == 0:
                print('%s (%d %d%%) %.4f' %(timeSince(start), iter, iter/self.n_iters * 100, loss))

            if iter%self.plot_every == 0:
                self.all_losses.append(total_loss / self.plot_every)
                total_loss = 0
                self.test()

    def sample(self, category, start_letter='A'):
        with torch.no_grad():  # no need to track history in sampling
            category_tensor = categoryTensor(category)
            input = inputTensor(start_letter)
            hidden_tensor = self.model.initHidden()
            cell_tensor = None
            if self.name == "LSTM":
                cell_tensor = self.model.initCell()
            output_name = start_letter

            for i in range(MAX_LEN):
                if self.name=="RNN" or self.name=="GRU":  # rnn gru
                    output, hidden_tensor = self.model(category_tensor, input[0], hidden_tensor)
                else:  # lstm
                    assert cell_tensor is not None
                    output, hidden_tensor, cell_tensor = self.model(category_tensor, input[0], hidden_tensor, cell_tensor)
                topv, topi = output.topk(1)
                topi = topi[0][0]
                if topi == n_letters - 1:
                    break
                else:
                    letter = all_letters[topi]
                    output_name += letter
                input = inputTensor(letter)
            return output_name

    def test(self):
        total_cnt = 0
        acc_cnt = 0
        for c in all_categories:
            for l in ascii_uppercase:
                total_cnt +=1
                name = self.sample(c,l)
                if name in category_names[c]:
                    acc_cnt+=1
        print(f"test result: {100*acc_cnt/total_cnt:.2f}%")
        self.all_accuracy.append(100*acc_cnt/total_cnt)

#create data
create_dataset()

# RNN
learning_rate = 0.001
n_hidden = 128
n_layer = 2
model = RNN(n_letters, n_hidden, len(all_categories), n_letters, n_layer).to(device)
rnn_trainer = Trainer(model, learning_rate, "RNN")
rnn_trainer.train()

# LSTM
learning_rate = 0.001
n_hidden = 128
n_layer = 2
model = LSTM(n_letters, n_hidden, len(all_categories), n_letters, n_layer).to(device)
lstm_trainer = Trainer(model, learning_rate, "LSTM")
lstm_trainer.train()

#GRU
learning_rate = 0.001
n_hidden = 128
n_layer = 2
model = GRU(n_letters, n_hidden, len(all_categories), n_letters, n_layer).to(device)
gru_trainer = Trainer(model, learning_rate, "GRU")
gru_trainer.train()

plt.figure(0)
plt.plot(rnn_trainer.all_losses, label="RNN", marker="o")
plt.plot(lstm_trainer.all_losses, label="LSTM", marker="^")
plt.plot(gru_trainer.all_losses, label="GRU", marker="*")
plt.legend()
plt.xlabel("Episode")
plt.ylabel("Loss")
plt.show()

plt.figure(1)
plt.plot(rnn_trainer.all_accuracy, label="RNN", marker="o")
plt.plot(lstm_trainer.all_accuracy, label="LSTM", marker="^")
plt.plot(gru_trainer.all_accuracy, label="GRU", marker="*")
plt.legend()
plt.xlabel("Episode")
plt.ylabel("Accuracy (%)")
plt.plot(gru_trainer.all_accuracy)
plt.show()




