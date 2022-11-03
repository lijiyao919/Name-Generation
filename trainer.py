import torch
import time
import matplotlib.pyplot as plt
from learner import Learner
from rnn import RNN

from utils import all_letters, n_letters, all_categories
from utils import randomTrainingExample, create_dataset
from utils import timeSince


class Trainer:
    def __init__(self, model, learning_rate, name):
        self.model = model
        self.learner = Learner(model, learning_rate)
        self.n_iters = 200000
        self.all_losses = []
        self.all_accuracy = []
        self.print_every = 5000
        self.plot_every = 500
        self.name = name

    def train(self):
        print(f"Training the {self.name}:")
        start = time.time()
        total_loss = 0
        for iter in range(1, self.n_iters+1):
            category_tensor, input_name_tensor, target_name_tensor = randomTrainingExample()
            target_name_tensor = target_name_tensor.unsqueeze(1)
            hidden_tensor = self.model.initHidden()
            loss = 0
            for i in range(len(input_name_tensor)):
                _, l = self.learner.update(input_name_tensor[i], category_tensor, target_name_tensor[i], hidden_tensor)
                loss+=l
            loss = loss/input_name_tensor.size()[0]
            total_loss += loss

            if iter%self.print_every == 0:
                print('%s (%d %d%%) %.4f' %(timeSince(start), iter, iter/self.n_iters * 100, loss))

            if iter%self.plot_every == 0:
                self.all_losses.append(total_loss / self.plot_every)
                total_loss = 0

#create data
create_dataset()

# RNN
learning_rate = 0.005
n_hidden = 128
n_layer = 2
model = RNN(n_letters, n_hidden, len(all_categories), n_letters, n_layer)
rnn_trainer = Trainer(model, learning_rate, "RNN")
rnn_trainer.train()

plt.figure()
plt.plot(rnn_trainer.all_losses)
plt.show()




