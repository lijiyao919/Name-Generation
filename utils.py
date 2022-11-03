from __future__ import unicode_literals, print_function, division
from io import open
import glob
import os
import unicodedata
import string
import random
import torch
import time
import math

all_letters = string.ascii_letters + " .,;'-"
n_letters = len(all_letters) + 1 # Plus EOS marker
category_names = {}
all_categories = []

def timeSince(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

# Turn a Unicode string to plain ASCII, thanks to https://stackoverflow.com/a/518232/2809427
def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
        and c in all_letters
    )

# Read a file and split into lines
def readLines(filename):
    with open(filename, encoding='utf-8') as some_file:
        return [unicodeToAscii(line.strip()) for line in some_file]

# Build the category_lines dictionary, a list of lines per category
def create_dataset():
    for filename in glob.glob('data/names/*.txt'):
        category = os.path.splitext(os.path.basename(filename))[0]
        all_categories.append(category)
        lines = readLines(filename)
        category_names[category] = lines


def randomTrainingExample():
    # Random item from a list
    def randomChoice(l):
        return l[random.randint(0, len(l) - 1)]

    # Get a random category and random line from that category
    def randomTrainingPair():
        category = randomChoice(all_categories)
        name = randomChoice(category_names[category])
        return category, name

    # One-hot vector for category
    def categoryTensor(category):
        li = all_categories.index(category)
        tensor = torch.zeros(1, len(all_categories))
        tensor[0][li] = 1
        return tensor

    # One-hot matrix of first to last letters (not including EOS) for input
    def inputTensor(line):
        tensor = torch.zeros(len(line), 1, n_letters)
        for li in range(len(line)):
            letter = line[li]
            tensor[li][0][all_letters.find(letter)] = 1
        return tensor

    # LongTensor of second letter to end (EOS) for target
    def targetTensor(line):
        letter_indexes = [all_letters.find(line[li]) for li in range(1, len(line))]
        letter_indexes.append(n_letters - 1)  # EOS
        return torch.LongTensor(letter_indexes)

    category, name = randomTrainingPair()
    category_tensor = categoryTensor(category)
    input_line_tensor = inputTensor(name)
    target_line_tensor = targetTensor(name)
    return category_tensor, input_line_tensor, target_line_tensor

if __name__ == "__main__":
    create_dataset()
    print(category_names["Chinese"])
    print(all_categories)
    print(len(all_categories))