import json
from nltk_utils import tokenize, stem, bag_of_words
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from model import NeuralNet

# Load intents.json
with open('intents.json', 'r') as f:
    intents = json.load(f)

all_words = []
tags = []
xy = []

# Loop through each intent and its patterns
for intent in intents['intents']:
    tag = intent['tag']
    tags.append(tag)
    for pattern in intent['patterns']:
        w = tokenize(pattern)
        all_words.extend(w)
        xy.append((w, tag))

ignore_words = ['?', '!', '.', ',']

# Stem and filter out ignored words
all_words = [stem(w) for w in all_words if w not in ignore_words]

# Sort and remove duplicates
all_words = sorted(set(all_words))
tags = sorted(set(tags))

# Print the results
print("All words:", all_words)
print("Tags:", tags)

X_train = []
Y_train = []

# Convert each pattern into a bag-of-words and match it with the corresponding tag
for (pattern_sentence, tag) in xy:
    bag = bag_of_words(pattern_sentence, all_words)  # Pass the pattern_sentence and all_words
    X_train.append(bag)

    label = tags.index(tag)  # Find the index of the tag
    Y_train.append(label)

X_train = np.array(X_train)
Y_train = np.array(Y_train)

print("Training data X:", X_train)
print("Training labels Y:", Y_train)

# the error will appear from the class ChatDataset

class ChatDataset(Dataset):

    def __init__(self):
        self.n_samples= len(X_train)
        self.x_data = X_train
        self.y_data = Y_train
    #dataset[idx]
    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.n_samples

#hyperparamters
batch_size = 8
hidden_size =8
output_size = len(tags)
input_size = len(X_train[0])
learning_rate = 0.001
num_epochs = 1000

dataset = ChatDataset()
train_loader = DataLoader(dataset = dataset,
                          batch_size =batch_size,
                          shuffle =True,
                          num_workers=0)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = NeuralNet(input_size, hidden_size, output_size).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

for epoch in range(num_epochs):
    for (words,labels) in train_loader:
        words = words.to(device)
        labels = labels.to(device)

        #foward
        outputs = model(words)
        loss = criterion(outputs,labels)

        #backward and optimizer step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    if (epoch+1) %100 ==0:
        print(f'epoch {epoch+1}/{num_epochs}, loss={loss.item():.4f}')
print(f'final loss, loss={loss.item():.4f}')

data = {
"model_state": model.state_dict(),
"input_size": input_size,
"hidden_size": hidden_size,
"output_size": output_size,
"all_words": all_words,
"tags": tags
}

FILE = "data.pth"
torch.save(data, FILE)

print(f'training complete. file saved to {FILE}')