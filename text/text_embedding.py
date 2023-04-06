"""Generate text embeddings."""
import numpy as np
import torch
import pandas as pd
from torch import nn
from torch import optim
from torch.utils.data import Dataset
from torch.nn import functional as F
from nltk import sent_tokenize, word_tokenize
from transformers import RobertaTokenizer, RobertaModel
from gensim.models import KeyedVectors
from nltk.corpus import stopwords

# hyperparameters
model_name = 'roberta-base'
tokenizer = RobertaTokenizer.from_pretrained(model_name)
lr = 0.001
batch_size = 32
epochs = 10
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
embed = KeyedVectors.load('text/embed.kv')

# return text embedding of a batch
def text_tokened(train):
    """Generate text embeddings."""
    encoded_input = tokenizer(train, padding=True, truncation=True, return_tensors='pt')
    return encoded_input

def cosine_similarity(a, b):
    """Calculate cosine similarity."""
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# Note: this function is not used in fine-tuning. It's used in general training dataloader
class dataloader(Dataset):
    """Generate data loader."""
    def __init__(self, train):
        super().__init__()
        model = Finetuner()
        model.load_state_dict(torch.load('text/roberta_ft1.pt'))
        model.eval()
        image_embedding = torch.load('image/VGG_basic.pt')
        self.items = []
        train['humour'].map({'funny': 1, 'not_funny': 0})
        train['sarcastic'].map({'sarcastic': 2, 'little_sarcastic': 1, 'not_sarcastic': 0})
        train['offensive'].map({'offensive': 2, 'slight': 1, 'not_offensive': 0})
        train['motivational'].map({'motivational': 1, 'not_motivational': 0})
        train['overall_sentiment'].map({'positive': 1, 'negative': -1, 'neutral': 0})
        for index, row in train.iterrows():
            self.items.append({
                'text_embedding': model.forward(row['ocr_text']),
                'image_embedding': image_embedding[index],
                'rating': [row['humour'], row['sarcastic'], row['offensive'], row['motivational'], row['overall_sentiment']],
            })
    
    def __len__(self):
        return len(self.items)
    
    def __getitem__(self, index):
        return self.items[index]

class Finetuner(nn.Module):
    def __init__(self):
        super().__init__()
        self.roberta = RobertaModel.from_pretrained(model_name)
        self.linear = nn.Linear(768, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = text_tokened(x)
        output = self.roberta(**x)
        x = output.last_hidden_state[:, -1, :]
        x = self.linear(x)
        x = self.sigmoid(x)
        return x.squeeze()

def basic_collate_fn(batch):
    """Collate function for basic setting."""
    # texts
    texts = [item['text_corrected'] for item in batch]
    # labels
    labels = [item['rating'] for item in batch]
    return texts, labels

def basic_collate_fn_all(batch):
    """Collate function for basic setting."""
    # texts
    embeds = torch.stack((torch.cat(item['text_embedding'],item['image_embedding']) for item in batch))
    # labels
    labels = [item['rating'] for item in batch]
    return embeds, labels

def incongruity(sentence):
    """Calculate incongruity."""
    words = word_tokenize(sentence)
    for word in words.copy():
        if embed.__contains__(word) == False:
            words.remove(word)
    min_sim = 1
    for word in words:
        for word2 in words:
            sim = cosine_similarity(embed[word], embed[word2])
            if sim < min_sim:
                min_sim = sim
    return min_sim

def get_optimizer(model):
    """Get optimizer."""
    return optim.Adam(model.parameters(), lr=lr)

def get_lossfunction():
    """Get loss function."""
    return nn.MSELoss()

def get_loss(output, target):
    """Get loss."""
    func = get_lossfunction()
    return func(output, target)

def train(model,train_loader, optimizer):
    model.train()
    for epoch in range(1, epochs + 1):
        for batch_idx, (data, target) in enumerate(train_loader):
            optimizer.zero_grad()
            output = model.forward(data)
            label = torch.Tensor(tuple(incongruity(i) for i in data))
            loss = get_loss(output, label)
            loss.backward()
            optimizer.step()
            if batch_idx % 100 == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(epoch, batch_idx * len(data), len(train_loader.dataset), 100. * batch_idx / len(train_loader), loss.item()))
    return model

def evaluate(model, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            tokened_data = text_tokened(data)
            output = model(tokened_data)
            test_loss += get_loss(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    test_loss /= len(test_loader.dataset)
    print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(test_loss, correct, len(test_loader.dataset), 100. * correct / len(test_loader.dataset)))

model = Finetuner()