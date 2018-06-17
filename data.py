import os
import torch
from collections import defaultdict
import numpy as np
from torch.autograd import Variable


class Dictionary(object):
    def __init__(self, min_freq):
        self.word2idx = {}
        self.idx2word = []
        self.vocab = defaultdict(float)
        self.min_freq = min_freq

    def add_vocab(self, word):
        self.vocab[word] += 1.0

    def create_w2id(self, create=False, path=None):
        if create:
            print ("Creating Vocab..................")
            for w, c in self.vocab.items():
                if c > self.min_freq:
                    self.idx2word.append(w)
                    self.word2idx[w] = len(self.idx2word) - 1
            self.word2idx['__UNK__'] = len(self.word2idx) + 1
            #print ("Size of vocabulary: " + str(len(self.idx2word)))
            np.save(os.path.join(path, 'w2id.npy'), self.word2idx)
            np.save(os.path.join(path, 'id2w.npy'), self.idx2word)
        else:
            self.idx2word = np.load(os.path.join(path, 'id2w.npy'))
            self.word2idx = np.load(os.path.join(path, 'w2id.npy')).item()

    def get_w2id(self, word):
        try:
            return self.word2idx[word]
        except KeyError:
            return self.word2idx['__UNK__']

    def __len__(self):
        return len(self.idx2word)


class Corpus(object):
    def __init__(self, path):
        self.path = path
        self.dictionary = Dictionary(10)
        self.train = self.tokenize(os.path.join(path, 'train.txt'), train=True)
        self.valid = self.tokenize(os.path.join(path, 'valid.txt'))
        self.test = self.tokenize(os.path.join(path, 'test.txt'))

    def tokenize(self, path, train=False):
        """Tokenizes a text file."""
        assert os.path.exists(path)
        # Add words to the dictionary
        if train:
            with open(path, 'r') as f:
                tokens = 0
                for line in f:
                    words = ['<go>'] + line.split() + ['<eos>']
                    tokens += len(words)
                    for word in words:
                        self.dictionary.add_vocab(word)
            self.dictionary.create_w2id(create=True, path=self.path)
        else:
            with open(path, 'r') as f:
                tokens = 0
                for line in f:
                    words = ['<go>'] + line.split() + ['<eos>']
                    tokens += len(words)
            self.dictionary.create_w2id(path=self.path)


        # Tokenize file content
        with open(path, 'r') as f:
            ids = Variable(torch.LongTensor(tokens))
            token = 0
            for line in f:
                words = ['<go>'] + line.split() + ['<eos>']
                for word in words:
                    ids[token] = self.dictionary.get_w2id(word)
                    token += 1

        return ids
