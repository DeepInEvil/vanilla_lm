import torch.nn as nn
import torch
from torch.autograd import Variable
from CustomLSTMCell import LSTMCell, LSTMCTop


class RNNModel(nn.Module):
    """Container module with an encoder, a recurrent module, and a decoder."""

    def __init__(self, rnn_type, ntoken, ninp, nhid, nlayers, dropout=0.5, tie_weights=False):
        super(RNNModel, self).__init__()
        self.drop = nn.Dropout(dropout)
        self.encoder = nn.Embedding(ntoken, ninp)
        self.nlayers = nlayers
        if rnn_type in ['LSTM', 'GRU']:
            #self.rnn = getattr(nn, rnn_type)(ninp, nhid, nlayers, dropout=dropout)
            self.rnn = LSTMCell(ninp, nhid, dropout=dropout, batch_first=False)
        else:
            try:
                nonlinearity = {'RNN_TANH': 'tanh', 'RNN_RELU': 'relu'}[rnn_type]
            except KeyError:
                raise ValueError( """An invalid option for `--model` was supplied,
                                 options are ['LSTM', 'GRU', 'RNN_TANH' or 'RNN_RELU']""")
            self.rnn = nn.RNN(ninp, nhid, nlayers, nonlinearity=nonlinearity, dropout=dropout)
        self.decoder = nn.Linear(nhid, ntoken)

        # Optionally tie weights as in:
        # "Using the Output Embedding to Improve Language Models" (Press & Wolf 2016)
        # https://arxiv.org/abs/1608.05859
        # and
        # "Tying Word Vectors and Word Classifiers: A Loss Framework for Language Modeling" (Inan et al. 2016)
        # https://arxiv.org/abs/1611.01462
        if tie_weights:
            if nhid != ninp:
                raise ValueError('When using the tied flag, nhid must be equal to emsize')
            self.decoder.weight = self.encoder.weight

        self.init_weights()

        self.rnn_type = rnn_type
        self.nhid = nhid
        self.nlayers = nlayers

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, input, hidden):
        emb = self.drop(self.encoder(input))
        output, hidden = self.rnn(emb, hidden)
        # sent_variable = emb
        # #outputs = []
        # for i in range(self.nlayers):
        #     output, hidden = self.rnns[i](sent_variable, hidden)
        #     #outputs.append(output)
        #     sent_variable = output
        output = self.drop(output)
        decoded = self.decoder(output.view(output.size(0)*output.size(1), output.size(2)))
        return decoded.view(output.size(0), output.size(1), decoded.size(1)), hidden

    def init_hidden(self, bsz):
        weight = next(self.parameters())
        if self.rnn_type == 'LSTM':
            # return (Variable(torch.zeros(self.nlayers, bsz, self.nhid)).cuda(),
            #         Variable(torch.zeros(self.nlayers, bsz, self.nhid)).cuda())
            return (Variable(torch.zeros(bsz, self.nhid)).cuda(),
                    Variable(torch.zeros(bsz, self.nhid)).cuda())
        else:
            return Variable(torch.zeros(self.nlayers, bsz, self.nhid).cuda())


class RNNModelmitTopic(nn.Module):
    """Container module with an encoder, a recurrent module, and a decoder."""

    def __init__(self, rnn_type, ntoken, ninp, nhid, nlayers, dropout=0.5, tie_weights=False, topic_dim=None):
        super(RNNModelmitTopic, self).__init__()
        self.drop = nn.Dropout(dropout)
        self.encoder = nn.Embedding(ntoken, ninp)
        self.nlayers = nlayers
        if rnn_type in ['LSTM', 'GRU']:
            #self.rnn = getattr(nn, rnn_type)(ninp, nhid, nlayers, dropout=dropout)
            self.rnn = LSTMCTop(ninp, nhid, dropout=dropout, batch_first=False, top_size=topic_dim)
            # self.rnn = nn.Sequential(OrderedDict([
            #                 ('LSTM1', nn.LSTM(ninp, nhid, 1),
            #                 ('LSTM2', nn.LSTM(ninp, nhid, 1)))]))
            # self.rnns = nn.ModuleList()
            # for i in range(nlayers):
            #     ninp = ninp if i == 0 else nhid
            #     self.rnns.append(LSTMCell(ninp, nhid, dropout=dropout, batch_first=False))
        else:
            try:
                nonlinearity = {'RNN_TANH': 'tanh', 'RNN_RELU': 'relu'}[rnn_type]
            except KeyError:
                raise ValueError( """An invalid option for `--model` was supplied,
                                 options are ['LSTM', 'GRU', 'RNN_TANH' or 'RNN_RELU']""")
            self.rnn = nn.RNN(ninp, nhid, nlayers, nonlinearity=nonlinearity, dropout=dropout)
        self.decoder = nn.Linear(nhid, ntoken)

        # Optionally tie weights as in:
        # "Using the Output Embedding to Improve Language Models" (Press & Wolf 2016)
        # https://arxiv.org/abs/1608.05859
        # and
        # "Tying Word Vectors and Word Classifiers: A Loss Framework for Language Modeling" (Inan et al. 2016)
        # https://arxiv.org/abs/1611.01462
        if tie_weights:
            if nhid != ninp:
                raise ValueError('When using the tied flag, nhid must be equal to emsize')
            self.decoder.weight = self.encoder.weight

        self.init_weights()

        self.rnn_type = rnn_type
        self.nhid = nhid
        self.nlayers = nlayers

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, input, hidden, top):
        emb = self.drop(self.encoder(input))
        output, hidden = self.rnn(emb, hidden, top)
        # sent_variable = emb
        # #outputs = []
        # for i in range(self.nlayers):
        #     output, hidden = self.rnns[i](sent_variable, hidden)
        #     #outputs.append(output)
        #     sent_variable = output
        output = self.drop(output)
        decoded = self.decoder(output.view(output.size(0)*output.size(1), output.size(2)))
        return decoded.view(output.size(0), output.size(1), decoded.size(1)), hidden

    def init_hidden(self, bsz):
        weight = next(self.parameters())
        if self.rnn_type == 'LSTM':
            # return (Variable(torch.zeros(self.nlayers, bsz, self.nhid)).cuda(),
            #         Variable(torch.zeros(self.nlayers, bsz, self.nhid)).cuda())
            return (Variable(torch.zeros(bsz, self.nhid)).cuda(),
                    Variable(torch.zeros(bsz, self.nhid)).cuda())
        else:
            return Variable(torch.zeros(self.nlayers, bsz, self.nhid).cuda())