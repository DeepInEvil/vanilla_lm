from torch import optim
import torch
from torch import nn
import torch.nn.functional as F


class LSTMCell(nn.Module):

    def __init__(self, input_size, hidden_size, dropout=None, batch_first=True):
        """Initialize params."""
        super(LSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = 1
        self.batch_first = batch_first
        self.dropout = dropout

        self.input_weights = nn.Linear(input_size, 4 * hidden_size)
        self.hidden_weights = nn.Linear(hidden_size, 4 * hidden_size)

    def forward(self, input, hidden):
        """Propogate input through the network."""
        # tag = None  #
        def recurrence(input, hidden):
            """Recurrence helper."""
            hx, cx = hidden  # n_b x hidden_dim
            gates = self.input_weights(input) + \
                self.hidden_weights(hx)
            ingate, forgetgate, cellgate, outgate = gates.chunk(4, 1)

            ingate = F.sigmoid(ingate)
            forgetgate = F.sigmoid(forgetgate)
            cellgate = F.tanh(cellgate)  # o_t
            outgate = F.sigmoid(outgate)

            cy = (forgetgate * cx) + (ingate * cellgate)
            hy = outgate * F.tanh(cy)  # n_b x hidden_dim
            if self.dropout:
                hy = F.dropout(hy, self.dropout, inplace=True)
            return hy, cy

        if self.batch_first:
            input = input.transpose(0, 1)

        output = []
        steps = range(input.size(0))
        for i in steps:
            hidden = recurrence(input[i], hidden)
            if isinstance(hidden, tuple):
                output.append(hidden[0])
            else:
                output.append(hidden)

        output = torch.cat(output, 0).view(input.size(0), *output[0].size())

        if self.batch_first:
            output = output.transpose(0, 1)

        return output, hidden


class LSTMCTop(nn.Module):

    def __init__(self, input_size, hidden_size, dropout=None, batch_first=True, top_size=None):
        """Initialize params."""
        super(LSTMCTop, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = 1
        self.batch_first = batch_first
        self.dropout = dropout

        self.input_weights = nn.Linear(input_size, 4 * hidden_size)
        self.hidden_weights = nn.Linear(hidden_size, 4 * hidden_size)
        if top_size:
            self.topic_w = nn.Linear(top_size, hidden_size)

    def forward(self, input, hidden, topic):
        """Propogate input through the network."""
        # tag = None  #
        def recurrence(input, hidden, topic):
            """Recurrence helper."""
            hx, cx = hidden  # n_b x hidden_dim
            gates = self.input_weights(input) + \
                self.hidden_weights(hx)
            ingate, forgetgate, cellgate, outgate = gates.chunk(4, 1)
            #print (ingate.size())
            ingate = F.sigmoid(ingate + self.topic_w(topic))
            forgetgate = F.sigmoid(forgetgate + self.topic_w(topic))
            cellgate = F.tanh(cellgate)  # o_t
            outgate = F.sigmoid(outgate)

            cy = (forgetgate * cx) + (ingate * cellgate)
            hy = outgate * F.tanh(cy)  # n_b x hidden_dim
            if self.dropout:
                hy = F.dropout(hy, self.dropout, inplace=True)
            return hy, cy

        if self.batch_first:
            input = input.transpose(0, 1)

        output = []
        steps = range(input.size(0))
        for i in steps:
            hidden = recurrence(input[i], hidden, topic)
            if isinstance(hidden, tuple):
                output.append(hidden[0])
            else:
                output.append(hidden)

        output = torch.cat(output, 0).view(input.size(0), *output[0].size())

        if self.batch_first:
            output = output.transpose(0, 1)

        return output, hidden