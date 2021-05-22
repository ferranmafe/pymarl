import torch.nn as nn
import torch.nn.functional as F
import torch
import math


class RNNAgentPairs(nn.Module):
    def __init__(self, input_shape, args):
        super(RNNAgentPairs, self).__init__()
        self.args = args
        self.input_shape = input_shape

        self.fc1 = nn.Linear(input_shape * 2, args.rnn_hidden_dim * 2)
        self.rnn = nn.GRUCell(args.rnn_hidden_dim * 2, args.rnn_hidden_dim * 2)
        self.fc2 = nn.Linear(args.rnn_hidden_dim * 2, args.n_actions ** 2)

    def init_hidden(self):
        # make hidden states on same device as model
        return self.fc1.weight.new(1, self.args.rnn_hidden_dim * 2).zero_()

    def forward(self, inputs, hidden_state):
        x = F.relu(self.fc1(inputs.reshape(-1, self.input_shape * 2)))
        h_in = hidden_state.reshape(-1, self.args.rnn_hidden_dim * 2)
        hp = self.rnn(x, h_in)
        qp = self.fc2(hp)
        qp = self.__decode_combined_output(qp)

        return qp, hp

    @staticmethod
    def __decode_combined_output(tensor):
        left_units = torch.max(
            tensor.view(
                tensor.size()[0],
                int(math.sqrt(tensor.size()[1])),
                int(math.sqrt(tensor.size()[1]))
            ),
            dim=2
        ).values
        right_units = torch.max(
            tensor.view(
                tensor.size()[0],
                int(math.sqrt(tensor.size()[1])),
                int(math.sqrt(tensor.size()[1]))
            ),
            dim=1
        ).values
        return torch.stack((left_units, right_units), dim=2)\
            .view(tensor.size()[0] * 2, int(math.sqrt(tensor.size()[1])))

