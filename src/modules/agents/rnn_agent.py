import torch.nn as nn
import torch.nn.functional as F
import torch
import math


class RNNAgent(nn.Module):
    def __init__(self, input_shape, args):
        super(RNNAgent, self).__init__()
        self.args = args
        self.input_shape = input_shape

        if self.args.agent_type == '2_units':
            self.fc1 = nn.Linear(input_shape * 2, args.rnn_hidden_dim * 2)
            self.rnn = nn.GRUCell(args.rnn_hidden_dim * 2, args.rnn_hidden_dim * 2)
            self.fc21 = nn.Linear(args.rnn_hidden_dim * 2, args.n_actions)
            self.fc22 = nn.Linear(args.rnn_hidden_dim * 2, args.n_actions)
        elif self.args.agent_type == '2_units_combined_output':
            # Pair of units
            self.fc11 = nn.Linear(input_shape, args.rnn_hidden_dim)
            self.rnn1 = nn.GRUCell(args.rnn_hidden_dim, args.rnn_hidden_dim)
            self.fc12 = nn.Linear(args.rnn_hidden_dim, args.n_actions)
            # Single unit
            self.fc21 = nn.Linear(input_shape * 2, args.rnn_hidden_dim * 2)
            self.rnn2 = nn.GRUCell(args.rnn_hidden_dim * 2, args.rnn_hidden_dim * 2)
            self.fc22 = nn.Linear(args.rnn_hidden_dim * 2, args.n_actions ** 2)
        else:
            self.fc1 = nn.Linear(input_shape, args.rnn_hidden_dim)
            self.rnn = nn.GRUCell(args.rnn_hidden_dim, args.rnn_hidden_dim)
            self.fc2 = nn.Linear(args.rnn_hidden_dim, args.n_actions)

    def init_hidden(self):
        # make hidden states on same device as model
        return self.fc11.weight.new(1, self.args.rnn_hidden_dim).zero_(), \
               self.fc21.weight.new(1, self.args.rnn_hidden_dim * 2).zero_()

    def forward(self, inputs, hidden_state):
        if self.args.agent_type == '2_units':
            x = F.relu(self.fc1(inputs.view(-1, self.input_shape * 2)))
            h_in = hidden_state.reshape(-1, self.args.rnn_hidden_dim * 2)
            h = self.rnn(x, h_in)
            q1 = self.fc21(h)
            q2 = self.fc22(h)
            return torch.stack((q1, q2), dim=2).view(q1.size()[0] * 2, q1.size()[1]), h
        elif self.args.agent_type == '2_units_combined_output':
            inputs = inputs.view(self.args.n_agents, self.input_shape, -1)

            individual_inputs = inputs[:self.args.n_agents - 2, :, :]
            individual_inputs = individual_inputs.view(-1, self.input_shape)

            x = F.relu(self.fc11(individual_inputs))
            h_in = hidden_state[0].reshape(-1, self.args.rnn_hidden_dim)
            h_ind = self.rnn1(x, h_in)
            q_ind = self.fc12(h_ind)

            pairs_inputs = inputs[self.args.n_agents - 2:, :, :]
            pairs_inputs = pairs_inputs.view(-1, self.input_shape * 2)

            x = F.relu(self.fc21(pairs_inputs))
            h_in = hidden_state[1].reshape(-1, self.args.rnn_hidden_dim * 2)
            h_p = self.rnn2(x, h_in)
            q_p = self.fc22(h_p)
            return torch.cat((q_ind, self.__decode_combined_output(q_p)), 0), (h_ind, h_p)
        else:
            x = F.relu(self.fc1(inputs))
            h_in = hidden_state.reshape(-1, self.args.rnn_hidden_dim)
            h = self.rnn(x, h_in)
            q = self.fc2(h)
            return q, h

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

