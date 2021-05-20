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
            self.fc11 = nn.Linear(input_shape, args.rnn_hidden_dim)
            self.rnn1 = nn.GRUCell(args.rnn_hidden_dim, args.rnn_hidden_dim)
            self.fc12 = nn.Linear(args.rnn_hidden_dim, args.n_actions)

            self.fc21 = nn.Linear(input_shape * 2, args.rnn_hidden_dim * 2)
            self.rnn2 = nn.GRUCell(args.rnn_hidden_dim * 2, args.rnn_hidden_dim * 2)
            self.fc22 = nn.Linear(args.rnn_hidden_dim * 2, args.n_actions ** 2)
        else:
            self.fc1 = nn.Linear(input_shape, args.rnn_hidden_dim)
            self.rnn = nn.GRUCell(args.rnn_hidden_dim, args.rnn_hidden_dim)
            self.fc2 = nn.Linear(args.rnn_hidden_dim, args.n_actions)

    def init_hidden(self):
        # make hidden states on same device as model
        return self.fc11.weight.new(1, self.args.rnn_hidden_dim).zero_(),\
               self.fc21.weight.new(1, self.args.rnn_hidden_dim).zero_()

    def forward(self, inputs, hidden_state):
        if self.args.agent_type == '2_units':
            x = F.relu(self.fc1(inputs.view(-1, self.input_shape * 2)))
            h_in = hidden_state.reshape(-1, self.args.rnn_hidden_dim * 2)
            h = self.rnn(x, h_in)
            q1 = self.fc21(h)
            q2 = self.fc22(h)
            return torch.stack((q1, q2), dim=2).view(q1.size()[0] * 2, q1.size()[1]), h
        elif self.args.agent_type == '2_units_combined_output':
            x = F.relu(self.fc11(inputs))
            h_in = hidden_state[0].reshape(-1, self.args.rnn_hidden_dim)
            hi = self.rnn1(x, h_in)
            qi = self.fc12(hi)
            qi = qi.view(-1, self.args.n_agents, qi.size()[1])[:, :self.args.n_agents - 2, :]
            
            x = F.relu(self.fc21(inputs.view(-1, self.input_shape * 2)))
            h_in = hidden_state[1].reshape(-1, self.args.rnn_hidden_dim * 2)
            hp = self.rnn2(x, h_in)
            qp = self.fc22(hp)
            qp = self.__decode_combined_output(qp)
            qp = qp.view(-1, self.args.n_agents, qp.size()[1])[:, self.args.n_agents - 2:, :]

            qip = torch.cat((qi, qp), dim=1)
            return qip.view(qip.size()[0] * qip.size()[1], -1), (hi, hp)
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

