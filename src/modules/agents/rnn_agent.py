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
            if self.args.n_agent_pairs > 0:
                self.fc11 = nn.Linear(input_shape * 2, args.rnn_hidden_dim * 2)
                self.rnn1 = nn.GRUCell(args.rnn_hidden_dim * 2, args.rnn_hidden_dim * 2)
                self.fc12 = nn.Linear(args.rnn_hidden_dim * 2, args.n_actions ** 2)
            if self.args.n_agent_pairs < int(self.args.n_agents / 2):
                self.fc21 = nn.Linear(input_shape, args.rnn_hidden_dim)
                self.rnn2 = nn.GRUCell(args.rnn_hidden_dim, args.rnn_hidden_dim)
                self.fc22 = nn.Linear(args.rnn_hidden_dim, args.n_actions)
        else:
            self.fc1 = nn.Linear(input_shape, args.rnn_hidden_dim)
            self.rnn = nn.GRUCell(args.rnn_hidden_dim, args.rnn_hidden_dim)
            self.fc2 = nn.Linear(args.rnn_hidden_dim, args.n_actions)

    def init_hidden(self):
        # make hidden states on same device as model
        if self.args.agent_type == "2_units_combined_output":
            if 0 < self.args.n_agent_pairs < int(self.args.n_agents / 2):
                self.fc21.weight.new(1, self.args.rnn_hidden_dim * 2).zero_()
                return self.fc11.weight.new(1, self.args.rnn_hidden_dim).zero_()
            elif self.args.n_agent_pairs < int(self.args.n_agents / 2):
                return self.fc21.weight.new(1, self.args.rnn_hidden_dim).zero_()
            else:
                return self.fc11.weight.new(1, self.args.rnn_hidden_dim).zero_()
        else:
            return self.fc1.weight.new(1, self.args.rnn_hidden_dim).zero_()

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
            hidden_state = hidden_state.reshape(self.args.n_agents, self.args.rnn_hidden_dim, -1)
            qp = None
            qi = None

            if self.args.n_agent_pairs > 0:
                n_pairs = self.args.n_agent_pairs
                if n_pairs > int(self.args.n_agents / 2):
                    n_pairs = int(self.args.n_agents / 2)

                pairs_inputs = inputs[:n_pairs * 2, :, :]
                pairs_inputs = pairs_inputs.view(-1, self.input_shape * 2)

                pairs_hidden_state = hidden_state[:n_pairs * 2, :, :]
                h_in = pairs_hidden_state.reshape(-1, self.args.rnn_hidden_dim * 2)

                x = F.relu(self.fc11(pairs_inputs))
                hp = self.rnn1(x, h_in)
                qp = self.fc12(hp)

            if self.args.n_agent_pairs < int(self.args.n_agents / 2):
                n_individuals = self.args.n_agents - 2 * self.args.n_agent_pairs
                individual_inputs = inputs[-n_individuals:, :, :]
                individual_inputs = individual_inputs.view(-1, self.input_shape)

                individual_hidden_state = hidden_state[-n_individuals:, :, :]
                h_in = individual_hidden_state.reshape(-1, self.args.rnn_hidden_dim)

                x = F.relu(self.fc21(individual_inputs))
                hi = self.rnn2(x, h_in)
                qi = self.fc22(hi)

            if qp is not None and qi is not None:
                return torch.cat((self.__decode_combined_output(qp), qi), 0), torch.cat((hp.view(hp.size()[0] * 2, -1), hi), 0)
            elif qp is not None:
                return self.__decode_combined_output(qp), hp.view(hp.size()[0] * 2, -1)
            else:
                return qi, hi

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

