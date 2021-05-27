import torch.nn as nn
import torch.nn.functional as F
import torch
import math


class RNNAgent(nn.Module):
    def __init__(self, input_shape, args):
        super(RNNAgent, self).__init__()
        self.args = args
        self.input_shape = input_shape

        self.fc1 = nn.Linear(input_shape * 2, args.rnn_hidden_dim * 2)
        self.rnn = nn.GRUCell(args.rnn_hidden_dim * 2, args.rnn_hidden_dim * 2)
        self.fc2 = nn.Linear(args.rnn_hidden_dim * 2, args.n_actions ** 2)

    def init_hidden(self):
        # make hidden states on same device as model
        return self.fc1.weight.new(1, self.args.rnn_hidden_dim * 2).zero_()

    def forward(self, inputs, hidden_state):
        inputs = torch.stack((inputs[:, :-1, :], inputs[:, 1:, :]), dim=1) \
            .view(-1, self.args.n_agents - 1, self.input_shape * 2).view(-1, self.input_shape * 2)

        x = F.relu(self.fc1(inputs))
        h_in = hidden_state.reshape(-1, self.args.rnn_hidden_dim * 2)
        hp = self.rnn(x, h_in)
        qp = self.fc2(hp)
        qp = self.__decode_combined_output(qp)
        qp = self.__get_max_values_from_decoded_output(qp)

        return qp, hp

    def __get_max_values_from_decoded_output(self, tensor):
        bs = math.floor(int(int(tensor.size()[0] / 2) / (self.args.n_agents - 1)))

        partial_tensor = tensor.view(-1, 2, self.args.n_agents - 1, tensor.size()[1])
        left_dimensions = partial_tensor[:, 0, 1:, :]
        right_dimensions = partial_tensor[:, 1, :-1, :]
        mixed_dimensions = torch.stack((left_dimensions, right_dimensions), dim=1)

        max_indices = torch.max(mixed_dimensions, dim=3)[0].max(dim=1)[1]

        mixed_dimensions_permuted = mixed_dimensions.permute(0, 2, 1, 3)
        mixed_dimensions_permuted = mixed_dimensions_permuted.reshape(-1, mixed_dimensions.size()[1], mixed_dimensions.size()[3])
        mixed_dimensions_permuted = mixed_dimensions_permuted[range(mixed_dimensions_permuted.size()[0]), max_indices.view(-1), :].view(bs, self.args.n_agents - 2, mixed_dimensions.size()[3])

        return torch.cat(
            (partial_tensor[:, 0, 0, :].reshape(bs, 1, -1), mixed_dimensions_permuted, partial_tensor[:, 1, -1, :].reshape(bs, 1, -1)),
            dim=1
        )

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

