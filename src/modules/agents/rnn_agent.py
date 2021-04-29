import torch.nn as nn
import torch.nn.functional as F
import torch

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
        else:
            self.fc1 = nn.Linear(input_shape, args.rnn_hidden_dim)
            self.rnn = nn.GRUCell(args.rnn_hidden_dim, args.rnn_hidden_dim)
            self.fc2 = nn.Linear(args.rnn_hidden_dim, args.n_actions)

    def init_hidden(self):
        # make hidden states on same device as model
        return self.fc1.weight.new(1, self.args.rnn_hidden_dim).zero_()

    def forward(self, inputs, hidden_state):
        if self.args.agent_type == '2_units':
            x = F.relu(self.fc1(inputs.view(-1, self.input_shape * 2)))
            h_in = hidden_state.reshape(-1, self.args.rnn_hidden_dim * 2)
            h = self.rnn(x, h_in)
            q1 = self.fc21(h)
            q2 = self.fc22(h)
            return self.__combine_torch_tensors(q1, q2), h
        else:
            x = F.relu(self.fc1(inputs))
            h_in = hidden_state.reshape(-1, self.args.rnn_hidden_dim)
            h = self.rnn(x, h_in)
            q = self.fc2(h)
            return q, h

    @staticmethod
    def __combine_torch_tensors(tensor1, tensor2):
        output = torch.empty([tensor1.size()[0] * 2, tensor1.size()[1]])
        for i in range(tensor1.size()[0]):
            for j in range(tensor1.size()[1]):
                output[i * 2][j] = tensor1[i][j]
                output[i * 2 + 1][j] = tensor2[i][j]
        return output
