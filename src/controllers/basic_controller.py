from modules.agents import REGISTRY as agent_REGISTRY
from modules.agents.rnn_agent_ind import RNNAgentInd
from modules.agents.rnn_agent_pairs import RNNAgentPairs
from components.action_selectors import REGISTRY as action_REGISTRY
import torch as th


# This multi-agent controller shares parameters between agents
class BasicMAC:
    def __init__(self, scheme, groups, args):
        self.n_agents = args.n_agents
        self.args = args
        input_shape = self._get_input_shape(scheme)
        self._build_agents(input_shape)
        self.agent_output_type = args.agent_output_type

        self.action_selector = action_REGISTRY[args.action_selector](args)

        self.hidden_states_ind = None
        self.hidden_states_pairs = None

    def select_actions(self, ep_batch, t_ep, t_env, bs=slice(None), test_mode=False):
        # Only select actions for the selected batch elements in bs
        avail_actions = ep_batch["avail_actions"][:, t_ep]
        agent_outputs = self.forward(ep_batch, t_ep, test_mode=test_mode)
        chosen_actions = self.action_selector.select_action(agent_outputs[bs], avail_actions[bs], t_env, test_mode=test_mode)
        return chosen_actions

    def forward(self, ep_batch, t, test_mode=False):
        agent_inputs_ind, agent_inputs_pairs = self._build_inputs(ep_batch, t)
        avail_actions = ep_batch["avail_actions"][:, t]
        agent_outs_ind, self.hidden_states_ind = self.agent_ind(agent_inputs_ind, self.hidden_states_ind)
        agent_outs_pairs, self.hidden_states_pairs = self.agent_pairs(agent_inputs_pairs, self.hidden_states_pairs)

        agent_outs = th.cat((agent_outs_ind, agent_outs_pairs), dim=0)
        agent_outs = agent_outs.view(ep_batch.batch_size * self.n_agents, -1)

        # Softmax the agent outputs if they're policy logits
        if self.agent_output_type == "pi_logits":

            if getattr(self.args, "mask_before_softmax", True):
                # Make the logits for unavailable actions very negative to minimise their affect on the softmax
                reshaped_avail_actions = avail_actions.reshape(ep_batch.batch_size * self.n_agents, -1)
                agent_outs[reshaped_avail_actions == 0] = -1e10

            agent_outs = th.nn.functional.softmax(agent_outs, dim=-1)
            if not test_mode:
                # Epsilon floor
                epsilon_action_num = agent_outs.size(-1)
                if getattr(self.args, "mask_before_softmax", True):
                    # With probability epsilon, we will pick an available action uniformly
                    epsilon_action_num = reshaped_avail_actions.sum(dim=1, keepdim=True).float()

                agent_outs = ((1 - self.action_selector.epsilon) * agent_outs
                               + th.ones_like(agent_outs) * self.action_selector.epsilon/epsilon_action_num)

                if getattr(self.args, "mask_before_softmax", True):
                    # Zero out the unavailable actions
                    agent_outs[reshaped_avail_actions == 0] = 0.0

        return agent_outs.view(ep_batch.batch_size, self.n_agents, -1)

    def init_hidden(self, batch_size):
        self.hidden_states_ind = self.agent_ind.init_hidden().unsqueeze(0).expand(batch_size, self.n_agents - 2, -1)
        self.hidden_states_pairs = self.agent_pairs.init_hidden().unsqueeze(0).expand(batch_size, 1, -1)

    def parameters(self):
        return self.agent_ind.parameters(), self.agent_pairs.parameters()

    def load_state(self, other_mac):
        self.agent_ind.load_state_dict(other_mac.agent_ind.state_dict())
        self.agent_pairs.load_state_dict(other_mac.agent_pairs.state_dict())

    def cuda(self):
        self.agent_ind.cuda()
        self.agent_pairs.cuda()

    def save_models(self, path):
        th.save(self.agent_ind.state_dict(), "{}/agent_ind.th".format(path))
        th.save(self.agent_pairs.state_dict(), "{}/agent_ind.th".format(path))

    def load_models(self, path):
        self.agent_ind.load_state_dict(
            th.load("{}/agent_ind.th".format(path), map_location=lambda storage, loc: storage))
        self.agent_pairs.load_state_dict(
            th.load("{}/agent_pairs.th".format(path), map_location=lambda storage, loc: storage))

    def _build_agents(self, input_shape):
        self.agent_ind = RNNAgentInd(input_shape, self.args)
        self.agent_pairs = RNNAgentPairs(input_shape, self.args)

    def _build_inputs(self, batch, t):
        # Assumes homogenous agents with flat observations.
        # Other MACs might want to e.g. delegate building inputs to each agent
        bs = batch.batch_size
        inputs = []
        inputs.append(batch["obs"][:, t])  # b1av
        if self.args.obs_last_action:
            if t == 0:
                inputs.append(th.zeros_like(batch["actions_onehot"][:, t]))
            else:
                inputs.append(batch["actions_onehot"][:, t-1])
        if self.args.obs_agent_id:
            inputs.append(th.eye(self.n_agents, device=batch.device).unsqueeze(0).expand(bs, -1, -1))

        inputs = th.cat([x.reshape(bs, self.n_agents, -1) for x in inputs], dim=2)
        inputs_ind = inputs[:, :self.n_agents - 2, :]
        inputs_pairs = inputs[:, self.n_agents - 2:, :]
        return inputs_ind, inputs_pairs

    def _get_input_shape(self, scheme):
        input_shape = scheme["obs"]["vshape"]
        if self.args.obs_last_action:
            input_shape += scheme["actions_onehot"]["vshape"][0]
        if self.args.obs_agent_id:
            input_shape += self.n_agents

        return input_shape
