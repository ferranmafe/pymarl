import torch as th
from torch.distributions import Categorical
from .epsilon_schedules import DecayThenFlatSchedule
import math

REGISTRY = {}


class MultinomialActionSelector():

    def __init__(self, args):
        self.args = args

        self.schedule = DecayThenFlatSchedule(args.epsilon_start, args.epsilon_finish, args.epsilon_anneal_time,
                                              decay="linear")
        self.epsilon = self.schedule.eval(0)
        self.test_greedy = getattr(args, "test_greedy", True)

    def select_action(self, agent_inputs, avail_actions, t_env, test_mode=False):
        masked_policies = agent_inputs.clone()
        masked_policies[avail_actions == 0.0] = 0.0

        self.epsilon = self.schedule.eval(t_env)

        if test_mode and self.test_greedy:
            picked_actions = masked_policies.max(dim=2)[1]
        else:
            picked_actions = Categorical(masked_policies).sample().long()

        return picked_actions


REGISTRY["multinomial"] = MultinomialActionSelector


class EpsilonGreedyActionSelector():

    def __init__(self, args):
        self.args = args

        self.schedule = DecayThenFlatSchedule(args.epsilon_start, args.epsilon_finish, args.epsilon_anneal_time,
                                              decay="linear")
        self.epsilon = self.schedule.eval(0)

    def select_action(self, agent_inputs, avail_actions, t_env, test_mode=False):

        # Assuming agent_inputs is a batch of Q-Values for each agent bav
        self.epsilon = self.schedule.eval(t_env)

        if test_mode:
            # Greedy action selection only
            self.epsilon = 0.0

        # mask actions that are excluded from selection
        masked_q_values = agent_inputs.clone()

        avail_actions_base = th.zeros((1, 3, 900))
        if th.cuda.is_available():
            avail_actions_base.to(device='cuda:0')
        for i in range(3):
            avail_actions_aux = th.cartesian_prod(avail_actions[:, 2 * i, :].view(-1), avail_actions[:, 2 * i + 1, :].view(-1))
            avail_actions_base[:, i, :] = avail_actions_aux[:, 0].mul(avail_actions_aux[:, 1])
        masked_q_values[avail_actions_base == 0.0] = -float("inf")  # should never be selected!

        random_numbers = th.rand_like(agent_inputs[:, :, 0])
        pick_random = (random_numbers < self.epsilon).long()
        random_actions = Categorical(avail_actions_base.float()).sample().long()

        picked_actions = pick_random * random_actions + (1 - pick_random) * masked_q_values.max(dim=2)[1]
        # return picked_actions

        picked_actions_final = []
        for x in list(picked_actions.view(-1)):
            picked_actions_final.append(math.floor(int(x) / 30))
            picked_actions_final.append(int(x) % 30)

        a = []
        for i in range(len(picked_actions_final)):
            a.append(int(avail_actions[:, i, picked_actions_final[i]]))
        r = sum(a)

        return th.tensor(picked_actions_final).view(1, 6)


REGISTRY["epsilon_greedy"] = EpsilonGreedyActionSelector
