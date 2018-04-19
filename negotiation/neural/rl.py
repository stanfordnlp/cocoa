import argparse
import random
import json
import numpy as np
import pdb

from core.controller import Controller

def add_rl_arguments(parser):
    group = parser.add_argument_group('Reinforce')
    group.add_argument('--max-turns', default=100, type=int, help='Maximum number of turns')
    group.add_argument('--num-dialogues', default=10000, type=int,
            help='Number of dialogues to generate/train')
    group.add_argument('--discount-factor', default=0.95, type=float,
            help='Amount to discount the reward for each timestep when \
            calculating the value, usually written as gamma')
    group.add_argument('--verbose', default=False, action='store_true',
            help='Whether or not to have verbose prints')

    group = parser.add_argument_group('Training')
    group.add_argument('--optim', default='sgd', help="""Optimization method.""",
                       choices=['sgd', 'adagrad', 'adadelta', 'adam'])
    group.add_argument('--report-every', type=int, default=5,
                       help="Print stats at this many batch intervals")
    group.add_argument('--epochs', type=int, default=14,
                       help='Number of training epochs')

    group.add_argument('--batch-size', type=int, default=64,
                       help='Maximum batch size for training')
    group.add_argument('--max-grad-norm', type=float, default=5,
                       help="""If the norm of the gradient vector exceeds this,
                       renormalize it to have the norm equal to max_grad_norm""")
    group.add_argument('--learning-rate', type=float, default=1.0,
                       help="""Starting learning rate. Recommended settings:
                       sgd = 1, adagrad = 0.1, adadelta = 1, adam = 0.001""")




class Reinforce(object):
    def __init__(self, agents, scenarios):
        self.agents = agents
        self.scenarios = scenarios

    def _get_controller(self):
        scenario = random.choice(self.scenarios)
        # Randomize because kb[0] is always seller
        if random.random() < 0.5:
            kbs = (scenario.kbs[0], scenario.kbs[1])
        else:
            kbs = (scenario.kbs[1], scenario.kbs[0])
        sessions = [self.agents[0].new_session(0, kbs[0], True),
                    self.agents[1].new_session(1, kbs[1], False)]
        return Controller(scenario, sessions)

    def get_reward(self, example):
        # reward = reward if agree else 0
        # self.all_rewards.append(reward)
        # standardize the reward
        # r = (reward - np.mean(self.all_rewards)) / max(1e-4, np.std(self.all_rewards))
        # pdb.set_trace()
        # check what is in agents, and what is in scenario

        # if not self.agent.has_deal(example.outcome):
        #     return 0   # there is no reward if there is no valid outcome
        # reward = example.outcome['reward']
        # offer = example.outcome['offer']

        agent_ids = {'seller': 0, 'buyer': 1}
        reward = {'seller': 0, 'buyer': 0}
        targets = {}
        for role in ('seller', 'buyer'):
            agent_id = agent_ids[role]
            targets[role] = self.scenarios[agent_id].kbs[agent_id].facts["personal"]["Target"]
        midpoint = (targets['seller'] + targets['buyer']) / 2.
        price = example.outcome['offer']['price']

        norm_factor = abs(midpoint - targets['seller'])
        reward['seller'] = (price - midpoint) / norm_factor
        # Zero sum
        reward['buyer'] = -1. * reward['seller']
        return reward

    def learn(self, args):
        for i in xrange(args.num_dialogues):
            controller = self._get_controller()
            example = controller.simulate(args.max_turns, verbose=args.verbose)
            rewards = self.get_reward(example)
            for session in controller.sessions:
                if hasattr(session, 'use_rl') and session.use_rl:
                    session.update(rewards[session.kb.role])
            # TODO: logging
