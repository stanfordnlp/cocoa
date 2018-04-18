import argparse
import random
import json
import numpy as np

from core.controller import Controller


def add_rl_arguments(parser):
    group = parser.add_argument_group('REINFORCE')
    parser.add_argument('--max-turns', default=100, type=int, help='Maximum number of turns')
    parser.add_argument('--num-dialogues', default=10000, type=int,
            help='Number of dialogues to generate/train')
    parser.add_argument('--discount_factor', default=0.95, type=float,
            help='Amount to discount the reward for each timestep when \
            calculating the value, usually written as gamma')
    parser.add_argument('--verbose', default=False, action='store_true',
            help='Whether or not to have verbose prints')


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
        sessions = [self.agents[0].new_session(0, kbs[0], rl=True),
                    self.agents[1].new_session(1, kbs[1], rl=False)]
        controller = Controller(scenario, sessions)
        return controller

    def get_reward(self, ex):
        outcome = get_outcome()
        reward = outcome['reward']
        offer = outcome['offer']
        # reward = reward if agree else 0
        # self.all_rewards.append(reward)
        # standardize the reward
        # r = (reward - np.mean(self.all_rewards)) / max(1e-4, np.std(self.all_rewards))



    def learn(self, opt):
        for i in xrange(opt.num_dialogues):
            controller = self._get_controller()
            ex = controller.simulate(max_turns, verbose=args.verbose)
            rewards = self.get_reward(ex)
            for session in controller.sessions:
                if hasattr(session, 'trainable') and session.trainable:
                    session.update(reward)
            # TODO: logging
