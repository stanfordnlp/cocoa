import argparse
import random
import json
import numpy as np
import copy

import torch
import torch.nn as nn
from torch.autograd import Variable

from cocoa.neural.rl_trainer import RLTrainer as BaseRLTrainer, \
        add_rl_arguments as base_add_rl_arguments

from core.controller import Controller
from neural.trainer import Trainer
from sessions.rl_session import RLSession
from utterance import UtteranceBuilder

def add_rl_arguments(parser):
    base_add_rl_arguments(parser)
    parser.add_argument('--reward', choices=['margin', 'length', 'fair'],
            help='Which reward function to use')


class RLTrainer(BaseRLTrainer):
    def __init__(self, agents, scenarios, train_loss, optim, training_agent=0, reward_func='margin'):
        super(RLTrainer, self).__init__(agents, scenarios, train_loss, optim, training_agent=training_agent)
        self.reward_func = reward_func

    def _margin_reward(self, example):
        rewards = {}
        targets = {}
        kbs = example.scenario.kbs
        for agent_id in (0, 1):
            kb = kbs[agent_id]
            targets[kb.role] = kb.target

        midpoint = (targets['seller'] + targets['buyer']) / 2.

        # No agreement
        if example.outcome['reward'] == 0 or example.outcome['offer'] is None:
            return {'seller': -0.5, 'buyer': -0.5}

        price = example.outcome['offer']['price']
        norm_factor = abs(midpoint - targets['seller'])
        rewards['seller'] = (price - midpoint) / norm_factor
        # Zero sum
        rewards['buyer'] = -1. * rewards['seller']
        return rewards

    def _length_reward(self, example):
        # Encourage long dialogue
        rewards = {}
        for role in ('buyer', 'seller'):
            rewards[role] = len(example.events)
        return rewards

    def _fair_reward(self, example):
        rewards = {}
        margin_rewards = self._margin_reward(example)
        for role in ('buyer', 'seller'):
            rewards[role] = -1. * abs(margin_rewards[role] - 0.)
        return rewards

    def get_reward(self, example):
        if self.reward_func == 'margin':
            return self._margin_reward(example)
        elif self.reward_func == 'fair':
            return self._fair_reward(example)
        elif self.reward_func == 'length':
            return self._length_reward(example)
        return reward
