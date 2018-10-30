import argparse
import random
import json
import numpy as np
import copy
from collections import defaultdict

import torch
import torch.nn as nn
from torch.autograd import Variable

from cocoa.neural.rl_trainer import RLTrainer as BaseRLTrainer, \
        add_rl_arguments as base_add_rl_arguments

from core.controller import Controller
from neural.trainer import Trainer
from utterance import UtteranceBuilder


class RLTrainer(BaseRLTrainer):
    def __init__(self, agents, scenarios, train_loss, optim, training_agent=0, reward_func='margin'):
        super(RLTrainer, self).__init__(agents, scenarios, train_loss, optim, training_agent=training_agent)
        self.reward_func = reward_func

    def _is_valid_dialogue(self, example):
        special_actions = defaultdict(int)
        for event in example.events:
            if event.action in ('offer', 'quit', 'accept', 'reject'):
                special_actions[event.action] += 1
                if special_actions[event.action] > 1:
                    return False
                # Cannot accept or reject before offer
                if event.action in ('accept', 'reject') and special_actions['offer'] == 0:
                    return False
        return True

    def _is_agreed(self, example):
        if example.outcome['reward'] == 0 or example.outcome['offer'] is None:
            return False
        return True

    def _margin_reward(self, example):
        # No agreement
        if not self._is_agreed(example):
            print 'No agreement'
            return {'seller': -0.5, 'buyer': -0.5}

        rewards = {}
        targets = {}
        kbs = example.scenario.kbs
        for agent_id in (0, 1):
            kb = kbs[agent_id]
            targets[kb.role] = kb.target

        midpoint = (targets['seller'] + targets['buyer']) / 2.

        price = example.outcome['offer']['price']
        norm_factor = abs(midpoint - targets['seller'])
        rewards['seller'] = (price - midpoint) / norm_factor
        # Zero sum
        rewards['buyer'] = -1. * rewards['seller']
        return rewards

    def _length_reward(self, example):
        # No agreement
        if not self._is_agreed(example):
            print 'No agreement'
            return {'seller': -0.5, 'buyer': -0.5}

        # Encourage long dialogue
        rewards = {}
        for role in ('buyer', 'seller'):
            rewards[role] = len(example.events) / 10.
        return rewards

    def _fair_reward(self, example):
        # No agreement
        if not self._is_agreed(example):
            print 'No agreement'
            return {'seller': -0.5, 'buyer': -0.5}

        rewards = {}
        margin_rewards = self._margin_reward(example)
        for role in ('buyer', 'seller'):
            rewards[role] = -1. * abs(margin_rewards[role] - 0.) + 2.
        return rewards

    def get_reward(self, example, session):
        if not self._is_valid_dialogue(example):
            print 'Invalid'
            rewards = {'seller': -1., 'buyer': -1.}
        if self.reward_func == 'margin':
            rewards = self._margin_reward(example)
        elif self.reward_func == 'fair':
            rewards = self._fair_reward(example)
        elif self.reward_func == 'length':
            rewards = self._length_reward(example)
        reward = rewards[session.kb.role]
        return reward
