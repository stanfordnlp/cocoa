from __future__ import division

import argparse
import random
import json
import numpy as np
import copy
import sys

import torch
import torch.nn as nn
from torch.autograd import Variable

from onmt.Trainer import Statistics as BaseStatistics

from core.controller import Controller
from utterance import UtteranceBuilder
from trainer import Trainer


class Statistics(BaseStatistics):
    def __init__(self, episode=0, loss=0, reward=0):
        self.episode = episode
        self.loss = loss
        self.reward = reward
        self.total_rewards = []

    def update(self, stat):
        self.loss += stat.loss
        self.reward += stat.reward
        self.episode += 1

    def mean_loss(self):
        return self.loss / self.episode

    def mean_reward(self):
        return self.reward / self.episode

    def output(self, episode):
        print ("Episode %2d; loss: %6.2f; reward: %6.2f;" %
              (episode,
               self.mean_loss(),
               self.mean_reward()))
        sys.stdout.flush()

# TODO: refactor
class RLTrainer(Trainer):
    pass
