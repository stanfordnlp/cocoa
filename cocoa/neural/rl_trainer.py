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
from sessions.rl_session import RLSession
from utterance import UtteranceBuilder
from trainer import Trainer

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
    group.add_argument('--train-from', default='', type=str,
                       help="""If training from a checkpoint then this is the
                       path to the pretrained model's state_dict.""")
    group.add_argument('--model-path', default='data/checkpoints',
                       help="""Which file the model checkpoints will be saved""")
    group.add_argument('--model-filename', default='model',
                       help="""Model filename (the model will be saved as
                       <filename>_acc_ppl_e.pt where ACC is accuracy, PPL is
                       the perplexity and E is the epoch""")


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
        #self.rewards.append(stat.reward)

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

class RLTrainer(Trainer):
    def __init__(self, agents, scenarios, train_loss, optim, training_agent=0):
        self.agents = agents
        self.scenarios = scenarios

        self.training_agent = training_agent
        self.model = agents[training_agent].env.model
        #self.models = [agent.env.model for agent in agents]
        self.train_loss = train_loss
        self.optim = optim
        self.cuda = False

        self.best_valid_reward = None

        # Set model in training mode.
        #for model in self.models:
        #    model.train()
        #self.model.train()

        self.all_rewards = [[], []]

    def update(self, batch_iter, reward, model, discount=0.95):
        model.train()
        model.generator.train()

        nll = []
        # batch_iter gives a dialogue
        dec_state = None
        for batch in batch_iter:
            if not model.stateful:
                dec_state = None
            enc_state = dec_state.hidden if dec_state is not None else None

            outputs, _, dec_state = self._run_batch(batch, None, enc_state)  # (seq_len, batch_size, rnn_size)
            loss, _ = self.train_loss.compute_loss(batch.targets, outputs)  # (seq_len, batch_size)
            nll.append(loss)

            # Don't backprop fully.
            if dec_state is not None:
                dec_state.detach()

        nll = torch.cat(nll)  # (total_seq_len, batch_size)

        rewards = [Variable(torch.zeros(1, 1).fill_(reward))]
        for i in xrange(1, nll.size(0)):
            rewards.append(rewards[-1] * discount)
        rewards = rewards[::-1]
        rewards = torch.cat(rewards)

        loss = nll.dot(rewards)
        model.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm(model.parameters(), 1.)
        self.optim.step()

        #model.eval()
        #model.generator.eval()

        # TODO
        #total_stats.update(batch_stats)
        #report_stats.update(batch_stats)

    def _get_scenario(self, scenario_id=None, split='train'):
        scenarios = self.scenarios[split]
        if scenario_id is None:
            scenario = random.choice(scenarios)
        else:
            scenario = scenarios[scenario_id % len(scenarios)]
        return scenario

    def _get_controller(self, scenario, split='train'):
        # Randomize because kb[0] is always seller
        if random.random() < 0.5:
            kbs = (scenario.kbs[0], scenario.kbs[1])
        else:
            kbs = (scenario.kbs[1], scenario.kbs[0])
        sessions = [self.agents[0].new_session(0, kbs[0]),
                    self.agents[1].new_session(1, kbs[1])]
        return Controller(scenario, sessions)

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
        raise NotImplementedError

    def validate(self, args):
        split = 'dev'
        self.model.eval()
        total_stats = Statistics()
        print '='*20, 'VALIDATION', '='*20
        for scenario in self.scenarios[split][:200]:
            controller = self._get_controller(scenario, split=split)
            example = controller.simulate(args.max_turns, verbose=args.verbose)
            rewards = self.get_reward(example)
            session = controller.sessions[self.training_agent]
            reward = rewards[session.kb.role]
            stats = Statistics(reward=reward)
            total_stats.update(stats)
        print '='*20, 'END VALIDATION', '='*20
        self.model.train()
        return total_stats

    def save_best_checkpoint(self, checkpoint, opt, valid_stats):
        if self.best_valid_reward is None or valid_stats.mean_reward() > self.best_valid_reward:
            self.best_valid_reward = valid_stats.mean_reward()
            path = '{root}/{model}_best.pt'.format(
                        root=opt.model_path,
                        model=opt.model_filename)

            print 'Save best checkpoint {path}'.format(path=path)
            torch.save(checkpoint, path)

    def checkpoint_path(self, episode, opt, stats):
        path = '{root}/{model}_reward{reward:.2f}_e{episode:d}.pt'.format(
                    root=opt.model_path,
                    model=opt.model_filename,
                    reward=stats.mean_reward(),
                    episode=episode)
        return path

    def learn(self, args):
        for i in xrange(args.num_dialogues):
            # Rollout
            scenario = self._get_scenario()
            controller = self._get_controller(scenario, split='train')
            example = controller.simulate(args.max_turns, verbose=args.verbose)

            #if i % 100 == 0:
            #    self.agents[1].env.model.load_state_dict(self.model.state_dict())

            for session_id, session in enumerate(controller.sessions):
                # Only train one agent
                if session_id != self.training_agent:
                    continue

                # Compute reward
                rewards = self.get_reward(example)
                reward = rewards[session.kb.role]
                # Standardize the reward
                all_rewards = self.all_rewards[session_id]
                all_rewards.append(reward)
                print 'step:', i
                print 'reward:', reward
                reward = (reward - np.mean(all_rewards)) / max(1e-4, np.std(all_rewards))
                print 'scaled reward:', reward
                print 'mean reward:', np.mean(all_rewards)

                batch_iter = session.iter_batches()
                T = batch_iter.next()
                self.update(batch_iter, reward, self.model, discount=args.discount_factor)

            if i > 0 and i % 100 == 0:
                valid_stats = self.validate(args)
                self.drop_checkpoint(args, i, valid_stats, model_opt=self.agents[self.training_agent].env.model_args)
