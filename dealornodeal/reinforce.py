"""
Takes two agent (Session) implementations, generates the dialogues,
and run REINFORCE.
"""

import argparse
import random
import json
import numpy as np

from onmt.Utils import use_gpu

from cocoa.core.util import read_json
from cocoa.core.schema import Schema
from cocoa.core.scenario_db import ScenarioDB
from cocoa.neural.loss import ReinforceLossCompute
import cocoa.options

from core.scenario import Scenario
from core.controller import Controller
from systems import get_system
from neural.rl_trainer import RLTrainer
from neural import build_optim
import options

def make_loss(opt, model, tgt_vocab):
    loss = ReinforceLossCompute(model.generator, tgt_vocab)
    if use_gpu(opt):
        loss.cuda()
    return loss


if __name__ == '__main__':
    parser = argparse.ArgumentParser(conflict_handler='resolve')
    parser.add_argument('--agents', help='What kind of agent to use. The first agent is always going to be updated and the second is fixed.', nargs='*', required=True)
    parser.add_argument('--agent-checkpoints', nargs='+', help='Directory to learned models')
    parser.add_argument('--random-seed', help='Random seed', type=int, default=1)
    parser.add_argument('--verbose', default=False, action='store_true', help='Whether or not to have verbose prints')
    parser.add_argument('--valid-scenarios-path', help='Output path for the validation scenarios')
    cocoa.options.add_scenario_arguments(parser)
    options.add_system_arguments(parser)
    options.add_rl_arguments(parser)
    options.add_model_arguments(parser)
    args = parser.parse_args()

    if args.random_seed:
        random.seed(args.random_seed)
        np.random.seed(args.random_seed)

    schema = Schema(args.schema_path)
    scenario_db = ScenarioDB.from_dict(schema, read_json(args.scenarios_path), Scenario)
    valid_scenario_db = ScenarioDB.from_dict(schema, read_json(args.valid_scenarios_path), Scenario)

    assert len(args.agent_checkpoints) <= len(args.agents)
    systems = [get_system(name, args, schema, False, args.agent_checkpoints[i]) for i, name in enumerate(args.agents)]

    rl_agent = 0
    system = systems[rl_agent]
    model = system.env.model
    loss = make_loss(args, model, system.mappings['tgt_vocab'])
    optim = build_optim(args, model, None)

    scenarios = {'train': scenario_db.scenarios_list, 'dev': valid_scenario_db.scenarios_list}
    trainer = RLTrainer(systems, scenarios, loss, optim, rl_agent, reward_func=args.reward)
    trainer.learn(args)
