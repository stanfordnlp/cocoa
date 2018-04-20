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
from cocoa.core.scenario_db import ScenarioDB, add_scenario_arguments
from cocoa.core.dataset import add_dataset_arguments

from core.scenario import Scenario
from core.controller import Controller
from systems import add_system_arguments, get_system
#from neural.rl import Reinforce, add_rl_arguments
from neural.model_builder import add_model_arguments
from neural.rl_trainer import RLTrainer, add_rl_arguments
from neural.loss import ReinforceLossCompute
from neural import build_optim
#from systems.rl_system import RLSystem

def make_loss(opt, model, tgt_vocab):
    loss = ReinforceLossCompute(model.generator, tgt_vocab)
    if use_gpu(opt):
        loss.cuda()
    return loss


if __name__ == '__main__':
    parser = argparse.ArgumentParser(conflict_handler='resolve')
    parser.add_argument('--agents', help='What kind of agent to use. The first agent is always going to be updated and the second is fixed.', nargs='*', required=True)
    parser.add_argument('--random-seed', help='Random seed', type=int, default=1)
    parser.add_argument('--verbose', default=False, action='store_true', help='Whether or not to have verbose prints')
    add_scenario_arguments(parser)
    add_system_arguments(parser)
    add_rl_arguments(parser)
    add_model_arguments(parser)
    args = parser.parse_args()

    if args.random_seed:
        random.seed(args.random_seed)
        np.random.seed(args.random_seed)

    schema = Schema(args.schema_path)
    scenario_db = ScenarioDB.from_dict(schema, read_json(args.scenarios_path), Scenario)

    assert len(args.checkpoint_files) <= len(args.agents)
    systems = [get_system(name, args, schema, False, args.checkpoint_files[i]) for i, name in enumerate(args.agents)]
    #systems[0] = RLSystem(systems[0], args)

    rl_agent = 0
    system = systems[rl_agent]
    model = system.env.model
    # TODO: tgt_vocab
    loss = make_loss(args, model, system.mappings['vocab'])
    optim = build_optim(args, model, None)

    trainer = RLTrainer(systems, scenario_db.scenarios_list, loss, optim, rl_agent)
    trainer.learn(args)
