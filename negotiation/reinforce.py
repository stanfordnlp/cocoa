"""
Takes two agent (Session) implementations, generates the dialogues,
and run REINFORCE.
"""

import argparse
import random
import json
import numpy as np

from cocoa.core.util import read_json
from cocoa.core.schema import Schema
from cocoa.core.scenario_db import ScenarioDB, add_scenario_arguments
from cocoa.core.dataset import add_dataset_arguments

from core.scenario import Scenario
from core.controller import Controller
from systems import add_system_arguments, get_system
from neural.rl import Reinforce, add_rl_arguments

if __name__ == '__main__':
    parser = argparse.ArgumentParser(conflict_handler='resolve')
    parser.add_argument('--agents', help='What kind of agent to use', nargs='*', required=True)
    group.add_argument('--checkpoint-files', nargs='*', required=True, help='Path to model .pt file')
    parser.add_argument('--random-seed', help='Random seed', type=int, default=1)
    parser.add_argument('--verbose', default=False, action='store_true', help='Whether or not to have verbose prints')
    add_scenario_arguments(parser)
    add_system_arguments(parser)
    add_rl_arguments(parser)
    args = parser.parse_args()

    if args.random_seed:
        random.seed(args.random_seed)
        np.random.seed(args.random_seed)

    schema = Schema(args.schema_path)
    scenario_db = ScenarioDB.from_dict(schema, read_json(args.scenarios_path), Scenario)

    # Match checkpoints with agents
    assert len(args.checkpoint_files) <= len(args.agents)

    agents = [get_system(name, args, schema,
        model_path=args.checkpoint_files[i] if i < len(args.checkpoint_files) else None)
        for i, name in enumerate(args.agents)]

    trainer = Reinforce(agents, scenario_db.scenarios_list)
    trainer.learn()
