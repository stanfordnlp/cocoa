from src.basic.lexicon import Lexicon
from src.basic.util import read_json
from src.model.preprocess import Preprocessor

__author__ = 'anushabala'

from argparse import ArgumentParser
from src.basic.scenario_db import ScenarioDB, add_scenario_arguments
from src.basic.schema import Schema
import json
from get_data_statistics import add_statistics_arguments, compute_statistics
from visualize_data import add_visualization_arguments, visualize_transcripts


if __name__ == "__main__":
    parser = ArgumentParser()
    add_scenario_arguments(parser)
    add_statistics_arguments(parser)
    add_visualization_arguments(parser)
    parser.add_argument('--transcripts', type=str, default='transcripts.json', help='Path to directory containing transcripts')
    parser.add_argument('--domain', type=str, choices=['MutualFriends', 'Matchmaking'])

    args = parser.parse_args()
    schema = Schema(args.schema_path, args.domain)
    scenario_db = ScenarioDB.from_dict(schema, read_json(args.scenarios_path))
    transcripts = json.load(open(args.transcripts, 'r'))
    lexicon = Lexicon(schema, False, scenarios_json=args.scenarios_path)

    visualize_transcripts(args, scenario_db, transcripts)
    compute_statistics(args, lexicon, schema, scenario_db, transcripts)
