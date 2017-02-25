from argparse import ArgumentParser
import os
import json
from src.basic.util import read_json, write_pickle, read_pickle
from src.basic.scenario_db import ScenarioDB, add_scenario_arguments
from src.basic.schema import Schema
from src.model.preprocess import Preprocessor
from dataset_statistics import *
from src.basic.lexicon import Lexicon, add_lexicon_arguments


def add_statistics_arguments(parser):
    parser.add_argument('--stats-output', type=str, required=True, help='Name of file to write JSON statistics to')
    parser.add_argument('--text-output', type=str, help='Name of file to write sentences line by line (for training a LM)')
    parser.add_argument('--alpha-stats', action='store_true', help='Get statistics grouped by alpha values')
    parser.add_argument('--item-stats', action='store_true',
                        help='Get statistics grouped by number of items in scenarios')
    parser.add_argument('--plot-item-stats', type=str, default=None,
                        help='If provided, and if --item-stats is specified, plots the relationship between # of items '
                             'and strategy stats to the provided path.')
    parser.add_argument('--plot-alpha-stats', type=str, default=None,
                        help='If provided, plots the relationship between alpha values and'
                             'strategy stats to the provided path.')
    parser.add_argument('--lm', help='Path to LM (.arpa)')


def compute_statistics(args, lexicon, schema, scenario_db, transcripts):
    if not os.path.exists(os.path.dirname(args.stats_output)) and len(os.path.dirname(args.stats_output)) > 0:
        os.makedirs(os.path.dirname(args.stats_output))

    stats = {}
    statsfile = open(args.stats_output, 'w')
    stats["total"] = total_stats = get_total_statistics(transcripts, scenario_db)
    print "Aggregated total dataset statistics"
    print_group_stats(total_stats)

    # LM
    if args.lm:
        import kenlm
        lm = kenlm.Model(args.lm)
    else:
        lm = None

    # Speech acts
    preprocessor = Preprocessor(schema, lexicon, 'canonical', 'canonical', 'canonical')
    strategy_stats = analyze_strategy(transcripts, scenario_db, preprocessor, args.text_output, lm)
    print_strategy_stats(strategy_stats)
    stats["speech_act"] = {k[0]: v for k, v in strategy_stats['speech_act'].iteritems() if len(k) == 1}
    stats["kb_strategy"] = strategy_stats['kb_strategy']
    stats["dialog_stats"] = strategy_stats['dialog_stats']
    stats["lm_score"] = strategy_stats['lm_score']
    stats["correct"] = strategy_stats['correct']
    stats["entity_mention"] = strategy_stats['entity_mention']
    stats['multi_speech_act'] = strategy_stats['multi_speech_act']

    if args.plot_alpha_stats:
        plot_alpha_stats(strategy_stats["alpha_stats"], args.plot_alpha_stats)

    if args.plot_item_stats:
        plot_num_item_stats(strategy_stats["num_items_stats"], args.plot_item_stats)
    json.dump(stats, statsfile)
    statsfile.close()

if __name__ == "__main__":
    parser = ArgumentParser()
    add_scenario_arguments(parser)
    add_lexicon_arguments(parser)
    parser.add_argument('--transcripts', type=str, default='transcripts.json', help='Path to directory containing transcripts')
    add_statistics_arguments(parser)

    parsed_args = parser.parse_args()
    schema = Schema(parsed_args.schema_path)
    scenario_db = ScenarioDB.from_dict(schema, read_json(parsed_args.scenarios_path))
    transcripts = json.load(open(parsed_args.transcripts, 'r'))
    # transcripts = transcripts[:100]
    lexicon = Lexicon(schema, False, scenarios_json=parsed_args.scenarios_path, stop_words=parsed_args.stop_words)
    compute_statistics(parsed_args, lexicon, schema, scenario_db, transcripts)
