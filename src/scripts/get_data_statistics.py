from argparse import ArgumentParser
import os
import json
from src.basic.util import read_json
from src.basic.scenario_db import ScenarioDB, add_scenario_arguments
from src.basic.schema import Schema
from src.model.preprocess import Preprocessor
from dataset_statistics import *
from src.basic.lexicon import Lexicon

if __name__ == "__main__":
    parser = ArgumentParser()
    add_scenario_arguments(parser)
    parser.add_argument('--transcripts', type=str, default='transcripts.json', help='Path to directory containing transcripts')
    parser.add_argument('--stats-output', type=str, required=True, help='Name of file to write JSON statistics to')
    parser.add_argument('--alpha-stats', action='store_true', help='Get statistics grouped by alpha values')
    parser.add_argument('--item-stats', action='store_true',
                        help='Get statistics grouped by number of items in scenarios')
    parser.add_argument('--plot-item-stats', type=str, default=None,
                        help='If provided, and if --item-stats is specified, plots the relationship between # of items '
                             'and various stats to the provided path.')

    args = parser.parse_args()
    schema = Schema(args.schema_path)
    scenario_db = ScenarioDB.from_dict(schema, read_json(args.scenarios_path))
    transcripts = json.load(open(args.transcripts, 'r'))
    if not os.path.exists(os.path.dirname(args.stats_output)) and len(os.path.dirname(args.stats_output)) > 0:
        os.makedirs(os.path.dirname(args.stats_output))

    stats = {}
    statsfile = open(args.stats_output, 'w')
    stats["total"] = total_stats = get_total_statistics(transcripts, scenario_db)
    print "Aggregated dataset statistics"
    print_group_stats(total_stats)

    if args.alpha_stats:
        print "-----------------------------------"
        print "-----------------------------------"
        print "Getting statistics grouped by alpha values...."
        stats["by_alpha"] = stats_by_alpha = get_statistics_by_alpha(transcripts, scenario_db)
        print_stats(stats_by_alpha, stats_type="alphas")
    if args.item_stats:
        print "-----------------------------------"
        print "-----------------------------------"
        print "Getting statistics grouped by number of items..."
        stats["by_num_items"] = stats_by_num_items = get_statistics_by_items(transcripts, scenario_db)
        print_stats(stats_by_num_items, stats_type="number of items")

        if args.plot_item_stats is not None:
            plot_num_items_stats(stats_by_num_items, args.plot_item_stats)

    # Speech acts
    lexicon = Lexicon(schema, False)
    preprocessor = Preprocessor(schema, lexicon, 'canonical', 'canonical', 'canonical', False)
    strategy_stats = analyze_strategy(transcripts, scenario_db, preprocessor)
    print_strategy_stats(strategy_stats)
    stats["speech_act"] = {k[0]: v for k, v in strategy_stats['speech_act'].iteritems() if len(k) == 1}
    stats["kb_strategy"] = strategy_stats['kb_strategy']

    json.dump(stats, statsfile)
    statsfile.close()
