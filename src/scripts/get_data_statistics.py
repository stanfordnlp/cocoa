from argparse import ArgumentParser
import os
import json
from src.basic.util import read_json, write_pickle
from src.basic.scenario_db import ScenarioDB, add_scenario_arguments
from src.basic.schema import Schema
from src.model.preprocess import Preprocessor
from dataset_statistics import *
from src.basic.lexicon import Lexicon, add_lexicon_arguments


def add_statistics_arguments(parser):
    parser.add_argument('--stats-output', type=str, required=True, help='Name of file to write JSON statistics to')
    parser.add_argument('--text-output', type=str, help='Name of file to write sentences line by line')
    parser.add_argument('--alpha-stats', action='store_true', help='Get statistics grouped by alpha values')
    parser.add_argument('--item-stats', action='store_true',
                        help='Get statistics grouped by number of items in scenarios')
    parser.add_argument('--plot-item-stats', type=str, default=None,
                        help='If provided, and if --item-stats is specified, plots the relationship between # of items '
                             'and various stats to the provided path.')
    parser.add_argument('--lm', help='Path to LM (.arpa)')


def compute_statistics(args, lexicon, schema, scenario_db, transcripts):
    if not os.path.exists(os.path.dirname(args.stats_output)) and len(os.path.dirname(args.stats_output)) > 0:
        os.makedirs(os.path.dirname(args.stats_output))

    stats = {}
    statsfile = open(args.stats_output, 'w')
    stats["total"] = total_stats = get_total_statistics(transcripts, scenario_db)
    print "Aggregated total dataset statistics"
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

    # LM
    if args.lm:
        import kenlm
        lm = kenlm.Model(args.lm)
    else:
        lm = None

    # Speech acts
    preprocessor = Preprocessor(schema, lexicon, 'canonical', 'canonical', 'canonical', False)
    strategy_stats = analyze_strategy(transcripts, scenario_db, preprocessor, args.text_output, lm)
    print_strategy_stats(strategy_stats)
    stats["speech_act"] = {k[0]: v for k, v in strategy_stats['speech_act'].iteritems() if len(k) == 1}
    stats["kb_strategy"] = strategy_stats['kb_strategy']
    stats["dialog_stats"] = strategy_stats['dialog_stats']
    stats["lm_score"] = strategy_stats['lm_score']
    stats["correct"] = strategy_stats['correct']
    #stats["ngram_counts"] = strategy_stats['ngram_counts']
    #stats["utterance_counts"] = strategy_stats['utterance_counts']
    outdir = os.path.dirname(args.stats_output)
    write_pickle(strategy_stats['ngram_counts'], os.path.join(outdir, 'ngram_counts.pkl'))
    write_pickle(strategy_stats['utterance_counts'], os.path.join(outdir, 'utterance_counts.pkl'))

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
    lexicon = Lexicon(schema, False, scenarios_json=parsed_args.scenarios_path, stop_words=parsed_args.stop_words)
    compute_statistics(parsed_args, lexicon, schema, scenario_db, transcripts)
