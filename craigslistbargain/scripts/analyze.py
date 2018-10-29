import argparse

from cocoa.core.schema import Schema
from cocoa.core.dataset import read_examples
from cocoa.model.dialogue_parser import parse_example
from cocoa.analysis.analyzer import Analyzer

from core.scenario import Scenario
from core.price_tracker import PriceTracker, add_price_tracker_arguments
from model.generator import Templates
from model.manager import Manager

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--transcripts', nargs='*', help='JSON transcripts to extract templates')
    parser.add_argument('--max-examples', default=-1, type=int)
    parser.add_argument('--templates', help='Path to load templates')
    parser.add_argument('--policy', help='Path to load model')
    parser.add_argument('--schema-path', help='Path to schema')
    parser.add_argument('--agent', help='Only consider examples with the given type of agent')
    add_price_tracker_arguments(parser)
    args = parser.parse_args()

    lexicon = PriceTracker(args.price_tracker_model)
    #templates = Templates.from_pickle(args.templates)
    templates = Templates()
    manager = Manager.from_pickle(args.policy)
    analyzer = Analyzer(lexicon)

    # TODO: skip examples
    examples = read_examples(args.transcripts, args.max_examples, Scenario)
    agent = args.agent
    if agent is not None:
        examples = [e for e in examples if agent in e.agents.values()]
    analyzer.example_stats(examples, agent=agent)
    #import sys; sys.exit()

    parsed_dialogues = []
    for example in examples:
        utterances = parse_example(example, lexicon, templates)
        parsed_dialogues.append(utterances)

    analyzer.parser_stats(parsed_dialogues, agent=agent)
    #analyzer.manager_stats(manager)
