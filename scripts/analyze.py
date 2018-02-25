import argparse

from cocoa.core.dataset import read_examples
from cocoa.model.dialogue_parser import parse_example
from cocoa.analysis.analyzer import Analyzer

from core.scenario import Scenario
from core.lexicon import Lexicon
from model.generator import Templates
from model.manager import Manager

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--lexicon', help='Path to pickled lexicon')
    parser.add_argument('--transcripts', nargs='*', help='JSON transcripts to extract templates')
    parser.add_argument('--max-examples', default=-1, type=int)
    parser.add_argument('--templates', help='Path to load templates')
    parser.add_argument('--policy', help='Path to load model')
    args = parser.parse_args()

    lexicon = Lexicon.from_pickle(args.lexicon)
    #templates = Templates.from_pickle(args.templates)
    templates = Templates()
    manager = Manager.from_pickle(args.policy)
    analyzer = Analyzer(lexicon)

    examples = read_examples(args.transcripts, args.max_examples, Scenario)

    parsed_dialogues = []
    for example in examples:
        utterances = parse_example(example, lexicon, templates)
        parsed_dialogues.append(utterances)

    analyzer.example_stats(examples)
    analyzer.parser_stats(parsed_dialogues)
    #analyzer.manager_stats(manager)
