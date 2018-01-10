import argparse
from collections import defaultdict

from cocoa.core.dataset import read_examples
from cocoa.model.manager import Manager
from cocoa.model.dialogue_parser import parse_example

from core.event import Event
from core.scenario import Scenario
from core.lexicon import Lexicon
from model.parser import Parser
from model.dialogue_state import DialogueState
from model.generator import Templates, Generator

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--lexicon', help='Path to pickled lexicon')
    parser.add_argument('--transcripts', nargs='*', help='JSON transcripts to extract templates')
    parser.add_argument('--max-examples', default=-1, type=int)
    parser.add_argument('--templates', help='Path to load templates')
    parser.add_argument('--reviews', help='Path to load templates')
    parser.add_argument('--templates-output', help='Path to save templates')
    parser.add_argument('--model', help='Path to load model')
    parser.add_argument('--model-output', help='Path to save the dialogue manager model')
    args = parser.parse_args()

    examples = read_examples(args.transcripts, args.max_examples, Scenario)
    parsed_dialogues = []
    templates = Templates()

    lexicon = Lexicon.from_pickle(args.lexicon)

    for example in examples:
        utterances = parse_example(example, lexicon, templates)
        parsed_dialogues.append(utterances)

    # Train n-gram model
    sequences = []
    for d in parsed_dialogues:
        sequences.append([u.lf.intent for u in d])
    manager = Manager.from_train(sequences)
    manager.save(args.model_output)

    if args.reviews:
        print 'read reviews from', args.reviews
        templates.read_reviews(args.reviews)

    templates.finalize()
    templates.save(args.templates_output)
    templates.dump(n=10)

    # Test model and generator
    generator = Generator(templates)
    actions = ['<start>', '<start>']
    for i in xrange(10):
        context = tuple(actions[-2:])
        prev_action = manager.choose_action(None, context=context)
        actions.append(prev_action)
    print 'Bigram model:'
    print actions
    #action = manager.choose_action(None, context=('<start>', '<start>'))
    #print action
    #print 'retrieve:'
    #print generator.retrieve('<start>', context_tag='<start>', tag=action).template

    #sequences = defaultdict(int)
    #full_sequences = defaultdict(int)
    #for u in parsed_dialogues:
    #    sequences[u.lf.intent] += 1
    #    full_sequences[u.lf.full_intent] += 1

    #total = sum(sequences.values())
    #for k, v in sequences.items():
    #    ratio = 100 * (float(v) / total)
    #    print("{0} intent occured {1} times which is {2:.2f}%".format(k, v, ratio) )
    #print "----------------"
    #full_total = sum(full_sequences.values())
    #for k, v in full_sequences.items():
    #    full_ratio = 100 * (float(v) / full_total)
    #    print("{0} occured {1} times which is {2:.2f}%".format(k, v, full_ratio) )

