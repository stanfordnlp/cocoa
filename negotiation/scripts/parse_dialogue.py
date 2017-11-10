import argparse

from cocoa.core.dataset import read_examples
from cocoa.core.util import write_pickle, read_pickle

from core.event import Event
from core.scenario import Scenario
from core.price_tracker import PriceTracker
from model.preprocess import Preprocessor
from model.parser import Parser
from model.dialogue_state import DialogueState

def parse_example(example, lexicon):
    kbs = example.scenario.kbs
    parsers = [Parser(agent, kbs[agent], lexicon) for agent in (0, 1)]
    states = [DialogueState(agent, kbs[agent]) for agent in (0, 1)]
    parsed_utterances = []
    for event in example.events:
        agent = event.agent  # Speaking agent
        parser = parsers[1 - agent]  # Receiving/Parsing agent
        state = states[1 - agent]
        utterance = parser.parse(event, state)
        if utterance:
            parsed_utterances.append(utterance)
            for state in states:
                state.update(agent, utterance)
    return parsed_utterances

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--transcripts', nargs='*', help='JSON transcripts to extract templates')
    parser.add_argument('--price-tracker-model')
    parser.add_argument('--max-examples', default=-1, type=int)
    args = parser.parse_args()

    price_tracker = PriceTracker(args.price_tracker_model)
    examples = read_examples(args.transcripts, args.max_examples, Scenario)
    parsed_dialogues = []
    for example in examples:
        if Preprocessor.skip_example(example):
            continue
        utterances = parse_example(example, price_tracker)
        parsed_dialogues.append(utterances)
    for d in parsed_dialogues[:10]:
        for u in d:
            print u
