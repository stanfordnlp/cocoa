import argparse
import copy

from cocoa.core.dataset import read_examples
from cocoa.model.manager import Manager

from core.event import Event
from core.scenario import Scenario
from core.lexicon import Lexicon
from model.parser import Parser
from model.dialogue_state import DialogueState
from model.generator import Templates, Generator

def parse_example(example, lexicon, templates):
    """Parse example and collect templates.
    """
    kbs = example.scenario.kbs
    parsers = [Parser(agent, kbs[agent], lexicon) for agent in (0, 1)]
    states = [DialogueState(agent, kbs[agent]) for agent in (0, 1)]
    # Add init utterance <start>
    parsed_utterances = [states[0].utterance[0], states[1].utterance[1]]
    for event in example.events:
        writing_agent = event.agent  # Speaking agent
        reading_agent = 1 - writing_agent
        #print event.agent

        received_utterance = parsers[reading_agent].parse(event, states[reading_agent])
        if received_utterance:
            sent_utterance = copy.deepcopy(received_utterance)
            if sent_utterance.tokens:
                sent_utterance.template = parsers[writing_agent].extract_template(sent_utterance.tokens, states[writing_agent])

            templates.add_template(sent_utterance, states[writing_agent])
            parsed_utterances.append(received_utterance)
            #print 'sent:', ' '.join(sent_utterance.template)
            #print 'received:', ' '.join(received_utterance.template)

            # Update states
            states[reading_agent].update(writing_agent, received_utterance)
            states[writing_agent].update(writing_agent, sent_utterance)
    return parsed_utterances

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--transcripts', nargs='*', help='JSON transcripts to extract templates')
    parser.add_argument('--max-examples', default=-1, type=int)
    parser.add_argument('--templates', help='Path to load templates')
    parser.add_argument('--templates-output', help='Path to save templates')
    parser.add_argument('--model', help='Path to load model')
    parser.add_argument('--model-output', help='Path to save the dialogue manager model')
    args = parser.parse_args()

    examples = read_examples(args.transcripts, args.max_examples, Scenario)
    parsed_dialogues = []
    templates = Templates()

    lexicon = Lexicon(['ball', 'hat', 'book'])
    for example in examples:
        utterances = parse_example(example, lexicon, templates)
        parsed_dialogues.append(utterances)

    templates.finalize()
    templates.save(args.templates_output)
    templates.dump(n=10)

    # Train n-gram model
    sequences = []
    for d in parsed_dialogues:
        sequences.append([u.lf.intent for u in d])

    manager = Manager.from_train(sequences)
    manager.save(args.model_output)

    generator = Generator(templates)
    action = manager.choose_action(None, context=('<start>', '<start>'))
    print action
    print generator.retrieve('<start>', context_tag='<start>', tag=action).template
