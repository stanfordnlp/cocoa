import copy

from model.parser import Parser
from model.dialogue_state import DialogueState

def parse_example(example, lexicon, templates=None):
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

        received_utterance = parsers[reading_agent].parse(event, states[reading_agent])
        if received_utterance:
            sent_utterance = copy.deepcopy(received_utterance)
            if sent_utterance.tokens:
                sent_utterance.template = parsers[writing_agent].extract_template(sent_utterance.tokens, states[writing_agent])

            if templates is not None:
                templates.add_template(sent_utterance, states[writing_agent])
            received_utterance.agent = example.agents[writing_agent]
            parsed_utterances.append(received_utterance)

            # Update states
            states[reading_agent].update(writing_agent, received_utterance)
            states[writing_agent].update(writing_agent, sent_utterance)
    return parsed_utterances
