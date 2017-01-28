from src.basic.event import Event

__author__ = 'anushabala'
from src.model.preprocess import tokenize, markers
import json


def preprocess_events(events, agent):
    """
    Tokenize every message and pad with EOS tags as needed. Selections are converted to a list of strings.
    :param events:
    :param agent:
    :return:
    """
    messages = [preprocess_event(e) for e in events]

    if len(messages) > 0 and agent == messages[0][0]:
        events.insert(0, Event(action='message', time=0, agent=1-agent, data="%s %s" % (markers.GO, markers.EOS)))
        messages.insert(0, (1-agent, [markers.GO, markers.EOS]))

    return messages


def preprocess_event(event):
    if event.action == 'message':
        utterance = tokenize(event.data)
        utterance.append(markers.EOS)
        return event.agent, utterance
    elif event.action == 'select':
        msg = [markers.SELECT, event.data, markers.EOS]
        return event.agent, msg


def dialog_to_message_sequence(tagged_dialog):
    msg_sequence = []
    for _, tagged_tokens in tagged_dialog:
        msg_sequence.extend(tagged_tokens)
    return msg_sequence
