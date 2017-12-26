from core.tokenizer import tokenize
from collections import defaultdict
import random

def get_turns_per_agent(transcript):
    turns = {0: 0, 1: 0}
    for event in transcript["events"]:
        if event["action"] == "message":
            turns[event["agent"]] += 1

    return turns

def get_total_turns(transcript):
    n = 0
    for event in transcript["events"]:
        if event["action"] == "message":
            n += 1
    return n

def get_avg_time_taken(transcript):
    events = transcript["events"]
    start_time = float(events[0]["time"])
    end_time = float(events[-1]["time"])
    return end_time - start_time


def get_avg_tokens_per_agent(transcript):
    tokens = {0: 0., 1: 0.}
    utterances = {0: 0., 1: 0.}
    for event in transcript["events"]:
        if event["action"] == "message":
            msg_tokens = tokenize(event["data"])
            tokens[event["agent"]] += len(msg_tokens)
            utterances[event["agent"]] += 1

    if utterances[0] != 0:
        tokens[0] /= utterances[0]
    if utterances[1] != 0:
        tokens[1] /= utterances[1]

    return tokens


def get_total_tokens_per_agent(transcript):
    tokens = {0: 0., 1: 0.}
    for event in transcript["events"]:
        if event["action"] == "message":
            msg_tokens = tokenize(event["data"])
            tokens[event["agent"]] += len(msg_tokens)

    return tokens

def reject_transcript(transcript, agent_idx=None, min_tokens=40):
    total_tokens = get_total_tokens_per_agent(transcript)
    if agent_idx is not None:
        if total_tokens[agent_idx] < min_tokens:
            return True
        return False

    if total_tokens[0] < min_tokens or total_tokens[1] < min_tokens:
        return True
    return False

def intent_breakdown(parsed_dialogues):
    sequences = defaultdict(int)
    for d in parsed_dialogues:
        for u in d:
            sequences[u.lf.intent] += 1

    total = sum(sequences.values())
    for k, v in sequences.items():
        ratio = 100 * (float(v) / total)
        print("{0} intent occured {1} times which is {2:.2f}%".format(k, v, ratio) )

def sample_intents(dialogues, desired_intent, size=10):
    counter = 0
    for d in dialogues:
        for u in d:
            if (u.lf.intent == desired_intent) and random.random() < 0.3:
                print u.text
                counter += 1
            if counter > size:
                return counter

def safe_div(numerator, denominator):
    return numerator / (denominator + 1e-5)
