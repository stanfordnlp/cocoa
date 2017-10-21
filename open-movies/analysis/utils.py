import numpy as np

from cocoa.core.dataset import Example
from core.scenario import Scenario

def filter_rejected_chats(transcripts):
    print("Tried to filter rejected in utils, but preprocessing is different, so modify")
    return transcripts
    # filtered = []
    # for chat in transcripts:
    #     ex = Example.from_dict(None, chat, Scenario)
    #     if not Preprocessor.skip_example(ex):
    #         filtered.append(chat)
    # return filtered

def filter_incomplete_chats(transcripts):
    print("Tried to filter incomplete in utils, but preprocessing is different, so modify")
    return transcripts
    # filtered = []
    # for chat in transcripts:
    #     winner = get_winner(chat)
    #     if winner is not None:
    #         filtered.append(chat)
    # return filtered

def get_split(transcript, agent=None):
    return transcript['data']['outcome'][agent]
