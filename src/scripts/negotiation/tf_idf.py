from collections import defaultdict
import os

__author__ = 'anushabala'
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import matplotlib.pyplot as plt
from src.turk.accept_negotiation_hits import is_chat_valid, is_partial_chat
from utils import *



def group_text_by_winner(transcripts, agent_type=None):
    grouped_chats = defaultdict(str)
    for t in transcripts:
        if (not is_chat_valid(t, 0) and not is_partial_chat(t, 0)) \
                or (not is_chat_valid(t, 1) and not is_partial_chat(t, 1)):
            continue
        winner = get_winner(t)
        if winner is None:
            continue
        chat_0 = json_to_text(t, agent_type, role=None, agent_id=0)
        chat_1 = json_to_text(t, agent_type, role=None, agent_id=1)
        if winner == -1:
            grouped_chats[WINNER] += " {:s}".format(chat_0)
            grouped_chats[WINNER] += " {:s}".format(chat_1)
            grouped_chats[LOSER] += " {:s}".format(chat_0)
            grouped_chats[LOSER] += " {:s}".format(chat_1)
        elif winner == 0:
            grouped_chats[WINNER] += " {:s}".format(chat_0)
            grouped_chats[LOSER] += " {:s}".format(chat_1)
        elif winner == 1:
            grouped_chats[WINNER] += " {:s}".format(chat_1)
            grouped_chats[LOSER] += " {:s}".format(chat_0)

    return grouped_chats


def group_text_by_category(transcripts, agent_type=None):
    grouped_chats = defaultdict(str)
    for t in transcripts:
        if (not is_chat_valid(t, 0) and not is_partial_chat(t, 0)) \
                or (not is_chat_valid(t, 1) and not is_partial_chat(t, 1)):
            continue
        chat = json_to_text(t, agent_type)
        category = t["scenario"]["category"]
        grouped_chats[category] += " {:s}".format(chat)
    return grouped_chats


def group_text_by_role(transcripts, agent_type=None):
    grouped_chats = defaultdict(str)
    for t in transcripts:
        if (not is_chat_valid(t, 0) and not is_partial_chat(t, 0)) \
                or (not is_chat_valid(t, 1) and not is_partial_chat(t, 1)):
            continue
        for r in ROLES:
            chat = json_to_text(t, agent_type, r)
            grouped_chats[r] += " {:s}".format(chat)
    return grouped_chats


def json_to_text(transcript, agent_type=None, role=None, agent_id=-1):
    """
    Convert a JSON chat transcript to a single string containing all the messages in the chat.
    :param json_chat:
    :return:
    """
    chat = []
    agents = transcript["agents"]
    roles = {0: transcript["scenario"]["kbs"][0]["personal"]["Role"],
             1: transcript["scenario"]["kbs"][1]["personal"]["Role"]}
    for event in transcript["events"]:
        if agent_type is not None and agents[str(event["agent"])] != agent_type:
            continue
        if role is not None and roles[event["agent"]] != role:
            continue
        if agent_id != -1 and agent_id != event["agent"]:
            continue
        if event["action"] == "message":
            chat.append(event["data"].strip())

    return " ".join(chat)


def plot_top_tokens(top_features_by_agent, agents=["human"], output_dir="./"):
    num_subplots = len(top_features_by_agent)
    categories = top_features_by_agent[0].keys()
    for cat in categories:
        plt.figure(1)
        for i in xrange(0, num_subplots):
            sp = int("{:d}1{:d}".format(num_subplots, i+1))
            plt.subplot(sp)

            top = top_features_by_agent[i][cat]
            tokens = [x[0] for x in top]
            scores = [x[1] for x in top]
            x_vals = [k+1 for k in xrange(len(top))]

            plt.xlabel("n-th most important token")
            plt.ylabel("tf-idf score")
            plt.scatter(x_vals, scores, c='m', marker='o', s=18)
            for (k, txt) in enumerate(tokens):
                plt.text(s="  "+txt, x=x_vals[k]+0.25, y=scores[k], rotation='vertical', ha='left')

            plt.gca().set_ylim(bottom=-0.05)
            plt.title("Agent type: {:s} Category: {:s}".format(agents[i], cat))

        plt.tight_layout()
        outpath = os.path.join(output_dir, "{:s}.png".format(cat))

        plt.savefig(outpath)
        plt.clf()


class TfIdfCalculator(object):
    """
    Computes tf-idf scores between a listing and all chats associated with that listing.
    """

    def __init__(self, grouped_chats, top_n=5, agent_type=None):
        self.grouped_chats = grouped_chats
        self.vectorizer = None
        self.listings = set()
        self.categories = set()
        self.tfs = None
        self.top_n = top_n
        self.agent_type = agent_type
        self.build(grouped_chats)

    def build(self, grouped_chats):
        self.categories = set(grouped_chats.keys())

        # self.listings = list(sorted(self.listings))
        self.categories = list(sorted(self.categories))
        self.grouped_chats = [grouped_chats[cat] for cat in self.categories]

        self.vectorizer = TfidfVectorizer(stop_words='english')
        self.tfs = self.vectorizer.fit_transform(self.grouped_chats)
        self.tfs = self.tfs.todense()

    def analyze(self):
        feature_names = self.vectorizer.get_feature_names()
        sorted_tfs = np.argsort(self.tfs, axis=1)
        top_features = {}
        for (idx, cat) in enumerate(self.categories):
            top_tfs = sorted_tfs[idx, :-self.top_n-1:-1].tolist()[0]
            top_features[cat] = [(feature_names[col], self.tfs[idx, col]) for col in top_tfs]

        for cat in self.categories:
            if self.agent_type is not None:
                print "Agent type: {:s}\t".format(self.agent_type),
            print "{:s}".format(cat)
            for (fname, col) in top_features[cat]:
                print "{:s}: {:.5f}".format(fname, col)
            print "---------------------------------------------------"

        return top_features
