from collections import defaultdict
import os

__author__ = 'anushabala'
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import matplotlib.pyplot as plt
from cocoa.turk.accept_negotiation_hits import is_chat_valid, is_partial_chat
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
        category = get_category(t)
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


def group_by_category_and_role(transcripts, agent_type=None):
    grouped_chats = defaultdict(str)
    for t in transcripts:
        if (not is_chat_valid(t, 0) and not is_partial_chat(t, 0)) \
                or (not is_chat_valid(t, 1) and not is_partial_chat(t, 1)):
            continue
        category = get_category(t)
        for r in ROLES:
            chat = json_to_text(t, agent_type, r)
            key = "{:s}_{:s}".format(category, r)
            grouped_chats[key] += u" {:s}".format(chat)
    return grouped_chats


def group_by_category_role_winner(transcripts, agent_type=None):
    grouped_chats = defaultdict(str)
    for t in transcripts:
        if (not is_chat_valid(t, 0) and not is_partial_chat(t, 0)) \
                or (not is_chat_valid(t, 1) and not is_partial_chat(t, 1)):
            continue
        roles = {
            0: t["scenario"]["kbs"][0]["personal"]["Role"],
            1: t["scenario"]["kbs"][1]["personal"]["Role"]
        }

        winner = get_winner(t)
        if winner is None or winner == -1:
            # ignore ties as well, there are so few of them
            continue
        category = get_category(t)
        chat_winner = json_to_text(t, agent_type, agent_id=winner)
        chat_loser = json_to_text(t, agent_type, agent_id=1-winner)
        if roles[winner] == BUYER:
            winner_key = "{:s}_{:s}_wins".format(category, BUYER)
            loser_key = "{:s}_{:s}_loses".format(category, SELLER)
        else:
            winner_key = "{:s}_{:s}_wins".format(category, SELLER)
            loser_key = "{:s}_{:s}_loses".format(category, BUYER)

        grouped_chats[winner_key] += u" {:s}".format(chat_winner)
        grouped_chats[loser_key] += u" {:s}".format(chat_loser)

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

    return u" ".join(chat)


def plot_top_tokens(top_features, agents=["human"], output_dir="./", suffix=""):
    num_subplots = len(top_features)
    categories = top_features[0].keys()
    for cat in categories:
        plt.clf()
        plt.figure(1)
        for i in xrange(0, num_subplots):
            sp = int("{:d}1{:d}".format(num_subplots, i+1))
            plt.subplot(sp)

            top = top_features[i][cat]
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
        outpath = os.path.join(output_dir, "{:s}{:s}.png".format(cat, suffix))

        plt.savefig(outpath)
        plt.clf()


def write_to_file(top_features, agents=["human"], output_dir="./", suffix=""):
    for (idx, agent) in enumerate(agents):
        top_features_by_cat = top_features[idx]
        for cat in top_features_by_cat.keys():
            outpath = os.path.join(output_dir, "top_ngrams_{:s}_{:s}{:s}.txt".format(agent, cat, suffix))
            outfile = open(outpath, 'w')
            for (ngram, score) in top_features_by_cat[cat]:
                outfile.write("{:.3f}\t{:s}\n".format(score, ngram))


class TfIdfCalculator(object):
    """
    Computes tf-idf scores between a listing and all chats associated with that listing.
    """

    def __init__(self, grouped_chats, top_n=5, ngrams=1, agent_type=None):
        self.grouped_chats = grouped_chats
        self.vectorizer = None
        self.listings = set()
        self.categories = set()
        self.tfs = None
        self.top_n = top_n
        self.agent_type = agent_type
        self.build(grouped_chats, n=ngrams)

    def build(self, grouped_chats, n=1):
        self.categories = set(grouped_chats.keys())

        # self.listings = list(sorted(self.listings))
        self.categories = list(sorted(self.categories))
        self.grouped_chats = [grouped_chats[cat] for cat in self.categories]

        self.vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(n, n))
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

    def plot_score_distribution(self, output_dir="/.", suffix=""):
        sorted_tfs = np.argsort(-self.tfs, axis=1)
        for (idx, cat) in enumerate(self.categories):
            plt.clf()
            top_tfs = sorted_tfs[idx, :].tolist()[0]
            tfs = [self.tfs[idx, col] for col in top_tfs]
            hist, bins = np.histogram(tfs)
            width = np.diff(bins)
            center = (bins[:-1] + bins[1:]) / 2

            fig, ax = plt.subplots(figsize=(8,3))
            ax.bar(center, hist, align='center', width=width)
            ax.set_xticks(bins)

            save_path = os.path.join(output_dir, '{:s}_distribution{:s}.png'.format(cat, suffix))
            plt.savefig(save_path)

